"""Unified API Provider Manager for centralized execution across all 7 verified free providers."""

import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.agents.registry import agent_registry
from app.performance_cache import perf_cache
from app.providers.base import ProviderMessage, ProviderRequest, ProviderResponse
from app.providers.gateway import model_gateway
from app.utils.logger import logger


class UnifiedExecutionRequest(BaseModel):
    """Universal execution request across all models and agents."""

    provider: Literal["auto", "gemini", "groq", "mistral", "openrouter", "nvidia", "cohere", "huggingface"] = Field(
        default="auto", description="Target provider name or 'auto' for intelligent capability matching"
    )
    agent_role: str | None = Field(
        default="system_architect",
        description="Specialist agent role or ID (e.g. trading_analyst, system_architect, code_generator, etc.)",
    )
    prompt: str = Field(..., description="Prompt or task instruction to execute")
    context: dict[str, Any] = Field(default_factory=dict, description="Additional structured context or parameters")
    max_tokens: int = Field(default=2000, ge=50, le=8192, description="Max output tokens")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    no_cache: bool = Field(default=False, description="Bypass L1 cache when fresh inference is required")
    fast_lane: bool = Field(default=False, description="Route to ultra-low latency providers (Groq / Gemini Flash)")
    speculative: bool = Field(default=False, description="Race top providers concurrently for lowest tail latency")


class UnifiedExecutionResponse(BaseModel):
    """Standardized response from unified provider manager."""

    provider_used: str
    model_used: str
    agent_role: str
    content: str
    latency_ms: float
    timestamp: float
    token_usage: dict[str, int] = Field(default_factory=dict)
    status: str = "success"
    # Honest degradation signalling: consumers MUST check this instead of assuming
    # a non-empty `content` means a model actually answered. When True, `content`
    # is empty and `error` carries the real reason no model could be reached.
    degraded: bool = False
    error: str | None = None
    # Why the model stopped generating ("stop", "length", ...). Callers that present
    # `content` as an answer need this: an answer cut off at the token ceiling is
    # materially incomplete and must be priced as such rather than published as if it
    # were finished.
    finish_reason: str | None = None
    # Set when the request was served by something other than the provider that was
    # asked for (the self-hosted tier under LOCAL_PREFERRED, or any fallback rung).
    served_by_provider: str | None = None
    # True when that substitution happened *by policy* rather than as fault recovery:
    # under LOCAL_PREFERRED the self-hosted tier is the intended primary, so serving
    # from it is not a degradation and must not be priced like one.
    served_by_policy: bool = False


class UnifiedProviderManager:
    """Central manager routing, balancing, and executing requests across all 7 cloud providers."""

    # Default fallback mapping from agent roles to best provider & model
    ROLE_PROVIDER_MAPPING = {
        "trading_analyst": ("groq", "openai/gpt-oss-120b"),
        "requirements_analyst": ("gemini", "gemini-3.8-flash"),
        "system_architect": ("nvidia", "nvidia/nemotron-3-super-120b-a12b"),
        "code_generator": ("groq", "openai/gpt-oss-120b"),
        "code_reviewer": ("gemini", "gemini-3.8-flash"),
        "test_generator": ("gemini", "gemini-3.8-flash"),
        "documentation_writer": ("gemini", "gemini-3.8-flash"),
        "devops_engineer": ("groq", "openai/gpt-oss-120b"),
        "researcher": ("gemini", "gemini-3.8-flash"),
        "critic": ("groq", "openai/gpt-oss-120b"),
    }

    async def execute(self, req: UnifiedExecutionRequest) -> UnifiedExecutionResponse:
        """Executes a unified prompt through designated/auto provider with L1 caching and speculative racing."""
        start_time = time.perf_counter()
        agent = None

        # 1. High-Speed L1 Cache Check (< 0.05ms)
        cache_mode = f"unified_{req.provider}_{req.agent_role or 'general'}"
        if not req.no_cache:
            cached_data = perf_cache.get_query(req.prompt, mode=cache_mode, caller_id=req.agent_role or "unified")
            if cached_data:
                cached_answer, cached_meta = cached_data
                cached_latency = round((time.perf_counter() - start_time) * 1000.0, 3)
                logger.info("Unified provider L1 cache hit for role '%s' in %.3fms", req.agent_role, cached_latency)
                return UnifiedExecutionResponse(
                    provider_used=cached_meta.get("provider", "cache"),
                    model_used=cached_meta.get("model", "l1-cached"),
                    agent_role=req.agent_role or "general",
                    content=cached_answer,
                    latency_ms=cached_latency,
                    timestamp=time.time(),
                    token_usage={"total_tokens": cached_meta.get("tokens", 0)},
                    status="cache_hit",
                )

        if req.agent_role:
            agent = agent_registry.get_agent(req.agent_role.lower())

        # Determine target provider and model
        target_provider: str = req.provider
        target_model = None
        system_prompt = "You are a helpful AI specialist in Inference."

        if agent:
            system_prompt = agent.system_instructions
            primary_config = agent.get_primary_model()
            if target_provider == "auto":
                target_provider = primary_config.provider
                target_model = primary_config.model
        elif req.agent_role and req.agent_role.lower() in self.ROLE_PROVIDER_MAPPING:
            default_prov, default_mod = self.ROLE_PROVIDER_MAPPING[req.agent_role.lower()]
            if target_provider == "auto":
                target_provider = default_prov
            target_model = default_mod
        elif target_provider == "auto":
            target_provider = "gemini"
            target_model = "gemini-3.8-flash"

        # Fast-lane override for auto routing
        extra_params: dict[str, Any] = {}
        if req.fast_lane and req.provider == "auto":
            target_provider = "groq"
            target_model = "openai/gpt-oss-20b"
            req.max_tokens = min(req.max_tokens, 60)
            extra_params["reasoning_effort"] = "low"
            system_prompt = (
                f"You are the {req.agent_role or 'expert'} specialist in the FRIDAY Universe. "
                "Provide the direct concise technical answer immediately in 1-2 sentences (maximum 40 words). "
                "Never repeat the question, never include internal reasoning or thinking traces, and output zero preamble."
            )
        elif target_provider == "groq" and "openai/gpt-oss" in (target_model or ""):
            extra_params["reasoning_effort"] = "low"

        # Capability drives the gateway's fallback model selection; derive it from the
        # agent that is speaking rather than defaulting everything to "general".
        capability = "general"
        if agent is not None:
            try:
                capability = agent.get_primary_model().capability or "general"
            except Exception:  # noqa: BLE001 - capability is advisory
                capability = "general"

        # Build provider request
        messages = [ProviderMessage(role="user", content=req.prompt)]
        if req.context:
            context_str = f"\n\nContext Metadata: {req.context}"
            messages[0].content += context_str

        prov_req = ProviderRequest(
            messages=messages,
            system_instruction=system_prompt,
            model=target_model,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            extra_params=extra_params,
        )

        # 2. Speculative Racing (Concurrent Execution across fastest providers)
        if req.speculative:
            try:
                spec_resp = await model_gateway.execute_speculative(
                    ["groq", "gemini"], prov_req, stage_name="unified_speculative"
                )
                elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
                race = (spec_resp.raw_response or {}).get("speculative_race", {}) or {}
                # ``winner`` is the raced candidate's LABEL; ``served_by`` is the provider
                # that actually produced the content. Under local preference or a missing
                # credential the raced label never runs, so reporting it as
                # ``provider_used`` attributed the work to a vendor that was never
                # called — and then wrote that fiction into the shared response cache.
                spec_winner = race.get("served_by") or spec_resp.provider or "speculative"
                if not req.no_cache:
                    perf_cache.set_query(
                        question=req.prompt,
                        mode=cache_mode,
                        value=(
                            spec_resp.content,
                            {
                                "provider": spec_winner,
                                "model": spec_resp.model,
                                "tokens": spec_resp.total_tokens or 0,
                            },
                        ),
                        caller_id=req.agent_role or "unified",
                    )
                return UnifiedExecutionResponse(
                    provider_used=spec_winner,
                    model_used=spec_resp.model or "speculative-model",
                    agent_role=req.agent_role or "general",
                    content=spec_resp.content,
                    latency_ms=elapsed_ms,
                    timestamp=time.time(),
                    token_usage={
                        "prompt_tokens": spec_resp.prompt_tokens or 0,
                        "completion_tokens": spec_resp.completion_tokens or 0,
                        "total_tokens": spec_resp.total_tokens or 0,
                    },
                    status="success",
                )
            except Exception as spec_exc:
                logger.warning("Speculative race in unified manager failed, falling back: %s", spec_exc)

        # 3. Standard Direct Provider Execution — routed through the ModelGateway.
        #
        # This used to call the provider adapter directly (``get_provider(...).generate``),
        # which quietly opted this API out of every guarantee the gateway provides:
        #
        #   * LOCAL_PREFERRED / LOCAL_ENABLED were ignored, so a deployment configured
        #     for self-hosted, no-egress operation could still send prompts to a cloud
        #     vendor — the exact opposite of what that setting is documented to
        #     guarantee. Measured: with the self-hosted tier running and healthy,
        #     POST /v1/ask produced **zero** model calls and returned a refusal, while
        #     POST /ask answered the same question from the local model in 0.01s.
        #   * provider health tracking, per-provider rate limiting, the fallback ladder
        #     and circuit-breaker behaviour did not apply to this route at all, so a
        #     failing provider here neither learned from nor contributed to the shared
        #     health state.
        #
        # Going through the gateway makes one provider stack govern every entry point.
        try:
            resp: ProviderResponse = await model_gateway.execute(
                provider_name=target_provider,
                request=prov_req,
                capability=capability,
                stage_name="unified_direct",
            )
            elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

            # Report the provider that actually served the request, not the one that was
            # asked for. Under LOCAL_PREFERRED (or any fallback) those differ, and
            # naming the requested provider would be a false claim about where the
            # prompt went.
            actual_provider = target_provider
            provenance = (resp.raw_response or {}).get("fallback_provenance") or {}
            if provenance.get("actual_provider"):
                actual_provider = str(provenance["actual_provider"])

            if not req.no_cache:
                perf_cache.set_query(
                    question=req.prompt,
                    mode=cache_mode,
                    value=(
                        resp.content,
                        {
                            "provider": actual_provider,
                            "model": resp.model or (target_model or "default"),
                            "tokens": resp.total_tokens or 0,
                        },
                    ),
                    caller_id=req.agent_role or "unified",
                )

            return UnifiedExecutionResponse(
                provider_used=actual_provider,
                model_used=resp.model or (target_model or "default"),
                agent_role=req.agent_role or "general",
                content=resp.content,
                latency_ms=elapsed_ms,
                timestamp=time.time(),
                token_usage={
                    "prompt_tokens": resp.prompt_tokens or 0,
                    "completion_tokens": resp.completion_tokens or 0,
                    "total_tokens": resp.total_tokens or 0,
                },
                status="success",
                finish_reason=resp.finish_reason,
                served_by_provider=actual_provider if actual_provider != target_provider else None,
                served_by_policy=bool(
                    provenance.get("fallback_reason") == "local_preferred"
                ),
            )
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            # NO SYNTHESIZED ANSWER. A provider failure is not a result: we return an
            # explicitly degraded response with empty content and no token counts so
            # callers cannot mistake "no model ran" for "the model answered".
            logger.warning(
                "Provider %s failed with no model output (degraded): %s", target_provider, exc
            )
            return UnifiedExecutionResponse(
                provider_used=target_provider,
                model_used=target_model or "none",
                agent_role=req.agent_role or "general",
                content="",
                latency_ms=elapsed_ms,
                timestamp=time.time(),
                token_usage={},
                status="degraded",
                degraded=True,
                error=f"{target_provider}: {type(exc).__name__}: {exc}",
            )


unified_provider_manager = UnifiedProviderManager()
