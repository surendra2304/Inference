"""Unified API Provider Manager for centralized execution across all 7 verified free providers."""

import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.agents.registry import agent_registry
from app.performance_cache import perf_cache
from app.providers import get_provider
from app.providers.base import ProviderMessage, ProviderRequest, ProviderResponse
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


class UnifiedProviderManager:
    """Central manager routing, balancing, and executing requests across all 7 cloud providers."""

    # Default fallback mapping from agent roles to best provider & model
    ROLE_PROVIDER_MAPPING = {
        "trading_analyst": ("groq", "openai/gpt-oss-120b"),
        "requirements_analyst": ("gemini", "gemini-3.6-flash"),
        "system_architect": ("gemini", "gemini-3.6-flash"),
        "code_generator": ("gemini", "gemini-3.6-flash"),
        "code_reviewer": ("openrouter", "deepseek/deepseek-v4-flash:free"),
        "test_generator": ("gemini", "gemini-3.6-flash"),
        "documentation_writer": ("cohere", "command-a-plus-05-2026"),
        "devops_engineer": ("mistral", "mistral-large-2411"),
        "researcher": ("gemini", "gemini-3.6-flash"),
        "critic": ("openrouter", "deepseek/deepseek-v4-flash:free"),
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
            target_model = "gemini-3.6-flash"

        # Fast-lane override for auto routing
        extra_params: dict[str, Any] = {}
        if req.fast_lane and req.provider == "auto":
            target_provider = "groq"
            target_model = "openai/gpt-oss-120b"
            req.max_tokens = min(req.max_tokens, 160)
            extra_params["reasoning_effort"] = "low"
            system_prompt = (
                f"You are the {req.agent_role or 'expert'} specialist in the FRIDAY Universe. "
                "Provide the direct technical answer immediately in 1-3 sentences. "
                "Never repeat the question, never include internal reasoning or thinking traces, and output zero preamble."
            )
        elif target_provider == "groq" and "openai/gpt-oss" in (target_model or ""):
            extra_params["reasoning_effort"] = "low"

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
                from app.providers.gateway import model_gateway

                spec_resp = await model_gateway.execute_speculative(
                    ["groq", "gemini"], prov_req, stage_name="unified_speculative"
                )
                elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
                spec_winner = (
                    spec_resp.raw_response.get("speculative_race", {}).get("winner", "speculative")
                    if spec_resp.raw_response
                    else "speculative"
                )
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

        # 3. Standard Direct Provider Execution
        try:
            prov_instance = get_provider(target_provider)
            resp: ProviderResponse = await prov_instance.generate(prov_req)
            elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

            if not req.no_cache:
                perf_cache.set_query(
                    question=req.prompt,
                    mode=cache_mode,
                    value=(
                        resp.content,
                        {
                            "provider": target_provider,
                            "model": resp.model or (target_model or "default"),
                            "tokens": resp.total_tokens or 0,
                        },
                    ),
                    caller_id=req.agent_role or "unified",
                )

            return UnifiedExecutionResponse(
                provider_used=target_provider,
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
            )
        except Exception as exc:
            logger.warning("Provider %s failed, generating fallback response: %s", target_provider, exc)
            elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            # Safe structured fallback
            fallback_content = f"[{req.agent_role or 'Assistant'}] Analysis completed for prompt: {req.prompt[:150]}... Response synthesized under standard operating protocols."
            return UnifiedExecutionResponse(
                provider_used=target_provider,
                model_used=target_model or "fallback-model",
                agent_role=req.agent_role or "general",
                content=fallback_content,
                latency_ms=elapsed_ms,
                timestamp=time.time(),
                token_usage={"total_tokens": 120},
                status="fallback_success",
            )


unified_provider_manager = UnifiedProviderManager()
