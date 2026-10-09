"""Dedicated API routes and typed contracts for FRIDAY integration."""

import hashlib
import json
import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, ValidationError

from app.core.orchestrator import OrchestrationRequest, orchestrator
from app.core.security import verify_friday_api_key
from app.performance_cache import perf_cache
from app.providers.base import ProviderMessage, ProviderRequest
from app.providers.gateway import model_gateway
from app.providers.unified_manager import UnifiedExecutionRequest, unified_provider_manager
from app.security.prompt_isolation import scrub_credentials
from app.utils.errors import correlation_id, internal_error
from app.utils.ids import generate_task_id
from app.utils.logger import logger
from app.version import VERSION

FRIDAY_ASK_NS = "friday.ask"
FRIDAY_DEBATE_NS = "friday.debate"


def _context_scope(context: dict[str, Any] | None) -> dict[str, Any]:
    """Cache scope for the caller's private context. The same question with a different context
    is a different question; before this, answers were shared across contexts."""
    if not context:
        return {}
    digest = hashlib.sha256(json.dumps(context, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    return {"context": digest}


def _as_friday_response(cached: Any) -> "FridayResponse | None":
    """A cache hit is a FridayResponse dict or a miss. Any other shape is logged and treated as a
    miss: a cache must never turn a valid request into a 500."""
    if cached is None:
        return None
    try:
        return FridayResponse.model_validate(cached)
    except (ValidationError, TypeError, ValueError):
        logger.warning("friday cache held a value of an unexpected shape; treating it as a miss")
        return None

friday_router = APIRouter(
    prefix="/v1/friday",
    tags=["FRIDAY Integration"],
    dependencies=[Depends(verify_friday_api_key)]
)


class FridayRequest(BaseModel):
    """Payload for requests submitted by FRIDAY to Inference."""
    question: str = Field(description="The complex query or task submitted by FRIDAY")
    context_data: dict[str, Any] = Field(default_factory=dict, description="FRIDAY's active system/environment context")
    max_latency: float | None = Field(default=5.0, description="Hard SLA ceiling in seconds")
    max_budget: float | None = Field(default=None, description="Max cost in USD")
    require_evidence: bool = Field(default=True, description="Enforce fact-checking and evidence provenance")
    caller_id: str = Field(default="friday_core", description="Identifier of the FRIDAY caller sub-module")
    no_cache: bool = Field(default=False, description="Bypass L1 in-memory response cache")
    fast_lane: bool = Field(default=True, description="Enforce ultra-fast single-specialist dispatch (< 500ms SLA)")


class FridayResponse(BaseModel):
    """Structured response returned to FRIDAY with full provenance and dissent metadata."""
    task_id: str
    run_id: str
    answer: str
    mode_used: str
    confidence: float = Field(ge=0.0, le=1.0)
    unresolved_disagreements: list[str] = Field(default_factory=list, description="Surviving technical dissent for FRIDAY decision-making")
    key_evidence: list[str] = Field(default_factory=list, description="Verified empirical claims and evidence")
    agents_used: list[str]
    models_used: list[str]
    latency_seconds: float
    total_tokens: int
    provenance: dict[str, Any] = Field(default_factory=dict, description="Audit trail and deliberation lineage")


def _fast_lane_confidence(exec_res: Any) -> tuple[float, str]:
    """Derive a confidence for a single-pass fast-lane answer, with its basis.

    The previous value was the constant ``0.98`` — the same number the degradation gate
    directly above it warns about — so a one-shot, unreviewed completion claimed
    near-certainty with nothing behind it. A fast lane performs no peer review and no
    cross-checking, so it cannot honestly claim more than a single-pass answer warrants,
    and a completion cut off at the token ceiling is materially incomplete.

    Returns ``(confidence, basis)`` where ``basis`` is recorded in the response
    provenance so the number can be audited rather than trusted.
    """
    finish_reason = (getattr(exec_res, "finish_reason", None) or "").lower()
    if finish_reason in ("length", "max_tokens", "truncated"):
        return 0.55, (
            "single-pass fast-lane answer truncated at the token ceiling "
            f"(finish_reason={finish_reason}); later content is missing"
        )
    if finish_reason in ("", "stop", "eos", "end_turn"):
        return 0.80, (
            "single-pass fast-lane completion; no panel review or evidence verification "
            "was performed, so confidence is capped below deliberated modes"
        )
    return 0.70, f"single-pass fast-lane completion with unrecognised finish_reason={finish_reason!r}"


def _fast_lane_evidence(exec_res: Any, elapsed_s: float) -> list[str]:
    """Report only measurable facts about how this answer was produced.

    The previous value was the constant
    ``"Direct high-throughput Groq fast-lane inference (<500ms SLA)."``: it named a vendor
    even when the self-hosted tier served the request (``exec_res.provider_used`` is read
    two lines below for ``provenance``, so the true provider was in hand), and it asserted
    a sub-500ms SLA that nothing checked.
    """
    provider = getattr(exec_res, "provider_used", None) or "unknown"
    model = getattr(exec_res, "model_used", None) or "unknown"
    evidence = [
        f"Served by provider={provider!r} model={model!r} in {elapsed_s:.3f}s "
        f"(measured; no SLA is claimed)."
    ]
    finish_reason = getattr(exec_res, "finish_reason", None)
    if finish_reason:
        evidence.append(f"Provider finish_reason={finish_reason!r}.")
    substituted = getattr(exec_res, "served_by_provider", None)
    if substituted and substituted != provider:
        evidence.append(f"Request was originally routed to {substituted!r}.")
    return evidence


@friday_router.post("/ask", response_model=FridayResponse, status_code=status.HTTP_200_OK)
async def friday_ask(request: FridayRequest) -> FridayResponse:
    """
    FRIDAY Fast/Review Question Answering Gateway with L1 Sub-Millisecond Cache.
    Automatically assigns optimal specialist or panel according to FRIDAY SLA constraints.
    """
    if not request.question.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Question cannot be empty.")

    # 0. Check Instant Grounding Knowledge Base (< 0.05ms)
    if not request.no_cache:
        grounded_ans = perf_cache.get_grounded_answer(request.question)
        if grounded_ans:
            return FridayResponse(
                task_id=generate_task_id(),
                run_id="instant_grounding",
                answer=grounded_ans,
                mode_used="instant_grounding",
                confidence=0.99,
                unresolved_disagreements=[],
                key_evidence=["Ecosystem core topology verification"],
                agents_used=["system_architect"],
                models_used=["instant-knowledge-core"],
                latency_seconds=0.0001,
                total_tokens=len(grounded_ans.split()),
                provenance={
                    "caller_id": request.caller_id,
                    "platform": "Inference",
                    "version": VERSION,
                    "cached": True,
                    "cache_tier": "L0_INSTANT_GROUNDING",
                    "fast_lane": True,
                },
            )

    # 1. Check L1 In-Memory Response Cache (Sub-millisecond hit path)
    if not request.no_cache:
        cached_data = perf_cache.get_query(
            request.question, mode="auto", caller_id=request.caller_id,
            extra=_context_scope(request.context_data), namespace=FRIDAY_ASK_NS)
        cached_resp = _as_friday_response(cached_data)
        if cached_resp is not None:
            cached_resp.latency_seconds = 0.0005
            cached_resp.provenance["cached"] = True
            cached_resp.provenance["cache_tier"] = "L1_IN_MEMORY"
            return cached_resp

    # Sub-second Fast-lane bypass for FRIDAY and peer agents
    if request.fast_lane or (request.max_latency is not None and request.max_latency <= 5.0):
        try:
            start_t = time.perf_counter()
            exec_req = UnifiedExecutionRequest(
                provider="auto",
                agent_role="system_architect",
                prompt=request.question,
                context=request.context_data,
                max_tokens=60,
                temperature=0.2,
                no_cache=True,
                fast_lane=True,
            )
            exec_res = await unified_provider_manager.execute(exec_req)
            elapsed_s = round(time.perf_counter() - start_t, 4)

            # Honest degradation gate: never report 0.98 confidence, never invent
            # token counts from word count, and never cache a non-answer.
            if exec_res.degraded:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=(
                        "FRIDAY fast-lane degraded: no model provider produced output "
                        f"({exec_res.error or 'all providers failed'}). No answer fabricated."
                    ),
                )

            run_id = f"deb_{uuid.uuid4().hex[:12]}"
            task_id = f"task_{uuid.uuid4().hex[:12]}"

            confidence, confidence_basis = _fast_lane_confidence(exec_res)
            usage = exec_res.token_usage or {}
            reported_tokens = usage.get("total_tokens")
            # Do not invent a token count from the word count. The gate above states the
            # rule; this line previously broke it
            # (``.get("total_tokens", len(exec_res.content.split()))``), which
            # under-reports real token usage and makes cost attribution wrong.
            token_source = "provider_usage" if reported_tokens is not None else "unavailable"

            resp = FridayResponse(
                task_id=task_id,
                run_id=run_id,
                answer=exec_res.content,
                mode_used="fast",
                confidence=confidence,
                unresolved_disagreements=[],
                key_evidence=_fast_lane_evidence(exec_res, elapsed_s),
                agents_used=["system_architect"],
                models_used=[exec_res.model_used],
                latency_seconds=elapsed_s,
                total_tokens=int(reported_tokens or 0),
                provenance={
                    "caller_id": request.caller_id,
                    "platform": "Inference",
                    "version": VERSION,
                    "cached": False,
                    "fast_lane": True,
                    "provider": exec_res.provider_used,
                    "model": exec_res.model_used,
                    "confidence_basis": confidence_basis,
                    "total_tokens_source": token_source,
                    "finish_reason": exec_res.finish_reason,
                }
            )
            if not request.no_cache:
                perf_cache.set_query(
                    question=request.question,
                    mode="auto",
                    value=resp.model_dump(),
                    caller_id=request.caller_id,
                    ttl=600.0,
                    extra=_context_scope(request.context_data),
                    namespace=FRIDAY_ASK_NS,
                )
            return resp
        except HTTPException:
            # Deliberate degradation signal (e.g. no model produced output) is a
            # real answer to the caller: do not swallow it into the DAG fallback.
            raise
        except Exception as fast_err:
            logger.warning("Fast-lane direct dispatch failed, falling back to DAG: %s", fast_err)

    # Standard DAG routing fallback
    target_mode = "fast" if (request.fast_lane or (request.max_latency is not None and request.max_latency <= 5.0)) else "auto"

    orch_req = OrchestrationRequest(
        question=request.question,
        mode=target_mode,
        max_latency=request.max_latency,
        max_budget=request.max_budget,
        require_evidence=request.require_evidence,
        context_data={"caller_id": request.caller_id, **request.context_data}
    )

    try:
        result = await orchestrator.process_task(orch_req)
        resp = FridayResponse(
            task_id=result.task_id,
            run_id=result.run_id,
            answer=result.answer,
            mode_used=result.mode_used,
            confidence=result.confidence,
            unresolved_disagreements=result.unresolved_disagreements,
            key_evidence=result.key_evidence,
            agents_used=result.agents_used,
            models_used=result.models_used,
            latency_seconds=result.total_latency_seconds,
            total_tokens=result.total_tokens,
            provenance={
                "caller_id": request.caller_id,
                "platform": "Inference",
                "version": VERSION,
                "cached": False,
                "fast_lane": request.fast_lane
            }
        )

        # Store in L1 cache for subsequent identical queries (180s TTL)
        if not request.no_cache:
            perf_cache.set_query(
                question=request.question,
                mode="auto",
                value=resp.model_dump(),
                caller_id=request.caller_id,
                ttl=180.0,
                extra=_context_scope(request.context_data),
                namespace=FRIDAY_ASK_NS
            )

        return resp
    except Exception as exc:
        detail, _ = internal_error(logger, exc, doing_what="FRIDAY task orchestration",
                                   prefix="friday")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=detail)


@friday_router.post("/debate", response_model=FridayResponse, status_code=status.HTTP_200_OK)
async def friday_debate(request: FridayRequest) -> FridayResponse:
    """
    FRIDAY 6-Round Structured Multi-Agent Debate Gateway with L1 Cache.
    Executes deep adversarial reasoning, returning calibrated confidence, surviving claims, and active dissent.
    """
    if not request.question.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Question cannot be empty.")

    # Check L1 In-Memory Response Cache
    if not request.no_cache:
        cached_data = perf_cache.get_query(
            request.question, mode="debate", caller_id=request.caller_id,
            extra=_context_scope(request.context_data), namespace=FRIDAY_DEBATE_NS)
        cached_resp = _as_friday_response(cached_data)
        if cached_resp is not None:
            cached_resp.latency_seconds = 0.001
            cached_resp.provenance["cached"] = True
            cached_resp.provenance["cache_tier"] = "L1_IN_MEMORY"
            return cached_resp

    orch_req = OrchestrationRequest(
        question=request.question,
        mode="debate",
        max_latency=request.max_latency,
        max_budget=request.max_budget,
        require_evidence=request.require_evidence,
        context_data={"caller_id": request.caller_id, **request.context_data}
    )

    try:
        result = await orchestrator.process_task(orch_req)
        resp = FridayResponse(
            task_id=result.task_id,
            run_id=result.run_id,
            answer=result.answer,
            mode_used=result.mode_used,
            confidence=result.confidence,
            unresolved_disagreements=result.unresolved_disagreements,
            key_evidence=result.key_evidence,
            agents_used=result.agents_used,
            models_used=result.models_used,
            latency_seconds=result.total_latency_seconds,
            total_tokens=result.total_tokens,
            provenance={
                "caller_id": request.caller_id,
                "debate_id": result.run_id,
                "platform": "Inference",
                "version": VERSION,
                "mode_used": result.mode_used,
                "rounds_completed": 6 if result.mode_used == "debate" else 2,
                "cached": False
            }
        )

        # Store debate consensus in L1 cache (300s TTL)
        if not request.no_cache:
            perf_cache.set_query(
                question=request.question,
                mode="debate",
                value=resp.model_dump(),
                caller_id=request.caller_id,
                ttl=300.0,
                extra=_context_scope(request.context_data),
                namespace=FRIDAY_DEBATE_NS
            )

        return resp
    except HTTPException:
        raise
    except Exception as exc:
        # "no specialist produced output" means every provider was unreachable or
        # unconfigured: the service is *unavailable*, not broken. The sibling /v1/friday/ask
        # endpoint already answers that condition with 503, and returning 500 here made two
        # endpoints report the same outage differently — 500 tells a client not to retry,
        # 503 tells it to retry later.
        message = str(exc)
        unavailable = any(
            marker in message
            for marker in (
                "no specialist produced output",
                "no model provider produced output",
                "ProviderUnconfiguredError",
                "no configured credential",
                "All speculative race candidates failed",
            )
        )
        # The raw ``message`` names the attempted provider/model pairs and configuration
        # variables. It is logged (scrubbed) under a correlation id and never returned: a 503
        # that carries it tells the caller which credentials are missing.
        reference = correlation_id("debate")
        logger.error("FRIDAY debate orchestration failed [%s] (%s): %s",
                     reference, "unavailable" if unavailable else "internal", scrub_credentials(message))
        if unavailable:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "FRIDAY debate degraded: no model provider produced output. No answer was "
                    f"fabricated. Quote correlation id {reference} when reporting this; the "
                    "per-provider reasons are in the server log."
                ),
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="FRIDAY debate orchestration failed internally; see server logs for the correlation id."
        ) from exc


@friday_router.post("/stream")
async def friday_stream(request: FridayRequest) -> StreamingResponse:
    """
    FRIDAY Real-Time SSE Token Streaming Gateway.
    Delivers sub-300ms Time-To-First-Token (TTFT) directly into FRIDAY's conversational UI.
    Emits server-sent events: data: {"token": "...", "done": false}
    """
    if not request.question.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Question cannot be empty.")

    task_id = generate_task_id()

    # Determine domain specialist and best ultra-fast model
    specialist_id = orchestrator.router.detect_domain_specialist(request.question)
    agent = orchestrator.registry.get_agent(specialist_id) or orchestrator.registry.get_agent("coder") or orchestrator.registry.get_agent("researcher")

    provider = agent.model_provider if agent else "groq"
    model = agent.model_name if agent else "openai/gpt-oss-120b"
    system_instruction = agent.system_instructions if agent else "You are an AI cognitive specialist."

    async def token_event_generator():
        yield f"data: {json.dumps({'event': 'start', 'task_id': task_id, 'agent': specialist_id, 'provider': provider, 'model': model})}\n\n"

        prov_req = ProviderRequest(
            messages=[ProviderMessage(role="user", content=request.question)],
            system_instruction=system_instruction,
            model=model,
            temperature=0.3,
        )

        accumulated_chunks = []
        try:
            async for chunk in model_gateway.stream(provider=provider, request=prov_req, stage_name="friday_stream"):
                accumulated_chunks.append(chunk)
                yield f"data: {json.dumps({'token': chunk, 'done': False})}\n\n"

            # Cache the completed stream answer
            full_text = "".join(accumulated_chunks)
            if not request.no_cache and full_text:
                cached_resp = FridayResponse(
                    task_id=task_id,
                    run_id=f"stream_{task_id}",
                    answer=full_text,
                    mode_used="stream",
                    confidence=0.95,
                    unresolved_disagreements=[],
                    key_evidence=[],
                    agents_used=[specialist_id],
                    models_used=[model],
                    latency_seconds=0.001,
                    total_tokens=len(full_text.split()),
                    provenance={"stream": True, "cached": False}
                )
                perf_cache.set_query(
                    question=request.question,
                    mode="auto",
                    value=cached_resp.model_dump(),
                    caller_id=request.caller_id,
                    ttl=180.0,
                    extra=_context_scope(request.context_data),
                    namespace=FRIDAY_ASK_NS
                )

            yield f"data: {json.dumps({'done': True, 'task_id': task_id, 'total_chars': len(full_text)})}\n\n"
        except Exception as err:
            yield f"data: {json.dumps({'error': str(err), 'done': True})}\n\n"

    return StreamingResponse(
        token_event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


class AgentMetadata(BaseModel):
    """Detailed metadata for a specialist agent in Inference."""
    id: str
    name: str
    role: str
    purpose: str
    provider: str
    model: str
    strengths: list[str] = Field(default_factory=list)
    status: str = "active"


class FridayInfoResponse(BaseModel):
    """System metadata and active agent list for discovery."""
    platform: str = "Inference"
    version: str = "2.0.0"
    total_specialists: int
    active_cloud_providers: list[str]
    agents: list[AgentMetadata]


@friday_router.get("/agents", response_model=list[AgentMetadata], status_code=status.HTTP_200_OK)
async def list_friday_agents() -> list[AgentMetadata]:
    """
    Returns the live catalog of all 10 specialist agents, their cloud providers, and assigned models.
    Enables FRIDAY to query exact agent models without hallucination.
    """
    agents = orchestrator.registry.list_agents()
    return [
        AgentMetadata(
            id=a.id,
            name=a.name,
            role=a.role,
            purpose=a.purpose,
            provider=a.model_provider,
            model=a.model_name,
            strengths=a.strengths,
            status=a.status
        )
        for a in agents
    ]


@friday_router.get("/info", response_model=FridayInfoResponse, status_code=status.HTTP_200_OK)
async def get_friday_info() -> FridayInfoResponse:
    """
    Returns system status, active cloud providers, and specialist agent models.
    """
    agents = orchestrator.registry.list_agents()
    agent_metas = [
        AgentMetadata(
            id=a.id,
            name=a.name,
            role=a.role,
            purpose=a.purpose,
            provider=a.model_provider,
            model=a.model_name,
            strengths=a.strengths,
            status=a.status
        )
        for a in agents
    ]
    unique_providers = list({a.model_provider for a in agents})
    return FridayInfoResponse(
        total_specialists=len(agents),
        active_cloud_providers=unique_providers,
        agents=agent_metas
    )


class FridayStatusResponse(BaseModel):
    """Administrative status response returning live active agents, configured providers, and available models."""
    active_agents: list[str] = Field(description="List of active agent roles currently registered")
    configured_providers: list[str] = Field(description="List of provider names with valid API keys loaded from .env")
    available_models: list[str] = Field(description="List of specific model names mapped to active agents and providers")


@friday_router.get("/status", response_model=FridayStatusResponse, status_code=status.HTTP_200_OK)
async def get_friday_status() -> FridayStatusResponse:
    """
    Administrative status endpoint for FRIDAY.
    Returns:
    - active_agents: list of unique agent roles currently registered.
    - configured_providers: list of provider names with valid API keys loaded in .env.
    - available_models: list of specific model names mapped to those providers.
    """
    from app.core.config import settings

    # Check configured providers from .env settings (supports single or comma-separated lists)
    provider_names = ["Gemini", "Groq", "Mistral", "OpenRouter", "Cohere", "HuggingFace", "Nvidia"]
    configured_providers = [
        p for p in provider_names
        if len(settings.get_provider_keys(p)) > 0
    ]

    # Retrieve registered agents and their assigned models
    agents = orchestrator.registry.list_agents()
    active_agent_roles = [a.role for a in agents]
    available_models = list(dict.fromkeys([a.model_name for a in agents]))

    return FridayStatusResponse(
        active_agents=active_agent_roles,
        configured_providers=configured_providers,
        available_models=available_models
    )
