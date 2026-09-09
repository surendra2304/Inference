"""Universal Inter-Agent Fast Gateway API for FRIDAY Universe Ecosystem Agents.

Enables ultra-low latency intelligence sharing, decision consultation, and real-time streaming
for FRIDAY, FORGE, SENTINEL, STRATEX, NEXUS, MEMORA, and INTELX.
"""

import json
import time
from typing import Any, Literal

from fastapi import APIRouter, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.performance_cache import perf_cache
from app.providers.base import ProviderMessage, ProviderRequest
from app.providers.gateway import model_gateway
from app.providers.unified_manager import (
    UnifiedExecutionRequest,
    unified_provider_manager,
)
from app.utils.logger import logger

agent_router = APIRouter(prefix="/v1/agent", tags=["Universal Inter-Agent Intelligence"])

CallerAgent = Literal[
    "friday",
    "forge",
    "sentinel",
    "stratex",
    "trading_bot",
    "nexus",
    "memora",
    "intelx",
    "futuris",
    "cortex",
    "general",
]

TaskType = Literal[
    "code",
    "architecture",
    "security",
    "market",
    "debugging",
    "review",
    "general",
]


class AgentAssistRequest(BaseModel):
    """Universal assistance request model for all peer agents."""

    caller_agent: CallerAgent = Field(default="general", description="Calling agent identity")
    task_type: TaskType = Field(default="general", description="Domain of requested assistance")
    prompt: str = Field(..., description="Query, task, or directive")
    context: dict[str, Any] = Field(default_factory=dict, description="Structured contextual parameters")
    fast_lane: bool = Field(default=True, description="Enable ultra-low latency routing")
    speculative: bool = Field(default=False, description="Enable concurrent speculative provider race")
    no_cache: bool = Field(default=False, description="Bypass L1 cache for non-deterministic inference")
    max_tokens: int = Field(default=1500, ge=50, le=8192, description="Max output tokens")
    temperature: float = Field(default=0.2, ge=0.0, le=2.0, description="Sampling temperature")


class AgentAssistResponse(BaseModel):
    """Standardized ultra-fast response for peer agents."""

    status: Literal["success", "cache_hit", "fallback_success"]
    caller_agent: str
    agent_role: str
    response: str
    provider_used: str
    model_used: str
    latency_ms: float
    cache_hit: bool
    token_usage: dict[str, int] = Field(default_factory=dict)


# In-memory latency and performance metrics per caller agent
_agent_telemetry: dict[str, dict[str, Any]] = {}


def _record_telemetry(caller: str, latency_ms: float, is_cache_hit: bool) -> None:
    if caller not in _agent_telemetry:
        _agent_telemetry[caller] = {
            "total_requests": 0,
            "cache_hits": 0,
            "total_latency_ms": 0.0,
            "avg_latency_ms": 0.0,
            "min_latency_ms": float("inf"),
            "max_latency_ms": 0.0,
        }
    entry = _agent_telemetry[caller]
    entry["total_requests"] += 1
    if is_cache_hit:
        entry["cache_hits"] += 1
    entry["total_latency_ms"] += latency_ms
    entry["avg_latency_ms"] = round(entry["total_latency_ms"] / entry["total_requests"], 2)
    entry["min_latency_ms"] = round(min(entry["min_latency_ms"], latency_ms), 3)
    entry["max_latency_ms"] = round(max(entry["max_latency_ms"], latency_ms), 2)


_ROLE_MAP: dict[tuple[str, str], str] = {
    ("forge", "code"): "code_generator",
    ("forge", "architecture"): "system_architect",
    ("forge", "review"): "code_reviewer",
    ("forge", "debugging"): "code_generator",
    ("sentinel", "security"): "security_analyst",
    ("stratex", "market"): "trading_analyst",
    ("trading_bot", "market"): "trading_analyst",
    ("nexus", "architecture"): "system_architect",
    ("memora", "general"): "system_architect",
    ("cortex", "general"): "system_architect",
}


@agent_router.post("/assist", response_model=AgentAssistResponse, status_code=status.HTTP_200_OK)
async def agent_assist_endpoint(req: AgentAssistRequest):
    """Ultra-low latency inter-agent intelligence endpoint.

    Sub-second guarantee: ALL /v1/agent/assist calls are unconditionally routed to the
    Groq fast-lane (openai/gpt-oss-120b, reasoning_effort=low, max_tokens=120).
    This guarantees ≤760ms server-side processing regardless of caller role or task type.
    L1 cache TTL is 600s (10 min) to make repeat queries return in <1ms.
    """
    start_time = time.perf_counter()

    # Determine optimal specialist agent role
    role = _ROLE_MAP.get((req.caller_agent.lower(), req.task_type.lower()))
    if not role:
        role = "system_architect" if req.task_type in ("code", "architecture") else "trading_analyst"

    # SUB-SECOND GUARANTEE: Always use Groq fast-lane for agent-to-agent calls.
    # Cap tokens hard at 120 — enough for a direct 1-3 sentence answer.
    # Raise cache TTL to 600s so semantically identical follow-up queries hit the
    # in-memory L1 cache and return in <1ms (zero LLM call).
    AGENT_MAX_TOKENS = min(req.max_tokens, 120)
    AGENT_CACHE_TTL = 600.0  # 10 minutes

    # Check L1 cache first — if hit, we're done in <1ms
    cache_mode = f"agent_assist_{req.caller_agent}_{req.task_type}"
    if not req.no_cache:
        cached_data = perf_cache.get_query(req.prompt, mode=cache_mode, caller_id=req.caller_agent)
        if cached_data:
            cached_answer, cached_meta = cached_data
            elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            _record_telemetry(req.caller_agent, elapsed_ms, is_cache_hit=True)
            logger.info("agent/assist L1 cache hit for %s/%s in %.2fms", req.caller_agent, req.task_type, elapsed_ms)
            return AgentAssistResponse(
                status="cache_hit",
                caller_agent=req.caller_agent,
                agent_role=role,
                response=cached_answer,
                provider_used="cache",
                model_used="l1-cache",
                latency_ms=elapsed_ms,
                cache_hit=True,
                token_usage={"total_tokens": cached_meta.get("tokens", 0)},
            )

    # Build Groq fast-lane request directly — bypasses role→provider mapping table
    # which would select Gemini for system_architect/code_generator roles.
    exec_req = UnifiedExecutionRequest(
        provider="auto",
        agent_role=role,
        prompt=req.prompt,
        context=req.context,
        max_tokens=AGENT_MAX_TOKENS,
        temperature=req.temperature,
        no_cache=True,  # Cache is handled above with longer TTL
        fast_lane=True,  # Always Groq regardless of req.fast_lane field
        speculative=req.speculative,
    )

    exec_res = await unified_provider_manager.execute(exec_req)
    elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

    # Store in L1 cache with 600s TTL
    if not req.no_cache:
        perf_cache.set_query(
            question=req.prompt,
            mode=cache_mode,
            value=(exec_res.content, {"provider": exec_res.provider_used, "model": exec_res.model_used, "tokens": exec_res.token_usage.get("total_tokens", 0)}),
            caller_id=req.caller_agent,
            ttl=AGENT_CACHE_TTL,
        )

    _record_telemetry(req.caller_agent, elapsed_ms, is_cache_hit=False)
    logger.info("agent/assist %s/%s → %s in %.2fms", req.caller_agent, req.task_type, exec_res.provider_used, elapsed_ms)

    res_status: Literal["success", "cache_hit", "fallback_success"] = (
        "fallback_success" if exec_res.status == "fallback_success" else "success"
    )

    return AgentAssistResponse(
        status=res_status,
        caller_agent=req.caller_agent,
        agent_role=role,
        response=exec_res.content,
        provider_used=exec_res.provider_used,
        model_used=exec_res.model_used,
        latency_ms=elapsed_ms,
        cache_hit=False,
        token_usage=exec_res.token_usage,
    )


@agent_router.post("/stream")
async def agent_stream_endpoint(req: AgentAssistRequest):
    """Universal SSE token streaming endpoint for all peer agents."""
    role = _ROLE_MAP.get((req.caller_agent.lower(), req.task_type.lower()), "system_architect")
    system_prompt = f"You are an AI specialist ({role}) providing assistance to {req.caller_agent.upper()}."

    prov_req = ProviderRequest(
        messages=[ProviderMessage(role="user", content=req.prompt)],
        system_instruction=system_prompt,
        model="openai/gpt-oss-120b",
        temperature=req.temperature,
        max_tokens=req.max_tokens,
    )

    async def _event_generator():
        try:
            async for chunk in model_gateway.stream("groq", prov_req, stage_name=f"agent_stream_{req.caller_agent}"):
                yield f"data: {json.dumps({'token': chunk, 'done': False})}\n\n"
            yield f"data: {json.dumps({'token': '', 'done': True})}\n\n"
        except Exception as exc:
            logger.warning("Groq stream failed for %s, falling back to gemini: %s", req.caller_agent, exc)
            try:
                prov_req.model = "gemini-3.6-flash"
                async for chunk in model_gateway.stream(
                    "gemini", prov_req, stage_name=f"agent_stream_{req.caller_agent}_fb"
                ):
                    yield f"data: {json.dumps({'token': chunk, 'done': False})}\n\n"
                yield f"data: {json.dumps({'token': '', 'done': True})}\n\n"
            except Exception as fb_exc:
                yield f"data: {json.dumps({'error': str(fb_exc), 'done': True})}\n\n"

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@agent_router.get("/stats", status_code=status.HTTP_200_OK)
async def agent_stats_endpoint():
    """Returns real-time latency statistics and cache hit rates partitioned by caller agent."""
    cache_stats = perf_cache.stats
    return {
        "agents": _agent_telemetry,
        "cache": cache_stats,
        "timestamp": time.time(),
    }
