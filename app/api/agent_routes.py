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
    """Ultra-low latency inter-agent intelligence endpoint with L1 caching and speculative racing."""
    start_time = time.perf_counter()

    # Determine optimal specialist agent role
    role = _ROLE_MAP.get((req.caller_agent.lower(), req.task_type.lower()))
    if not role:
        role = "system_architect" if req.task_type in ("code", "architecture") else "trading_analyst"

    # Execute through unified provider manager with caching and speculative options
    exec_req = UnifiedExecutionRequest(
        provider="auto",
        agent_role=role,
        prompt=req.prompt,
        context=req.context,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        no_cache=req.no_cache,
        fast_lane=req.fast_lane,
        speculative=req.speculative,
    )

    exec_res = await unified_provider_manager.execute(exec_req)
    elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
    is_hit = exec_res.status == "cache_hit"

    _record_telemetry(req.caller_agent, elapsed_ms, is_hit)

    res_status: Literal["success", "cache_hit", "fallback_success"] = (
        "cache_hit"
        if exec_res.status == "cache_hit"
        else ("fallback_success" if exec_res.status == "fallback_success" else "success")
    )

    return AgentAssistResponse(
        status=res_status,
        caller_agent=req.caller_agent,
        agent_role=role,
        response=exec_res.content,
        provider_used=exec_res.provider_used,
        model_used=exec_res.model_used,
        latency_ms=elapsed_ms,
        cache_hit=is_hit,
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
