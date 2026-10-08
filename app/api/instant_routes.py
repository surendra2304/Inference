"""Instant Answer Fast-Path API for Inference.

Delivers sub-millisecond grounded and cached responses, or ~100-250ms
speculative LPU completions across Groq and Gemini Flash.
"""

import json
import time
from typing import Any

from fastapi import APIRouter, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.performance_cache import perf_cache
from app.providers.base import ProviderMessage, ProviderRequest
from app.providers.gateway import model_gateway
from app.utils.ids import generate_task_id
from app.utils.logger import logger

instant_router = APIRouter(prefix="/v1/instant", tags=["Instant Ultra-Low Latency Answers"])


class InstantRequest(BaseModel):
    """Payload for instant ultra-low latency queries."""

    prompt: str = Field(..., description="The user or agent query")
    caller_id: str = Field(default="human", description="Calling identity (e.g. human, friday, forge, sentinel)")
    no_cache: bool = Field(default=False, description="Bypass L1/L0 cache")
    max_tokens: int = Field(default=1024, ge=50, le=4096, description="Max tokens")


class InstantResponse(BaseModel):
    """Standardized response returned in sub-millisecond or sub-second time."""

    answer: str
    latency_ms: float
    source: str  # "instant_grounding", "l1_cache", "speculative_lpu", "fallback"
    # ``provider``/``model`` describe what ACTUALLY served the text. ``speculative_race``
    # carries the race bookkeeping separately, so a caller can see which candidate won
    # the race and whether the answer was in fact served locally after a fallback,
    # without the two facts being conflated into a vendor name that never ran.
    provider: str
    model: str
    task_id: str
    cached: bool
    speculative_race: dict[str, Any] | None = None


@instant_router.post("/ask", response_model=InstantResponse, status_code=status.HTTP_200_OK)
async def instant_ask(req: InstantRequest) -> InstantResponse:
    """Instant question answering endpoint with L0 grounding, L1 multi-tier cache, and speculative LPU execution."""
    start = time.perf_counter()
    task_id = generate_task_id()

    # 1. Check L0 Instant Grounding Knowledge Base (< 0.05ms)
    if not req.no_cache:
        grounded = perf_cache.get_grounded_answer(req.prompt)
        if grounded:
            lat_ms = round((time.perf_counter() - start) * 1000.0, 3)
            return InstantResponse(
                answer=grounded,
                latency_ms=lat_ms,
                source="instant_grounding",
                provider="local_grounding",
                model="instant-knowledge-core",
                task_id=task_id,
                cached=True,
            )

    # 2. Check L1 Multi-Tier Cache (< 0.02ms)
    if not req.no_cache:
        cached = perf_cache.get_query(req.prompt, mode="auto", caller_id=req.caller_id)
        if cached is not None:
            lat_ms = round((time.perf_counter() - start) * 1000.0, 3)
            ans = cached.get("answer", "") if isinstance(cached, dict) else str(cached)
            return InstantResponse(
                answer=ans,
                latency_ms=lat_ms,
                source="l1_cache",
                provider="cache",
                model="l1-in-memory",
                task_id=task_id,
                cached=True,
            )

    # 3. Speculative Fastest-Provider Racing (Groq + Gemini Flash)
    prov_req = ProviderRequest(
        messages=[ProviderMessage(role="user", content=req.prompt)],
        system_instruction="You are Inference, the ultra-low latency intelligence engine for the FRIDAY Universe. Be direct, accurate, and concise.",
        model="openai/gpt-oss-120b",
        temperature=0.3,
        max_tokens=req.max_tokens,
    )

    try:
        race_resp = await model_gateway.execute_speculative(
            ["groq", "gemini"], prov_req, stage_name="instant_race"
        )
        lat_ms = round((time.perf_counter() - start) * 1000.0, 2)
        race = (race_resp.raw_response or {}).get("speculative_race", {}) or {}
        # Report the provider that produced the content, never the raced candidate's
        # label. Measured before this fix: the local rig answered
        # ("GATEWAY: Served capability 'general' from the self-hosted tier") while the
        # response advertised provider="groq" / model="gemini-3.8-flash" — a vendor that
        # never ran. Anything billing or routing on this field was being told a fiction.
        served_by = race.get("served_by") or race_resp.provider

        if not req.no_cache:
            perf_cache.set_query(
                question=req.prompt,
                mode="auto",
                value={"answer": race_resp.content},
                caller_id=req.caller_id,
                ttl=180.0,
            )

        return InstantResponse(
            answer=race_resp.content,
            latency_ms=lat_ms,
            source="speculative_lpu" if not race.get("fell_back") else "speculative_lpu_local_fallback",
            provider=served_by,
            model=race_resp.model,
            task_id=task_id,
            cached=False,
            speculative_race=race or None,
        )
    except Exception as exc:
        logger.warning("Speculative race failed in instant_ask: %s", exc)
        lat_ms = round((time.perf_counter() - start) * 1000.0, 2)
        # Do not claim verification that never happened: this string is returned when no
        # model answered at all, and the previous wording ("Verified operational status
        # active") asserted a check that was never performed.
        fallback = (
            f"No instant answer available for: {req.prompt[:100]} "
            f"(no inference provider responded; this is a placeholder, not an answer)"
        )
        return InstantResponse(
            answer=fallback,
            latency_ms=lat_ms,
            source="fallback",
            provider="local_fallback",
            model="fallback-core",
            task_id=task_id,
            cached=False,
        )


@instant_router.post("/stream")
async def instant_stream(req: InstantRequest):
    """Real-time SSE token stream delivering lowest Time-To-First-Token."""
    # Check L0 / L1 first for 0ms token stream
    if not req.no_cache:
        grounded = perf_cache.get_grounded_answer(req.prompt)
        if grounded:
            async def _grounded_gen():
                yield f"data: {json.dumps({'token': grounded, 'done': True, 'source': 'instant_grounding'})}\n\n"

            return StreamingResponse(_grounded_gen(), media_type="text/event-stream")

    prov_req = ProviderRequest(
        messages=[ProviderMessage(role="user", content=req.prompt)],
        system_instruction="You are Inference, the ultra-low latency intelligence engine for the FRIDAY Universe. Be direct, accurate, and concise.",
        model="openai/gpt-oss-120b",
        temperature=0.3,
        max_tokens=req.max_tokens,
    )

    async def _event_gen():
        try:
            async for chunk in model_gateway.stream("groq", prov_req, stage_name="instant_stream"):
                yield f"data: {json.dumps({'token': chunk, 'done': False})}\n\n"
        except Exception as exc:
            logger.warning("Groq stream failed in instant_stream, falling back: %s", exc)
            try:
                prov_req.model = "gemini-3.6-flash"
                async for chunk in model_gateway.stream("gemini", prov_req, stage_name="instant_stream_fb"):
                    yield f"data: {json.dumps({'token': chunk, 'done': False})}\n\n"
                yield f"data: {json.dumps({'token': '', 'done': True})}\n\n"
            except Exception as fb_exc:
                yield f"data: {json.dumps({'error': str(fb_exc), 'done': True})}\n\n"

    return StreamingResponse(
        _event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
