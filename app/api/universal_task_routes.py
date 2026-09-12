"""Universal Task Execution and Deliberative Reasoning Routes for FRIDAY Universe."""

from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import APIRouter, Header, HTTPException, status
from pydantic import BaseModel, Field

from app.core.config import settings
from app.providers.unified_manager import UnifiedExecutionRequest, unified_provider_manager
from app.utils.logger import logger

task_router = APIRouter(tags=["FRIDAY Universe Universal Task Protocol"])


class TaskEnvelopeModel(BaseModel):
    """Universal Task Envelope accepted from any agent in the FRIDAY Universe."""

    task_id: str = Field(default_factory=lambda: f"task_{uuid.uuid4().hex[:12]}")
    source_agent: str = Field(default="friday")
    target_agent: str = Field(default="inference")
    action: str = Field(default="reason")
    payload: dict[str, Any] = Field(default_factory=dict)
    priority: str = Field(default="normal")
    created_at: str | None = None
    trace_id: str | None = None


class TaskResultModel(BaseModel):
    """Universal Task Result returned to calling agent."""

    task_id: str
    target_agent: str = "inference"
    status: str = "SUCCESS"
    result: dict[str, Any] = Field(default_factory=dict)
    summary: str = ""
    error: str | None = None
    execution_time_ms: int = 0


class AstraReasonRequest(BaseModel):
    """Deliberative reasoning consultation request."""

    prompt: str = Field(..., description="The query or problem requiring multi-perspective reasoning")
    context: dict[str, Any] = Field(default_factory=dict)
    mode: str = Field(default="deliberative", description="deliberative | consensus | fast")
    caller: str = Field(default="friday")


class AstraReasonResponse(BaseModel):
    """Deliberative reasoning synthesis."""

    task_id: str
    synthesis: str
    confidence: float
    model_used: str
    provider_used: str
    latency_ms: int
    perspective: str


@task_router.post("/v1/task/execute", response_model=TaskResultModel, status_code=status.HTTP_200_OK)
async def execute_task(
    envelope: TaskEnvelopeModel,
    x_friday_api_key: str | None = Header(None, alias="X-FRIDAY-API-Key"),
    authorization: str | None = Header(None),
) -> TaskResultModel:
    """Execute a structured TaskEnvelope from any peer agent in the FRIDAY Universe."""
    t0 = time.time()
    logger.info(f"[TASK_EXECUTE] Received action '{envelope.action}' from '{envelope.source_agent}'")

    # Extract prompt from payload
    prompt = (
        envelope.payload.get("prompt")
        or envelope.payload.get("question")
        or envelope.payload.get("query")
        or str(envelope.payload)
    )

    try:
        req = UnifiedExecutionRequest(
            provider="auto",
            agent_role="system_architect",
            prompt=prompt,
            context=envelope.payload.get("context", {}),
            fast_lane=True,
            max_tokens=1500,
        )
        resp = await unified_provider_manager.execute(req)
        lat = int((time.time() - t0) * 1000)

        return TaskResultModel(
            task_id=envelope.task_id,
            target_agent="inference",
            status="SUCCESS",
            result={
                "response": resp.content,
                "model_used": resp.model_used,
                "provider_used": resp.provider_used,
                "token_usage": resp.token_usage,
            },
            summary=resp.content[:250],
            execution_time_ms=lat,
        )
    except Exception as e:
        lat = int((time.time() - t0) * 1000)
        logger.error(f"[TASK_EXECUTE] Execution error: {e}")
        return TaskResultModel(
            task_id=envelope.task_id,
            target_agent="inference",
            status="ERROR",
            error=str(e),
            execution_time_ms=lat,
        )


@task_router.post("/v1/astra/reason", response_model=AstraReasonResponse, status_code=status.HTTP_200_OK)
async def astra_reason(request: AstraReasonRequest) -> AstraReasonResponse:
    """Deliberative high-level reasoning gateway using optimal free model pool."""
    t0 = time.time()
    logger.info(f"[ASTRA_REASON] Reasoning query from '{request.caller}': {request.prompt[:80]}")

    try:
        req = UnifiedExecutionRequest(
            provider="auto",
            agent_role="system_architect",
            prompt=f"[ASTRA DELIBERATIVE REASONING]\nEvaluate context and provide structured resolution:\n{request.prompt}",
            context=request.context,
            fast_lane=request.mode == "fast",
            max_tokens=2000,
        )
        resp = await unified_provider_manager.execute(req)
        lat = int((time.time() - t0) * 1000)

        return AstraReasonResponse(
            task_id=f"astra_{uuid.uuid4().hex[:8]}",
            synthesis=resp.content,
            confidence=0.95,
            model_used=resp.model_used,
            provider_used=resp.provider_used,
            latency_ms=lat,
            perspective="Central Reasoning Council (ASTRA)",
        )
    except Exception as e:
        logger.error(f"[ASTRA_REASON] Error: {e}")
        lat = int((time.time() - t0) * 1000)
        return AstraReasonResponse(
            task_id=f"astra_err_{uuid.uuid4().hex[:8]}",
            synthesis=f"Reasoning failure: {e}",
            confidence=0.0,
            model_used="none",
            provider_used="none",
            latency_ms=lat,
            perspective="Error Handler",
        )
