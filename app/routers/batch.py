"""Batch Code Generation Endpoint for FORGE."""

import asyncio
import time
from typing import Any

from fastapi import APIRouter, status
from pydantic import BaseModel, Field

from app.services.code_generation import (
    CodeGenerationRequest,
    CodeGenerationResponse,
    code_generation_service,
)
from app.utils.errors import internal_error
from app.utils.logger import logger

batch_router = APIRouter(prefix="/v1/forge", tags=["FORGE Batch Processing"])


class BatchGenerateRequest(BaseModel):
    requests: list[CodeGenerationRequest] = Field(..., max_length=10, description="Up to 10 parallel code generation requests")


class BatchGenerateResponse(BaseModel):
    results: list[CodeGenerationResponse]
    failed: list[dict[str, Any]]
    total_latency_ms: float
    total_tokens: int


@batch_router.post("/batch-generate", response_model=BatchGenerateResponse, status_code=status.HTTP_200_OK)
async def batch_generate_code(req: BatchGenerateRequest):
    """Processes up to 10 code generation requests in parallel with partial failure resilience."""
    start_time = time.perf_counter()
    results: list[CodeGenerationResponse] = []
    failed: list[dict[str, Any]] = []

    async def _process_item(item: CodeGenerationRequest):
        try:
            return await code_generation_service.generate_code(item)
        except Exception as exc:
            detail, _reference = internal_error(
                logger, exc, doing_what="code generation", prefix="batch", extra=item.filename,
            )
            return {"filename": item.filename, "error": detail}

    outcomes = await asyncio.gather(*[_process_item(r) for r in req.requests])

    total_tokens = 0
    for outcome in outcomes:
        if isinstance(outcome, CodeGenerationResponse):
            results.append(outcome)
            total_tokens += outcome.token_usage
        else:
            failed.append(outcome)

    elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
    return BatchGenerateResponse(
        results=results,
        failed=failed,
        total_latency_ms=elapsed_ms,
        total_tokens=total_tokens
    )
