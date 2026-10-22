"""FastAPI Router for Sentinel Security Intelligence Endpoints."""

from fastapi import APIRouter, HTTPException, Path, status

from app.services.sentinel_intelligence import (
    SentinelAnalysisRequest,
    SentinelAnalysisResponse,
    sentinel_intelligence_service,
)
from app.utils.bounded_store import missing_entry_detail

sentinel_router = APIRouter(prefix="/v1/sentinel", tags=["Sentinel Security Intelligence"])


@sentinel_router.post("/analyze", response_model=SentinelAnalysisResponse, status_code=status.HTTP_200_OK)
async def analyze_security_posture(request: SentinelAnalysisRequest):
    """Processes cybersecurity posture, vulnerability assessment, attack path reasoning, and remediation prioritization."""
    return await sentinel_intelligence_service.analyze(request)


@sentinel_router.get("/analyze/{request_id}", status_code=status.HTTP_200_OK)
async def get_sentinel_analysis_record(request_id: str = Path(..., description="Unique ID of previous Sentinel analysis request")):
    """Retrieves full request and response record with provenance for security audit and compliance."""
    record = sentinel_intelligence_service.get_provenance(request_id)
    if not record:
        # Distinguishes "never recorded" from "recording expired out of the retention
        # window"; the two are different statements (see app/utils/bounded_store.py).
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=missing_entry_detail(sentinel_intelligence_service.provenance_store, request_id, "security analysis provenance record"),
        )
    return record
