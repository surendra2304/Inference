"""FastAPI Router for IntelX Deep Research Intelligence Endpoints."""

from fastapi import APIRouter, HTTPException, Path, status

from app.services.intelx_intelligence import (
    IntelXResearchRequest,
    IntelXResearchResponse,
    intelx_intelligence_service,
)
from app.utils.bounded_store import missing_entry_detail

intelx_router = APIRouter(prefix="/v1/intelx", tags=["IntelX Deep Research Intelligence"])


@intelx_router.post("/research", response_model=IntelXResearchResponse, status_code=status.HTTP_200_OK)
async def process_intelx_research_role(request: IntelXResearchRequest):
    """Processes role-specific research requests for IntelX (planner, extractor, verifier, analyst, critic, synthesizer)."""
    return await intelx_intelligence_service.execute_research_role(request)


@intelx_router.get("/research/{request_id}", status_code=status.HTTP_200_OK)
async def get_intelx_research_record(request_id: str = Path(..., description="Unique ID of previous IntelX research request")):
    """Retrieves full request and response record with provenance for research audit and verification."""
    record = intelx_intelligence_service.get_provenance(request_id)
    if not record:
        # Distinguishes "never recorded" from "recording expired out of the retention
        # window"; the two are different statements (see app/utils/bounded_store.py).
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=missing_entry_detail(intelx_intelligence_service.provenance_store, request_id, "research provenance record"),
        )
    return record
