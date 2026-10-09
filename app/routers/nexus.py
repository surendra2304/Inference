"""FastAPI Router for Nexus Intelligence Endpoints."""

from fastapi import APIRouter, HTTPException, Path, status

from app.services.nexus_intelligence import (
    IntelligenceRequest,
    IntelligenceResponse,
    nexus_intelligence_service,
)
from app.utils.bounded_store import missing_entry_detail

nexus_router = APIRouter(prefix="/v1/nexus", tags=["Nexus Intelligence"])


@nexus_router.post("/intelligence", response_model=IntelligenceResponse, status_code=status.HTTP_200_OK)
async def process_nexus_intelligence(request: IntelligenceRequest):
    """Processes structured multi-mode intelligence requests with calibrated confidence and full provenance."""
    return await nexus_intelligence_service.process_request(request)


@nexus_router.get("/intelligence/{request_id}", status_code=status.HTTP_200_OK)
async def get_nexus_intelligence_record(request_id: str = Path(..., description="Unique ID of previous intelligence request")):
    """Retrieves full request and response record with complete provenance for audit and explanation."""
    record = nexus_intelligence_service.get_provenance(request_id)
    if not record:
        # Distinguishes "never recorded" from "recording expired out of the retention
        # window"; the two are different statements (see app/utils/bounded_store.py).
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=missing_entry_detail(nexus_intelligence_service.provenance_store, request_id, "intelligence provenance record"),
        )
    return record
