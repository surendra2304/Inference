"""FastAPI Router for Futuris Predictive Forecasting and Statistical Grounding."""

from fastapi import APIRouter, HTTPException, Path, status

from app.services.futuris_enhancement import (
    FuturisEnhanceRequest,
    FuturisEnhanceResponse,
    futuris_enhancement_service,
)
from app.utils.bounded_store import missing_entry_detail

futuris_router = APIRouter(prefix="/v1/futuris", tags=["Futuris Statistical Grounding"])


@futuris_router.post("/enhance", response_model=FuturisEnhanceResponse, status_code=status.HTTP_200_OK)
async def enhance_statistical_forecast(request: FuturisEnhanceRequest):
    """Enhances raw quantitative/statistical forecasts with multi-agent qualitative analysis and risk adjustment."""
    return await futuris_enhancement_service.enhance_forecast(request)


@futuris_router.get("/enhance/{request_id}", status_code=status.HTTP_200_OK)
async def get_futuris_enhancement_record(request_id: str = Path(..., description="Unique ID of previous Futuris enhancement request")):
    """Retrieves full request and response record with provenance for statistical audit."""
    record = futuris_enhancement_service.get_provenance(request_id)
    if not record:
        # Distinguishes "never recorded" from "recording expired out of the retention
        # window"; the two are different statements (see app/utils/bounded_store.py).
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=missing_entry_detail(futuris_enhancement_service.provenance_store, request_id, "forecast provenance record"),
        )
    return record
