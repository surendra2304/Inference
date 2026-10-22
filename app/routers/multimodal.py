"""FastAPI Router for Multi-Modal Intelligence, Temporal Analysis, and Counterfactual Reasoning."""

from fastapi import APIRouter, status

from app.services.multimodal_intelligence import (
    MultiModalIntelligenceRequest,
    MultiModalIntelligenceResponse,
    multimodal_service,
)

multimodal_router = APIRouter(prefix="/v1/intelligence", tags=["Multi-Modal & Advanced Intelligence"])


@multimodal_router.post("/multimodal", response_model=MultiModalIntelligenceResponse, status_code=status.HTTP_200_OK)
async def process_multimodal_intelligence(request: MultiModalIntelligenceRequest):
    """Processes multi-modal intelligence requests supporting text, code, structured data, URLs, and images."""
    return await multimodal_service.analyze_multimodal(request)
