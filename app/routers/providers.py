"""FastAPI Router for Unified Provider Execution serving Trading and Software Engineering."""

from fastapi import APIRouter, status

from app.providers.unified_manager import (
    UnifiedExecutionRequest,
    UnifiedExecutionResponse,
    unified_provider_manager,
)

providers_router = APIRouter(prefix="/v1/providers", tags=["Unified Provider Management"])


@providers_router.post("/execute", response_model=UnifiedExecutionResponse, status_code=status.HTTP_200_OK)
async def execute_provider_request(req: UnifiedExecutionRequest):
    """
    Unified execution endpoint serving both algorithmic trading bots and FORGE software engineering.
    Routes to the requested or optimal cloud provider (Gemini, Groq, Mistral, OpenRouter, NVIDIA, Cohere, HuggingFace).
    """
    return await unified_provider_manager.execute(req)
