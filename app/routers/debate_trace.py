"""FastAPI Router for Enhanced Debate Protocol and Reasoning Chain Explainability."""

from fastapi import APIRouter, HTTPException, Path, status

from app.debate.enhanced_debate_protocol import enhanced_debate_engine
from app.utils.bounded_store import missing_entry_detail

debate_router = APIRouter(prefix="/v1", tags=["Debate Protocol & Explainability"])


@debate_router.get("/intelligence/{request_id}/trace", status_code=status.HTTP_200_OK)
async def get_intelligence_reasoning_trace(request_id: str = Path(..., description="Request ID of debate intelligence query")):
    """Retrieves full 4-round reasoning chain trace, cross-examinations, assumptions, and evidence scores."""
    trace = enhanced_debate_engine.get_trace(request_id)
    if not trace:
        # Evicted and never-recorded are different answers, and the retention window is
        # reported so the caller can tell which one they are looking at.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=missing_entry_detail(
                enhanced_debate_engine.reasoning_traces, request_id, "reasoning chain trace"
            ),
        )
    return trace.model_dump()


@debate_router.get("/debate/statistics", status_code=status.HTTP_200_OK)
async def get_debate_statistics():
    """Returns empirical statistics on debate compositions, provider diversity impact, and objection rates."""
    return enhanced_debate_engine.get_debate_statistics()
