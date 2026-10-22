"""FastAPI Router for Ecosystem Intelligence, Continuous Learning, Meta-Intelligence, and Ecosystem Consultations."""

from fastapi import APIRouter, status
from pydantic import BaseModel, Field

from app.intelligence.meta_intel import meta_intelligence
from app.learning.continuous_learning import continuous_learning_engine
from app.services.ecosystem_intel import ecosystem_hub

ecosystem_router = APIRouter(prefix="/v1/ecosystem", tags=["Ecosystem Intelligence & Continuous Learning"])


class EcosystemConsultRequest(BaseModel):
    portfolio_positions: dict[str, float] = Field(
        default={"BTCUSDT": 20000.0, "ETHUSDT": 10000.0, "SOLUSDT": 4000.0},
        description="Active portfolio USD notional positions"
    )
    active_strategies: list[str] = Field(
        default=["ADX_EMA_Trend", "Bollinger_Reversion", "ML_Ensemble"],
        description="List of currently executing strategies"
    )


@ecosystem_router.get("/intelligence", status_code=status.HTTP_200_OK)
async def get_ecosystem_intelligence():
    """Returns real-time model of entire trading ecosystem, proactive warnings, and learning states."""
    return ecosystem_hub.get_ecosystem_intelligence_report()


@ecosystem_router.get("/learning", status_code=status.HTTP_200_OK)
async def get_learning_status():
    """Returns continuous learning progression, recommendation outcome rates, and adapted agent weights."""
    return continuous_learning_engine.get_learning_status()


@ecosystem_router.get("/meta", status_code=status.HTTP_200_OK)
async def get_meta_intelligence():
    """Returns meta-intelligence self-assessment, calibration metrics, and agent performance rankings."""
    return meta_intelligence.generate_meta_intelligence_report()


@ecosystem_router.post("/consult", status_code=status.HTTP_200_OK)
async def consult_ecosystem(req: EcosystemConsultRequest):
    """Conducts full ecosystem-level consultation spanning portfolio risk, regime state, and strategy allocations."""
    return ecosystem_hub.conduct_ecosystem_consultation(
        portfolio_positions=req.portfolio_positions,
        active_strategies=req.active_strategies
    )
