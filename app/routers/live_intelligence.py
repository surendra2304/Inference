"""FastAPI Router for Live Capital Intelligence, Crisis Status, Stress Testing, and Live Attribution."""

import time

from fastapi import APIRouter, Query, status
from pydantic import BaseModel, Field

from app.analysis.live_attribution import live_attribution_engine
from app.analysis.stress_intelligence import stress_intelligence_engine
from app.services.conservative_engine import conservative_engine
from app.services.crisis_detector import crisis_detector

live_router = APIRouter(prefix="/v1/trading/live", tags=["Live Capital Intelligence"])


class StressTestRequest(BaseModel):
    # allow_inf_nan=False states in the published contract what the arithmetic already
    # assumes: equity and notional are finite quantities. Without it Pydantic accepts
    # NaN/inf for a plain float, the stress engine arithmetic propagates them, and the
    # response then cannot be encoded — a 500 caused by a value the schema said was fine.
    portfolio_equity: float = Field(
        default=10000.0, allow_inf_nan=False, description="Active portfolio equity in USD"
    )
    active_notional: float = Field(
        default=3000.0, allow_inf_nan=False, description="Open notional position size in USD"
    )


@live_router.get("/intelligence", status_code=status.HTTP_200_OK)
async def get_live_intelligence(
    drawdown_pct: float = Query(default=2.5, description="Current drawdown percentage"),
    consecutive_losses: int = Query(default=1, description="Current consecutive loss streak"),
    bid_ask_spread_pct: float = Query(default=0.0005, description="Active bid-ask spread percentage")
):
    """Returns comprehensive live capital intelligence report including crisis state, stress indices, and conservative guidance."""
    crisis_eval = crisis_detector.evaluate_crisis_level(
        current_drawdown_pct=drawdown_pct,
        consecutive_losses=consecutive_losses
    )
    stress_eval = stress_intelligence_engine.evaluate_market_stress(
        bid_ask_spread_pct=bid_ask_spread_pct,
        cross_asset_correlation=0.62,
        volatility_atr_pct=0.018
    )
    rec = conservative_engine.generate_conservative_recommendation(
        strategy_name="Live_Multi_Strategy",
        current_drawdown_pct=drawdown_pct,
        profit_factor=1.45,
        confidence=0.82
    )

    return {
        "status": "ONLINE",
        "trading_mode": "LIVE",
        "crisis_evaluation": crisis_eval,
        "market_stress": stress_eval,
        "conservative_guidance": rec,
        "timestamp": time.time()
    }


@live_router.get("/crisis-status", status_code=status.HTTP_200_OK)
async def get_crisis_status(
    drawdown_pct: float = Query(default=0.0, description="Current drawdown percentage"),
    consecutive_losses: int = Query(default=0, description="Consecutive losses")
):
    """Returns active crisis evaluation and defensive protocol actions."""
    return crisis_detector.evaluate_crisis_level(
        current_drawdown_pct=drawdown_pct,
        consecutive_losses=consecutive_losses
    )


@live_router.post("/stress-test", status_code=status.HTTP_200_OK)
async def run_portfolio_stress_test(req: StressTestRequest):
    """Runs portfolio stress test across historical crisis scenarios."""
    return stress_intelligence_engine.run_historical_stress_test(
        portfolio_equity=req.portfolio_equity,
        active_notional=req.active_notional
    )


@live_router.get("/attribution", status_code=status.HTTP_200_OK)
async def get_live_attribution():
    """Returns live execution slippage and strategy reliability scoring."""
    sample_live_trades = [
        {"pnl": 45.0, "expected_price": 65000.0, "fill_price": 65002.5, "qty": 0.05},
        {"pnl": -22.0, "expected_price": 65120.0, "fill_price": 65123.0, "qty": 0.05},
        {"pnl": 88.0, "expected_price": 64900.0, "fill_price": 64901.0, "qty": 0.08}
    ]
    return live_attribution_engine.evaluate_live_attribution(
        live_trades=sample_live_trades,
        testnet_metrics={"win_rate": 0.65}
    )
