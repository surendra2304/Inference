"""FastAPI Router for Live Capital Intelligence, Crisis Status, Stress Testing, and Live Attribution."""

import time

from fastapi import APIRouter, Query, status
from pydantic import BaseModel, Field

from app.analysis.live_attribution import live_attribution_engine
from app.analysis.stress_intelligence import stress_intelligence_engine
from app.services.conservative_engine import conservative_engine
from app.services.crisis_detector import crisis_detector
from app.utils.confidence import DEGRADED_CONFIDENCE

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
    # Drawdown and the loss streak are required: a default of 2.5% / 1 loss described a state
    # nobody measured. A caller must say what the book looks like.
    drawdown_pct: float = Query(..., description="Current drawdown percentage"),
    consecutive_losses: int = Query(..., ge=0, description="Current consecutive loss streak"),
    bid_ask_spread_pct: float | None = Query(default=None, ge=0.0, description="Active bid-ask spread percentage"),
    cross_asset_correlation: float | None = Query(default=None, ge=-1.0, le=1.0, description="Measured cross-asset correlation"),
    volatility_atr_pct: float | None = Query(default=None, ge=0.0, description="Measured ATR as a percentage"),
    profit_factor: float | None = Query(default=None, ge=0.0, description="Measured live profit factor"),
    model_confidence: float | None = Query(default=None, ge=0.0, le=1.0, description="Measured confidence in the proposed change"),
):
    """Live capital intelligence: crisis state, market stress and conservative guidance.

    Stress, guidance and confidence are computed only from the measured inputs that were supplied.
    The previous version fed fixed correlation (0.62), ATR (0.018), profit factor (1.45) and
    confidence (0.82) into the engines. Because 0.82 clears the 0.80 threshold in the conservative
    engine, every live call recommended a parameter optimisation on real capital.
    """
    crisis_eval = crisis_detector.evaluate_crisis_level(
        current_drawdown_pct=drawdown_pct,
        consecutive_losses=consecutive_losses
    )
    if bid_ask_spread_pct is not None and cross_asset_correlation is not None and volatility_atr_pct is not None:
        stress_eval = stress_intelligence_engine.evaluate_market_stress(
            bid_ask_spread_pct=bid_ask_spread_pct,
            cross_asset_correlation=cross_asset_correlation,
            volatility_atr_pct=volatility_atr_pct,
        )
    else:
        stress_eval = {
            "status": "not_measured",
            "reason": "bid_ask_spread_pct, cross_asset_correlation and volatility_atr_pct are all required for a stress index",
        }
    if profit_factor is not None:
        # With no supplied confidence the engine gets DEGRADED (0.0), below its 0.80 threshold, so
        # the guidance is NO_CHANGE rather than an optimisation on an unmeasured number.
        rec = conservative_engine.generate_conservative_recommendation(
            strategy_name="Live_Multi_Strategy",
            current_drawdown_pct=drawdown_pct,
            profit_factor=profit_factor,
            confidence=model_confidence if model_confidence is not None else DEGRADED_CONFIDENCE,
        )
        rec["confidence_status"] = "supplied" if model_confidence is not None else "not_supplied: treated as 0.0 (no change advised)"
    else:
        rec = {"status": "not_measured", "reason": "profit_factor is required for conservative guidance"}

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
    drawdown_pct: float = Query(..., description="Current drawdown percentage"),
    consecutive_losses: int = Query(..., ge=0, description="Consecutive losses")
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
    """Live execution slippage and strategy reliability; reports not measured without a trade ledger.

    The previous version scored three hard-coded trades (BTC at 65,000 with +45/-22/+88 PnL) and a
    fixed testnet win rate of 0.65, presenting them as the live book. No trade ledger is read here,
    so there is nothing to attribute.
    """
    return {
        "status": "not_measured",
        "reason": "no live trade ledger is connected to this endpoint; supply trades to POST /v1/trading/live/attribution",
        "strategy_reliability_score": None,
        "is_recommended_for_capital_allocation": None,
    }


class ExecutedTrade(BaseModel):
    """One executed trade. Every field is required: a missing price or quantity is not zero."""

    pnl: float = Field(allow_inf_nan=False)
    expected_price: float = Field(gt=0.0, allow_inf_nan=False)
    fill_price: float = Field(gt=0.0, allow_inf_nan=False)
    qty: float = Field(gt=0.0, allow_inf_nan=False)


class AttributionRequest(BaseModel):
    trades: list[ExecutedTrade] = Field(min_length=1, max_length=5000)
    testnet_win_rate: float | None = Field(default=None, ge=0.0, le=1.0, description="Measured testnet win rate")


@live_router.post("/attribution", status_code=status.HTTP_200_OK)
async def post_live_attribution(req: AttributionRequest):
    """Attribution for caller-supplied executed trades; reliability only when a testnet win rate is given."""
    testnet = {"win_rate": req.testnet_win_rate} if req.testnet_win_rate is not None else None
    return live_attribution_engine.evaluate_live_attribution(
        live_trades=[t.model_dump() for t in req.trades],
        testnet_metrics=testnet,
    )
