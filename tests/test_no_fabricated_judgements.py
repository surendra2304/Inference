"""Judgements must come from their inputs, never from fixed numbers or invented facts.

Found by the sensitivity probe (``scripts/sensitivity_probe.py``), which sends two different
bodies to each POST operation and flags judgement fields that do not change. Each test below
names the fabrication it guards. Every one of them failed before the fix, and the file is
falsified against the pre-fix code.
"""

from __future__ import annotations

import pathlib
import re
from types import SimpleNamespace

import pytest

from app.analysis.onchain_analytics import onchain_engine
from app.analysis.overfitting_intel import overfitting_engine
from app.analysis.regime_robustness import regime_robustness_engine
from app.analysis.sentiment_analysis import sentiment_engine
from app.debate.enhanced_trading_debate import enhanced_trading_debate
from app.debate.strategy_evaluation import strategy_evaluation_debate
from app.integrations import market_data
from app.ml.price_prediction import ml_prediction_model
from app.utils.confidence import DEGRADED_CONFIDENCE, UNVERIFIED_MODEL_CONFIDENCE

ROOT = pathlib.Path(__file__).resolve().parents[1]

GOOD_REGIMES = {
    "bull": {"profit_factor": 2.1, "max_drawdown_pct": 3.5},
    "bear": {"profit_factor": 1.4, "max_drawdown_pct": 5.8},
    "chop": {"profit_factor": 0.95, "max_drawdown_pct": 7.2},
}
WEAK_REGIMES = {
    "bull": {"profit_factor": 0.8, "max_drawdown_pct": 14.0},
    "bear": {"profit_factor": 0.5, "max_drawdown_pct": 22.0},
    "chop": {"profit_factor": 0.3, "max_drawdown_pct": 30.0},
}


# --- regime robustness: no built-in regime results --------------------------------------------

def test_regime_without_data_reports_no_score_instead_of_the_builtin_profile():
    """Pre-fix: an empty request was scored on four hard-coded regimes (score 14.7 for all)."""
    out = regime_robustness_engine.test_regime_robustness("s", {})
    assert out["status"] == "not_measured"
    assert out["robustness_score"] is None
    assert out["regime_breakdown"] == {}


def test_regime_entries_missing_their_measures_are_not_defaulted():
    """Pre-fix: a missing profit_factor was silently 1.0 and a missing drawdown 5.0."""
    out = regime_robustness_engine.test_regime_robustness("s", {"bull": {"win_rate": 0.7}})
    assert out["status"] == "insufficient_inputs"
    assert out["robustness_score"] is None
    assert out["missing_fields_for_regimes"] == ["bull"]


def test_regime_score_depends_on_the_supplied_regimes():
    good = regime_robustness_engine.test_regime_robustness("s", GOOD_REGIMES)
    weak = regime_robustness_engine.test_regime_robustness("s", WEAK_REGIMES)
    assert good["status"] == weak["status"] == "measured"
    assert good["robustness_score"] > weak["robustness_score"]


# --- strategy evaluation: no invented backtest numbers ----------------------------------------

def test_strategy_without_backtest_metrics_is_not_evaluated():
    """Pre-fix: missing Sharpe/PF/drawdown/trades were replaced by 1.8/1.6/6.5/120 and a verdict issued."""
    out = strategy_evaluation_debate.evaluate_strategy_candidate("s", {}, GOOD_REGIMES)
    assert out["final_verdict"] == "NOT_EVALUATED"
    assert out["composite_evaluation_score"] is None
    assert set(out["missing_backtest_fields"]) == {
        "sharpe_ratio", "profit_factor", "max_drawdown_pct", "total_trades"
    }


def test_strategy_without_regime_metrics_is_not_evaluated():
    """A regime score cannot be invented, so the composite and verdict cannot be either."""
    backtest = {"sharpe_ratio": 1.9, "profit_factor": 1.7, "max_drawdown_pct": 5.8, "total_trades": 140}
    out = strategy_evaluation_debate.evaluate_strategy_candidate("s", backtest, {})
    assert out["final_verdict"] == "NOT_EVALUATED"
    assert out["composite_evaluation_score"] is None


def test_strategy_verdict_follows_the_supplied_backtest():
    strong = {"sharpe_ratio": 2.2, "profit_factor": 2.0, "max_drawdown_pct": 4.0, "total_trades": 300}
    poor = {"sharpe_ratio": 0.2, "profit_factor": 0.9, "max_drawdown_pct": 35.0, "total_trades": 25}
    a = strategy_evaluation_debate.evaluate_strategy_candidate("s", strong, GOOD_REGIMES)
    b = strategy_evaluation_debate.evaluate_strategy_candidate("s", poor, GOOD_REGIMES)
    assert a["composite_evaluation_score"] > b["composite_evaluation_score"]


def test_overfitting_output_says_it_is_a_proxy_not_measured_pbo():
    """Pre-fix: the key probability_of_backtest_overfitting_pbo read as Bailey's CSCV PBO."""
    out = overfitting_engine.evaluate_strategy_overfitting("s", 1.5, 1.4, 120, 50)
    assert out["estimator"] == "summary_statistics_proxy"
    assert out["pbo_is_cscv_measured"] is False


def test_overfitting_request_fields_are_required_not_defaulted():
    """Pre-fix: an omitted Sharpe/PF/trades/trials was filled with 2.1/1.85/120/50."""
    from app.routers.evolution_intel import OverfittingCheckRequest

    with pytest.raises(Exception):
        OverfittingCheckRequest(strategy_name="s")


def test_strategy_request_has_no_default_metrics():
    from app.routers.evolution_intel import StrategyEvaluationRequest

    assert StrategyEvaluationRequest().backtest_metrics == {}


# --- market data: validated symbols, labelled provenance, no invented news ---------------------

@pytest.mark.parametrize("bad", ["BTCUSDT&limit=1000", "../../admin", "BTC USDT", "", "B", "X" * 21, "BTC;DROP"])
def test_symbol_is_validated_before_it_reaches_a_url(bad):
    """Pre-fix: the raw symbol was interpolated into the Binance query string."""
    with pytest.raises(market_data.InvalidMarketParameter):
        market_data.normalize_symbol(bad)


def test_valid_symbol_is_normalised():
    assert market_data.normalize_symbol("btcusdt") == "BTCUSDT"


async def test_failed_fetch_is_labelled_synthetic_not_market_data(monkeypatch):
    """Pre-fix: the zig-zag fallback was returned and cached like real candles."""

    class Boom:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            raise RuntimeError("offline")

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(market_data.httpx, "AsyncClient", Boom)
    fetcher = market_data.MarketDataFetcher(cache_ttl_sec=0.0)
    candles, source = await fetcher.get_ohlcv_sourced("BTCUSDT", "1h", 30)
    assert source == market_data.SYNTHETIC_SOURCE
    assert len(candles) == 30


async def test_news_feed_is_empty_rather_than_invented():
    """Pre-fix: three headlines attributed to CryptoPanic and Reddit were returned on every call."""
    assert await market_data.MarketDataFetcher().get_news_and_social_feed("BTC") == []


def test_empty_news_is_no_data_not_a_neutral_zero():
    """Pre-fix: an empty feed scored 0.0 NEUTRAL, which the trajectory used as a sentiment input."""
    out = sentiment_engine.analyze_news([])
    assert out["overall_score"] is None
    assert out["classification"] == "NO_DATA"


def test_onchain_metrics_are_not_measured_and_carry_no_numbers():
    """Pre-fix: fixed network figures and whale transfers were served as on-chain data."""
    out = onchain_engine.get_onchain_metrics("ETH")
    assert out["status"] == "not_measured"
    assert out["exchange_flows"] is None
    assert out["network_health"] is None
    assert out["whale_movements"] == []


# --- price trajectory: attributions from inputs, missing inputs stay missing ------------------

def _indicators(rsi: float, macd: float, adx: float, atr: float) -> dict:
    return {"rsi_14": rsi, "macd": {"histogram": macd}, "adx": adx, "atr_14": atr}


def test_attributions_follow_the_inputs_instead_of_a_fixed_split():
    """Pre-fix: feature_attributions were always 50/30/20."""
    a = ml_prediction_model.predict_price_trajectory(
        100.0, _indicators(70, 2.0, 30, 1.5), {"overall_score": 0.8}, {"exchange_flows": {"net_flow_usd": -5}})
    b = ml_prediction_model.predict_price_trajectory(
        100.0, _indicators(30, -2.0, 30, 1.5), {"overall_score": -0.1}, {"exchange_flows": {"net_flow_usd": 5}})
    assert a["feature_attributions"] != b["feature_attributions"]
    assert a["feature_attributions"] != {"technical_momentum_pct": 50.0, "sentiment_nlp_pct": 30.0, "onchain_flow_pct": 20.0}


def test_attributions_are_shares_of_the_measured_contributions():
    out = ml_prediction_model.predict_price_trajectory(
        100.0, _indicators(60, 1.0, 25, 1.5), {"overall_score": 0.5}, {"exchange_flows": {"net_flow_usd": -1}})
    assert round(sum(out["feature_attributions"].values()), 0) == 100


def test_missing_inputs_are_listed_and_not_assumed_neutral():
    """Pre-fix: missing RSI/ADX/ATR/sentiment were replaced by neutral defaults and used."""
    out = ml_prediction_model.predict_price_trajectory(100.0, {}, {}, {})
    assert {"rsi_14", "adx", "atr_14", "sentiment_score", "onchain_net_flow_usd"} <= set(out["missing_inputs"])
    assert out["overall_confidence"] is None
    assert all(h["confidence_interval"] is None for h in out["horizons"].values())


# --- deliberation: bias from inputs, no literal confidences -----------------------------------

async def test_onchain_specialist_is_unknown_without_flow_data_not_always_bullish():
    """Pre-fix: the on-chain specialist was hard-wired BULLISH with confidence 0.88."""
    out = await enhanced_trading_debate.conduct_advanced_market_deliberation("BTCUSDT", [], [], {})
    onchain = next(s for s in out["specialist_deliberations"] if s["specialist"] == "On-Chain Analyst")
    assert onchain["bias"] == "UNKNOWN"
    assert onchain["confidence"] is None
    assert out["overall_confidence"] is None
    assert out["overall_consensus"] == "NO_EVIDENCE"


async def test_deliberation_does_not_crash_without_candles():
    """Pre-fix: float(None) and a :,.2f format on a missing price raised with no candles."""
    out = await enhanced_trading_debate.conduct_advanced_market_deliberation("BTCUSDT", [], [], {})
    assert out["current_price"] is None


# --- route-level: predict refuses to forecast on synthetic data -------------------------------

async def test_predict_refuses_on_synthetic_market_data(monkeypatch, auth):
    """Pre-fix: a failed fetch produced a 24h trajectory from a zig-zag with a fixed 65000 start."""
    from httpx import ASGITransport, AsyncClient

    from app.main import app

    async def synthetic(self, symbol="BTCUSDT", interval="1h", limit=100):
        return market_data.MarketDataFetcher._synthetic_candles("BTCUSDT", limit, 0.0), market_data.SYNTHETIC_SOURCE

    monkeypatch.setattr(market_data.MarketDataFetcher, "get_ohlcv_sourced", synthetic)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver",
                           headers=auth, timeout=60) as client:
        r = await client.post("/v1/trading/predict", json={"symbol": "BTCUSDT"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "no_market_data"
    assert body["prediction"] is None


async def test_predict_rejects_an_injected_symbol_with_422(auth):
    from httpx import ASGITransport, AsyncClient

    from app.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver",
                           headers=auth, timeout=60) as client:
        r = await client.post("/v1/trading/predict", json={"symbol": "BTC&limit=1000"})
    assert r.status_code == 422


# --- model-backed confidence is a named constant, never a literal -----------------------------

def test_the_named_confidence_states_are_ordered():
    assert DEGRADED_CONFIDENCE == 0.0
    assert 0.0 < UNVERIFIED_MODEL_CONFIDENCE < 0.9


@pytest.mark.parametrize("relpath", [
    "app/api/universal_task_routes.py",
    "app/services/architecture_planning.py",
    "app/services/multimodal_intelligence.py",
    "app/debate/enhanced_trading_debate.py",
    "app/api/friday_routes.py",
])
def test_no_fixed_high_confidence_literal_remains(relpath):
    """Pre-fix: confidence=0.94, 0.95, 0.88, 0.85, 0.82, 0.80 literals were returned on every success."""
    text = (ROOT / relpath).read_text(encoding="utf-8")
    code_lines = [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]
    hits = [ln.strip() for ln in code_lines if re.search(r"confidence\w*[\"']?\s*[:=]\s*0\.[89]\d*\b", ln)]
    assert hits == [], hits


async def test_architecture_degraded_call_reports_no_spec_and_zero_confidence(monkeypatch):
    """Pre-fix: a degraded call still returned confidence 0.94 with no spec."""
    from app.services import architecture_planning as ap

    class Fake:
        async def execute(self, req):
            return SimpleNamespace(content="", degraded=True, model_used="none", provider_used="none")

    monkeypatch.setattr(ap, "unified_provider_manager", Fake())
    req = ap.ArchitecturePlanRequest(goal="a small API")
    out = await ap.ArchitecturePlanningService().plan_architecture(req)
    assert out.confidence == DEGRADED_CONFIDENCE
    assert out.architecture_spec == ""


async def test_architecture_model_answer_is_unverified_not_0_94(monkeypatch):
    from app.services import architecture_planning as ap

    class Fake:
        async def execute(self, req):
            return SimpleNamespace(content="Use FastAPI.", degraded=False, model_used="m", provider_used="p")

    monkeypatch.setattr(ap, "unified_provider_manager", Fake())
    out = await ap.ArchitecturePlanningService().plan_architecture(ap.ArchitecturePlanRequest(goal="x"))
    assert out.confidence == UNVERIFIED_MODEL_CONFIDENCE
    assert out.architecture_spec == "Use FastAPI."


async def test_multimodal_reports_no_decision_and_no_fixed_estimate():
    """Pre-fix: every request returned OPTIMIZE_STRATEGY, +12% (CI +5..+19%) and 0.88."""
    from app.services.multimodal_intelligence import MultiModalIntelligenceRequest, multimodal_service

    req = MultiModalIntelligenceRequest(request_id="mm-1", goal="reduce churn", attached_contents=[])
    out = await multimodal_service.analyze_multimodal(req)
    assert out.decision == "NO_DECISION"
    assert "+12" not in out.point_estimate_with_ci
    assert out.confidence == DEGRADED_CONFIDENCE


# --- futuris: the forecast summary is measured, never a template or an invented panel ---------

async def test_futuris_does_not_invent_risks_or_claim_a_panel_without_a_model(monkeypatch):
    """Pre-fix: a fixed sentence about a 'macro policy announcement', three 'consulted' agents,
    a fixed dissent note and confidence_adjustment 0.08 were returned for every request."""
    from app.providers import unified_manager as um
    from app.services.futuris_enhancement import FuturisEnhanceRequest, futuris_enhancement_service

    class Degraded:
        async def execute(self, req):
            return SimpleNamespace(content="", degraded=True, model_used="none", provider_used="none")

    monkeypatch.setattr(um.unified_provider_manager, "execute", Degraded().execute)
    req = FuturisEnhanceRequest(
        request_id="fut-no-model-1",
        statistical_forecast={"model_used": "GARCH", "point_estimate": 0.05,
                              "confidence_interval": [0.04, 0.06]},
    )
    out = await futuris_enhancement_service.enhance_forecast(req)
    dumped = out.model_dump_json()
    assert "macro policy" not in dumped.lower()
    assert "Critic Note" not in dumped
    assert out.confidence_adjustment is None
    assert out.provenance["agents_consulted"] == []
    assert out.provenance["qualitative_review_status"] in {"not_performed"}
    assert any("Interval width" in risk for risk in out.enhanced_assessment.key_risks)


async def test_futuris_has_no_invented_grounding_series():
    """Pre-fix: three hard-coded forecasts (volatility_btc 0.045, p=0.88 ...) were served as grounding."""
    from app.services.futuris_enhancement import StatisticalGroundingEngine

    assert StatisticalGroundingEngine().cached_forecasts == {}
    assert StatisticalGroundingEngine().get_grounding_context("volatility_btc") is None


def test_futuris_probability_is_not_defaulted_to_0_80():
    """Pre-fix: an omitted probability became 0.80, which fed the usage analytics as a measurement."""
    from app.services.futuris_enhancement import StatisticalForecastInput

    f = StatisticalForecastInput(point_estimate=1.0, confidence_interval=[0.9, 1.1], model_used="x")
    assert f.probability is None


async def test_futuris_prose_reply_is_unparsed_not_a_500(monkeypatch):
    """Found live: a model that answered in prose made the review raise JSONDecodeError and the
    whole endpoint returned HTTP 500. The suite never reached this path because it has no model."""
    from app.providers import unified_manager as um
    from app.services.futuris_enhancement import FuturisEnhanceRequest, futuris_enhancement_service

    async def prose(req):
        return SimpleNamespace(content="Here is my view, no JSON at all.", degraded=False,
                               model_used="m", provider_used="p", finish_reason="stop")

    monkeypatch.setattr(um.unified_provider_manager, "execute", prose)
    req = FuturisEnhanceRequest(
        request_id="fut-prose-1",
        statistical_forecast={"model_used": "GARCH", "point_estimate": 0.05, "confidence_interval": [0.04, 0.06]},
    )
    out = await futuris_enhancement_service.enhance_forecast(req)
    assert out.provenance["qualitative_review_status"] == "unparsed"
    assert out.provenance["agents_consulted"] == []


async def test_futuris_json_reply_is_used_and_attributed(monkeypatch):
    from app.providers import unified_manager as um
    from app.services.futuris_enhancement import FuturisEnhanceRequest, futuris_enhancement_service

    reply = ('{"key_risks":["Liquidity thins below 0.04"],"contextual_drivers":["Volume rising"],'
             '"uncertainty_factors":["Sample is 30 days"],"qualitative_adjustments":["Tighten stops"],'
             '"dissent":["Regime may have shifted"]}')

    async def json_reply(req):
        return SimpleNamespace(content=reply, degraded=False, model_used="m", provider_used="p", finish_reason="stop")

    monkeypatch.setattr(um.unified_provider_manager, "execute", json_reply)
    req = FuturisEnhanceRequest(
        request_id="fut-json-1",
        statistical_forecast={"model_used": "GARCH", "point_estimate": 0.05, "confidence_interval": [0.04, 0.06]},
    )
    out = await futuris_enhancement_service.enhance_forecast(req)
    assert out.provenance["qualitative_review_status"] == "parsed"
    assert out.provenance["agents_consulted"] == ["qualitative_model_review"]
    assert "Liquidity thins below 0.04" in out.enhanced_assessment.key_risks
    assert out.dissent == ["Regime may have shifted"]
