"""Second fabrication sweep: static analytics, fixed confidences, invented live inputs.

Found by ``scripts/find_literal_judgements.py``, an AST scan of every float literal bound to a
judgement name (confidence, score, probability, likelihood, credibility, ...). Before this batch it
reported 130 bindings. Each area below names the fabrication it removes; the gate test at the end
keeps the count of reviewed literals fixed so a new one cannot land unnoticed.

Each behavioural test asserts the honest state (None / not_measured / a named constant), so it
fails against the pre-batch code. The file is falsified against that code before the commit.
"""

from __future__ import annotations

import asyncio
import importlib.util
import pathlib
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _scanner():
    spec = importlib.util.spec_from_file_location("find_literal_judgements", ROOT / "scripts" / "find_literal_judgements.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# --- the gate ------------------------------------------------------------------------------------

def test_no_unallowed_literal_judgement_remains_in_app():
    """Pre-batch: 130 unallowed bindings. Post-batch: every remaining one is a reviewed, counted exception."""
    scanner = _scanner()
    hits = scanner.scan_all()
    unallowed = [f"{h['file']}:{h['line']} {h['name']}={h['value']} ({h.get('note', '')})"
                 for h in hits if not h["allowed"]]
    assert unallowed == []


def test_every_allowance_has_a_reason_and_a_count():
    scanner = _scanner()
    for rel, (expected, reason) in scanner.ALLOWED_FILES.items():
        assert isinstance(expected, int) and expected > 0, rel
        assert len(reason) > 20, rel


# --- static analytics that was presented as measured ---------------------------------------------

def test_meta_intelligence_reports_not_measured_instead_of_constants():
    """Pre-fix: quality 94.8, high-confidence accuracy 86.4%, agent scores 96.2/92.5/90.1/81.4."""
    from app.intelligence.meta_intel import MetaIntelligenceEngine

    out = MetaIntelligenceEngine().generate_meta_intelligence_report()
    assert out["status"] == "not_measured"
    assert out["meta_intelligence_quality_score"] is None
    assert out["self_calibration_analysis"]["high_confidence_accuracy_pct"] is None
    assert out["agent_performance_ranking"] == []


def test_observability_business_metrics_are_not_constants():
    """Pre-fix: recommendation_quality_score 98.4 and prediction_direction_accuracy_pct 76.5, always."""
    from app.observability_system import ObservabilityCollector

    metrics = ObservabilityCollector().get_observability_snapshot()["business_metrics"]
    assert metrics["recommendation_quality_score"] is None
    assert metrics["prediction_direction_accuracy_pct"] is None
    assert metrics["average_debate_consensus_confidence"] is None


def test_liquidity_is_not_measured_without_a_venue_feed():
    """Pre-fix: global_liquidity_score 92.5, best venue BINANCE and crisis flag False for every asset."""
    from app.analysis.liquidity_intel import CrossExchangeLiquidityIntel

    out = CrossExchangeLiquidityIntel().analyze_asset_liquidity("DOGEUSDT")
    assert out["status"] == "not_measured"
    assert out["global_liquidity_score"] is None
    assert out["best_execution_venue"] is None
    assert out["slippage_estimates"] is None


def test_market_debate_specialists_carry_no_fixed_confidence():
    """Pre-fix: specialists stated 0.85 / 0.89 / 0.88 and the overall was 0.87, always."""
    from app.debate.market_debate import MultiMarketDebateEngine

    out = MultiMarketDebateEngine().conduct_cross_market_debate({"BTCUSDT": 2.0})
    assert out["overall_confidence"] is None
    assert all(s["confidence"] is None for s in out["specialist_deliberations"])
    liquidity = next(s for s in out["specialist_deliberations"] if s["specialist"] == "Liquidity Analyst")
    assert "92.5" not in liquidity["findings"]


def test_ecosystem_status_is_not_a_constant_when_quality_is_unmeasured(monkeypatch):
    """Pre-fix: ecosystem_status was always OPTIMAL_AUTONOMOUS_OPERATION."""
    from app.services import ecosystem_intel as ei

    fake_meta = {"meta_intelligence_quality_score": None}
    monkeypatch.setattr(ei.meta_intelligence, "generate_meta_intelligence_report", lambda: fake_meta)
    report = ei.EcosystemIntelligenceHub().get_ecosystem_intelligence_report()
    assert report["ecosystem_status"] == "NOT_MEASURED"
    assert report["ecosystem_status"] != "OPTIMAL_AUTONOMOUS_OPERATION"


def test_continuous_learning_has_no_seeded_outcomes_or_invented_rate():
    """Pre-fix: three invented outcomes, a helpful rate of 85.0 when empty, fixed agent weights."""
    from app.learning.continuous_learning import ContinuousLearningEngine

    engine = ContinuousLearningEngine()
    assert engine.recommendation_outcomes == []
    status = engine.get_learning_status()
    assert status["helpful_recommendation_rate_pct"] is None
    assert status["learned_agent_weights"] is None


def test_long_term_memory_has_no_invented_episode_and_no_unrelated_fallback():
    """Pre-fix: an invented 2026 episode (effectiveness 0.92), a '84% of episodes' claim, and a fallback of unrelated episodes."""
    from app.memory.long_term import LongTermMemoryArchitecture

    memory = LongTermMemoryArchitecture()
    assert memory.episodic_memories == []
    assert memory.semantic_memories == {}
    assert memory.procedural_memories == []
    assert memory.retrieve_relevant_learnings("high_volatility") == []


def test_distillation_has_no_seeded_empirical_rules():
    """Pre-fix: RULE-001..003 carried empirical confidences 0.89-0.95 that no outcome data produced."""
    from app.learning.distillation import KnowledgeDistillationEngine

    assert KnowledgeDistillationEngine().rules == []


def test_provider_intelligence_is_measured_from_observed_calls(monkeypatch):
    """Pre-fix: a fixed matrix (groq 98.2% success, 35 ms, 'confidence 0.93') and 'Groq is 40% faster'."""
    from app.analytics import provider_intel as pi
    from app.analytics.usage_analytics import UsageAnalyticsEngine

    engine = UsageAnalyticsEngine()
    monkeypatch.setattr(pi, "usage_analytics", engine)
    assert pi.ProviderPerformanceIntelligence().get_performance_matrix()["status"] == "no_observed_calls"

    for ok in [True, True, True, False]:  # 4 calls: below MIN_CALLS_PER_CELL (5)
        engine.log_request(consumer="t", service="code_generation", provider="groq", latency_ms=100.0, success=ok)
    matrix = pi.ProviderPerformanceIntelligence().get_performance_matrix()
    cell = matrix["provider_service_matrix"]["groq"]["code_generation"]
    assert cell["status"] == "insufficient_calls"
    assert "success_rate_pct" not in cell

    for _ in range(6):
        engine.log_request(consumer="t", service="code_generation", provider="groq", latency_ms=100.0, success=True)
    cell = pi.ProviderPerformanceIntelligence().get_performance_matrix()["provider_service_matrix"]["groq"]["code_generation"]
    assert cell["status"] == "measured"
    assert cell["success_rate_pct"] == pytest.approx(9 / 10 * 100, abs=0.1)  # 9 of 10 calls
    assert cell["avg_latency_ms"] == 100.0


# --- synthetic inputs must not move a directional call -------------------------------------------

def test_synthetic_alternative_data_does_not_set_the_direction(monkeypatch):
    """Pre-fix: a bearish-looking synthetic fixture (negative sentiment, positive netflow) turned a
    bullish model read into NEUTRAL, so the fixture decided the call. Only measured legs may vote."""
    from app.ml import prediction_aggregator as pa

    monkeypatch.setattr(pa.deep_models_engine, "predict_horizons",
                        lambda *a, **k: {"horizons": {"24h": {"predicted_direction": "BULLISH"}}})
    bearish_fixture = {
        "evidence_class": "synthetic_fixture",
        "news_intelligence": {"sentiment_score": -0.5},
        "onchain_intelligence": {"exchange_netflow_24h_usd": 1_000_000.0},
    }
    monkeypatch.setattr(pa.alt_data_engine, "get_consolidated_alternative_data", lambda symbol: bearish_fixture)
    out = pa.PredictionAggregationEngine().aggregate_prediction("BTCUSDT", 65000.0, [0.01, -0.002, 0.004])
    assert out["unified_direction"] == "BULLISH"
    assert out["unified_confidence"] is None
    assert out["synthetic_legs"]


def test_neutral_model_read_is_not_published_as_bearish(monkeypatch):
    """Pre-fix: the LSTM leg was ``1.0 if BULLISH else -1.0``, so a NEUTRAL 24h read counted as bearish.
    Live: /v1/predict/btc with rising returns gave BEARISH (-0.40) because the 24h horizon was NEUTRAL."""
    from app.ml import prediction_aggregator as pa

    monkeypatch.setattr(pa.deep_models_engine, "predict_horizons",
                        lambda *a, **k: {"horizons": {"24h": {"predicted_direction": "NEUTRAL"}}})
    fixture = {
        "evidence_class": "synthetic_fixture",
        "news_intelligence": {"sentiment_score": 0.0},
        "onchain_intelligence": {"exchange_netflow_24h_usd": 0.0},
    }
    monkeypatch.setattr(pa.alt_data_engine, "get_consolidated_alternative_data", lambda symbol: fixture)
    out = pa.PredictionAggregationEngine().aggregate_prediction("BTCUSDT", 65000.0, [0.004, 0.005, 0.006])
    assert out["unified_direction"] == "NEUTRAL"


# --- sentiment ------------------------------------------------------------------------------------

def test_sentiment_without_keywords_is_neutral_and_social_is_not_measured():
    """Pre-fix: neutral headlines scored +0.05; reddit/twitter were the news score scaled; news_confidence 0.88."""
    from app.analysis.sentiment_analysis import SentimentAnalysisEngine

    out = SentimentAnalysisEngine().analyze_news([
        {"title": "Quarterly office relocation announced", "timestamp": time.time(), "credibility": 0.9},
    ])
    assert out["overall_score"] == 0.0
    social = out["social_breakdown"]
    assert social["reddit_sentiment"] is None
    assert social["twitter_sentiment"] is None
    assert social["news_confidence"] is None


# --- temporal and counterfactual -----------------------------------------------------------------

def test_temporal_empty_series_is_no_data_not_stable():
    """Pre-fix: an empty series was reported as a STABLE trend with confidence 0.70."""
    from app.intelligence.temporal import temporal_reasoning_engine

    out = temporal_reasoning_engine.analyze_temporal_series([])
    assert out.trend == "NO_DATA"
    assert out.confidence is None


def test_temporal_confidence_is_not_a_fixed_literal():
    """Pre-fix: any non-empty series reported confidence 0.88."""
    from app.intelligence.temporal import TimeSeriesPoint, temporal_reasoning_engine

    series = [TimeSeriesPoint(timestamp=float(i), value=100.0 + i, metric_name="px") for i in range(10)]
    out = temporal_reasoning_engine.analyze_temporal_series(series)
    assert out.confidence is None
    assert out.trend != "NO_DATA"  # the direction is the engine's call; the confidence must not be a literal


def test_counterfactual_effect_does_not_depend_on_the_spelling_of_a_variable():
    """Pre-fix: any counterfactual variable containing 'b' gave +12.0% and every other gave -8.5%."""
    from app.intelligence.counterfactual import CounterfactualScenario, counterfactual_engine

    def run(variable: str):
        return counterfactual_engine.evaluate_what_if(CounterfactualScenario(
            scenario_name="s", proposed_intervention="tighten stops",
            baseline_variable="stop_pct", counterfactual_variable=variable,
        ))

    with_b = run("breakeven_stop")
    without_b = run("trailing_stop")
    assert with_b.estimated_outcome_delta_pct is None
    assert without_b.estimated_outcome_delta_pct is None
    assert with_b.counterfactual_confidence is None
    assert with_b.estimate_status.startswith("not_estimated")


# --- technical patterns --------------------------------------------------------------------------

def test_technical_pattern_confidence_is_not_a_fixed_literal():
    """Pre-fix: 'Double Bottom' 0.82, 'Bull Flag' 0.78, 'Channel' 0.70 regardless of the candles."""
    from app.analysis import technical_analysis as ta

    source = pathlib.Path(ta.__file__).read_text(encoding="utf-8")
    assert '"confidence": 0.' not in source
    assert '"confidence": None' in source


# --- sentinel and intelx: rule-based analyses -----------------------------------------------------

def test_sentinel_attack_path_probabilities_are_not_estimated():
    """Pre-fix: chain probability 0.78, step likelihoods 0.85/0.72, a fixed 0.88 confidence and a
    'Critic challenged...' dissent, with no debate having run."""
    from app.services.sentinel_intelligence import (
        SecurityFinding,
        SentinelAnalysisRequest,
        SentinelIntelligenceService,
        TargetContext,
    )

    req = SentinelAnalysisRequest(
        request_id="sen-batch-1",
        analysis_type="attack_path_reasoning",
        target_context=TargetContext(asset_type="api_gateway", exposure_level="public_internet"),
        findings=[SecurityFinding(finding_id="F1", title="Outdated TLS", severity="HIGH", description="old cipher")],
    )
    resp = asyncio.run(SentinelIntelligenceService().analyze(req))
    analysis = resp.analysis
    assert analysis.confidence is None
    assert analysis.dissent == []
    assert analysis.attack_paths and analysis.attack_paths[0].overall_probability is None
    assert all(n.likelihood_score is None for n in analysis.attack_paths[0].nodes)
    assert "not derived from findings" in analysis.attack_paths[0].title


def test_intelx_unrated_source_is_unverified_and_planner_has_no_fixed_confidence():
    """Pre-fix: an unrated source defaulted to credibility 0.85; the planner stated 0.94 always."""
    from app.services.intelx_intelligence import (
        ExtractedClaimSpan,
        IntelXIntelligenceService,
        IntelXResearchContext,
        IntelXResearchRequest,
    )

    span = ExtractedClaimSpan(claim="x", verbatim_span="x", document_source="blog.example")
    assert span.credibility_score is None

    req = IntelXResearchRequest(
        request_id="ix-batch-planner-1",
        role="planner",
        context=IntelXResearchContext(question="When was the release?"),
    )
    out = asyncio.run(IntelXIntelligenceService().execute_research_role(req))
    assert out.confidence is None


# --- live capital surface ------------------------------------------------------------------------

def test_live_intelligence_requires_the_book_state_and_invents_nothing(auth):
    """Pre-fix: defaults of 2.5% drawdown / 1 loss, fixed correlation 0.62, ATR 0.018, PF 1.45 and
    confidence 0.82 -> every call recommended a parameter optimisation on real capital."""
    from app.main import app

    client = TestClient(app)
    assert client.get("/v1/trading/live/intelligence", headers=auth).status_code == 422

    resp = client.get(
        "/v1/trading/live/intelligence",
        params={"drawdown_pct": 1.0, "consecutive_losses": 0, "profit_factor": 1.45},
        headers=auth,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["market_stress"]["status"] == "not_measured"
    assert body["conservative_guidance"]["confidence"] == 0.0
    assert body["conservative_guidance"]["confidence_status"].startswith("not_supplied")
    assert "OPTIMIZE" not in resp.text


def test_live_attribution_without_a_ledger_is_not_measured(auth):
    """Pre-fix: three hard-coded trades (BTC at 65,000) and a fixed testnet win rate of 0.65."""
    from app.main import app

    client = TestClient(app)
    body = client.get("/v1/trading/live/attribution", headers=auth).json()
    assert body["status"] == "not_measured"
    assert body["strategy_reliability_score"] is None
    assert "65000" not in client.get("/v1/trading/live/attribution", headers=auth).text


def test_live_attribution_reliability_needs_a_measured_testnet_win_rate(auth):
    """Pre-fix: an absent testnet win rate was silently 0.60, so reliability was always computed."""
    from app.analysis.live_attribution import live_attribution_engine

    trades = [{"pnl": 10.0, "expected_price": 100.0, "fill_price": 100.1, "qty": 1.0}] * 3
    no_benchmark = live_attribution_engine.evaluate_live_attribution(live_trades=trades, testnet_metrics=None)
    assert no_benchmark["strategy_reliability_score"] is None
    assert no_benchmark["is_recommended_for_capital_allocation"] is None

    with_benchmark = live_attribution_engine.evaluate_live_attribution(
        live_trades=trades, testnet_metrics={"win_rate": 1.0})
    assert with_benchmark["strategy_reliability_score"] is not None


def test_live_attribution_post_rejects_trades_with_missing_prices(auth):
    """Pre-fix: a missing fill_price was treated as 0.0, which invented a slippage figure."""
    from app.main import app

    client = TestClient(app)
    resp = client.post(
        "/v1/trading/live/attribution",
        json={"trades": [{"pnl": 5.0, "expected_price": 100.0, "qty": 1.0}]},
        headers=auth,
    )
    assert resp.status_code == 422


def test_conservative_rationale_does_not_claim_a_consensus_it_never_computed():
    """Pre-fix: any confidence >= 0.80 got 'High confidence statistical validation across multi-agent
    consensus.' regardless of input. Live: the /v1/trading/live/intelligence response carried it."""
    from app.services.conservative_engine import conservative_engine

    rec = conservative_engine.generate_conservative_recommendation(
        strategy_name="s", current_drawdown_pct=1.0, profit_factor=1.45, confidence=0.9)
    assert rec["recommended_action"] == "OPTIMIZE_PARAMETERS"
    assert "multi-agent consensus" not in rec["rationale"].split("No multi-agent")[0]
    assert "0.90" in rec["rationale"] and "1.45" in rec["rationale"]
    assert "No multi-agent consensus was computed" in rec["rationale"]


# --- agents and adjudication --------------------------------------------------------------------

def test_agent_default_confidence_is_the_unverified_prior_not_0_90():
    """Pre-fix: an agent answer without a stated confidence was recorded as 0.90."""
    from app.agents.base import AgentResponse
    from app.utils.confidence import UNVERIFIED_MODEL_CONFIDENCE

    assert AgentResponse(summary="s").confidence == UNVERIFIED_MODEL_CONFIDENCE
    assert AgentResponse.parse_raw_or_json("plain prose").confidence == UNVERIFIED_MODEL_CONFIDENCE


def test_adjudicator_with_no_assessments_has_no_confidence_and_is_labelled_heuristic():
    """Pre-fix: no assessments gave 0.50, and the output called itself 'empirically calibrated'."""
    from app.agents.adjudication import Adjudicator

    conf, factors = Adjudicator.calculate_system_confidence(
        assessments=[], contradictions=[], evidence_count=0, complexity_str="simple")
    assert conf == 0.0
    assert factors["calibration_status"] == "not_applicable"

    conf, factors = Adjudicator.calculate_system_confidence(
        assessments=[SimpleNamespace(model_confidence=0.7)], contradictions=[], evidence_count=0,
        complexity_str="simple")
    assert factors["calibration_status"].startswith("uncalibrated heuristic")


# --- performance tracker -------------------------------------------------------------------------

def test_performance_tracker_computes_from_recorded_outcomes_only():
    """Pre-fix: record_task_outcome only logged, and compute_model_statistics named 'gemini-3.8-flash'."""
    from app.learning.performance import PerformanceTracker
    from app.utils.bounded_store import BoundedList

    tracker = PerformanceTracker.__new__(PerformanceTracker)
    tracker._outcomes = BoundedList("test_outcomes", max_entries=100)

    empty = asyncio.run(tracker.compute_model_statistics())
    assert empty["top_reasoning_model"] is None
    assert empty["basis"].startswith("not_measured")
    assert "gemini" not in str(empty)

    for i in range(6):
        asyncio.run(tracker.record_task_outcome(f"t{i}", "code", "fast", ["critic"], 0.8, 1.0, 100))
    stats = asyncio.run(tracker.compute_model_statistics())
    assert stats["outcomes_recorded"] == 6
    assert stats["per_agent"]["critic"]["average_score"] == 0.8
    assert stats["most_effective_critic"] == "critic"


# --- multimodal sub-results ---------------------------------------------------------------------

def test_temporal_result_serialises_with_null_confidence_for_multimodal_consumers():
    """The multimodal route embeds temporal sub-results: an unmeasured one must serialise with null."""
    from app.intelligence.temporal import temporal_reasoning_engine

    temporal = temporal_reasoning_engine.analyze_temporal_series([]).model_dump()
    assert temporal["confidence"] is None
    assert temporal["trend"] == "NO_DATA"
