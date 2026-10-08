"""No fabricated numbers on the prediction and monitoring surfaces.

Both surfaces published constants that a reader could not distinguish from measurements.

* ``/v1/intelligence/accuracy`` and ``/v1/predict/{asset}/history`` returned
  ``lstm_transformer_model: 78.4``, ``news_sentiment_nlp: 71.2``,
  ``onchain_whale_signals: 81.5``, ``technical_momentum: 69.8``,
  ``calibration_score_0_to_1: 0.88`` and ``accuracy_status:
  "HIGH_CONFIDENCE_PRODUCTION_READY"``, computed from four hardcoded predictions, with a
  ``75.0`` fallback for an empty store. The tracker had no method to record a prediction,
  so no observation could ever have replaced them.
* ``GET /v1/predict/{asset}`` invented its inputs (``base_price = 65200.0 if "BTC" in
  symbol else 3450.0`` plus a fixed return series) and still published a directional call
  with ``unified_confidence >= 0.60`` — the floor meant three contradicting signals still
  produced a confident-looking number.
* ``monitor.get_debate_metrics()`` returned ``recommendation_quality_score: 100.0`` with the
  comment "Certified perfect 100/100 by audit suite", and called every served HTTP request a
  "deliberation".
* ``monitor.get_api_metrics()`` reported ``error_rate_pct: 0.0`` and ``p50..p99: 0.0`` with
  no sample count when nothing had been measured, which reads as a clean bill of health; the
  Prometheus exporter then rendered those ``None`` values as the literal text ``None``,
  producing a malformed exposition line (``inference_error_rate_percent None``).
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.ml.prediction_tracking import MIN_OBSERVATIONS_FOR_SCORE, PredictionTrackingEngine

FABRICATED_CONSTANTS = ("78.4", "71.2", "81.5", "69.8", "0.88", "PRODUCTION_READY", "75.0")


def test_accuracy_report_claims_nothing_without_observations():
    engine = PredictionTrackingEngine()
    report = engine.get_source_accuracy_report()

    assert report["overall_directional_accuracy_pct"] is None
    assert report["accuracy_status"] == "INSUFFICIENT_DATA"
    assert report["sub_model_accuracies"] == {}
    assert report["calibration_score_0_to_1"] is None
    assert report["fixture_rows_excluded"] == 4, "the four demonstration rows must be declared"
    assert report["total_predictions_evaluated"] == 0

    blob = json.dumps(report)
    for constant in FABRICATED_CONSTANTS:
        assert constant not in blob, f"unmeasured constant {constant} still published"


def test_accuracy_is_computed_from_recorded_observations():
    engine = PredictionTrackingEngine()
    # 15 hits and 5 misses = 75.0% — the same value the old constant produced, but now
    # derived, with the sample size and the baseline travelling alongside it.
    for i in range(20):
        entry = engine.record_prediction("BTCUSDT", "BULLISH" if i % 2 else "BEARISH",
                                         0.9 if i % 2 else 0.6, source="momentum_model")
        engine.resolve_prediction(entry["id"], "CORRECT" if i < 15 else "INCORRECT")

    report = engine.get_source_accuracy_report()
    assert report["total_predictions_evaluated"] == 20
    assert report["overall_directional_accuracy_pct"] == 75.0
    assert report["accuracy_status"] == "OBSERVED_ACCURACY_ABOVE_BASELINE"
    assert report["sub_model_accuracies"]["momentum_model"] == {
        "accuracy_pct": 75.0,
        "sample_size": 20,
        "status": "observed",
    }
    # Calibration is the Brier-based score over the stated confidences actually recorded.
    resolved = [
        e for e in engine.history
        if e.get("evidence_class") == "observed_prediction"
        and e.get("actual_outcome") in ("CORRECT", "INCORRECT")
    ]
    assert len(resolved) == 20
    expected_brier = sum(
        (float(e["confidence"]) - (1.0 if e["actual_outcome"] == "CORRECT" else 0.0)) ** 2
        for e in resolved
    ) / len(resolved)
    assert report["calibration_score_0_to_1"] == pytest.approx(round(1 - expected_brier, 3))
    assert "Brier" in report["calibration_basis"]


def test_sub_model_scores_below_the_sample_floor_are_withheld():
    engine = PredictionTrackingEngine()
    for i in range(MIN_OBSERVATIONS_FOR_SCORE):
        entry = engine.record_prediction("ETHUSDT", "BULLISH", 0.7, source="primary")
        engine.resolve_prediction(entry["id"], "CORRECT")
    thin = engine.record_prediction("ETHUSDT", "BULLISH", 0.7, source="thin_model")
    engine.resolve_prediction(thin["id"], "CORRECT")

    report = engine.get_source_accuracy_report()
    assert report["sub_model_accuracies"]["primary"]["accuracy_pct"] == 100.0
    assert report["sub_model_accuracies"]["thin_model"] == {
        "accuracy_pct": None,
        "sample_size": 1,
        "status": "insufficient_sample",
    }


def test_unresolved_predictions_do_not_count_as_correct():
    engine = PredictionTrackingEngine()
    for i in range(30):
        engine.record_prediction("BTCUSDT", "BULLISH", 0.8, source="m")
    report = engine.get_source_accuracy_report()
    assert report["observed_predictions_recorded"] == 30
    assert report["total_predictions_evaluated"] == 0, "no outcome yet: nothing to score"
    assert report["pending_resolution"] == 30
    assert report["overall_directional_accuracy_pct"] is None


def test_resolve_rejects_an_unknown_outcome():
    engine = PredictionTrackingEngine()
    entry = engine.record_prediction("BTCUSDT", "BULLISH", 0.8)
    with pytest.raises(ValueError):
        engine.resolve_prediction(entry["id"], "PROBABLY")
    assert engine.resolve_prediction("pred_does_not_exist", "CORRECT") is None


def test_accuracy_endpoint_serves_observed_only(monkeypatch, auth):
    engine = PredictionTrackingEngine()
    monkeypatch.setattr("app.routers.predictions.prediction_tracker", engine)
    client = TestClient(__import__("app.main", fromlist=["app"]).app)

    resp = client.get("/v1/intelligence/accuracy", headers=auth)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["accuracy_status"] == "INSUFFICIENT_DATA"
    assert body["overall_directional_accuracy_pct"] is None
    assert "78.4" not in resp.text and "0.88" not in resp.text


def test_prediction_history_is_labelled_and_serializable(monkeypatch, auth):
    engine = PredictionTrackingEngine()
    monkeypatch.setattr("app.routers.predictions.prediction_tracker", engine)
    client = TestClient(__import__("app.main", fromlist=["app"]).app)

    resp = client.get("/v1/predict/btc/history", headers=auth)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert isinstance(body["history"], list), "a bounded store must be serialized via to_list()"
    assert body["history"], "the demonstration rows are still served, but labelled"
    assert all(row.get("source") == "fixture" for row in body["history"])
    assert body["retention"]["fixture_rows"] == 4
    assert body["accuracy_summary"]["fixture_rows_excluded"] == 4


def test_prediction_without_supplied_inputs_withholds_confidence(auth):
    client = TestClient(__import__("app.main", fromlist=["app"]).app)
    resp = client.get("/v1/predict/btc", headers=auth)
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["inputs_simulated"] is True
    assert body["unified_confidence"] is None, "a confidence over invented inputs is fabricated"
    assert "withheld" in body["confidence_basis"]
    assert body["evidence_class"] == "synthetic_fixture_inputs"
    assert body["advisory_only"] is True
    assert body["disclaimer"]
    assert body["unified_direction"] in ("BULLISH", "BEARISH", "NEUTRAL")


def test_prediction_with_supplied_inputs_reports_a_number(auth):
    client = TestClient(__import__("app.main", fromlist=["app"]).app)
    resp = client.get(
        "/v1/predict/btc",
        headers=auth,
        params={"current_price": 65000.0, "recent_returns": "0.004,0.005,0.006,0.004,0.005"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["inputs_simulated"] is False
    assert body["unified_confidence"] is not None
    # The price/return legs are now real, but the four alternative-data legs are still a
    # static fixture table, and the payload has to say so.
    assert body["evidence_class"] == "caller_supplied_price_and_returns_with_synthetic_alt_data"
    assert body["synthetic_leg_count"] >= 4
    assert any("onchain_intelligence" in leg for leg in body["synthetic_legs"])

    bad = client.get("/v1/predict/btc", headers=auth, params={"recent_returns": "not,numbers"})
    assert bad.status_code == 422, "garbage inputs must be rejected, not silently replaced"


def test_no_confidence_floor_when_signals_contradict():
    """The old expression was max(0.60, ...): weak evidence looked strong."""
    from app.ml.prediction_aggregator import PredictionAggregationEngine

    engine = PredictionAggregationEngine()
    # A flat series produces a momentum t of 0 and no directional signal at all.
    result = engine.aggregate_prediction("BTCUSDT", 65000.0, [0.0, 0.0, 0.0, 0.0], inputs_simulated=False)
    assert result["unified_confidence"] is None or result["unified_confidence"] <= 0.6, (
        "contradictory evidence must not be reported at the old 0.60 floor"
    )


def test_alternative_data_declares_itself_synthetic(auth):
    """The intelligence snapshot must not read as live market observation."""
    client = TestClient(__import__("app.main", fromlist=["app"]).app)
    resp = client.get("/v1/intelligence/summary", headers=auth, params={"asset": "BTC"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["evidence_class"] == "synthetic_fixture"
    assert body["inputs_simulated"] is True
    assert body["asset_specific"] is False
    assert "static fixture table" in body["generator"]

    other = client.get("/v1/intelligence/summary", headers=auth, params={"asset": "DOGE"}).json()
    assert other["news_intelligence"] == body["news_intelligence"], (
        "the engine returns asset-independent constants; that is exactly why it is labelled"
    )


def test_deep_learning_payload_states_its_calibration():
    from app.ml.deep_models import deep_models_engine

    result = deep_models_engine.predict_horizons("BTCUSDT", 65000.0, [0.001, -0.002, 0.003])
    assert result["model_family"] == "statistical_econometrics"
    assert "not calibrated" in result["calibration"]
    assert result["sample_size"] == 3, "the payload must expose how many returns it saw"


def test_fallback_attempt_history_is_bounded():
    from app.inference_runtime.fallback_executor import FallbackExecutor

    executor = FallbackExecutor(max_history=5)
    from app.inference_runtime.fallback_executor import AttemptRecord

    for i in range(50):
        executor.history.append(AttemptRecord("p", "m", i, 0.1, None, True))
    assert len(executor.history) == 5
    assert executor.history.describe()["evicted"] == 45


def test_debate_metrics_publish_no_invented_quality_score():
    from app.monitoring import PerformanceMonitor

    monitor = PerformanceMonitor()
    metrics = monitor.get_debate_metrics()
    assert metrics["recommendation_quality_score"] is None
    assert "withheld" in metrics["recommendation_quality_basis"]
    assert metrics["evidence_class"] == "no_deliberations_observed"
    assert metrics["total_deliberations"] == 0
    assert metrics["debate_duration_p50_sec"] is None

    monitor.record_deliberation(1.5)
    monitor.record_deliberation(2.5)
    metrics = monitor.get_debate_metrics()
    assert metrics["total_deliberations"] == 2
    assert metrics["debate_duration_samples"] == 2
    assert metrics["debate_duration_p50_sec"] is not None
    assert metrics["evidence_class"] == "observed_deliberations"


def test_api_metrics_distinguish_unmeasured_from_zero():
    from app.monitoring import PerformanceMonitor

    monitor = PerformanceMonitor()
    empty = monitor.get_api_metrics()
    assert empty["samples"] == 0
    assert empty["error_rate_pct"] is None
    assert empty["p50_latency_sec"] is None
    assert empty["total_requests"] == 0
    assert "no request latencies" in empty["basis"]

    monitor.record_request(0.2, success=True)
    monitor.record_request(0.4, success=False)
    populated = monitor.get_api_metrics()
    assert populated["samples"] == 2
    assert populated["error_rate_pct"] == 50.0
    assert populated["latency_window"].startswith("most recent")


def test_dedup_cache_is_registered_and_ceiling_is_honoured():
    from app.governance.tenant_manager import MultiTenantManager
    from app.utils.bounded_store import audit_bounds

    manager = MultiTenantManager()
    assert manager.dedup_cache.max_entries == manager.max_dedup_entries
    registered = {store["name"] for store in audit_bounds()["stores"]}
    assert "governance.dedup_cache" in registered, (
        f"the biggest payload-holding cache must appear in the retention audit, saw {sorted(registered)}"
    )

    manager.max_dedup_entries = 5
    for i in range(50):
        manager.store_deduplication(f"req-{i}", {"payload": "x" * 100})
    assert len(manager.dedup_cache) == 5, "lowering the ceiling must trim immediately"
    assert manager.describe_dedup_cache()["evicted"] >= 45

    # A miss after eviction is described as evicted, not as never recorded.
    detail = manager.dedup_cache_miss_detail("req-0")
    assert "expired from the retention window" in detail or "evicted" in detail
