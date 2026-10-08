"""Bounded stores, and the analytics that must not invent numbers.

Two defect families are covered here, both measured on the live service before the fixes.

**Unbounded accumulators.** A 60s soak at concurrency 20 (17,765 requests, 0 errors) grew
the agent's RSS from 77.9 MB to 143.2 MB, and it did not come back down (144.3 MB three
seconds after the load stopped). Per-route attribution with a ``gc.collect()`` before each
sample gave::

    nexus      +9.03 MB / 1k requests   retained +19.7 MB
    sentinel   +5.19 MB / 1k requests   retained +13.7 MB
    analytics  +2.42 MB / 1k requests   retained +10.4 MB
    instant    +0.50 MB / 1k requests
    market     +0.20 MB / 1k requests
    stress     +0.02 MB / 1k requests
    assist     +0.02 MB / 1k requests

Every leak was a module-level singleton with a plain ``{}``/``[]`` that requests appended
to and nothing removed. ``app/utils/bounded_store.py`` replaces them, and ``audit_bounds()``
makes the whole process auditable so a store added later cannot silently regress.

**Invented numbers in the analytics.** ``90.0`` pass rates for providers with zero
samples, ``92/94/95.0`` consumer rates for consumers with zero outcomes, four headline
"cross-consumer patterns" that were string literals ("improves accuracy 23% over
single-model across ALL consumers"), a four-row calibration curve labelled
``HONEST_CALIBRATED``, ``success_rate=0.90`` stamped on every recorded success, and a
pre-seeded routing log claiming "98% verification pass rate observed from FORGE builds".
Each test below pins one of them to a computed value or an explicit absence.
"""

import asyncio
import json

import pytest

from app.utils.bounded_store import (
    BoundedList,
    BoundedSeries,
    BoundedStore,
    audit_bounds,
)

# Small ceilings keep the tests fast; the process defaults are far larger.
TINY = 8


# ── the primitives ───────────────────────────────────────────────────────────


def test_bounded_store_evicts_least_recently_used_and_counts_it():
    store: BoundedStore[int] = BoundedStore("test.lru", max_entries=3)
    for i in range(5):
        store[i] = i
    assert len(store) == 3
    assert store.evictions == 2
    assert 0 not in store and 1 not in store
    assert store.was_evicted("0") is True, "an evicted key must be distinguishable from absent"
    assert store.was_evicted("4") is False

    # Reading a key protects it from the next eviction.
    store[2] = 20
    store[99] = 99
    assert (store[2] if "2" in store else None) == 20, "a recently read key should survive"

    described = store.describe()
    assert described["used"] <= described["max_entries"]
    assert described["within_bound"] is True
    assert described["evicted"] == store.evictions


def test_bounded_store_rejects_a_zero_ceiling():
    with pytest.raises(ValueError):
        BoundedStore("test.bad", max_entries=0)


def test_bounded_list_keeps_the_tail_and_reports_drops():
    items: BoundedList[int] = BoundedList("test.tail", max_entries=4)
    for i in range(10):
        items.append(i)
    assert len(items) == 4
    assert items.to_list() == [6, 7, 8, 9], "the newest entries must be the ones kept"
    assert items.dropped == 6
    assert items.describe()["within_bound"] is True


def test_bounded_list_accepts_initial_values():
    items: BoundedList[str] = BoundedList("test.initial", max_entries=4, initial=["a", "b"])
    assert items.to_list() == ["a", "b"]


def test_bounded_series_keeps_a_window_but_counts_every_sample():
    series = BoundedSeries("test.series", max_samples=5)
    for i in range(50):
        series.add(float(i))
    assert len(series) == 5
    assert series.dropped == 45
    assert series.count_total == 50, (
        "a percentile over a window must be reportable as 'of N observed'"
    )


def test_every_registered_store_is_within_its_bound():
    audit = audit_bounds()
    assert audit["store_count"] > 0
    assert audit["within_bound"] is True, f"stores exceeded their ceilings: {audit['overflowing']}"
    for row in audit["stores"]:
        assert row["used"] <= row["max_entries"], row


# ── the engines that leaked ──────────────────────────────────────────────────


def test_nexus_provenance_is_bounded_and_says_what_it_dropped():
    from app.services.nexus_intelligence import NexusIntelligenceService

    engine = NexusIntelligenceService()
    engine.provenance_store._max_entries = TINY  # type: ignore[attr-defined]
    for i in range(TINY * 3):
        engine.provenance_store[f"req-{i}"] = {"i": i}
    described = engine.provenance_retention()
    assert described["used"] <= TINY
    assert described["evicted"] == TINY * 2
    assert engine.provenance_store.was_evicted("req-0") is True
    assert engine.get_provenance("not-a-request") is None


def test_sentinel_provenance_is_bounded():
    from app.services.sentinel_intelligence import SentinelIntelligenceService

    engine = SentinelIntelligenceService()
    engine.provenance_store._max_entries = TINY  # type: ignore[attr-defined]
    for i in range(TINY * 4):
        engine.provenance_store[f"s-{i}"] = {"i": i}
    described = engine.provenance_retention()
    assert described["used"] <= TINY
    assert described["evicted"] == TINY * 3


def test_debate_traces_are_bounded():
    from app.debate.enhanced_debate_protocol import enhanced_debate_engine

    engine = enhanced_debate_engine
    engine.reasoning_traces._max_entries = TINY  # type: ignore[attr-defined]

    async def run(i: int):
        return await engine.execute_structured_debate(
            request_id=f"bounded-{i}",
            task_type="incident_analysis",
            goal="probe",
            evidence=[{"claim": "x", "trust_label": "system_fact"}],
            agents=["debugger", "critic"],
        )

    for i in range(TINY * 3):
        asyncio.run(run(i))

    retention = engine.trace_retention()
    assert retention["used"] <= TINY, "traces must not accumulate without limit"
    assert retention["evicted"] >= TINY * 2
    assert engine.trace_was_evicted("bounded-0") is True


def test_outcome_ledgers_are_bounded():
    from app.analytics.outcome_learning import OutcomeLearningEngine

    engine = OutcomeLearningEngine()
    engine.outcome_records._data = type(engine.outcome_records._data)(maxlen=TINY)  # type: ignore[attr-defined]
    from app.analytics.outcome_learning import DetailedOutcomeReport

    for i in range(TINY * 3):
        engine.record_outcome(
            DetailedOutcomeReport(
                consumer="forge", request_id=f"r-{i}", outcome="success",
                provider_used="groq", task_type="code_generation",
                agent_composition=["coder"], source="observed",
            )
        )
    assert len(engine.outcome_records) <= TINY
    assert engine.outcome_records.dropped >= TINY * 2
    # The drop count must reach the payload, not just the object.
    stats = engine.compute_provider_performance()
    assert stats["records_dropped_for_capacity"] >= TINY * 2
    assert stats["records_retained"] <= TINY


# ── invented analytics ───────────────────────────────────────────────────────


def test_no_provider_is_given_a_pass_rate_without_samples():
    from app.analytics.outcome_learning import OutcomeLearningEngine

    engine = OutcomeLearningEngine()
    for record in list(engine.outcome_records):  # fixtures only
        assert record.source == "fixture"

    stats = engine.compute_provider_performance()
    for provider, row in stats["providers"].items():
        if row["total_samples"] == 0:
            assert row["success_rate_pct"] is None, (
                f"{provider} has no observed outcomes and must not be given a rate "
                f"(was 90.0 before the fix): {row}"
            )
            assert "insufficient_data" in row
        else:
            assert row["success_rate_pct"] is not None
    assert "90.0" not in json.dumps(stats["providers"]), "the fabricated prior must be gone"


def test_no_consumer_is_given_a_rate_without_evaluations():
    from app.analytics.outcome_learning import OutcomeLearningEngine

    engine = OutcomeLearningEngine()
    insights = engine.get_cross_consumer_insights()
    for consumer, row in insights["consumer_quality_metrics"].items():
        if row["total_evaluations"] == 0:
            assert row["success_rate_pct"] is None, (
                f"{consumer} has no observed outcomes; 92/94/95% were previously invented: {row}"
            )
            assert row["alert_status"] == "NO_DATA"


def test_cross_consumer_patterns_are_derived_or_withheld():
    from app.analytics.outcome_learning import OutcomeLearningEngine

    engine = OutcomeLearningEngine()
    insights = engine.get_cross_consumer_insights()

    blob = json.dumps(insights["cross_consumer_patterns"])
    for invented in ("23%", "91%", "71%", "85%", "78%"):
        assert invented not in blob, f"the invented pattern statistic {invented!r} survived"

    if insights["cross_consumer_patterns"]:
        # If a pattern is stated at all it must carry its sample size.
        assert all("observed report" in p for p in insights["cross_consumer_patterns"])
    else:
        assert "patterns_insufficient_data" in insights


def test_calibration_curve_is_computed_from_reported_confidence():
    from app.analytics.outcome_learning import DetailedOutcomeReport, OutcomeLearningEngine

    engine = OutcomeLearningEngine()
    # With only fixtures on record, every bin must abstain rather than borrow a number.
    baseline = engine.get_confidence_calibration()
    for row in baseline["calibration_curve"]:
        assert row["empirical_success_rate"] is None, (
            f"a calibration row with no data must not state a rate: {row}"
        )
        assert row["status"] == "INSUFFICIENT_DATA"

    # Now feed it enough real reports for one bin and check the arithmetic.
    for i in range(12):
        engine.record_outcome(
            DetailedOutcomeReport(
                consumer="forge", request_id=f"cal-{i}", outcome="success" if i < 6 else "failure",
                stated_confidence=0.95, provider_used="groq", task_type="code_generation",
                agent_composition=["coder"], source="observed",
            )
        )
    curve = {row["confidence_bin"]: row for row in engine.get_confidence_calibration()["calibration_curve"]}
    bin_90 = curve["0.90 - 1.00"]
    assert bin_90["sample_size"] == 12
    assert bin_90["empirical_success_rate"] == pytest.approx(0.5, abs=0.01)
    assert bin_90["stated_avg"] == pytest.approx(0.95, abs=0.001)
    assert bin_90["status"] == "OVERCONFIDENT", (
        "stating 0.95 and succeeding 50% of the time is overconfidence, whatever the old "
        "hardcoded table said"
    )


def test_recorded_success_does_not_claim_a_90_percent_rate():
    from app.analytics.outcome_learning import DetailedOutcomeReport, OutcomeLearningEngine

    engine = OutcomeLearningEngine()
    engine.record_outcome(
        DetailedOutcomeReport(
            consumer="forge", request_id="rate-1", outcome="success",
            detail="build_passed", provider_used="groq", task_type="code_generation",
            agent_composition=["coder"], source="observed",
        )
    )
    entry = [e for e in engine.strategy_bank if e.pattern_id.startswith("PAT-") and "-" in e.pattern_id][-1]
    assert entry.success_rate is None, (
        "an outcome report states whether it succeeded, not how often the pattern does"
    )
    assert entry.recommendation is None, "no recommendation was given; none may be invented"
    assert entry.source == "observed"

    # When a rate *is* reported, it is used.
    engine.record_outcome(
        DetailedOutcomeReport(
            consumer="forge", request_id="rate-2", outcome="success",
            detail="build_passed", provider_used="groq", task_type="code_generation",
            agent_composition=["coder"], source="observed",
            measured_metrics={"success_rate_pct": 62.5},
        )
    )
    measured = [e for e in engine.strategy_bank if e.outcome_summary.find("62.5") >= 0]
    assert measured, "a measured rate must be recorded rather than discarded"
    assert measured[0].success_rate == pytest.approx(0.625, abs=0.001)


def test_pattern_ids_are_unique_within_a_millisecond():
    from app.analytics.outcome_learning import DetailedOutcomeReport, OutcomeLearningEngine

    engine = OutcomeLearningEngine()
    for i in range(25):
        engine.record_outcome(
            DetailedOutcomeReport(
                consumer="forge", request_id=f"dup-{i}", outcome="success",
                provider_used="groq", task_type="code_generation",
                agent_composition=["coder"], source="observed",
            )
        )
    ids = [e.pattern_id for e in engine.strategy_bank]
    assert len(ids) == len(set(ids)), "PAT-<seconds> collided for same-second outcomes"


def test_fixture_outcomes_are_labelled_and_excluded_from_statistics():
    from app.analytics.outcome_learning import outcome_learning_engine

    fixtures = [r for r in outcome_learning_engine.outcome_records if r.source == "fixture"]
    assert len(fixtures) >= 4, "the demonstration rows should still be present, labelled"
    for record in fixtures:
        assert record.request_id.startswith("init-")

    composition = outcome_learning_engine.compute_agent_composition_performance()
    assert composition["records_fixture_excluded"] >= 4
    assert composition["solo_sample_size"] + composition["debate_sample_size"] == composition["records_observed"] - composition["unstated_composition_runs"]


def test_unreported_fields_are_not_defaulted_to_claims():
    """Defaults must mean "not reported", never a specific vendor or task type."""
    from app.analytics.outcome_learning import DetailedOutcomeReport
    from app.analytics.outcomes import OutcomeReportRequest

    report = DetailedOutcomeReport(consumer="forge", request_id="minimal", outcome="success")
    assert report.provider_used is None, "a report that named no provider must not claim gemini"
    assert report.task_type is None
    assert report.agent_composition is None
    assert report.stated_confidence is None
    assert report.detail is None

    legacy = OutcomeReportRequest(consumer="forge", request_id="minimal", outcome="success")
    assert legacy.provider_used is None
    assert legacy.service is None
    assert legacy.detail is None


def test_router_reports_no_data_instead_of_assuming_a_ninety_percent_pass_rate():
    from app.routing.self_optimizer import SelfOptimizingRouter

    router = SelfOptimizingRouter()
    assert router.optimization_logs.to_list() == [], "no rebalance may be pre-seeded"

    before = dict(router.provider_weights["code_generation"])
    router.adapt_weights_from_outcomes()
    after = dict(router.provider_weights["code_generation"])
    assert after == before, "weights must not move on fabricated data"

    entry = router.optimization_logs.to_list()[-1]
    assert entry["action"] == "REBALANCE_SKIPPED"
    assert "90.0" not in json.dumps(entry)
    assert entry["rationale"], "a skip must state why"


def test_router_rejects_a_pre_seeded_verification_claim():
    from app.routing.self_optimizer import SelfOptimizingRouter

    status = SelfOptimizingRouter().get_routing_status()
    blob = json.dumps(status)
    assert "98%" not in blob, "the seeded '98% verification pass rate' log entry must be gone"


def test_legacy_outcome_tracker_has_no_default_prior():
    from app.analytics.outcomes import ConsumerOutcomeTracker

    tracker = ConsumerOutcomeTracker()
    summary = tracker.get_outcome_summary()
    assert summary["total_outcomes_reported"] == 0, "fixture rows are not observations"
    assert summary["fixture_rows_excluded"] == 3
    for provider, row in summary["provider_verification_pass_rates"].items():
        assert row["verification_pass_rate_pct"] is None, (
            f"{provider} had no observed outcomes and previously defaulted to 90.0: {row}"
        )
    assert summary["overall_downstream_success_rate_pct"] is None


def test_legacy_tracker_history_is_bounded():
    from app.analytics.outcomes import ConsumerOutcomeTracker, OutcomeReportRequest

    tracker = ConsumerOutcomeTracker()
    tracker.outcome_history._data = type(tracker.outcome_history._data)(maxlen=TINY)  # type: ignore[attr-defined]
    for i in range(TINY * 3):
        tracker.record_outcome(
            OutcomeReportRequest(consumer="forge", request_id=f"legacy-{i}", outcome="success")
        )
    assert len(tracker.outcome_history) <= TINY
    assert tracker.outcome_history.dropped >= TINY * 2


def test_consumer_outcomes_are_not_relabelled_on_the_way_in():
    """A nexus outcome recorded as a forge outcome corrupts per-consumer statistics."""
    import inspect

    from app.routers import admin_analytics

    source = inspect.getsource(admin_analytics.report_consumer_outcome)
    assert 'consumer="forge" if req.consumer not in' not in source, (
        "the relabelling branch must be gone"
    )
    assert 'or "gemini"' not in source and 'or "code_generation"' not in source, (
        "an unreported provider/task must not be filled in with a concrete value"
    )


def test_provenance_misses_distinguish_eviction_from_absence():
    from app.utils.bounded_store import BoundedStore, missing_entry_detail

    store: BoundedStore[int] = BoundedStore("test.detail", max_entries=2)
    store["a"] = 1
    store["b"] = 2
    store["c"] = 3  # evicts "a"

    evicted = missing_entry_detail(store, "a", "record")
    never = missing_entry_detail(store, "zzz", "record")
    assert "expired" in evicted and "retention" in evicted
    # This store has evicted a key, so a miss for another key is ambiguous: it must say
    # both possibilities rather than claim the key never existed.
    assert "either" in never and "evicted" in never
    assert evicted != never


def test_a_miss_message_cannot_claim_never_recorded_after_evictions():
    """After evictions the store cannot tell "expired" from "never existed" for old keys.

    Measured live: 2,601 unique nexus request ids evicted 1,106 provenance entries, and
    the lookup for the first (successful, HTTP 200) request answered "No ... was ever
    recorded" because the key had aged out of a 64-key eviction memory. A response must
    not assert a fact the store does not have.
    """
    from app.utils.bounded_store import BoundedStore, missing_entry_detail

    store: BoundedStore[int] = BoundedStore("test.ambiguous", max_entries=2)
    # Fill the eviction memory with capacity evictions of other keys.
    for i in range(store.EVICTED_KEY_MEMORY + 5):
        store[f"k{i}"] = i
    assert store.evictions > 0
    message = missing_entry_detail(store, "k0", "record")
    assert "either" in message and "evicted" in message, (
        f"an ambiguous miss must present both possibilities, not assert one: {message}"
    )
    assert "Re-run" in message, "the caller must be told how to regenerate it"

    # A store that has evicted nothing can state absence confidently — that is the only
    # case where "never recorded" is a fact the store actually has.
    fresh: BoundedStore[int] = BoundedStore("test.absent", max_entries=4)
    fresh["a"] = 1
    fresh_message = missing_entry_detail(fresh, "zzz", "record")
    assert "No record was recorded" in fresh_message and "none evicted" in fresh_message, (
        fresh_message
    )


def test_a_recently_evicted_key_is_reported_as_expired():
    from app.utils.bounded_store import BoundedStore, missing_entry_detail

    store: BoundedStore[int] = BoundedStore("test.recent", max_entries=2)
    store["a"], store["b"], store["c"] = 1, 2, 3  # evicts "a"
    message = missing_entry_detail(store, "a", "record")
    assert "expired" in message and "Re-run" in message


def test_consumer_usage_does_not_invent_token_counts():
    """The consumer ledger used to receive fixed token counts from four services.

    [FACT] nexus/intelx/sentinel/futuris called ``record_usage`` with
    ``tokens=650/600/550/500`` while none of them contains a provider call, and the admin view
    (``GET /v1/admin/usage``) summed those constants into ``total_tokens`` and derived
    ``estimated_cost_usd`` from them. An unmeasured call now reports ``tokens=None`` and
    increments ``unmeasured_calls`` instead.
    """
    from app.routing.consumer_router import MultiConsumerRouter

    router = MultiConsumerRouter()
    fresh = router.get_usage("nexus")
    assert fresh["total_tokens"] is None, "an unmeasured consumer must not report a token total"
    assert fresh["estimated_cost_usd"] is None
    assert fresh["total_calls"] == 0

    router.record_usage("nexus", tokens=None, latency_sec=0.12)
    unmeasured = router.get_usage("nexus")
    assert unmeasured["total_calls"] == 1
    assert unmeasured["total_tokens"] is None
    assert unmeasured["unmeasured_calls"] == 1
    assert unmeasured["estimated_cost_usd"] is None

    router.record_usage("nexus", tokens=150, latency_sec=0.2)
    measured = router.get_usage("nexus")
    assert measured["total_tokens"] == 150
    assert measured["tokens_measured_calls"] == 1
    assert measured["unmeasured_calls"] == 1
    # Only the measured call contributes cost: 150 tokens at $0.0005/1k.
    assert measured["estimated_cost_usd"] == 7.5e-05


def test_services_stop_sending_placeholder_token_counts():
    """The call sites themselves must not carry the constants back in."""
    import inspect

    from app.services import (
        futuris_enhancement,
        intelx_intelligence,
        nexus_intelligence,
        sentinel_intelligence,
    )

    for module in (nexus_intelligence, intelx_intelligence, sentinel_intelligence, futuris_enhancement):
        source = inspect.getsource(module)
        for constant in ("tokens=650", "tokens=600", "tokens=550", "tokens=500"):
            assert constant not in source, f"{module.__name__} still reports {constant}"


def test_cost_report_states_its_basis_and_does_not_project_from_one_sample():
    """The cost report used to publish ``total_cost * 30`` as a monthly projection and a
    hardcoded ``anomaly_detected: False``.

    With one call recorded, that produced a month-end figure from a single sample; with no
    calls it printed $0.00, which reads like "this deployment is free". It now needs an
    observation window, names the price basis of every included record, and reports the
    anomalies it actually raised.
    """
    from app.analytics.cost_tracking import ProviderCostTracker
    from app.utils.bounded_store import _REGISTRY  # noqa: PLC2701 - test-only registration check

    tracker = ProviderCostTracker()
    empty = tracker.get_cost_report()
    assert empty["projected_monthly_spend_usd"] is None
    assert "guess" in empty["projection_basis"]
    assert empty["spend_basis"] == ["no observed call"]
    assert empty["anomaly_detected"] is False

    tracker.log_cost_event("groq", "trading_bot", "trading_consult", 0.0005, True)
    one_sample = tracker.get_cost_report()
    assert one_sample["projected_monthly_spend_usd"] is None, (
        "a single call is not an observation window"
    )
    assert one_sample["spend_basis"] == ["declared_estimate"]
    assert one_sample["all_observed_costs_are_estimates"] is True

    # With a real observation window (two measured calls two hours apart) a projection is
    # allowed, and is still labelled with its basis. The trailing records are the observed
    # ones; the first four are labelled fixtures and are excluded from every aggregate.
    tracker.log_cost_event("groq", "trading_bot", "trading_consult", 0.0005, True)
    tracker.records[-2].timestamp -= 7200
    windowed = tracker.get_cost_report()
    assert windowed["projected_monthly_spend_usd"] is not None
    assert "spanning" in windowed["projection_basis"]
    assert windowed["observed_calls"] == 2
    # The tracker registers itself in the process-wide audit; drop this instance's row so the
    # test does not make every later `audit_bounds()` call see a duplicate store.
    _REGISTRY.remove(tracker.records)


# --------------------------------------------------------------------------------------
# #63: usage analytics must not invent a confidence for a caller that reported none.
# --------------------------------------------------------------------------------------
# The old default was ``confidence: float = 0.90``. Every request logged through
# ``log_request`` therefore carried a measured-looking 0.90 even when the caller measured
# nothing, and any average over the field was an average over a constant dressed up as a
# measurement. The field is now ``float | None = None`` and ``get_overview`` publishes how
# many records actually carry one.


def test_usage_analytics_does_not_default_confidence_to_anything():
    from app.analytics.usage_analytics import UsageAnalyticsEngine

    engine = UsageAnalyticsEngine()
    engine.records.clear()
    record = engine.log_request(consumer="trading", service="consult", tokens_in=100, tokens_out=20)
    assert record.confidence is None, (
        "a request whose caller reported no confidence must not be assigned one; "
        f"got {record.confidence!r}"
    )


def test_usage_analytics_reports_confidence_coverage_not_a_filled_in_default():
    from app.analytics.usage_analytics import UsageAnalyticsEngine

    engine = UsageAnalyticsEngine()
    engine.records.clear()
    engine.log_request(consumer="trading", service="consult", tokens_in=100, tokens_out=20)
    engine.log_request(consumer="trading", service="consult", tokens_in=50, tokens_out=10,
                       confidence=0.72)
    overview = engine.get_overview()
    assert overview["measurement_coverage"]["confidence"] == 1, (
        "coverage must count the one record that actually carried a confidence, so a reader "
        "can tell 0.72 was measured and the other record was not"
    )
    assert overview["total_calls"] == 2


def test_usage_analytics_round_trips_a_reported_confidence():
    from app.analytics.usage_analytics import UsageAnalyticsEngine

    engine = UsageAnalyticsEngine()
    engine.records.clear()
    record = engine.log_request(consumer="forge", service="code_generation", confidence=0.61)
    assert record.confidence == 0.61
