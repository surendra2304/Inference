"""The self-optimizing router's weights must decide something, and the decision must be honest.

[FACT] Before this suite, ``SelfOptimizingRouter`` published ``active_weights``,
a ``diversity_constraint`` ("Minimum 2 active providers per service; max weight cap at 0.70")
and ``explore_traffic_pct: 10.0`` on ``GET /v1/admin/routing/status`` while no code path read
any of it: the only consumers were the endpoints that displayed the table. The constraints were
also not enforced by the router's own adaptation code, which clamped each provider to 0.70
*before* normalising, so a service's shares could sum to more than 1.

The router now selects a provider (used by the trading-consult service when the agent's
preferred provider is unavailable) and these tests pin the arithmetic of that choice.
"""

from __future__ import annotations

import collections

import pytest

from app.routing.self_optimizer import (
    EXPLORE_TRAFFIC_PCT,
    MAX_WEIGHT_CAP,
    MIN_OBSERVED_CALLS_FOR_ADAPTATION,
    SelfOptimizingRouter,
)


def test_effective_weights_sum_to_one_and_respect_the_cap():
    router = SelfOptimizingRouter()
    for service in router.provider_weights:
        effective = router.normalize_weights(service)
        assert effective, f"{service} lost all weights during normalisation"
        assert sum(effective.values()) == pytest.approx(1.0, abs=0.01), (
            f"{service} effective weights sum to {sum(effective.values())}, not 1"
        )
        for provider, weight in effective.items():
            assert weight <= MAX_WEIGHT_CAP + 0.01, (
                f"{provider} in {service} exceeds the published cap: {weight}"
            )


def test_normalisation_survives_a_table_that_would_break_the_cap():
    """Two providers declared at 0.8 each: the cap must still hold after renormalisation."""
    router = SelfOptimizingRouter()
    router.provider_weights["stress"] = {"a": 0.8, "b": 0.8, "c": 0.4}
    effective = router.normalize_weights("stress")
    assert sum(effective.values()) == pytest.approx(1.0, abs=0.01)
    # With three providers the cap is satisfiable, so no provider may exceed it.
    assert max(effective.values()) <= MAX_WEIGHT_CAP + 0.01


def test_choose_provider_follows_the_weights():
    router = SelfOptimizingRouter()
    router._rng.seed(11)
    counts = collections.Counter(
        router.choose_provider("trading_consult", ["groq", "gemini", "openrouter"])[0]
        for _ in range(4000)
    )
    declared = router.normalize_weights("trading_consult")
    total = sum(counts.values())
    for provider, weight in declared.items():
        assert counts[provider] / total == pytest.approx(weight, abs=0.05), (
            f"{provider} received {counts[provider] / total:.2f} of decisions, declared {weight}"
        )


def test_exploration_budget_is_bounded_and_keeps_an_unlisted_provider_reachable():
    router = SelfOptimizingRouter()
    router._rng.seed(3)
    decisions = [
        router.choose_provider("trading_consult", ["groq", "gemini", "openrouter"])[1]
        for _ in range(2000)
    ]
    explored = sum(1 for d in decisions if d["method"] == "explore")
    assert explored / len(decisions) == pytest.approx(EXPLORE_TRAFFIC_PCT / 100.0, abs=0.03)

    # A provider with no declared weight must still be selectable, or a healthy provider the
    # table has never heard of can never receive traffic.
    picks = collections.Counter(
        router.choose_provider("trading_consult", ["nvidia", "groq"])[0] for _ in range(500)
    )
    assert picks["nvidia"] > 0, "an unlisted candidate was unreachable"


def test_choose_provider_reports_how_the_decision_was_made():
    router = SelfOptimizingRouter()
    _, single = router.choose_provider("trading_consult", ["groq"])
    assert single["method"] == "single_candidate"
    _, none = router.choose_provider("trading_consult", [])
    assert none["method"] == "no_candidates" and none["chosen"] is None

    chosen, weighted = router.choose_provider("trading_consult", ["groq", "gemini"])
    assert weighted["method"] in ("weighted", "explore")
    assert weighted["chosen"] == chosen
    assert chosen in weighted["candidates"]


def test_decisions_are_counted_so_idle_is_distinguishable_from_dead():
    router = SelfOptimizingRouter()
    assert router.get_routing_status()["decisions"] == {}, "a fresh router has made no decisions"
    router.choose_provider("trading_consult", ["groq", "gemini"])
    router.choose_provider("trading_consult", ["groq", "gemini"])
    status = router.get_routing_status()
    assert status["decisions"]["trading_consult"]["total"] == 2
    assert status["applied_to_request_routing"] is True
    assert "trading_consult" in status["applied_where"]
    assert status["last_decision"]["trading_consult"]["chosen"] in ("groq", "gemini")


@pytest.fixture(autouse=True)
def _isolate_observed_provider_calls():
    """The router's "observed health" evidence is the process-wide monitor, which every test
    that drives a provider call writes into. This file's health tests assert on an *empty*
    evidence base ("no observations must not move a weight"), so they must own that base:
    clear it for the duration of each test and restore it afterwards. Measured before this
    fixture: a successful call recorded by an unrelated test earlier in the suite made
    ``adapt_weights_from_observed_health`` report ``ADAPTED_FROM_OBSERVED_HEALTH``."""
    import copy

    from app.monitoring import monitor

    saved = copy.deepcopy(dict(monitor.provider_stats))
    monitor.provider_stats.clear()
    yield
    monitor.provider_stats.clear()
    monitor.provider_stats.update(saved)


def test_health_adaptation_needs_measured_calls():
    """No observations must not move a weight, and must not be reported as an adaptation."""
    router = SelfOptimizingRouter()
    before = dict(router.provider_weights["trading_consult"])
    result = router.adapt_weights_from_observed_health("trading_consult")
    assert result["action"] == "NO_SUFFICIENT_EVIDENCE", (
        "crediting an adaptation that changed nothing is the fabricated-prior mistake again"
    )
    assert router.provider_weights["trading_consult"] == before
    for provider, info in result["evidence"].items():
        assert info["applied"] is False
        assert (info["calls"] or 0) < MIN_OBSERVED_CALLS_FOR_ADAPTATION


def test_health_adaptation_moves_weights_only_for_measured_providers():
    from app.monitoring import monitor

    router = SelfOptimizingRouter()
    saved = dict(monitor.provider_stats)
    try:
        for _ in range(MIN_OBSERVED_CALLS_FOR_ADAPTATION):
            monitor.record_provider_call("groq", 0.2, success=True)
        for _ in range(MIN_OBSERVED_CALLS_FOR_ADAPTATION):
            monitor.record_provider_call("gemini", 0.2, success=False)

        result = router.adapt_weights_from_observed_health("trading_consult")
        assert result["action"] == "ADAPTED_FROM_OBSERVED_HEALTH"
        assert result["evidence"]["groq"]["applied"] is True
        assert result["evidence"]["gemini"]["applied"] is True
        assert result["evidence"]["openrouter"]["applied"] is False  # never observed
        effective = router.normalize_weights("trading_consult")
        assert effective["groq"] > effective["gemini"], (
            "a provider with a 100% observed success rate must outweigh one at 0%"
        )
        assert sum(effective.values()) == pytest.approx(1.0, abs=0.01)
    finally:
        monitor.provider_stats.clear()
        monitor.provider_stats.update(saved)
