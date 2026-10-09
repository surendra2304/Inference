"""Regression tests: the system must not publish measurements it never made.

Every defect here was found by driving the running service and reading the *values* it
returned, not by reading code. They share one root shape: a number or a claim that was
hardcoded, seeded, or templated, and was then presented to a caller as evidence,
performance, or deliberation.

Measured before the fixes (captured with ``scripts``-style probes against the live app):

* ``GET /v1/debate/statistics`` -> ``total_structured_debates: 24`` before one debate had
  been requested; ``objection_rate_pct: 24.5``; ``provider_diversity_impact`` claiming
  93.8% (multi-model) vs 81.2% (single-model) with ``diversity_lift_pct: 12.6``; and a
  ``composition_success_distribution`` of three invented compositions with invented rates
  (94.2 / 92.8 / 91.5).
* ``POST /v1/nexus/intelligence`` ``{"mode": "debate"}`` -> ``"Multi-round structured
  adversarial deliberation across debugger, security_analyst, critic (4 rounds
  executed)."`` with ``latency_ms: 0.1`` and **zero** model calls, plus ``confidence``
  taken from the literal list ``[0.82, 0.76, 0.88, 0.86]``.
* ``POST /v1/nexus/intelligence`` -> ``key_evidence`` citing
  ``[STRATEGY_BANK] Similar past situation for strategic_decision: Accelerated SDR routing
  ... (+14.5% conversion to closed-won ... 92% success)`` for an unrelated Postgres/SQLite
  architecture question, because ``query_strategy_bank`` fell back to the first two records
  when nothing matched.
* ``app/debate/enhanced_debate_protocol.py`` round-1 "agent outputs" were prefixed
  ``[GEMINI]`` / ``[GROQ]`` / ``[MISTRAL]`` from a list indexed by position — attributing
  text to providers that were never called.
"""

import json
import re

from fastapi.testclient import TestClient

from app.debate.enhanced_debate_protocol import enhanced_debate_engine
from app.main import app

# ── the debate engine must not invent its own history or performance ──────────


def test_engine_starts_with_no_recorded_history():
    """Counters must start at zero; a seeded history is a fabricated track record."""
    fresh = type(enhanced_debate_engine)()
    assert fresh.total_debates == 0, (
        "the engine seeded total_debates with 24 before any request — a track record that "
        "no run produced"
    )
    stats = fresh.get_debate_statistics()
    assert stats["total_structured_debates"] == 0
    assert stats["insufficient_data"], "zero observations must be reported as insufficient"
    assert stats["average_rounds_conducted"] is None, (
        "an average over zero samples must be null, not a number"
    )
    assert stats["objection_rate_pct"] is None


def test_statistics_report_observations_and_sample_sizes_only():
    """Every rate must be derived from real accumulations and carry its sample size."""
    fresh = type(enhanced_debate_engine)()

    async def run_one(rid: str) -> None:
        await fresh.execute_structured_debate(
            request_id=rid,
            task_type="incident_analysis",
            goal="Why did the nightly batch double in duration?",
            evidence=[{"evidence_id": "e1", "claim": "duration doubled", "trust_label": "system_fact"}],
            agents=["debugger", "security_analyst", "critic"],
        )

    import asyncio

    for i in range(3):
        asyncio.run(run_one(f"stat-{i}"))

    stats = fresh.get_debate_statistics()
    assert stats["total_structured_debates"] == 3
    assert stats["total_rounds_conducted"] == 12, "4 rounds x 3 debates, counted not assumed"
    assert stats["average_rounds_conducted"] == 4.0
    # Still below the reporting threshold, so no rate may be published.
    assert stats["insufficient_data"]
    assert stats["provider_diversity_impact"] is None

    # The invented study must be gone in every form.
    blob = json.dumps(stats)
    for fabricated in ("93.8", "81.2", "12.6", "24.5", "94.2", "92.8", "91.5"):
        assert fabricated not in blob, (
            f"{fabricated} was a hardcoded metric published as measurement: {blob[:300]}"
        )


def test_statistics_declare_that_no_model_is_consulted():
    """The endpoint must state what the engine is, next to the numbers it reports."""
    stats = enhanced_debate_engine.get_debate_statistics()
    assert stats["execution_engine"] == "deterministic_heuristic_template"
    assert stats["model_calls"] == 0
    assert "no model provider is invoked" in stats["measurement_note"].lower(), (
        "a consumer deciding whether panels help must be told this engine cannot observe it"
    )


def test_provider_allocation_is_flagged_unexercised_on_the_trace():
    """A planned routing table must not read as proof that providers ran."""
    import asyncio

    trace = asyncio.run(
        enhanced_debate_engine.execute_structured_debate(
            request_id="prov-alloc-1",
            task_type="strategic_decision",
            goal="Keep Postgres and SQLite in one service?",
            evidence=[],
            agents=["strategist", "critic"],
        )
    )
    assert trace.provider_allocation, "the plan is still recorded"
    assert trace.provider_allocation_exercised is False
    assert trace.model_calls == 0
    assert trace.execution_engine == "deterministic_heuristic_template"

    # And no round may wear a vendor tag it did not earn.
    for round_ in trace.rounds:
        for output in round_.agent_outputs.values():
            for vendor in ("GEMINI", "GROQ", "MISTRAL", "NVIDIA", "OPENROUTER"):
                assert f"[{vendor}]" not in output, (
                    f"round text attributes itself to {vendor}, which was never called: {output!r}"
                )


def test_confidence_evolution_is_derived_from_the_supplied_evidence():
    """A constant cannot be a trajectory.

    The previous value was the literal ``[0.82, 0.76, 0.88, 0.86]``; callers read
    ``confidence_evolution[-1]`` as the debate's confidence, so the reported confidence was
    0.86 for every input, including input with no evidence at all.
    """
    import asyncio

    async def trace_for(evidence):
        return await enhanced_debate_engine.execute_structured_debate(
            request_id=f"conf-{len(evidence)}",
            task_type="strategic_decision",
            goal="probe",
            evidence=evidence,
            agents=["strategist", "critic"],
        )

    strong = asyncio.run(trace_for([
        {"evidence_id": "e1", "claim": "measured", "trust_label": "system_fact"},
    ]))
    weak = asyncio.run(trace_for([
        {"evidence_id": "e1", "claim": "user said so", "trust_label": "untrusted_user_input"},
    ]))
    empty = asyncio.run(trace_for([]))

    assert strong.confidence_evolution != [0.82, 0.76, 0.88, 0.86], (
        "the hardcoded sequence is still present"
    )
    assert strong.confidence_evolution[-1] > weak.confidence_evolution[-1], (
        "trustworthy evidence must yield more confidence than untrusted input"
    )
    assert empty.confidence_evolution[-1] < weak.confidence_evolution[-1], (
        "no evidence must yield less confidence than weak evidence"
    )
    assert strong.confidence_evolution != empty.confidence_evolution


# ── the strategy bank must not present fixtures or strangers as precedent ─────


def test_seeded_strategy_entries_are_labelled_as_fixtures():
    from app.analytics.outcome_learning import outcome_learning_engine

    seeded = [e for e in outcome_learning_engine.strategy_bank if e.pattern_id.startswith("PAT-00")]
    assert seeded, "the shipped illustrative entries should still exist"
    for entry in seeded:
        assert entry.source == "illustrative_fixture", (
            f"{entry.pattern_id} describes a scenario that was never run and must say so"
        )


def test_unmatched_strategy_query_returns_nothing_not_a_stranger():
    """Returning an unrelated record as "similar" is worse than returning nothing."""
    from app.analytics.outcome_learning import outcome_learning_engine

    matches = outcome_learning_engine.query_strategy_bank(
        "strategic_decision", "Decide whether to keep Postgres and SQLite in one service."
    )
    assert matches == [], (
        f"an unmatched task type must match nothing, but returned {[m.get('pattern_id') for m in matches]}"
    )

    # A genuinely matching query still works.
    assert [m["pattern_id"] for m in outcome_learning_engine.query_strategy_bank(
        "lead_qualification", "inbound enterprise lead"
    )] == ["PAT-001"]


def test_composition_performance_is_computed_or_absent():
    """The comparison must come from recorded outcomes, with sample sizes."""
    from app.analytics.outcome_learning import outcome_learning_engine

    result = outcome_learning_engine.compute_agent_composition_performance()
    assert result["basis"] == "observed outcome reports only (source='observed')"
    # Fixture records must be excluded from the comparison and disclosed.
    assert result["records_fixture_excluded"] >= 4, (
        "the seeded demonstration reports must not be counted as runs"
    )
    assert "solo_sample_size" in result and "debate_sample_size" in result
    # The invented study must not reappear.
    blob = json.dumps(result)
    for fabricated in ("78.4", "91.2", "92.5", "90.8", "94.0"):
        assert fabricated not in blob, f"{fabricated} was never measured: {blob[:300]}"
    if result.get("insufficient_data"):
        assert result["diversity_lift_pct"] is None, (
            "no lift may be claimed without enough of both kinds of run"
        )
    for comp in result["top_performing_compositions"]:
        assert "sample_size" in comp, "a rate without its sample size invites overreach"


# ── end to end: the service must not claim deliberation it did not perform ────


def test_nexus_does_not_claim_model_deliberation(auth):
    """A 0.1 ms answer must not describe itself as multi-round adversarial deliberation."""
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(
        "/v1/nexus/intelligence",
        json={
            "request_id": "honesty-1",
            "task_type": "incident_analysis",
            "mode": "debate",
            "goal": "Why did the nightly batch double in duration after the schema change?",
        },
        headers={**auth, "Content-Type": "application/json"},
    )
    assert response.status_code < 500, response.text[:300]
    body = response.json()
    summary = body["summary"]
    provenance = body["provenance"]

    assert "adversarial deliberation across" not in summary, (
        f"no model was invoked, so this wording overstates what ran: {summary!r}"
    )
    assert provenance["model_calls"] == 0
    assert provenance["model_invoked"] is False
    assert provenance["execution_engine"] == "deterministic_heuristic_template"

    # The round count in the prose must equal the round count in the structured field.
    match = re.search(r"\((\d+) rounds executed", summary)
    assert match, f"the summary must state the round count it actually ran: {summary!r}"
    assert int(match.group(1)) == provenance["rounds_conducted"], (
        f"prose says {match.group(1)} rounds, provenance says "
        f"{provenance['rounds_conducted']} — the literal '4' used to contradict the field"
    )


def test_nexus_does_not_cite_unrelated_precedent(auth):
    """Evidence attached to a database question must not be a sales-funnel record."""
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(
        "/v1/nexus/intelligence",
        json={
            "request_id": "honesty-2",
            "task_type": "strategic_decision",
            "mode": "review",
            "goal": "Decide whether to keep Postgres and SQLite in the same service.",
        },
        headers={**auth, "Content-Type": "application/json"},
    )
    body = response.json()
    evidence = json.dumps(body.get("key_evidence", []))
    assert "SDR" not in evidence and "closed-won" not in evidence, (
        f"an unrelated sales record was cited as precedent for an architecture question: {evidence}"
    )
    if "STRATEGY_BANK" in evidence:
        assert "illustrative_fixture" in evidence, (
            "a shipped fixture must be labelled as such wherever it is cited"
        )


def test_nexus_fast_and_review_modes_still_answer(auth):
    """The honesty labels must not break the other two modes (regression guard)."""
    client = TestClient(app, raise_server_exceptions=False)
    for mode in ("fast", "review", "debate"):
        response = client.post(
            "/v1/nexus/intelligence",
            json={
                "request_id": f"modes-{mode}",
                "task_type": "strategic_decision",
                "mode": mode,
                "goal": "probe",
            },
            headers={**auth, "Content-Type": "application/json"},
        )
        assert response.status_code == 200, f"{mode}: {response.status_code} {response.text[:200]}"
        body = response.json()
        assert body["provenance"]["model_calls"] == 0


def test_cross_examination_and_objections_are_derived_from_the_evidence():
    """Rounds must say something specific to the request, not replay canned text.

    Measured before the fix: every debate — a database incident, a compliance question,
    anything — was cross-examined for "constant market liquidity", defended with "dynamic
    stop boundaries", and returned the single hardcoded objection "Telemetry must be
    re-sampled within 6 hours to confirm absence of drift" for every request, including
    requests carrying no telemetry at all.
    """
    import asyncio

    async def trace(label: str, evidence: list[dict]):
        return await enhanced_debate_engine.execute_structured_debate(
            request_id=f"rounds-{label}",
            task_type="incident_analysis",
            goal="Why did the nightly batch double in duration?",
            evidence=evidence,
            agents=["debugger", "security_analyst", "critic"],
        )

    strong = asyncio.run(trace("strong", [
        {"evidence_id": "e1", "claim": "duration 2.1h -> 4.3h", "trust_label": "system_fact"},
        {"evidence_id": "e2", "claim": "row count unchanged", "trust_label": "verified_telemetry"},
    ]))
    weak = asyncio.run(trace("weak", [
        {"evidence_id": "e9", "claim": "someone said it felt slow", "trust_label": "untrusted_user_input"},
    ]))
    empty = asyncio.run(trace("empty", []))

    blob = json.dumps([r.model_dump() for r in strong.rounds])
    for canned in ("market liquidity", "stop boundaries", "dynamic stop", "6 hours"):
        assert canned not in blob, f"canned text survived in the rounds: {canned!r}"

    # A challenge must name the evidence it is about.
    weak_challenges = json.dumps(weak.rounds[1].challenges)
    assert "e9" in weak_challenges, f"the challenge must identify the weak item: {weak_challenges}"

    # No evidence must be treated as exactly that, and must register an objection.
    assert empty.unresolved_objections, "an unevidenced conclusion must carry an objection"
    assert "no evidence" in empty.unresolved_objections[0].lower()

    # Strong evidence that is not contradictory leaves no unresolved objection.
    assert strong.unresolved_objections == [], (
        f"a panel with no unresolved objection must report none, not an invented one: "
        f"{strong.unresolved_objections}"
    )

    # The three inputs must not produce the same trace text. Round 4 is compared on the
    # objection wording rather than the round payload: "weak evidence" and "no evidence"
    # both register exactly one objection, so their round payloads are legitimately equal
    # while the objections themselves differ.
    assert json.dumps(strong.rounds[1].model_dump()) != json.dumps(weak.rounds[1].model_dump())
    assert weak.unresolved_objections != empty.unresolved_objections, (
        "a distrust objection and an absent-evidence objection are different statements"
    )

    # Assumptions must reference the evidence they rest on.
    assert any("e1" in a.hypothesis for a in strong.stated_assumptions), (
        "an assumption must name what it is assumed about"
    )


def test_a_challenge_never_targets_its_own_challenger():
    """A second exchange must come from a different agent than the first."""
    import asyncio

    trace = asyncio.run(
        enhanced_debate_engine.execute_structured_debate(
            request_id="self-challenge",
            task_type="incident_analysis",
            goal="probe",
            evidence=[
                {"evidence_id": "e1", "claim": "a", "trust_label": "system_fact"},
                {"evidence_id": "e2", "claim": "b", "trust_label": "verified_telemetry"},
            ],
            agents=["debugger", "security_analyst", "critic"],
        )
    )
    for challenge in trace.rounds[1].challenges:
        assert challenge["challenger"] != challenge["target"], (
            f"an agent cannot cross-examine itself: {challenge}"
        )
