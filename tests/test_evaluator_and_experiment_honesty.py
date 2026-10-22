"""The evaluation harness must not invent the judgement it failed to obtain.

Measured before the fix: when the LLM judge was unavailable — provider unreachable,
timeout, or a reply that was not JSON — ``app/evaluation/evaluator.py`` substituted six
hardcoded semantic scores (correctness 0.85, relevance 0.90, completeness 0.80,
reasoning_quality 0.85, evidence_quality 0.80, safety 1.0, strength
"Structured output generated successfully", confidence 0.80), averaged them together with
the deterministic latency/efficiency scores, and returned the report under
``evaluator_model="gemini:gemini-2.5-pro"`` — a model that had not run, scoring an answer
it had not seen. ``app/experiments/harness.py`` then published
``winner: "fast", score_difference: -0.15`` from those numbers, which reads as
"single-agent beat debate on reasoning quality".

Live evidence for the fixed path (local rig, judge reply is prose, not JSON)::

    winner                   : fast | score_difference: -0.15
    quality_verdict_withheld : True
    hypothesis_tested        : False
    verdict_basis            : deterministic dimensions only (latency/efficiency); the
                               semantic judge did not run, so no comparison of reasoning
                               quality was made
    fast_baseline            score=1.0 judge_ran=False dims=['latency','usage_efficiency']
                             judge_error: JSONDecodeError: Expecting value: line 1 column 2

These tests drive the evaluator with a stubbed gateway so both the failure path and the
recovery path are deterministic.
"""

import asyncio
import json

import pytest

from app.evaluation.evaluator import Evaluator
from app.providers.gateway import ModelGateway

# The six scores the old failure fallback invented, with their exact reasoning strings.
FABRICATED = {
    0.85: ["Heuristic baseline score", "Coherent structure"],
    0.90: ["Addressed core query"],
    0.80: ["Adequate coverage", "Consistent claims"],
    1.0: ["No unsafe operations detected"],
}


def _judge_reply(content: str):
    """Stub ``ModelGateway.execute`` to return ``content`` as the judge's answer.

    Patched on the **class**, matching ``tests/test_multi_agent_self_healing.py``. Patching
    the string ``"app.evaluation.evaluator.model_gateway.execute"`` writes an instance
    attribute on the shared gateway singleton, and monkeypatch's teardown restores it as a
    *bound method* — which then shadows the class attribute for the rest of the session. Any
    later ``monkeypatch.setattr(ModelGateway, "execute", ...)`` is bypassed, and six tests
    in test_multi_agent_self_healing/test_no_fabricated_results failed only when this file
    had been imported first.
    """
    from app.providers.base import ProviderResponse

    async def _execute(_self, *_args, **_kwargs):
        return ProviderResponse(
            content=content, model="stub-judge", provider="stub",
            total_tokens=10, finish_reason="stop",
        )

    return _execute


async def _noop_save(_record):
    return None


def _context():
    return {"mode_used": "fast", "latency_seconds": 0.5, "total_tokens": 300}


def test_a_failed_judge_produces_no_semantic_scores(monkeypatch):
    def raise_connection(_self, *_args, **_kwargs):
        raise ConnectionError("provider unreachable")

    monkeypatch.setattr(ModelGateway, "execute", raise_connection)
    report = asyncio.run(Evaluator().evaluate_answer("q", "a", _context()))

    assert report.judge_ran is False
    assert report.confidence is None, "a judge that did not run cannot calibrate a confidence"
    assert report.evaluator_model is None, "no model may be credited for the scores"
    assert "ConnectionError" in (report.judge_error or "")

    criteria = {s.criterion for s in report.scores}
    assert criteria == {"latency", "usage_efficiency"}, (
        f"only deterministic dimensions were measured, got {criteria}"
    )
    assert report.judge_dimensions == []
    assert report.deterministic_dimensions == ["latency", "usage_efficiency"]

    # None of the six invented scores, and none of their reasoning strings, may appear.
    blob = json.dumps([s.model_dump() for s in report.scores])
    for score, reasons in FABRICATED.items():
        for reason in reasons:
            assert reason not in blob, f"the invented reasoning {reason!r} survived"
    assert "Structured output generated successfully" not in json.dumps(report.model_dump())
    assert report.metadata["partial_evaluation"] is True
    assert "deterministic dimensions only" in report.metadata["overall_score_basis"]


def test_a_non_json_judge_reply_is_reported_as_a_partial_evaluation(monkeypatch):
    """This is the live case: the local rig answers the judge prompt with prose."""
    prose = "[specialist] round analysis\n\nThis is a boundary problem before it is an implementation problem."
    monkeypatch.setattr(ModelGateway, "execute", _judge_reply(prose))
    report = asyncio.run(Evaluator().evaluate_answer("q", "a", _context()))

    assert report.judge_ran is False
    assert "JSONDecodeError" in (report.judge_error or "")
    assert report.confidence is None
    assert {s.criterion for s in report.scores} == {"latency", "usage_efficiency"}


def test_a_prose_wrapped_json_judgement_is_recovered(monkeypatch):
    """A usable judgement must not be thrown away because of a leading sentence."""
    payload = {
        "scores": [
            {"criterion": "correctness", "score": 0.7, "reasoning": "mostly right"},
            {"criterion": "completeness", "score": 0.6, "reasoning": "missing edge cases"},
        ],
        "strengths": ["clear"],
        "flaws_identified": ["no measurement"],
        "calibrated_confidence": 0.66,
    }
    reply = f"Here is my assessment:\n{json.dumps(payload)}\nLet me know if you need more detail."
    monkeypatch.setattr(ModelGateway, "execute", _judge_reply(reply))
    report = asyncio.run(Evaluator().evaluate_answer("q", "a", _context()))

    assert report.judge_ran is True, report.judge_error
    assert report.judge_dimensions == ["correctness", "completeness"]
    assert report.confidence == pytest.approx(0.66)
    assert report.strengths == ["clear"]
    assert report.flaws_identified == ["no measurement"]
    assert "recovered JSON object" in (report.judge_error or ""), (
        "a recovered judgement should be flagged, not silently accepted as a clean parse"
    )


def test_out_of_range_judge_scores_are_rejected_not_clamped(monkeypatch):
    payload = {"scores": [{"criterion": "correctness", "score": 1.4, "reasoning": "x"}]}
    monkeypatch.setattr(ModelGateway, "execute", _judge_reply(json.dumps(payload)))
    report = asyncio.run(Evaluator().evaluate_answer("q", "a", _context()))

    assert report.judge_ran is False, "a score outside 0-1 is an unusable judgement"
    assert "outside 0.0-1.0" in (report.judge_error or "")
    assert 1.4 not in [s.score for s in report.scores]


def test_a_judge_payload_without_a_confidence_does_not_get_a_default(monkeypatch):
    payload = {"scores": [{"criterion": "correctness", "score": 0.5, "reasoning": "x"}]}
    monkeypatch.setattr(ModelGateway, "execute", _judge_reply(json.dumps(payload)))
    report = asyncio.run(Evaluator().evaluate_answer("q", "a", _context()))

    assert report.judge_ran is True
    assert report.confidence is None, "0.85 was previously invented for this field"


def test_the_benchmark_verdict_declares_when_quality_was_not_compared(monkeypatch):
    """The comparison payload must say what the score measured."""
    prose = "not json at all"
    monkeypatch.setattr(ModelGateway, "execute", _judge_reply(prose))

    from types import SimpleNamespace

    from app.experiments.harness import BenchmarkHarness

    harness = BenchmarkHarness()

    # Stub the orchestrator: this test is about how the *verdict* is built from two
    # evaluations, not about the provider path (which the other tests in this file cover
    # through the gateway stub above).
    async def _process_task(_request):
        return SimpleNamespace(
            answer="stubbed answer", confidence=0.8, total_latency_seconds=0.25,
            total_tokens=120, unresolved_disagreements=[],
        )

    # Via monkeypatch: assigning onto the shared orchestrator singleton directly leaked the
    # stub into every later test in the session (six tests in test_multi_agent_self_healing
    # and test_no_fabricated_results failed only in the full run, passing in isolation).
    monkeypatch.setattr(harness.orchestrator, "process_task", _process_task)
    monkeypatch.setattr(harness.memory, "save_experiment", _noop_save)
    record = asyncio.run(harness.run_baseline_vs_debate_comparison(question="Why do p99 spikes appear at 02:00?"))
    result = record.result

    assert result["quality_verdict_withheld"] is True
    assert result["hypothesis_tested"] is False
    assert "deterministic dimensions only" in result["verdict_basis"]
    for side in ("fast_baseline", "multi_agent_debate"):
        assert result[side]["judge_ran"] is False
        assert set(result[side]["scored_dimensions"]) == {"latency", "usage_efficiency"}
    # The hypothesis is about reasoning quality; a latency-only comparison cannot answer it.
    assert "reasoning quality" in result["verdict_basis"]


def test_the_shared_gateway_has_no_instance_level_execute_override():
    """Guard the trap above at the source.

    If anything sets ``execute`` on the gateway *instance* (a monkeypatch on the string
    "app.evaluation.evaluator.model_gateway.execute" does exactly that, and its teardown
    reinstates a bound method), the instance attribute shadows the class attribute forever:
    later class-level patches stop working and the real method can be invoked in tests that
    believe they stubbed it. Keep the class as the single patch point.
    """
    from app.providers.gateway import ModelGateway, model_gateway

    assert "execute" not in vars(model_gateway), (
        "the gateway singleton carries an instance-level 'execute', which shadows the class "
        f"method: {vars(model_gateway).get('execute')!r}"
    )
    assert ModelGateway.execute.__qualname__.startswith("ModelGateway."), (
        f"the class method must be the real implementation, found {ModelGateway.execute!r}"
    )
