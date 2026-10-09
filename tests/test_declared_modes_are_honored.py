"""/v1/ask must honour the mode it accepts, and must not default to invented confidence.

Measured live before the fix, with the same question and the same payload:

==============================  ===========  ===========  ===============  ==========
mode                            mode_used    agents       confidence       distinct?
==============================  ===========  ===========  ===============  ==========
fast                            (absent)     2            0.65             -
deliberative                    (absent)     2            0.65             same answer
consensus                       (absent)     2            0.65             same answer
review                          (absent)     2            0.65             same answer
banana                          (absent)     2            0.70             HTTP 200
==============================  ===========  ===========  ===============  ==========

``mode`` reached exactly one line — ``fast_lane=(request.mode == "fast")`` — so four of
the five documented modes produced identical single-call answers, nothing in the response
said which mode had run, and ``{"mode": "banana"}`` was answered with HTTP 200. The
response model also defaulted ``confidence`` to 0.90 and ``uncertainty`` to 0.10, so any
construction site that omitted them published an invented claim.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.schemas.v1_models import InferenceTaskResponse

QUESTION = "should we shard the write path?"


def _client() -> TestClient:
    from app.main import app

    return TestClient(app)


@pytest.fixture
def stub_deliberation(monkeypatch):
    """Capture the orchestrator call a multi-agent mode makes, and skip DB writes."""
    calls: dict = {}

    async def _process_task(request):
        calls["mode"] = request.mode
        calls["question"] = request.question
        return SimpleNamespace(
            task_id="task_stubbed",
            run_id="deb_stubbed",
            answer="stubbed deliberation answer",
            confidence=0.83,
            mode_used=request.mode,
            deliberation_outcome="consensus",
            agents_used=["researcher", "critic"],
            models_used=["rig-8b"],
            unresolved_disagreements=[],
            key_evidence=["stub evidence"],
            total_tokens=42,
            provider_used="local",
            degraded=False,
            degradation_reasons=[],
            failed_agents=[],
            agent_coverage={"researcher": "ok", "critic": "ok"},
            total_latency_seconds=0.1,
        )

    recorded: list[dict] = []

    async def _record_task(task_record):
        recorded.append({"id": task_record.id, "mode": task_record.mode, "status": task_record.status})

    from app.core.orchestrator import orchestrator

    monkeypatch.setattr(orchestrator, "process_task", _process_task)
    monkeypatch.setattr(orchestrator, "record_task", _record_task)
    return calls, recorded


def test_an_unknown_mode_is_rejected(auth):
    """The field documents four values; a fifth must not be silently accepted."""
    resp = _client().post("/v1/ask", headers=auth, json={"question": QUESTION, "mode": "banana"})
    assert resp.status_code == 422, resp.text
    detail = resp.json()["detail"][0]
    assert detail["type"] == "literal_error"
    assert "review" in detail["msg"] or "review" in str(detail.get("ctx", {}))


def test_multi_agent_modes_reach_the_deliberation_engine(stub_deliberation, auth):
    calls, recorded = stub_deliberation
    for mode, expected_engine_mode in (("review", "review"), ("debate", "debate")):
        calls.clear()
        resp = _client().post("/v1/ask", headers=auth, json={"question": QUESTION, "mode": mode})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert calls["mode"] == expected_engine_mode, (
            f"mode={mode} must run the engine as {expected_engine_mode}, saw {calls.get('mode')}"
        )
        assert body["mode_requested"] == mode
        assert body["mode_used"] == expected_engine_mode
        assert body["confidence"] == 0.83, "the engine's own confidence must be published"
        assert recorded and recorded[-1]["mode"] == expected_engine_mode


def test_deliberative_and_consensus_disclose_their_mapping(stub_deliberation, auth):
    calls, _ = stub_deliberation
    for mode in ("deliberative", "consensus"):
        calls.clear()
        resp = _client().post("/v1/ask", headers=auth, json={"question": QUESTION, "mode": mode})
        body = resp.json()
        assert body["mode_requested"] == mode
        assert body["mode_used"] == "review"
        assert body["mode_mapping_note"], "a mapped mode must say so in the response"
        assert mode in body["mode_mapping_note"]


def test_answers_differ_between_fast_and_debate(stub_deliberation, auth):
    """The original complaint: three modes, one answer. Now the paths differ."""
    fast = _client().post("/v1/ask", headers=auth, json={"question": QUESTION, "mode": "fast"}).json()
    debate = _client().post("/v1/ask", headers=auth, json={"question": QUESTION, "mode": "debate"}).json()

    assert fast["mode_used"] == "fast"
    assert debate["mode_used"] == "debate"
    assert fast["answer"] != debate["answer"]


def test_confidence_must_be_stated_not_defaulted():
    fields = InferenceTaskResponse.model_fields
    assert fields["confidence"].is_required(), "a default confidence is an invented claim"
    assert fields["uncertainty"].is_required()
    with pytest.raises(Exception):
        InferenceTaskResponse(answer="x")  # type: ignore[call-arg]
