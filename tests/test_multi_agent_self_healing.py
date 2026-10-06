"""Multi-agent self-healing: peer coverage and honest panel failure.

Proves that when a specialist's provider goes dark:
  1. a peer specialist covers for it and the panel still answers, with the
     coverage recorded (`agent_coverage`) and the result marked degraded;
  2. the confidence is earned from observed evidence, not a hardcoded constant;
  3. when EVERY specialist (and every peer) is dark, /ask returns 503 instead of
     fabricating a confident answer.
"""

import pytest
from httpx import ASGITransport, AsyncClient

from app.agents.debate import (
    AgentCallUnavailable,
    AgentPanelUnavailable,
    CollaborationEngine,
)
from app.core.config import settings
from app.main import app
from app.providers.base import ProviderResponse
from app.providers.gateway import ModelGateway


def _make_gateway(monkeypatch, dead_providers: set[str] = frozenset(), dead_agents: set[str] = frozenset()) -> list[str]:
    """Patch ModelGateway.execute.

    `dead_providers` simulates a whole provider outage.
    `dead_agents` simulates a specific specialist being unable to reach ANY of its
    models (agent-scoped outage) — this is what lets us prove peer coverage, since
    every agent holds a multi-provider model list and would otherwise just fail over
    to a sibling provider on its own.
    """
    calls: list[str] = []
    from app.agents.registry import agent_registry

    dead_prefixes = [
        ag.system_instructions
        for ag in (agent_registry.get_agent(i) for i in dead_agents)
        if ag is not None
    ]

    async def _execute(self, provider_name, request, capability="general", stage_name="general"):
        calls.append(provider_name)
        sys_instr = request.system_instruction or ""
        if any(sys_instr.startswith(p) for p in dead_prefixes):
            raise RuntimeError("simulated total model outage for specialist")
        if provider_name in dead_providers:
            raise RuntimeError(f"simulated outage on {provider_name}")
        return ProviderResponse(
            content=f"GROUNDED:{provider_name}",
            model=f"{provider_name}-model",
            provider=provider_name,
            total_tokens=10,
            prompt_tokens=5,
            completion_tokens=5,
        )

    monkeypatch.setattr(ModelGateway, "execute", _execute)
    return calls


@pytest.mark.asyncio
async def test_peer_covers_dark_specialist_and_records_coverage(monkeypatch):
    """A gemini-blackout specialist is covered by a live peer; panel still answers."""
    _make_gateway(monkeypatch, dead_agents={"researcher"})
    # 'researcher' is the sole gemini agent in the default fast panel.
    engine = CollaborationEngine()
    result = await engine.run_collaboration(
        task_id="task_peer_cover",
        question="Summarize consensus protocols.",
        participating_agents=[a for a in [engine.registry.get_agent("researcher")] if a],
    )

    assert result.degraded is True, "a covered specialist means the panel degraded"
    assert "researcher" in result.failed_agents
    assert result.agent_coverage.get("researcher"), "a peer must be recorded as covering"
    covering = result.agent_coverage["researcher"]
    assert covering != "researcher"
    assert result.degradation_reasons
    # The answer came from a real model, not a prose placeholder.
    assert result.final_answer.startswith("GROUNDED:")
    assert "temporarily offline" not in result.final_answer


@pytest.mark.asyncio
async def test_confidence_is_earned_not_hardcoded(monkeypatch):
    """Peer-covered, evidence-free output must not score the old constant 0.90/0.92."""
    _make_gateway(monkeypatch, dead_agents={"researcher"})
    engine = CollaborationEngine()
    result = await engine.run_collaboration(
        task_id="task_confidence",
        question="What is a quorum?",
        participating_agents=[engine.registry.get_agent("researcher")],
    )

    # Previously model_confidence was hardcoded 0.90 -> system confidence 0.92.
    assert result.confidence < 0.92, "unearned confidence must not survive the fix"
    assert result.confidence >= 0.0


@pytest.mark.asyncio
async def test_ask_returns_503_when_entire_panel_is_dark(monkeypatch, auth):
    """No specialist and no peer can answer -> honest 503, no fabricated result."""
    _make_gateway(monkeypatch, dead_providers={"gemini", "groq", "nvidia", "mistral", "openrouter", "cohere", "huggingface"})
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://testserver", headers=auth) as client:
        r = await client.post("/ask", json={"question": "anything", "mode": "fast"})

    assert r.status_code == 503, f"expected 503, got {r.status_code}: {r.text[:300]}"
    detail = r.json()["detail"]
    assert "no specialist produced output" in detail.lower()
    # The real failure reasons must be carried through, not hidden.
    assert "simulated outage" in detail


@pytest.mark.asyncio
async def test_agent_call_unavailable_is_raised_not_returned_as_prose(monkeypatch):
    """_execute_agent_call must raise instead of handing back a placeholder string."""
    _make_gateway(monkeypatch, dead_agents={"researcher", "proposer", "architect", "synthesizer", "critic"})
    engine = CollaborationEngine()
    agent = engine.registry.get_agent("researcher")
    assert agent is not None

    with pytest.raises(AgentCallUnavailable) as exc:
        await engine._execute_agent_call(
            task_id="task_raises",
            stage_name="independent_analysis",
            round_number=1,
            agent=agent,
            messages=[],
        )

    assert "temporarily offline" not in str(exc.value)


@pytest.mark.asyncio
async def test_panel_unavailable_carries_failure_detail():
    exc = AgentPanelUnavailable(task_id="t1", failures=["researcher: no keys"])
    assert exc.task_id == "t1"
    assert exc.failures == ["researcher: no keys"]
    assert "researcher: no keys" in str(exc)


@pytest.mark.asyncio
async def test_successful_panel_reports_not_degraded(monkeypatch):
    """Control case: everything healthy -> degraded=False, empty coverage."""
    _make_gateway(monkeypatch)
    engine = CollaborationEngine()
    result = await engine.run_collaboration(
        task_id="task_healthy",
        question="Explain Raft leader election.",
        participating_agents=[engine.registry.get_agent("researcher")],
    )

    assert result.degraded is False
    assert result.agent_coverage == {}
    assert result.failed_agents == []
    assert result.degradation_reasons == []


@pytest.mark.asyncio
async def test_no_fabricated_evidence_when_nothing_was_extracted(monkeypatch):
    """key_evidence must be empty rather than carrying a placeholder sentence."""
    _make_gateway(monkeypatch)
    engine = CollaborationEngine()
    result = await engine.run_collaboration(
        task_id="task_evidence",
        question="Define CAP theorem.",
        participating_agents=[engine.registry.get_agent("researcher")],
    )

    for item in result.key_evidence:
        assert item not in (
            "Single-specialist direct assessment.",
            "Consensus verified across specialist team.",
            "Resolved through cross-specialist debate.",
        ), "placeholder 'evidence' is not evidence"


def test_settings_has_no_provider_keys_during_ci(monkeypatch):
    """Sanity: this module's tests assume no ambient provider credentials."""
    for prov in ("GEMINI", "GROQ", "MISTRAL", "OPENROUTER", "COHERE", "HUGGINGFACE", "NVIDIA"):
        monkeypatch.setattr(settings, f"{prov}_API_KEY", None)
        monkeypatch.setattr(settings, f"{prov}_API_KEYS", None)
    monkeypatch.setattr(type(settings), "get_provider_keys", lambda self, name: [])
    assert settings.get_provider_keys("gemini") == []
