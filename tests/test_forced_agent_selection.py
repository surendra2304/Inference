from __future__ import annotations

from app.agents.base import Agent
from app.core.orchestrator import MAX_FORCED_AGENT_CANDIDATES, Orchestrator


def _agent(agent_id: str) -> Agent:
    return Agent(
        id=agent_id,
        name=agent_id.title(),
        role=f"{agent_id} specialist",
        purpose="Test fixture",
        system_instructions="Be accurate.",
    )


class _Registry:
    def __init__(self, agents: list[Agent]) -> None:
        self.agents = {agent.id: agent for agent in agents}
        self.lookups: list[str] = []

    def get_agent(self, agent_id: str) -> Agent | None:
        self.lookups.append(agent_id)
        return self.agents.get(agent_id)


def _orchestrator(*agents: Agent) -> Orchestrator:
    orchestrator = object.__new__(Orchestrator)
    orchestrator.registry = _Registry(list(agents))
    return orchestrator


def test_forced_agent_selection_deduplicates_ids_and_respects_max_agents():
    researcher = _agent("researcher")
    synthesizer = _agent("synthesizer")
    orchestrator = _orchestrator(researcher, synthesizer)

    selected = orchestrator._resolve_forced_agents(
        ["researcher", "researcher", "unknown", 42, "synthesizer"],
        max_agents=1,
    )

    assert [agent.id for agent in selected] == ["researcher"]
    assert orchestrator.registry.lookups == ["researcher"]


def test_forced_agent_selection_resolves_known_unique_agents_in_order():
    researcher = _agent("researcher")
    synthesizer = _agent("synthesizer")
    orchestrator = _orchestrator(researcher, synthesizer)

    selected = orchestrator._resolve_forced_agents(
        ["unknown", "synthesizer", "researcher", "synthesizer"],
        max_agents=2,
    )

    assert [agent.id for agent in selected] == ["synthesizer", "researcher"]


def test_no_forced_assignment_preserves_automatic_routing():
    orchestrator = _orchestrator(_agent("researcher"))

    assert orchestrator._resolve_forced_agents(None, max_agents=5) is None
    assert orchestrator._resolve_forced_agents([], max_agents=5) is None


def test_malformed_forced_assignment_does_not_crash_agent_lookup():
    orchestrator = _orchestrator(_agent("researcher"))

    assert orchestrator._resolve_forced_agents({"not": "a list"}, max_agents=5) == []
    assert orchestrator.registry.lookups == []


def test_forced_assignment_candidate_scan_is_bounded():
    researcher = _agent("researcher")
    orchestrator = _orchestrator(researcher)
    candidates = [f"unknown-{i}" for i in range(MAX_FORCED_AGENT_CANDIDATES)]
    candidates.extend(["researcher"] * 100_000)

    selected = orchestrator._resolve_forced_agents(candidates, max_agents=5)

    assert selected == []
    assert len(orchestrator.registry.lookups) == MAX_FORCED_AGENT_CANDIDATES
    assert orchestrator.registry.lookups == candidates[:MAX_FORCED_AGENT_CANDIDATES]



def test_forced_agent_selection_returns_empty_for_non_positive_max_agents():
    researcher = _agent("researcher")
    orchestrator = _orchestrator(researcher)

    for max_agents in (0, -1):
        assert orchestrator._resolve_forced_agents(["researcher"], max_agents=max_agents) == []

    assert orchestrator.registry.lookups == []
