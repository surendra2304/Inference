from __future__ import annotations

from app.agents.base import Agent
from app.core.dag import TaskComplexity
from app.core.orchestrator import Orchestrator


def _agent(agent_id: str) -> Agent:
    return Agent(
        id=agent_id,
        name=agent_id.replace("_", " ").title(),
        role=f"{agent_id} specialist",
        purpose="Test fixture for DAG construction",
        system_instructions="Be accurate and concise.",
    )


def test_dag_synthesis_node_does_not_collide_with_synthesizer_specialist():
    orchestrator = object.__new__(Orchestrator)
    specialists = [_agent("synthesizer"), _agent("researcher")]

    dag = orchestrator.build_execution_dag(
        question="Summarize this research",
        participating_agents=specialists,
        complexity=TaskComplexity.COMPLEX,
    )

    assert "node_synthesizer" in dag.nodes
    synthesis_nodes = [
        item for item in dag.nodes.values()
        if item.stage_name == "consensus_synthesis"
    ]
    assert len(synthesis_nodes) == 1
    synthesis = synthesis_nodes[0]
    assert synthesis.node_id != "node_synthesizer"
    assert set(synthesis.dependencies) == {"node_synthesizer", "node_researcher"}
    assert dag.layers == [
        ["node_synthesizer", "node_researcher"],
        [synthesis.node_id],
    ]
