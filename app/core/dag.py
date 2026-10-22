"""DAG Execution, Complexity Classifier, and Dependency Models for Inference."""

import re
from enum import Enum

from pydantic import BaseModel, Field


class TaskComplexity(str, Enum):
    SIMPLE = "simple"       # Easy: Single primary model per agent
    COMPLEX = "complex"     # Moderate/Complex: 2-3 models per agent in parallel
    STRATEGIC = "strategic" # Strategic/Deep: Full DAG with multi-model parallel debate


class DAGNode(BaseModel):
    """Represents an agent execution stage in the Directed Acyclic Graph."""

    node_id: str
    agent_id: str
    agent_role: str
    dependencies: list[str] = Field(
        default_factory=list,
        description="IDs of nodes that must finish before this node runs",
    )
    stage_name: str = "analysis"
    complexity: TaskComplexity = TaskComplexity.SIMPLE


class ExecutionDAG(BaseModel):
    """Directed Acyclic Graph defining parallel execution stages for multi-agent workflows."""

    nodes: dict[str, DAGNode] = Field(default_factory=dict)
    layers: list[list[str]] = Field(
        default_factory=list,
        description="Topologically sorted execution layers for parallel asyncio.gather",
    )

    def add_node(self, node: DAGNode) -> None:
        """Add a uniquely keyed node; silently replacing it would invalidate dependencies."""
        if node.node_id in self.nodes:
            raise ValueError(f"Duplicate DAG node id: {node.node_id}")
        self.nodes[node.node_id] = node
        # A previously computed layer list is no longer a valid snapshot of this graph.
        self.layers = []

    def build_layers(self) -> list[list[str]]:
        """Build deterministic topological layers or reject an invalid graph.

        Missing dependency references and cycles are configuration errors, not a reason
        to run dependent nodes concurrently. The former implementation put all unresolved
        nodes into one final layer, which violated the advertised dependency contract.
        """
        self.layers = []
        for key, node in self.nodes.items():
            if key != node.node_id:
                raise ValueError(
                    f"DAG mapping key '{key}' does not match node id '{node.node_id}'"
                )
            if len(node.dependencies) != len(set(node.dependencies)):
                raise ValueError(f"DAG node '{key}' contains duplicate dependencies")
            missing = [dep for dep in node.dependencies if dep not in self.nodes]
            if missing:
                raise ValueError(
                    f"DAG node '{key}' references missing dependencies: {', '.join(missing)}"
                )

        in_degree: dict[str, int] = {
            node_id: len(node.dependencies) for node_id, node in self.nodes.items()
        }
        dependents: dict[str, list[str]] = {node_id: [] for node_id in self.nodes}
        for node_id, node in self.nodes.items():
            for dependency in node.dependencies:
                dependents[dependency].append(node_id)

        layers: list[list[str]] = []
        visited: set[str] = set()
        while len(visited) < len(self.nodes):
            current_layer = [
                node_id
                for node_id in self.nodes
                if node_id not in visited and in_degree[node_id] == 0
            ]
            if not current_layer:
                blocked = [node_id for node_id in self.nodes if node_id not in visited]
                raise ValueError(
                    "Cycle detected in DAG; nodes cannot be scheduled: " + ", ".join(blocked)
                )

            layers.append(current_layer)
            for node_id in current_layer:
                visited.add(node_id)
                for dependent in dependents[node_id]:
                    in_degree[dependent] -= 1

        self.layers = layers
        return [layer.copy() for layer in layers]


def classify_task_complexity(question: str, requested_mode: str = "auto") -> TaskComplexity:
    """
    Classifies an incoming query into SIMPLE, COMPLEX, or STRATEGIC complexity.
    - SIMPLE: Basic lookup, concise code query, direct factual question, or fast mode.
    - STRATEGIC: Multi-system architectural trade-offs, security threat modeling, high-stakes decisions.
    - COMPLEX: Refactoring, debugging tricky errors, comparing components, deep analysis.
    """
    if requested_mode == "fast":
        return TaskComplexity.SIMPLE
    if requested_mode in ["debate", "strategic"]:
        return TaskComplexity.STRATEGIC

    q_lower = question.lower()
    strategic_keywords = [
        "architecture", "architectural", "system design", "trade-off", "tradeoff",
        "vs", "versus", "threat model", "vulnerability", "microservice", "monolith",
        "roadmap", "governance", "distributed system", "failover", "consensus"
    ]
    complex_keywords = [
        "implement", "refactor", "debug", "deadlock", "race condition", "optimize",
        "benchmark", "sql query", "verify claims", "pipeline", "security analysis",
        "traceback", "exception", "async", "schema"
    ]

    if any(re.search(rf"\b{re.escape(kw)}\b", q_lower) for kw in strategic_keywords):
        return TaskComplexity.STRATEGIC
    if any(re.search(rf"\b{re.escape(kw)}\b", q_lower) for kw in complex_keywords):
        return TaskComplexity.COMPLEX

    if len(question.split()) < 12:
        return TaskComplexity.SIMPLE

    return TaskComplexity.COMPLEX
