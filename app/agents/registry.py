"""Agent registry implementation for managing specialist agents."""


from app.agents.base import Agent, BaseAgentRegistry


class InMemoryAgentRegistry(BaseAgentRegistry):
    """In-memory implementation of BaseAgentRegistry."""

    def __init__(self) -> None:
        self._agents: dict[str, Agent] = {}

    def register_agent(self, agent: Agent) -> None:
        """Register or update an agent definition."""
        self._agents[agent.id] = agent

    def get_agent(self, agent_id: str) -> Agent | None:
        """Retrieve an agent by ID."""
        return self._agents.get(agent_id)

    def list_agents(self, role: str | None = None, status: str | None = None) -> list[Agent]:
        """List all agents with optional filtering."""
        results = list(self._agents.values())
        if role:
            results = [a for a in results if a.role.lower() == role.lower()]
        if status:
            results = [a for a in results if a.status.lower() == status.lower()]
        return results

    def get_agents_by_capability(self, capability: str) -> list[Agent]:
        """Discover agents matching a required capability or strength."""
        cap_lower = capability.lower()
        return [
            a for a in self._agents.values()
            if any(cap_lower in s.lower() for s in a.strengths)
        ]


# Global default registry instance
agent_registry = InMemoryAgentRegistry()
