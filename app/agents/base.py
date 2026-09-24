"""Base data models, schemas, and contracts for Inference agents."""

import json
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class AgentModelConfig(BaseModel):
    """Specification of a provider, model identifier, and capability assigned to an agent."""
    provider: str = Field(description="Cloud provider name, e.g. gemini, openrouter, nvidia, cohere, mistral")
    model: str = Field(description="Model identifier, e.g. gemini-3.8-flash, deepseek/deepseek-v4-flash:free")
    capability: str = Field(description="Specialized capability tag, e.g. research, reasoning, coding, safety, synthesis")


class AgentResponse(BaseModel):
    """
    Strict Pydantic schema for inter-agent communication.
    All agents output structured information adhering to this schema.
    """
    summary: str = Field(description="Core concise technical recommendation and summary")
    rationale: str = Field(default="", description="Detailed technical justification and evidence")
    code: str | None = Field(default=None, description="Concrete code snippets, schemas, or pseudocode if applicable")
    trade_offs: list[str] = Field(default_factory=list, description="Explicitly identified trade-offs or alternatives")
    assumptions: list[str] = Field(default_factory=list, description="Underlying assumptions made in the proposal")
    confidence: float = Field(default=0.90, ge=0.0, le=1.0, description="Confidence score in the recommendation")
    dissent: str | None = Field(default=None, description="Identified counterpoints, risks, or dissent from other perspectives")

    @classmethod
    def parse_raw_or_json(cls, text: str) -> "AgentResponse":
        """
        Parses JSON if available in text (supporting markdown code fences),
        otherwise extracts a structured AgentResponse gracefully.
        """
        cleaned = text.strip()
        # Handle markdown JSON fences
        if "```json" in cleaned:
            start = cleaned.find("```json") + 7
            end = cleaned.find("```", start)
            cleaned = cleaned[start:end].strip()
        elif "```" in cleaned:
            start = cleaned.find("```") + 3
            end = cleaned.find("```", start)
            cleaned = cleaned[start:end].strip()

        try:
            data = json.loads(cleaned)
            if isinstance(data, dict):
                return cls(**data)
        except Exception:
            pass

        # Fallback: treat entire string as summary
        return cls(
            summary=text[:1000].strip(),
            rationale=text.strip(),
            confidence=0.85
        )


class Agent(BaseModel):
    """Pydantic model representing an individual specialist agent identity."""
    id: str = Field(description="Unique agent identifier, e.g., 'researcher', 'architect'")
    name: str = Field(description="Human-readable agent display name")
    role: str = Field(description="Primary specialist role title")
    purpose: str = Field(description="Core purpose and mission of the agent")
    system_instructions: str = Field(description="Base prompt/system instructions guiding reasoning")
    allowed_tools: list[str] = Field(default_factory=list, description="List of tools the agent is permitted to call")
    model_provider: str = Field(default="gemini", description="Primary underlying provider")
    model_name: str = Field(default="gemini-3.6-flash", description="Primary model identifier")
    models: list[AgentModelConfig] = Field(
        default_factory=list,
        description="Ranked list of specialized models and capabilities assigned to this agent"
    )
    personality_style: str | None = Field(default=None, description="Optional reasoning or interaction style")
    strengths: list[str] = Field(default_factory=list, description="Key domain capabilities and strengths")
    weaknesses: list[str] = Field(default_factory=list, description="Known limitations or failure modes")
    memory_scope: str = Field(default="agent_private", description="Memory visibility: agent_private or shared")
    status: str = Field(default="active", description="Operational status: active, paused, deprecated")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional custom properties")

    @property
    def preferred_models(self) -> list[AgentModelConfig]:
        """Alias for models list."""
        return self.models

    def get_primary_model(self) -> AgentModelConfig:
        """Returns the primary (1st) model configuration for standard tasks."""
        if self.models:
            return self.models[0]
        return AgentModelConfig(provider=self.model_provider, model=self.model_name, capability="general")

    def get_preferred_models(self, limit: int = 3) -> list[AgentModelConfig]:
        """Returns the top N preferred model configurations for this agent."""
        if self.models:
            return self.models[:limit]
        return [self.get_primary_model()]


class BaseAgentRegistry(ABC):
    """Abstract interface for agent registration and discovery."""

    @abstractmethod
    def register_agent(self, agent: Agent) -> None:
        """Register a new specialist agent or update an existing one."""

    @abstractmethod
    def get_agent(self, agent_id: str) -> Agent | None:
        """Retrieve an agent by its unique identifier."""

    @abstractmethod
    def list_agents(self, role: str | None = None, status: str | None = None) -> list[Agent]:
        """List all registered agents, optionally filtered by role or status."""

    @abstractmethod
    def get_agents_by_capability(self, capability: str) -> list[Agent]:
        """Discover agents matching a required capability or strength."""
