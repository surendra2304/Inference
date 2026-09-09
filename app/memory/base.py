"""Base interface and data contracts for persistent memory storage."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class MemoryRecord(BaseModel):
    """An individual persistent memory record scoped to an agent or system."""
    id: str
    agent_id: str
    content: str
    memory_type: str = Field(default="fact", description="fact, experience, preference, decision, reflection")
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    context_tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskRecord(BaseModel):
    """Audit record for a submitted user/system task."""
    id: str
    question: str
    mode: str = Field(default="auto", description="auto, fast, review, debate")
    status: str = Field(default="pending", description="pending, running, completed, failed")
    result: str | None = None
    confidence: float | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunRecord(BaseModel):
    """Audit record for a specific reasoning stage or agent invocation."""
    id: str
    task_id: str
    agent_id: str
    provider: str
    model: str
    stage: str = Field(description="round_0_framing, round_1_analysis, round_2_critique, etc.")
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_seconds: float = 0.0
    status: str = "completed"
    error: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class MessageRecord(BaseModel):
    """Record for conversation or debate messages."""
    id: str
    run_id: str
    task_id: str
    role: str
    agent_id: str | None = None
    content: str
    stage: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class StrategyRecord(BaseModel):
    """Learned routing and orchestration pattern."""
    id: str
    task_type: str
    strategy: str
    score: float = Field(ge=0.0, le=1.0)
    sample_size: int = 1
    recommended_agents: list[str] = Field(default_factory=list)
    recommended_provider: str = "gemini"
    recommended_model: str = "gemini-3.5-flash-lite"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExperimentRecord(BaseModel):
    """Controlled multi-model or multi-agent experiment."""
    id: str
    hypothesis: str
    configuration: dict[str, Any] = Field(default_factory=dict)
    status: str = "completed"
    result: dict[str, Any] | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class BaseMemory(ABC):
    """Abstract base class for persistent memory operations."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize database schema, tables, and indexes."""

    @abstractmethod
    async def save_agent(self, agent_data: dict[str, Any]) -> None:
        """Persist or update an agent configuration record."""

    @abstractmethod
    async def save_task(self, task: TaskRecord) -> None:
        """Create or update a task record."""

    @abstractmethod
    async def get_task(self, task_id: str) -> TaskRecord | None:
        """Retrieve a task record by its ID."""

    @abstractmethod
    async def save_run(self, run: RunRecord) -> None:
        """Persist an execution run audit record."""

    @abstractmethod
    async def save_message(self, message: MessageRecord) -> None:
        """Persist a conversation or debate message."""

    @abstractmethod
    async def get_task_messages(self, task_id: str) -> list[MessageRecord]:
        """Retrieve all messages associated with a task ID."""

    @abstractmethod
    async def save_memory(self, memory: MemoryRecord) -> None:
        """Save a scoped persistent memory item."""

    @abstractmethod
    async def get_agent_memories(
        self,
        agent_id: str,
        limit: int = 10,
        memory_type: str | None = None
    ) -> list[MemoryRecord]:
        """Retrieve memories strictly scoped to a specific agent_id."""

    @abstractmethod
    async def search_memories(
        self,
        query: str,
        agent_id: str | None = None,
        limit: int = 5
    ) -> list[MemoryRecord]:
        """Search memory records by semantic/text query, optionally filtered by agent_id."""

    @abstractmethod
    async def save_strategy(self, strategy: StrategyRecord) -> None:
        """Save or update a learned strategy record."""

    @abstractmethod
    async def get_strategy(self, task_type: str) -> StrategyRecord | None:
        """Retrieve the best learned strategy for a specific task type."""

    @abstractmethod
    async def list_strategies(self) -> list[StrategyRecord]:
        """List all learned strategies."""

    @abstractmethod
    async def save_experiment(self, experiment: ExperimentRecord) -> None:
        """Save an experiment record."""

    @abstractmethod
    async def get_experiment(self, experiment_id: str) -> ExperimentRecord | None:
        """Retrieve experiment details by ID."""
