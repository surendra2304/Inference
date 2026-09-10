import asyncio
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from app.agents.base import Agent
from app.agents.debate import CollaborationEngine
from app.agents.reasoning import AdjudicationResult, AtomicClaim, StructuredEvidence
from app.agents.registry import agent_registry
from app.agents.roles import register_all_specialists
from app.agents.router import router as task_router
from app.core.dag import DAGNode, ExecutionDAG, TaskComplexity, classify_task_complexity
from app.learning.performance import PerformanceTracker
from app.learning.strategy_store import StrategyStore
from app.memory.base import BaseMemory, TaskRecord
from app.memory.sqlite import SQLiteMemory
from app.utils.ids import generate_task_id
from app.utils.logger import logger


class OrchestrationRequest(BaseModel):
    """Input payload for a high-level orchestration task."""
    question: str
    mode: str = Field(default="auto", description="auto, fast, review, debate")
    max_agents: int = Field(default=5, ge=1, le=10)
    require_evidence: bool = True
    max_budget: float | None = Field(default=None, description="Max budget in USD for this task")
    max_latency: float | None = Field(default=None, description="Max desired latency in seconds")
    context_data: dict[str, Any] = Field(default_factory=dict)


class OrchestrationResult(BaseModel):
    """Final result output from an orchestrated workflow."""
    task_id: str
    run_id: str
    question: str
    answer: str
    mode_used: str
    provider_used: str = "multi_provider"
    agents_used: list[str]
    models_used: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    unresolved_disagreements: list[str] = Field(default_factory=list)
    key_evidence: list[str] = Field(default_factory=list)
    structured_evidence: list[StructuredEvidence] = Field(default_factory=list)
    claims: list[AtomicClaim] = Field(default_factory=list)
    adjudication: AdjudicationResult | None = None
    complexity: str = "simple"
    total_tokens: int = 0
    total_latency_seconds: float = 0.0


class BaseOrchestrator(ABC):
    """Abstract base class for coordinating tasks from start to finish."""

    @abstractmethod
    async def process_task(self, request: OrchestrationRequest) -> OrchestrationResult:
        """Execute full end-to-end task routing, reasoning/debate, and answer synthesis."""

    @abstractmethod
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel an in-flight orchestration task."""

    @abstractmethod
    async def get_task_status(self, task_id: str) -> dict[str, Any] | None:
        """Retrieve live execution progress and state of an ongoing or completed task."""


class Orchestrator(BaseOrchestrator):
    """Coordinates task routing, DAG construction, agent assignment, memory persistence, and synthesis."""

    def __init__(
        self,
        memory: BaseMemory | None = None,
    ) -> None:
        self.memory = memory or SQLiteMemory()
        self.router = task_router
        self.registry = agent_registry
        self.debate_engine = CollaborationEngine(memory=self.memory, registry=self.registry)
        self.strategy_store = StrategyStore(memory=self.memory)
        self.performance_tracker = PerformanceTracker(memory=self.memory)
        self._active_cancellations: dict[str, asyncio.Event] = {}
        self._recent_tasks: dict[str, TaskRecord] = {}

        # Ensure all 10 specialist roles are registered
        register_all_specialists()

    def build_execution_dag(
        self,
        question: str,
        participating_agents: list[Agent],
        complexity: TaskComplexity
    ) -> ExecutionDAG:
        """
        Builds a Directed Acyclic Graph for the given agents and complexity:
        - Layer 0: Independent specialist analysis nodes (run in parallel via asyncio.gather)
        - Layer 1: Synthesizer consensus node (depends on all Layer 0 nodes)
        - Layer 2: (Optional if conflict occurs) Critic Rebuttal & Final Synthesizer
        """
        dag = ExecutionDAG()

        # Add specialist analysis nodes (Layer 0)
        for agent in participating_agents:
            node = DAGNode(
                node_id=f"node_{agent.id}",
                agent_id=agent.id,
                agent_role=agent.role,
                dependencies=[],
                stage_name="independent_analysis",
                complexity=complexity
            )
            dag.add_node(node)

        # Add synthesis node (Layer 1)
        synth_node = DAGNode(
            node_id="node_synthesizer",
            agent_id="synthesizer",
            agent_role="Synthesizer",
            dependencies=[f"node_{agent.id}" for agent in participating_agents],
            stage_name="consensus_synthesis",
            complexity=complexity
        )
        dag.add_node(synth_node)
        dag.build_layers()
        return dag

    async def process_task(
        self,
        request: OrchestrationRequest,
        task_id: str | None = None
    ) -> OrchestrationResult:
        """Execute full end-to-end task with Dynamic DAG Orchestration."""
        task_id = task_id or generate_task_id()
        start_time = time.perf_counter()

        self.strategy_store.memory = self.memory
        self.performance_tracker.memory = self.memory
        self.debate_engine.memory = self.memory

        # 1. Classify Task Complexity (Simple, Complex, Strategic)
        complexity = classify_task_complexity(request.question, request.mode)

        # Check for learned strategy recommendations if mode is 'auto' and not strictly latency-bounded
        learned_strat = None
        is_latency_critical = request.max_latency is not None and request.max_latency <= 5.0
        if request.mode == "auto" and not is_latency_critical:
            task_domain = self.router.detect_domain_specialist(request.question)
            try:
                learned_strat = await self.strategy_store.recommend_strategy(task_domain)
            except Exception as e:
                logger.debug("Strategy store lookup skipped: %s", str(e))

        # 2. Route task and select specialist agents with telemetry guardrails
        decision = self.router.route_task(
            question=request.question,
            requested_mode=learned_strat.recommended_mode if learned_strat else request.mode,
            max_agents=request.max_agents,
            max_budget=request.max_budget,
            max_latency=request.max_latency
        )

        mode_used = decision.mode
        route_reason = decision.reason
        selected_agent_ids = decision.selected_agent_ids
        forced_agents = request.context_data.get("assigned_agents") if request.context_data else None
        if forced_agents:
            selected_agent_ids = forced_agents if isinstance(forced_agents, list) else [forced_agents]
            route_reason = f"Forced agent assignment via context_data: {selected_agent_ids}"

        participating_agents: list[Agent] = [
            a for aid in selected_agent_ids
            if (a := self.registry.get_agent(aid)) is not None
        ]
        if not participating_agents:
            fallback_agent = self.registry.get_agent("researcher") or self.registry.get_agent("synthesizer")
            if fallback_agent:
                participating_agents = [fallback_agent]

        # 3. Create initial task record with telemetry metadata (in-memory + async background persistence)
        task_record = TaskRecord(
            id=task_id,
            question=request.question,
            mode=mode_used,
            status="running",
            metadata={
                "route_reason": route_reason,
                "selected_agents": [a.id for a in participating_agents],
                "complexity": complexity.value,
                "routing_telemetry": decision.telemetry,
                "learned_strategy_applied": bool(learned_strat),
                "request_context": request.context_data
            }
        )
        self._recent_tasks[task_id] = task_record
        try:
            asyncio.create_task(self.memory.save_task(task_record))
        except RuntimeError:
            await self.memory.save_task(task_record)

        try:
            # For review/debate modes, pad specialist agents if needed
            if mode_used == "review" and len(participating_agents) < 2:
                extra_candidates = ["critic", "fact_checker", "strategist", "researcher"]
                existing_ids = [a.id for a in participating_agents]
                for cid in extra_candidates:
                    if len(participating_agents) >= 2:
                        break
                    agent = self.registry.get_agent(cid)
                    if agent and agent.id not in existing_ids:
                        participating_agents.append(agent)
                        existing_ids.append(agent.id)

            # 4. Build Dynamic DAG for execution
            dag = self.build_execution_dag(
                question=request.question,
                participating_agents=participating_agents,
                complexity=complexity
            )

            logger.info(
                "DAG Orchestrator: task %s | mode '%s' | complexity '%s' | %d agents | %d DAG layers",
                task_id, mode_used, complexity.value, len(participating_agents), len(dag.layers)
            )

            cancel_event = self._active_cancellations.setdefault(task_id, asyncio.Event())

            # 5. Run Collaboration via CollaborationEngine with DAG Complexity & Cooperative Cancellation
            self.debate_engine.memory = self.memory
            collab_result = await self.debate_engine.run_collaboration(
                task_id=task_id,
                question=request.question,
                participating_agents=participating_agents,
                require_evidence=request.require_evidence,
                complexity=complexity,
                cancellation_event=cancel_event
            )
            latency = time.perf_counter() - start_time

            if cancel_event.is_set():
                task_record.status = "cancelled"
                task_record.completed_at = datetime.now(timezone.utc)
                await self.memory.save_task(task_record)
                self._active_cancellations.pop(task_id, None)
                raise asyncio.CancelledError(f"Task {task_id} was cancelled during execution.")

            actual_mode = mode_used if mode_used == "debate" else getattr(collab_result, "mode_used", mode_used)
            task_record.status = "completed"
            task_record.result = collab_result.final_answer
            task_record.confidence = collab_result.confidence
            task_record.mode = actual_mode
            task_record.completed_at = datetime.now(timezone.utc)
            task_record.metadata["debate_id"] = collab_result.debate_id
            task_record.metadata["unresolved_disagreements"] = collab_result.unresolved_disagreements
            task_record.metadata["complexity"] = complexity.value
            task_record.metadata["models_used"] = collab_result.models_used
            self._recent_tasks[task_id] = task_record
            try:
                asyncio.create_task(self.memory.save_task(task_record))
            except RuntimeError:
                await self.memory.save_task(task_record)

            self._active_cancellations.pop(task_id, None)

            return OrchestrationResult(
                task_id=task_id,
                run_id=collab_result.debate_id,
                question=request.question,
                answer=collab_result.final_answer,
                mode_used=actual_mode,
                provider_used="multi_provider",
                agents_used=collab_result.participating_agents,
                models_used=collab_result.models_used if collab_result.models_used else [a.model_name for a in participating_agents],
                confidence=collab_result.confidence,
                unresolved_disagreements=collab_result.unresolved_disagreements,
                key_evidence=collab_result.key_evidence,
                structured_evidence=collab_result.structured_evidence,
                claims=collab_result.claims,
                adjudication=collab_result.adjudication,
                complexity=complexity.value,
                total_tokens=collab_result.total_tokens,
                total_latency_seconds=round(latency, 4)
            )

        except (asyncio.CancelledError, Exception) as exc:
            latency = time.perf_counter() - start_time
            is_cancel = isinstance(exc, asyncio.CancelledError) or (task_id in self._active_cancellations and self._active_cancellations[task_id].is_set())
            logger.error("Task %s %s during execution: %s", task_id, "cancelled" if is_cancel else "failed", str(exc))

            task_record.status = "cancelled" if is_cancel else "failed"
            task_record.completed_at = datetime.now(timezone.utc)
            task_record.metadata["error"] = str(exc)
            self._recent_tasks[task_id] = task_record
            try:
                await self.memory.save_task(task_record)
            except Exception:
                pass
            self._active_cancellations.pop(task_id, None)

            raise exc

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel an in-flight task and notify active execution tokens."""
        signaled = False
        if task_id in self._active_cancellations:
            self._active_cancellations[task_id].set()
            logger.info("Signaled in-flight cancellation event for task %s", task_id)
            signaled = True

        if task_id in self._recent_tasks:
            self._recent_tasks[task_id].status = "cancelled"
            self._recent_tasks[task_id].completed_at = datetime.now(timezone.utc)

        task = await self.memory.get_task(task_id)
        if task and task.status in ("running", "pending"):
            task.status = "cancelled"
            task.completed_at = datetime.now(timezone.utc)
            await self.memory.save_task(task)
            return True
        return signaled

    async def get_task_status(self, task_id: str) -> dict[str, Any] | None:
        """Retrieve task details and progress with zero-latency in-memory cache lookup."""
        if task_id in self._recent_tasks:
            return self._recent_tasks[task_id].model_dump()
        task = await self.memory.get_task(task_id)
        return task.model_dump() if task else None


# Global default orchestrator instance
orchestrator = Orchestrator()
