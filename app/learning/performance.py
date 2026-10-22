"""Performance analytics and tracking for models, agents, and debate combinations."""

from typing import Any

from pydantic import BaseModel

from app.memory.base import BaseMemory
from app.memory.sqlite import SQLiteMemory
from app.utils.bounded_store import BoundedList
from app.utils.logger import logger


class ModelPerformanceStats(BaseModel):
    """Aggregated performance metrics for a specific model or agent role.

    Fields are None when no outcome was recorded for the entity. The previous defaults of 0.0
    read as a measured average of zero (the worst possible score).
    """
    entity_id: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    average_latency_s: float | None = None
    average_score: float | None = None
    total_tokens: int = 0


#: Recorded outcomes kept in process. Bounded so a long-running server cannot grow without limit.
MAX_RECORDED_OUTCOMES = 5000
#: Outcomes an agent needs before it is ranked; below this a ranking is noise.
MIN_OUTCOMES_FOR_RANKING = 5


class PerformanceTracker:
    """Records task outcomes and computes per-agent statistics from them.

    ``record_task_outcome`` previously only logged, and ``compute_model_statistics`` returned a fixed
    dictionary naming a model ("gemini-3.8-flash") that is not an observed result. Statistics are now
    computed from the recorded outcomes; with none recorded every figure is None.
    """

    def __init__(self, memory: BaseMemory | None = None) -> None:
        self.memory = memory or SQLiteMemory()
        self._outcomes: BoundedList[dict[str, Any]] = BoundedList(
            "performance_outcomes", max_entries=MAX_RECORDED_OUTCOMES
        )

    async def record_task_outcome(
        self,
        task_id: str,
        task_type: str,
        mode: str,
        agents: list[str],
        score: float | None,
        latency_s: float | None,
        tokens: int | None
    ) -> None:
        """Records telemetry outcome from a completed task; unmeasured values stay None."""
        logger.info(
            "Tracking performance for task %s (Type: %s, Mode: %s, Score: %s, Latency: %s)",
            task_id, task_type, mode, score, latency_s
        )
        for agent in agents:
            self._outcomes.append({
                "task_id": task_id,
                "task_type": task_type,
                "mode": mode,
                "agent": agent,
                "score": score,
                "latency_s": latency_s,
                "tokens": tokens,
            })

    async def compute_model_statistics(self) -> dict[str, Any]:
        """Per-agent statistics from recorded outcomes; ranking needs MIN_OUTCOMES_FOR_RANKING outcomes."""
        per_agent: dict[str, list[dict[str, Any]]] = {}
        for row in self._outcomes:
            per_agent.setdefault(row["agent"], []).append(row)

        stats: dict[str, ModelPerformanceStats] = {}
        for agent, rows in per_agent.items():
            scores = [r["score"] for r in rows if r["score"] is not None]
            latencies = [r["latency_s"] for r in rows if r["latency_s"] is not None]
            stats[agent] = ModelPerformanceStats(
                entity_id=agent,
                total_calls=len(rows),
                successful_calls=len(scores),
                failed_calls=len(rows) - len(scores),
                average_latency_s=round(sum(latencies) / len(latencies), 3) if latencies else None,
                average_score=round(sum(scores) / len(scores), 3) if scores else None,
                total_tokens=sum(r["tokens"] or 0 for r in rows),
            )

        rankable = {a: s for a, s in stats.items() if s.total_calls >= MIN_OUTCOMES_FOR_RANKING and s.average_score is not None}

        def best(key: Any, reverse: bool) -> str | None:
            if not rankable:
                return None
            ordered = sorted(rankable, key=key, reverse=reverse)
            return ordered[0]

        return {
            "outcomes_recorded": len(self._outcomes),
            "top_reasoning_model": best(lambda a: rankable[a].average_score or 0.0, True),
            "fastest_execution_model": best(lambda a: rankable[a].average_latency_s or float("inf"), False),
            "most_effective_critic": next((a for a in ("critic", "critic_agent") if a in rankable), None),
            "per_agent": {a: s.model_dump() for a, s in stats.items()},
            "basis": "computed from recorded task outcomes" if stats else "not_measured: no task outcomes recorded",
        }
