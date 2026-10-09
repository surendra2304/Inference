"""Continuous Learning Engine: Attribution of Advisory Recommendations and Model Tuning.

Previously seeded with three invented outcomes (a "85.0" helpful rate when empty, fixed agent
weights 1.25/1.30/0.90 that nothing measured, and a score per outcome label that was a guess).
Now it starts empty; every rate is None until an outcome is recorded, and no agent weight is
reported as learned unless a learning step produced it (none does yet).
"""

from typing import Any

from app.memory.long_term import long_term_memory


class ContinuousLearningEngine:
    """Evaluates whether AI advice helped or hurt, from recorded outcomes only."""

    def __init__(self) -> None:
        self.recommendation_outcomes: list[dict[str, Any]] = []

    def record_outcome(
        self,
        consultation_id: str,
        action: str,
        drawdown_reduction_pct: float,
        outcome: str
    ) -> None:
        """Records the post-execution outcome of an applied recommendation."""
        self.recommendation_outcomes.append({
            "consultation_id": consultation_id,
            "action_taken": action,
            "drawdown_reduction_pct": drawdown_reduction_pct,
            "outcome": outcome,
        })

    def get_learning_status(self) -> dict[str, Any]:
        """Learning progression computed from the recorded outcomes."""
        total = len(self.recommendation_outcomes)
        helped = sum(1 for r in self.recommendation_outcomes if r["outcome"] == "HELPED")
        helpful_rate = round(helped / total * 100.0, 1) if total else None
        return {
            "total_outcomes_evaluated": total,
            "helpful_recommendation_rate_pct": helpful_rate,
            "learned_agent_weights": None,
            "learned_agent_weights_status": "not_learned: no learning step has run",
            "recent_semantic_insights": long_term_memory.semantic_memories,
            "continuous_learning_status": "ACTIVE_WITH_OUTCOMES" if total else "NO_OUTCOMES_RECORDED",
        }


continuous_learning_engine = ContinuousLearningEngine()
