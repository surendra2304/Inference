"""Long-Term Episodic, Semantic, and Procedural Memory Architecture.

The previous version was seeded with an invented episode (``EP-2026-08-20``, effectiveness 0.92),
a semantic claim that "84% of historical episodes" showed a whipsaw signature, and a procedural
rule with a "historical_success_rate" of 0.88. No episode was recorded and no history was
measured, so the stores start empty and hold only what ``record_episodic_event`` is given.
"""

import time
from typing import Any

from app.utils.logger import logger


class LongTermMemoryArchitecture:
    """Episodic (past consultations), semantic (learned patterns), procedural (advisory lessons) memory."""

    def __init__(self) -> None:
        self.episodic_memories: list[dict[str, Any]] = []
        self.semantic_memories: dict[str, Any] = {}
        self.procedural_memories: list[dict[str, Any]] = []

    def record_episodic_event(
        self,
        scenario: str,
        conditions: dict[str, Any],
        action: str,
        outcome: str,
        effectiveness: float | None,
    ) -> None:
        """Stores an episodic consultation memory. ``effectiveness`` is None unless measured."""
        mem = {
            "event_id": f"EP-{int(time.time())}",
            "scenario": scenario,
            "market_conditions": conditions,
            "ai_advisory_action": action,
            "outcome_pnl_delta": outcome,
            "effectiveness_score": effectiveness,
        }
        self.episodic_memories.append(mem)
        logger.info("Recorded new episodic memory: %s", scenario)

    def retrieve_relevant_learnings(self, current_regime: str) -> list[dict[str, Any]]:
        """Past consultations matching the current regime; nothing is returned when none match.

        The previous fallback returned the first three episodes whatever their regime, which
        presented unrelated history as relevant learning.
        """
        return [
            m for m in self.episodic_memories
            if m["market_conditions"].get("regime") == current_regime or current_regime in m["scenario"]
        ]


long_term_memory = LongTermMemoryArchitecture()
