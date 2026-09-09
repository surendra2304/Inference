"""Learned routing and orchestration strategy store."""

from pydantic import BaseModel, Field

from app.memory.base import BaseMemory, StrategyRecord
from app.memory.sqlite import SQLiteMemory
from app.utils.ids import generate_id
from app.utils.logger import logger


class RecommendedStrategy(BaseModel):
    """Strategy recommendation for a new query based on historical empirical outcomes."""
    task_type: str
    recommended_mode: str = Field(description="fast, review, debate")
    recommended_agents: list[str]
    recommended_provider: str = "gemini"
    recommended_model: str = "gemini-3.5-flash-lite"
    confidence: float = Field(ge=0.0, le=1.0)
    historical_score: float = Field(ge=0.0, le=1.0)
    sample_size: int = 1
    reasoning: str


class StrategyStore:
    """Saves, updates, and retrieves learned orchestration patterns and agent compositions."""

    def __init__(self, memory: BaseMemory | None = None) -> None:
        self.memory = memory or SQLiteMemory()

    async def save_learned_pattern(
        self,
        task_type: str,
        mode: str,
        agents: list[str],
        score: float,
        provider: str = "gemini",
        model: str = "gemini-3.5-flash-lite"
    ) -> None:
        """Stores or updates a winning orchestration pattern for a task category."""
        existing = await self.memory.get_strategy(task_type)
        if existing:
            # Moving average score update
            new_samples = existing.sample_size + 1
            updated_score = round(((existing.score * existing.sample_size) + score) / new_samples, 3)
            existing.score = updated_score
            existing.sample_size = new_samples
            if score >= existing.score:
                existing.strategy = mode
                existing.recommended_agents = agents
                existing.recommended_provider = provider
                existing.recommended_model = model
            await self.memory.save_strategy(existing)
            logger.info("Updated learned strategy for task type '%s' (New score: %.3f, N=%d)", task_type, updated_score, new_samples)
        else:
            strat = StrategyRecord(
                id=generate_id("strat"),
                task_type=task_type,
                strategy=mode,
                score=score,
                sample_size=1,
                recommended_agents=agents,
                recommended_provider=provider,
                recommended_model=model
            )
            await self.memory.save_strategy(strat)
            logger.info("Created new learned strategy for task type '%s' (Mode: %s, Score: %.3f)", task_type, mode, score)

    async def recommend_strategy(
        self,
        task_type: str,
        complexity: str = "auto"
    ) -> RecommendedStrategy | None:
        """Queries historical outcomes to recommend the optimal mode and specialist panel."""
        strat = await self.memory.get_strategy(task_type)
        if strat and strat.score >= 0.75:
            return RecommendedStrategy(
                task_type=task_type,
                recommended_mode=strat.strategy,
                recommended_agents=strat.recommended_agents,
                recommended_provider=strat.recommended_provider,
                recommended_model=strat.recommended_model,
                confidence=min(0.95, 0.70 + (strat.sample_size * 0.05)),
                historical_score=strat.score,
                sample_size=strat.sample_size,
                reasoning=f"Empirically validated pattern from {strat.sample_size} historical tasks with avg score {strat.score:.2f}."
            )
        return None


# Global default strategy store
strategy_store = StrategyStore()
