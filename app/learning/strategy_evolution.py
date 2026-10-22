"""Strategy Evolution Engine: Genetic Strategy Variants, Elitism, Mutation & Adaptation."""

import random
from typing import Any, Literal

from pydantic import BaseModel


class StrategyVariant(BaseModel):
    strategy_id: str
    task_type: str
    agent_composition: list[str]
    provider_selection: str
    prompt_template: str
    mode: Literal["fast", "review", "debate"]
    confidence_threshold: float
    outcomes_evaluated: int = 0
    success_count: int = 0
    total_cost_usd: float = 0.0
    is_elite: bool = False


class StrategyEvolutionEngine:
    """Maintains a population of 20 strategy variants, allocating 80% traffic to top elites and 20% to mutating explorers."""

    def __init__(self) -> None:
        self.population: list[StrategyVariant] = [
            StrategyVariant(
                strategy_id=f"strat-{i+1:02d}",
                task_type="strategic_decision" if i % 2 == 0 else "lead_qualification",
                agent_composition=["strategist", "critic", "fact_checker"] if i % 3 == 0 else ["data_analyst", "strategist"],
                provider_selection="gemini" if i % 2 == 0 else "groq",
                prompt_template="standard_structured" if i % 2 == 0 else "chain_of_thought",
                mode="debate" if i % 3 == 0 else "review",
                confidence_threshold=0.80 + (i * 0.005),
                outcomes_evaluated=40 + (i * 2),
                success_count=35 + int(i * 1.8),
                total_cost_usd=0.045,
                is_elite=(i < 3)
            )
            for i in range(20)
        ]

    def select_strategy(self, task_type: str) -> StrategyVariant:
        """Selects strategy variant using 80/20 exploitation/exploration balance."""
        matching = [s for s in self.population if s.task_type == task_type or task_type in s.task_type]
        if not matching:
            matching = self.population

        # 80% Exploitation of top performers
        if random.random() < 0.80:
            matching.sort(key=lambda s: (s.success_count / max(1, s.outcomes_evaluated)), reverse=True)
            return matching[0]

        # 20% Exploration of underused variants
        return random.choice(matching)

    def evolve_population(self) -> dict[str, Any]:
        """Mutates the bottom 25% underperforming variants while protecting top elites."""
        self.population.sort(key=lambda s: (s.success_count / max(1, s.outcomes_evaluated)), reverse=True)

        # Mark top 3 as elite
        for idx, s in enumerate(self.population):
            s.is_elite = (idx < 3)

        # Mutate bottom 5 variants (bottom 25%)
        mutated_count = 0
        providers = ["gemini", "groq", "nvidia", "mistral", "openrouter"]
        for s in self.population[-5:]:
            if s.outcomes_evaluated >= 30:
                s.provider_selection = random.choice(providers)
                s.confidence_threshold = round(random.uniform(0.75, 0.88), 2)
                s.prompt_template = "adaptive_compressed"
                s.outcomes_evaluated = 0
                s.success_count = 0
                mutated_count += 1

        return {
            "total_population": len(self.population),
            "elite_variants_preserved": 3,
            "underperformers_mutated": mutated_count,
            "top_strategy_id": self.population[0].strategy_id,
            "top_strategy_success_rate": round((self.population[0].success_count / max(1, self.population[0].outcomes_evaluated)) * 100.0, 1)
        }

    def get_population_dashboard(self) -> list[dict[str, Any]]:
        return [s.model_dump() for s in self.population]


strategy_evolution_engine = StrategyEvolutionEngine()
