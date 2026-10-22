from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostRate:
    input_per_1k: float = 0.0
    output_per_1k: float = 0.0


class CostCalculator:
    def __init__(self, rates: dict[tuple[str, str], CostRate] | None = None):
        self.rates = rates or {}

    def estimate(self, provider: str, model: str, prompt_tokens: int, max_tokens: int) -> float:
        r = self.rates.get((provider, model), CostRate())
        return prompt_tokens / 1000 * r.input_per_1k + max_tokens / 1000 * r.output_per_1k

    def actual(self, provider: str, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        r = self.rates.get((provider, model), CostRate())
        return prompt_tokens / 1000 * r.input_per_1k + completion_tokens / 1000 * r.output_per_1k
