from __future__ import annotations

import math
from dataclasses import dataclass

from .contracts import CompletionRequest, ProviderEndpoint
from .health import ProviderHealth


@dataclass
class Arm:
    attempts: int = 0
    successes: int = 0
    latency: float = 0.0


class AdaptiveRouter:
    def __init__(self, health: ProviderHealth):
        self.health = health
        self._arms: dict[tuple[str, str], Arm] = {}

    def _arm(self, ep):
        return self._arms.setdefault((ep.provider, ep.model), Arm())

    def score(self, ep: ProviderEndpoint, total_attempts: int) -> float:
        a = self._arm(ep)
        rate = (a.successes + 1) / (a.attempts + 2)
        exploration = math.sqrt(2 * math.log(max(2, total_attempts)) / (a.attempts + 1))
        latency = max(0.01, self.health.snapshot(ep.provider).ewma_latency or ep.base_latency_seconds)
        return rate + 0.15 * exploration - 0.03 * latency - 0.01 * (ep.cost_per_1k_input + ep.cost_per_1k_output)

    def choose(self, request: CompletionRequest, endpoints: list[ProviderEndpoint]) -> ProviderEndpoint:
        candidates = [
            e
            for e in endpoints
            if request.model in {"auto", e.model}
            and request.capabilities.issubset(e.capabilities)
            and self.health.snapshot(e.provider).healthy
        ]
        if not candidates:
            raise RuntimeError("no adaptive routing candidate")
        total = sum(self._arm(e).attempts for e in candidates)
        return max(candidates, key=lambda e: (self.score(e, total), e.provider, e.model))

    def record(self, ep: ProviderEndpoint, success: bool, latency: float):
        a = self._arm(ep)
        a.attempts += 1
        a.successes += int(success)
        a.latency = latency if not a.latency else 0.8 * a.latency + 0.2 * latency
