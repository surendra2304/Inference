from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AutoscaleDecision:
    target_replicas: int
    reason: str


class AutoscalePlanner:
    def __init__(self, min_replicas: int = 1, max_replicas: int = 10, target_queue_per_replica: float = 2.0):
        self.min = min_replicas
        self.max = max_replicas
        self.target = target_queue_per_replica

    def decide(self, queue_depth: int, replicas: int, ongoing: int) -> AutoscaleDecision:
        desired = max(self.min, min(self.max, int((queue_depth + ongoing) / max(self.target, 0.1) + 0.999)))
        reason = "scale_up" if desired > replicas else "scale_down" if desired < replicas else "hold"
        return AutoscaleDecision(desired, reason)
