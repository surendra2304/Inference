from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LoadProfile:
    target_rps: float
    p95_latency_seconds: float
    max_queue_depth: int
    max_inflight: int

    def pressure(self, current_rps: float, current_p95: float, queue: int, inflight: int) -> float:
        factors = [
            current_rps / max(0.1, self.target_rps),
            current_p95 / max(0.001, self.p95_latency_seconds),
            queue / max(1, self.max_queue_depth),
            inflight / max(1, self.max_inflight),
        ]
        return max(factors)
