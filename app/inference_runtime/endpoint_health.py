from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class EndpointHealth:
    successes: int = 0
    failures: int = 0
    latency_ewma: float = 0
    last_error: str | None = None
    last_failure_at: float = 0

    def success_rate(self) -> float:
        return self.successes / max(1, self.successes + self.failures)

    def record_success(self, latency: float):
        self.successes += 1
        self.latency_ewma = latency if not self.latency_ewma else 0.8 * self.latency_ewma + 0.2 * latency
        self.last_error = None

    def record_failure(self, error: str, latency: float = 0):
        self.failures += 1
        self.last_error = error[:500]
        self.last_failure_at = time.time()
        self.latency_ewma = latency if not self.latency_ewma else 0.8 * self.latency_ewma + 0.2 * latency
