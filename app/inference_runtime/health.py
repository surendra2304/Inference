from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass
class HealthSnapshot:
    healthy: bool = True
    consecutive_failures: int = 0
    ewma_latency: float = 0.0
    last_error: str | None = None
    checked_at: float = 0.0


class ProviderHealth:
    def __init__(self) -> None:
        self._data: dict[str, HealthSnapshot] = {}
        self._lock = threading.RLock()

    def ensure(self, name: str) -> HealthSnapshot:
        with self._lock:
            return self._data.setdefault(name, HealthSnapshot(checked_at=time.time()))

    def success(self, name: str, latency: float) -> None:
        with self._lock:
            s = self.ensure(name)
            s.healthy = True
            s.consecutive_failures = 0
            s.last_error = None
            s.checked_at = time.time()
            s.ewma_latency = latency if not s.ewma_latency else 0.8 * s.ewma_latency + 0.2 * latency

    def failure(self, name: str, error: str, latency: float = 0.0) -> None:
        with self._lock:
            s = self.ensure(name)
            s.consecutive_failures += 1
            s.healthy = s.consecutive_failures < 3
            s.last_error = error[:500]
            s.checked_at = time.time()
            s.ewma_latency = latency if latency and not s.ewma_latency else s.ewma_latency

    def snapshot(self, name: str) -> HealthSnapshot:
        with self._lock:
            return self.ensure(name)
