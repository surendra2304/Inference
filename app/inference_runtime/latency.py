from __future__ import annotations

import statistics
import threading


class LatencyWindow:
    def __init__(self, size: int = 500):
        self.size = max(10, size)
        self._values: list[float] = []
        self._lock = threading.Lock()

    def add(self, value: float) -> None:
        with self._lock:
            self._values.append(max(0.0, value))
            self._values = self._values[-self.size :]

    def percentile(self, p: int) -> float:
        with self._lock:
            if not self._values:
                return 0.0
            data = sorted(self._values)
            index = min(len(data) - 1, max(0, int((p / 100) * len(data)) - 1))
            return data[index]

    def snapshot(self) -> dict[str, float]:
        with self._lock:
            if not self._values:
                return {"count": 0, "p50": 0, "p95": 0, "p99": 0, "avg": 0}
            return {
                "count": len(self._values),
                "p50": self.percentile(50),
                "p95": self.percentile(95),
                "p99": self.percentile(99),
                "avg": statistics.fmean(self._values),
            }
