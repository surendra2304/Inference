from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class Counter:
    value: int = 0


class Metrics:
    def __init__(self) -> None:
        self._c: dict[tuple[str, tuple], Counter] = defaultdict(Counter)
        self._lock = threading.Lock()
        self._lat: dict[tuple[str, tuple], list[float]] = {}
        self._tokens: dict[str, int] = defaultdict(int)
        self._cost: dict[str, float] = defaultdict(float)

    def inc(self, name: str, value: int = 1, labels: tuple = ()):
        with self._lock:
            self._c[(name, labels)].value += value

    def observe_latency(self, name: str, value: float, labels: tuple = ()):
        with self._lock:
            self._lat.setdefault((name, labels), []).append(value)

    def observe_usage(self, tenant: str, tokens: int, cost: float):
        with self._lock:
            self._tokens[tenant] += tokens
            self._cost[tenant] += cost

    def snapshot(self):
        with self._lock:
            return {
                "counters": {str(k): v.value for k, v in self._c.items()},
                "latency_count": {str(k): len(v) for k, v in self._lat.items()},
                "tokens": dict(self._tokens),
                "cost": dict(self._cost),
            }
