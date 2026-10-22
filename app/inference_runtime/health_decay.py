from __future__ import annotations

import math
import time
from dataclasses import dataclass


@dataclass
class Signal:
    value: float = 0
    updated: float = 0


class HealthDecay:
    def __init__(self, half_life: float = 60):
        self.half_life = max(0.1, half_life)
        self._signals: dict[str, Signal] = {}

    def set(self, name: str, value: float):
        self._signals[name] = Signal(value, time.time())

    def get(self, name: str, default: float = 0) -> float:
        s = self._signals.get(name)
        if s is None:
            return default
        age = max(0, time.time() - s.updated)
        return s.value * math.exp(-math.log(2) * age / self.half_life)
