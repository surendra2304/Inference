from __future__ import annotations

from .circuit import CircuitBreaker


class CircuitPool:
    def __init__(self):
        self._items = {}

    def get(self, key: str) -> CircuitBreaker:
        return self._items.setdefault(key, CircuitBreaker())

    def allow(self, key: str) -> bool:
        return self.get(key).allow()

    def success(self, key: str) -> None:
        self.get(key).success()

    def failure(self, key: str) -> None:
        self.get(key).failure()
