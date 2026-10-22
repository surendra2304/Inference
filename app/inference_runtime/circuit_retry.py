from __future__ import annotations

from .circuit import CircuitBreaker
from .retry_policy import RetryPolicy


class ProviderState:
    def __init__(self):
        self.breaker = CircuitBreaker()
        self.retry = RetryPolicy()

    def allow(self) -> bool:
        return self.breaker.allow()

    def success(self) -> None:
        self.breaker.success()

    def failure(self) -> None:
        self.breaker.failure()
