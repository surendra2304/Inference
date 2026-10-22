from __future__ import annotations

from dataclasses import dataclass

from .retry import retryable


@dataclass(frozen=True)
class RetryDecision:
    retry: bool
    delay_seconds: float
    reason: str


class RetryPolicy:
    def __init__(self, max_attempts: int = 3, max_delay: float = 8.0):
        self.max_attempts = max(1, max_attempts)
        self.max_delay = max_delay

    def decide(self, exc: Exception, attempt: int, remaining: float | None) -> RetryDecision:
        if attempt >= self.max_attempts:
            return RetryDecision(False, 0.0, "attempt_limit")
        if not retryable(exc):
            return RetryDecision(False, 0.0, "non_retryable")
        delay = min(self.max_delay, 0.25 * (2 ** (attempt - 1)))
        if remaining is not None and delay >= remaining:
            return RetryDecision(False, 0.0, "deadline")
        return RetryDecision(True, delay, "transient_failure")
