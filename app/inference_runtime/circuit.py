from __future__ import annotations

import math
import threading
import time
from enum import Enum


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Thread-safe breaker with one recovery probe at a time."""

    def __init__(self, threshold: int = 5, recovery_seconds: float = 30.0) -> None:
        if isinstance(threshold, bool) or not isinstance(threshold, int) or threshold < 1:
            raise ValueError("threshold must be a positive integer")
        if (
            isinstance(recovery_seconds, bool)
            or not isinstance(recovery_seconds, (int, float))
            or not math.isfinite(recovery_seconds)
            or recovery_seconds < 0
        ):
            raise ValueError("recovery_seconds must be finite and non-negative")
        self.threshold = threshold
        self.recovery_seconds = float(recovery_seconds)
        self.failures = 0
        self.opened_at = 0.0
        self.state = CircuitState.CLOSED
        self._probe_in_flight = False
        self._lock = threading.RLock()

    def allow(self) -> bool:
        """Return whether this call is admitted, reserving the sole half-open probe."""
        with self._lock:
            now = time.monotonic()
            if self.state == CircuitState.OPEN:
                if now - self.opened_at < self.recovery_seconds:
                    return False
                self.state = CircuitState.HALF_OPEN
                self._probe_in_flight = False

            if self.state == CircuitState.HALF_OPEN:
                if self._probe_in_flight:
                    return False
                self._probe_in_flight = True
                return True

            return True

    def success(self) -> None:
        with self._lock:
            self.failures = 0
            self.state = CircuitState.CLOSED
            self.opened_at = 0.0
            self._probe_in_flight = False

    def failure(self) -> None:
        with self._lock:
            self.failures += 1
            if self.state == CircuitState.HALF_OPEN or self.failures >= self.threshold:
                self.state = CircuitState.OPEN
                self.opened_at = time.monotonic()
            self._probe_in_flight = False

    def release_probe(self) -> None:
        """Release a half-open permit when the probe is cancelled before completion."""
        with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self._probe_in_flight = False
                # Retain the original open timestamp. If recovery already elapsed,
                # another caller may immediately take the probe instead of deadlocking.
                self.state = CircuitState.OPEN
