from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from .errors import ProviderUnavailable


@dataclass
class KeyState:
    key: str
    failures: int = 0
    quarantined_until: float = 0.0


class SafeKeyPool:
    def __init__(self, provider: str, keys: list[str] | None = None, quarantine_seconds: float = 60.0) -> None:
        self.provider = provider
        self._states = [KeyState(k) for k in (keys or []) if k]
        self._index = 0
        self._quarantine = quarantine_seconds
        self._lock = threading.RLock()

    def replace(self, keys: list[str]) -> None:
        with self._lock:
            self._states = [KeyState(k) for k in keys if k]
            self._index = 0

    def available_count(self) -> int:
        now = time.monotonic()
        return sum(s.quarantined_until <= now for s in self._states)

    def choose(self) -> str | None:
        with self._lock:
            if not self._states:
                return None
            now = time.monotonic()
            for _ in range(len(self._states)):
                s = self._states[self._index % len(self._states)]
                self._index = (self._index + 1) % len(self._states)
                if s.quarantined_until <= now:
                    return s.key
            return None  # fail closed; caller must not hammer quarantined credentials

    def quarantine(self, key: str, seconds: float | None = None) -> None:
        with self._lock:
            for s in self._states:
                if s.key == key:
                    s.failures += 1
                    s.quarantined_until = time.monotonic() + (seconds if seconds is not None else self._quarantine)
                    return

    def next_available_delay(self) -> float:
        with self._lock:
            if not self._states:
                return float("inf")
            now = time.monotonic()
            return max(0.0, min(s.quarantined_until for s in self._states) - now)

    def require_key(self) -> str:
        key = self.choose()
        if key is None:
            raise ProviderUnavailable(f"No healthy key available for provider {self.provider}")
        return key
