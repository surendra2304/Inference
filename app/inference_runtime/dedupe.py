from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass
class _InFlight:
    """One shared operation and the number of callers still awaiting it."""

    task: asyncio.Task[Any]
    waiters: int = 0


class RequestDeduper:
    """Single-flight deduplication with explicit waiter cancellation rules.

    Cancelling one waiter must not cancel a provider operation that other callers
    are awaiting. When the final waiter leaves, unfinished work is cancelled and
    the entry is removed, keeping the in-flight table bounded by active requests.
    """

    def __init__(self) -> None:
        self._tasks: dict[str, _InFlight] = {}
        self._lock = asyncio.Lock()

    @staticmethod
    def fingerprint(payload: Any) -> str:
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()

    @staticmethod
    def _consume_exception(task: asyncio.Task[Any]) -> None:
        """Retrieve detached-task exceptions without hiding them from awaiters."""
        if task.cancelled():
            return
        try:
            task.exception()
        except asyncio.CancelledError:
            return

    async def run(self, key: str, factory: Callable[[], Awaitable[T]]) -> T:
        """Return the shared result for a key while preserving waiter cancellation."""
        if not isinstance(key, str) or not key.strip():
            raise ValueError("deduplication key must be a non-empty string")

        async with self._lock:
            entry = self._tasks.get(key)
            if entry is None:
                task = asyncio.create_task(factory())
                task.add_done_callback(self._consume_exception)
                entry = _InFlight(task=task)
                self._tasks[key] = entry
            entry.waiters += 1

        task = entry.task
        try:
            # Awaiting the task directly lets cancellation of any one caller
            # propagate into shared work. Shield only the shared operation; each
            # waiter still receives its own CancelledError.
            return await asyncio.shield(task)
        finally:
            async with self._lock:
                entry.waiters -= 1
                if entry.waiters == 0 and self._tasks.get(key) is entry:
                    self._tasks.pop(key, None)
                    # No caller needs the result any longer. Avoid leaking a
                    # provider request when the final waiter was cancelled.
                    if not task.done():
                        task.cancel()
