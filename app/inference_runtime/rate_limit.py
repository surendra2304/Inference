from __future__ import annotations

import asyncio
import time


class AsyncTokenBucket:
    def __init__(self, rate: float, capacity: float) -> None:
        self.rate = max(rate, 0.0001)
        self.capacity = max(capacity, 1.0)
        self.tokens = self.capacity
        self.updated = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, cost: float = 1.0) -> None:
        cost = max(cost, 0.0)
        while True:
            async with self._lock:
                now = time.monotonic()
                self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
                self.updated = now
                if self.tokens >= cost:
                    self.tokens -= cost
                    return
                delay = (cost - self.tokens) / self.rate
            await asyncio.sleep(delay)  # never hold lock while waiting


class ConcurrencyGate:
    def __init__(self, limit: int) -> None:
        self._sem = asyncio.Semaphore(max(1, limit))

    async def __aenter__(self):
        await self._sem.acquire()
        return self

    async def __aexit__(self, *exc):
        self._sem.release()
