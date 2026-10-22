from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass


@dataclass
class Lease:
    key: str
    expires_at: float


class ConcurrencyLeasePool:
    def __init__(self, limit: int):
        self._sem = asyncio.Semaphore(max(1, limit))
        self._seq = 0

    async def acquire(self, timeout: float | None = None) -> Lease:
        if timeout is None:
            await self._sem.acquire()
        else:
            await asyncio.wait_for(self._sem.acquire(), timeout)
        self._seq += 1
        return Lease(str(self._seq), time.monotonic() + 30)

    def release(self, lease: Lease):
        self._sem.release()
