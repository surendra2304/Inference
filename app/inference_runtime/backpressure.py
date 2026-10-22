from __future__ import annotations

import asyncio


class BackpressureGate:
    def __init__(self, limit: int):
        self.limit = max(1, limit)
        self._inflight = 0
        self._cv = asyncio.Condition()

    async def enter(self) -> None:
        async with self._cv:
            while self._inflight >= self.limit:
                await self._cv.wait()
            self._inflight += 1

    async def leave(self) -> None:
        async with self._cv:
            self._inflight = max(0, self._inflight - 1)
            self._cv.notify()

    async def __aenter__(self):
        await self.enter()
        return self

    async def __aexit__(self, *exc):
        await self.leave()
