from __future__ import annotations

import asyncio
import heapq
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(order=True)
class QueueItem:
    priority: int
    sequence: int
    enqueued_at: float = field(compare=False)
    tenant: str = field(compare=False)
    payload: Any = field(compare=False)
    future: asyncio.Future = field(compare=False)


class FairScheduler:
    def __init__(self) -> None:
        self._heap: list[QueueItem] = []
        self._seq = 0
        self._cv = asyncio.Condition()
        self._tenant_inflight: dict[str, int] = {}
        self._stopped = False

    async def submit(self, payload: Any, tenant: str = "default", priority: int = 0) -> Any:
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        async with self._cv:
            self._seq += 1
            heapq.heappush(self._heap, QueueItem(priority, self._seq, time.monotonic(), tenant, payload, fut))
            self._cv.notify()
        return await fut

    async def run(self, worker, max_inflight_per_tenant: int = 4) -> None:
        while not self._stopped:
            async with self._cv:
                while not self._heap and not self._stopped:
                    await self._cv.wait()
                if self._stopped:
                    break
                idx = next(
                    (
                        i
                        for i, x in enumerate(self._heap)
                        if self._tenant_inflight.get(x.tenant, 0) < max_inflight_per_tenant
                    ),
                    None,
                )
                if idx is None:
                    await self._cv.wait()
                    continue
                item = self._heap.pop(idx)
                heapq.heapify(self._heap)
                self._tenant_inflight[item.tenant] = self._tenant_inflight.get(item.tenant, 0) + 1
            asyncio.create_task(self._execute(item, worker))

    async def _execute(self, item, worker) -> None:
        try:
            item.future.set_result(await worker(item.payload))
        except asyncio.CancelledError:
            item.future.cancel()
        except Exception as exc:
            item.future.set_exception(exc)
        finally:
            async with self._cv:
                self._tenant_inflight[item.tenant] = max(0, self._tenant_inflight.get(item.tenant, 1) - 1)
                self._cv.notify_all()

    async def stop(self) -> None:
        async with self._cv:
            self._stopped = True
            self._cv.notify_all()
