from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Generic, TypeVar

T = TypeVar("T")


@dataclass
class BatchItem(Generic[T]):
    payload: T
    future: asyncio.Future[object]
    submitted_at: float


class MicroBatcher(Generic[T]):
    """Collect inputs into bounded batches and resolve every submission exactly once.

    Closing is graceful: already accepted submissions are drained before the worker
    exits, while new submissions are rejected. If the worker task is cancelled
    unexpectedly, active and queued futures are completed with cancellation/errors
    instead of being left pending forever.
    """

    def __init__(self, max_batch: int = 16, max_wait_ms: int = 15) -> None:
        if isinstance(max_batch, bool) or not isinstance(max_batch, int) or max_batch < 1:
            raise ValueError("max_batch must be a positive integer")
        if (
            isinstance(max_wait_ms, bool)
            or not isinstance(max_wait_ms, (int, float))
            or not math.isfinite(max_wait_ms)
            or max_wait_ms < 0
        ):
            raise ValueError("max_wait_ms must be a finite non-negative number")

        self.max_batch = max_batch
        self.max_wait = float(max_wait_ms) / 1000
        self._q: list[BatchItem[T]] = []
        self._cv = asyncio.Condition()
        self._task: asyncio.Task[None] | None = None
        self._worker: Callable[[list[T]], Awaitable[list[object]]] | None = None
        self._started = False
        self._closed = False

    async def start(self, worker: Callable[[list[T]], Awaitable[list[object]]]) -> None:
        if self._closed:
            raise RuntimeError("cannot start a closed batcher")
        if self._started:
            raise RuntimeError("batcher has already been started")
        if not callable(worker):
            raise TypeError("worker must be callable")

        self._worker = worker
        self._started = True
        self._task = asyncio.create_task(self._run())

    async def submit(self, payload: T) -> object:
        if not self._started:
            raise RuntimeError("batcher must be started before submit")
        future: asyncio.Future[object] = asyncio.get_running_loop().create_future()
        item = BatchItem(payload=payload, future=future, submitted_at=time.monotonic())

        async with self._cv:
            if self._closed:
                raise RuntimeError("cannot submit to a closed batcher")
            self._q.append(item)
            self._cv.notify()

        try:
            return await future
        except asyncio.CancelledError:
            # Avoid doing provider work for an item whose caller cancelled while it
            # was still queued. If a worker already took it into a batch, the future
            # remains cancelled and the worker discards its eventual result.
            async with self._cv:
                try:
                    self._q.remove(item)
                except ValueError:
                    pass
                self._cv.notify_all()
            raise

    async def _run(self) -> None:
        active_batch: list[BatchItem[T]] = []
        try:
            while True:
                async with self._cv:
                    while not self._q and not self._closed:
                        await self._cv.wait()
                    if self._closed and not self._q:
                        break

                    start = time.monotonic()
                    while (
                        not self._closed
                        and len(self._q) < self.max_batch
                        and time.monotonic() - start < self.max_wait
                    ):
                        remaining = self.max_wait - (time.monotonic() - start)
                        if remaining <= 0:
                            break
                        try:
                            await asyncio.wait_for(self._cv.wait(), remaining)
                        except asyncio.TimeoutError:
                            break

                    active_batch = self._q[: self.max_batch]
                    del self._q[: self.max_batch]

                worker = self._worker
                if worker is None:
                    raise RuntimeError("batch worker was not configured")
                try:
                    results = await worker([item.payload for item in active_batch])
                    if len(results) != len(active_batch):
                        raise RuntimeError("batch worker returned incorrect result count")
                    for item, result in zip(active_batch, results):
                        if not item.future.done():
                            item.future.set_result(result)
                except asyncio.CancelledError:
                    for item in active_batch:
                        if not item.future.done():
                            item.future.cancel()
                    raise
                except Exception as exc:
                    for item in active_batch:
                        if not item.future.done():
                            item.future.set_exception(exc)
                finally:
                    active_batch = []
        finally:
            # This also covers unexpected task cancellation and fatal worker errors:
            # no submitter should remain asleep on a Future that can never resolve.
            async with self._cv:
                queued = self._q
                self._q = []
                self._closed = True
                self._cv.notify_all()

            for item in active_batch + queued:
                if item.future.done():
                    continue
                if asyncio.current_task() and asyncio.current_task().cancelled():
                    item.future.cancel()
                else:
                    item.future.set_exception(
                        RuntimeError("batcher stopped before processing this submission")
                    )

    async def close(self) -> None:
        async with self._cv:
            self._closed = True
            self._cv.notify_all()
        if self._task:
            await self._task
