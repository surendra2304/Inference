from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import TypeVar

T = TypeVar("T")


class CancellationToken:
    """Cooperative cancellation signal with race helpers that own their task cleanup."""

    def __init__(self) -> None:
        self.event = asyncio.Event()

    def cancel(self) -> None:
        self.event.set()

    @property
    def cancelled(self) -> bool:
        return self.event.is_set()

    async def wait(self) -> None:
        await self.event.wait()

    async def race(self, awaitable: Awaitable[T]) -> T:
        """Return the operation result, or raise cancellation when the token is set.

        The operation and the token waiter are both owned by this call. They are
        cancelled and awaited on every exit path, including cancellation of the
        caller itself; otherwise a disconnected caller could leave provider work
        and a token-wait task running in the background.
        """
        task = asyncio.ensure_future(awaitable)
        cancel_task = asyncio.create_task(self.wait(), name="cancellation-token-wait")
        try:
            done, _ = await asyncio.wait(
                {task, cancel_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if cancel_task in done:
                raise asyncio.CancelledError("cancellation token was set")
            return task.result()
        finally:
            if not task.done():
                task.cancel()
            if not cancel_task.done():
                cancel_task.cancel()
            # Wait for both cleanup paths before race() returns or propagates an
            # exception. gather(return_exceptions=True) consumes cancellation/errors
            # from owned children without hiding the primary outcome above.
            await asyncio.gather(task, cancel_task, return_exceptions=True)
