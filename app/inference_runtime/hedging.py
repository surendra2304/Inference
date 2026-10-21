from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")


async def _run(fn: Callable[[], Awaitable[T]]) -> T:
    return await fn()


async def hedge(
    callers: list[Callable[[], Awaitable[T]]],
    delay_seconds: float = 0.2,
) -> T:
    """Return the first successful hedged call, not the first completed call.

    Calls start with delay_seconds between normal launches. If an attempt fails
    before that delay expires, the next candidate starts immediately. Failed attempts
    never mask a slower candidate that may still succeed. Once one succeeds, all
    remaining attempts are cancelled and awaited so their cleanup runs before return.
    """
    if not callers:
        raise ValueError("at least one caller required")
    if (
        isinstance(delay_seconds, bool)
        or not isinstance(delay_seconds, (int, float))
        or not math.isfinite(delay_seconds)
        or delay_seconds < 0
    ):
        raise ValueError("delay_seconds must be a finite non-negative number")

    tasks: set[asyncio.Task[T]] = set()
    errors: list[Exception] = []
    next_index = 0

    def launch_next() -> None:
        nonlocal next_index
        task = asyncio.create_task(_run(callers[next_index]))
        tasks.add(task)
        next_index += 1

    launch_next()
    try:
        while tasks:
            can_launch_more = next_index < len(callers)
            done, _ = await asyncio.wait(
                tasks,
                timeout=float(delay_seconds) if can_launch_more else None,
                return_when=asyncio.FIRST_COMPLETED,
            )

            if not done:
                # Existing attempts are still running; launch the next hedge only
                # because the stagger interval expired with no completion.
                if next_index < len(callers):
                    launch_next()
                continue

            for task in done:
                tasks.discard(task)
                if task.cancelled():
                    errors.append(RuntimeError("hedged provider attempt was cancelled"))
                    continue

                error = task.exception()
                if error is None:
                    return task.result()
                if not isinstance(error, Exception):
                    raise error
                errors.append(error)

            # A failure is known, so do not sleep out another hedge interval before
            # trying the next candidate. Other already-running candidates remain alive.
            if next_index < len(callers):
                launch_next()

        if errors:
            raise errors[-1]
        raise RuntimeError("all hedged provider attempts ended without a result")
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
