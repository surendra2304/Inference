from __future__ import annotations

import asyncio
import random
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")


def retryable(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    return any(
        x in name or x in msg for x in ("timeout", "ratelimit", "temporar", "unavailable", "429", "503", "connection")
    )


async def with_retries(fn: Callable[[], Awaitable[T]], max_retries: int, base_delay: float = 0.25) -> T:
    last: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return await fn()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            last = exc
            if attempt >= max_retries or not retryable(exc):
                raise
            await asyncio.sleep(min(8.0, base_delay * (2**attempt) + random.random() * 0.1))
    assert last is not None
    raise last
