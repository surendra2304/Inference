from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from app.utils.bounded_store import BoundedList

from .contracts import CompletionRequest, ProviderEndpoint
from .errors import ProviderUnavailable
from .retry_policy import RetryPolicy


@dataclass(frozen=True)
class AttemptRecord:
    provider: str
    model: str
    attempt: int
    latency: float
    error: str | None
    selected: bool


class FallbackExecutor:
    def __init__(self, policy: RetryPolicy | None = None, max_history: int = 2000):
        self.policy = policy or RetryPolicy(3)
        # Bounded: one record is appended per provider attempt, and nothing in this service
        # reads the list, so an unbounded list was pure accumulation on the failure path
        # (the path a degraded provider drives hardest).
        self.history: BoundedList[AttemptRecord] = BoundedList(
            "inference_runtime.fallback_attempts", max_entries=max_history
        )

    async def run(self, request: CompletionRequest, candidates: list[ProviderEndpoint], call):
        last = None
        for idx, ep in enumerate(candidates, 1):
            try:
                start = time.perf_counter()
                result = await call(ep)
                self.history.append(AttemptRecord(ep.provider, ep.model, idx, time.perf_counter() - start, None, True))
                return result
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                latency = time.perf_counter() - start
                self.history.append(AttemptRecord(ep.provider, ep.model, idx, latency, str(exc)[:300], False))
                last = exc
        raise ProviderUnavailable(f"all inference endpoints failed: {last}") from last
