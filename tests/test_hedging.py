from __future__ import annotations

import asyncio
import math

import pytest

from app.inference_runtime.hedging import hedge


@pytest.mark.asyncio
async def test_hedge_continues_after_fast_failure_until_later_attempt_succeeds():
    started: list[str] = []

    async def fails_first():
        started.append("first")
        raise RuntimeError("first provider failed")

    async def fails_second():
        started.append("second")
        raise ValueError("second provider failed")

    async def succeeds_third():
        started.append("third")
        return "healthy provider answer"

    result = await hedge([fails_first, fails_second, succeeds_third], delay_seconds=10.0)

    assert result == "healthy provider answer"
    assert started == ["first", "second", "third"]


@pytest.mark.asyncio
async def test_hedge_returns_success_while_other_attempt_is_still_running_and_cleans_loser():
    slow_started = asyncio.Event()
    slow_cancelled = asyncio.Event()
    fast_started = asyncio.Event()

    async def slow_provider():
        slow_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            slow_cancelled.set()
            raise

    async def fast_provider():
        fast_started.set()
        return "fast winner"

    result = await hedge([slow_provider, fast_provider], delay_seconds=0.01)

    assert result == "fast winner"
    assert slow_started.is_set()
    assert fast_started.is_set()
    assert slow_cancelled.is_set()


@pytest.mark.asyncio
async def test_hedge_waits_for_later_inflight_success_after_an_earlier_error():
    second_started = asyncio.Event()
    release_second = asyncio.Event()

    async def first_provider():
        raise RuntimeError("first failed immediately")

    async def second_provider():
        second_started.set()
        await release_second.wait()
        return "second succeeds"

    task = asyncio.create_task(hedge([first_provider, second_provider], delay_seconds=1.0))
    await asyncio.wait_for(second_started.wait(), timeout=1.0)
    await asyncio.sleep(0)
    assert not task.done()

    release_second.set()
    assert await asyncio.wait_for(task, timeout=1.0) == "second succeeds"


@pytest.mark.asyncio
async def test_hedge_raises_only_after_all_candidates_fail():
    async def first_provider():
        raise RuntimeError("first error")

    async def second_provider():
        raise ValueError("last error")

    with pytest.raises(ValueError, match="last error"):
        await hedge([first_provider, second_provider], delay_seconds=1.0)


@pytest.mark.asyncio
async def test_hedge_propagates_caller_cancellation_and_cancels_all_attempts():
    started = [asyncio.Event(), asyncio.Event()]
    cancelled = [asyncio.Event(), asyncio.Event()]

    def provider(index: int):
        async def call():
            started[index].set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled[index].set()
                raise
        return call

    task = asyncio.create_task(hedge([provider(0), provider(1)], delay_seconds=0.01))
    await asyncio.wait_for(started[0].wait(), timeout=1.0)
    await asyncio.wait_for(started[1].wait(), timeout=1.0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert all(event.is_set() for event in cancelled)


@pytest.mark.parametrize("delay", [-1, math.nan, math.inf, -math.inf, True, "0.1"])
@pytest.mark.asyncio
async def test_hedge_rejects_invalid_stagger_interval(delay):
    async def caller():
        return "unused"

    with pytest.raises(ValueError, match="delay_seconds"):
        await hedge([caller], delay_seconds=delay)


@pytest.mark.asyncio
async def test_hedge_requires_at_least_one_candidate():
    with pytest.raises(ValueError, match="at least one caller"):
        await hedge([])
