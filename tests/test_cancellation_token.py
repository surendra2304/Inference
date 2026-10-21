from __future__ import annotations

import asyncio

import pytest

from app.inference_runtime.cancellation import CancellationToken


def _pending_token_waiters() -> list[asyncio.Task]:
    current = asyncio.current_task()
    return [
        task for task in asyncio.all_tasks()
        if task is not current
        and task.get_name() == "cancellation-token-wait"
        and not task.done()
    ]


@pytest.mark.asyncio
async def test_race_returns_completed_operation_and_cleans_token_waiter():
    token = CancellationToken()

    async def operation():
        return "answer"

    assert await token.race(operation()) == "answer"
    await asyncio.sleep(0)
    assert _pending_token_waiters() == []


@pytest.mark.asyncio
async def test_token_cancellation_interrupts_operation_and_awaits_its_cleanup():
    token = CancellationToken()
    started = asyncio.Event()
    cleaned = asyncio.Event()

    async def operation():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    task = asyncio.create_task(token.race(operation()))
    await asyncio.wait_for(started.wait(), timeout=1.0)
    token.cancel()

    with pytest.raises(asyncio.CancelledError, match="cancellation token was set"):
        await task

    assert cleaned.is_set()
    assert _pending_token_waiters() == []


@pytest.mark.asyncio
async def test_external_caller_cancellation_cancels_operation_and_token_waiter():
    token = CancellationToken()
    started = asyncio.Event()
    cleaned = asyncio.Event()

    async def operation():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    task = asyncio.create_task(token.race(operation()))
    await asyncio.wait_for(started.wait(), timeout=1.0)

    for _ in range(50):
        if _pending_token_waiters():
            break
        await asyncio.sleep(0)
    else:
        pytest.fail("token waiter was not started")

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert cleaned.is_set()
    assert _pending_token_waiters() == []


@pytest.mark.asyncio
async def test_operation_exception_propagates_after_waiter_cleanup():
    token = CancellationToken()

    async def operation():
        raise ValueError("provider failed")

    with pytest.raises(ValueError, match="provider failed"):
        await token.race(operation())

    await asyncio.sleep(0)
    assert _pending_token_waiters() == []


@pytest.mark.asyncio
async def test_race_accepts_future_awaitables():
    token = CancellationToken()
    future = asyncio.get_running_loop().create_future()
    future.set_result(42)

    assert await token.race(future) == 42
    assert _pending_token_waiters() == []
