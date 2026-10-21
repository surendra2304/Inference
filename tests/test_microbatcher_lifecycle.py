from __future__ import annotations

import asyncio
import math

import pytest

from app.inference_runtime.batch import MicroBatcher


@pytest.mark.asyncio
async def test_close_drains_all_accepted_batches_before_returning():
    batcher: MicroBatcher[str] = MicroBatcher(max_batch=2, max_wait_ms=1000)
    first_batch_started = asyncio.Event()
    release_first_batch = asyncio.Event()
    calls: list[list[str]] = []

    async def worker(payloads: list[str]) -> list[object]:
        calls.append(payloads)
        if len(calls) == 1:
            first_batch_started.set()
            await release_first_batch.wait()
        return [item.upper() for item in payloads]

    await batcher.start(worker)
    submissions = [
        asyncio.create_task(batcher.submit("a")),
        asyncio.create_task(batcher.submit("b")),
        asyncio.create_task(batcher.submit("c")),
    ]
    await asyncio.wait_for(first_batch_started.wait(), timeout=1.0)

    for _ in range(100):
        if len(batcher._q) == 1:
            break
        await asyncio.sleep(0)
    else:
        pytest.fail("third submission was not queued while the first batch was running")

    close_task = asyncio.create_task(batcher.close())
    await asyncio.sleep(0)
    assert not close_task.done(), "close must wait until an active worker and queued items finish"

    with pytest.raises(RuntimeError, match="closed batcher"):
        await batcher.submit("late")

    release_first_batch.set()
    await asyncio.wait_for(close_task, timeout=1.0)

    assert await asyncio.gather(*submissions) == ["A", "B", "C"]
    assert calls == [["a", "b"], ["c"]]


@pytest.mark.asyncio
async def test_submit_requires_start_and_closed_batcher_rejects_new_work():
    batcher: MicroBatcher[str] = MicroBatcher()

    with pytest.raises(RuntimeError, match="must be started"):
        await batcher.submit("too early")

    async def worker(items: list[str]) -> list[object]:
        return items

    await batcher.start(worker)
    with pytest.raises(RuntimeError, match="already been started"):
        await batcher.start(worker)
    await batcher.close()

    with pytest.raises(RuntimeError, match="closed batcher"):
        await batcher.submit("too late")
    with pytest.raises(RuntimeError, match="closed batcher"):
        await batcher.start(worker)


@pytest.mark.parametrize("max_batch", [0, -1, 1.5, True, "2"])
def test_microbatcher_rejects_invalid_batch_capacity(max_batch):
    with pytest.raises(ValueError, match="max_batch"):
        MicroBatcher(max_batch=max_batch)


@pytest.mark.parametrize("max_wait_ms", [-1, math.nan, math.inf, -math.inf, True, "20"])
def test_microbatcher_rejects_invalid_wait_window(max_wait_ms):
    with pytest.raises(ValueError, match="max_wait_ms"):
        MicroBatcher(max_wait_ms=max_wait_ms)


@pytest.mark.asyncio
async def test_worker_result_count_mismatch_fails_every_item_without_hanging():
    batcher: MicroBatcher[str] = MicroBatcher(max_batch=2, max_wait_ms=1000)

    async def bad_worker(items: list[str]) -> list[object]:
        return ["only one result"]

    await batcher.start(bad_worker)
    first = asyncio.create_task(batcher.submit("first"))
    second = asyncio.create_task(batcher.submit("second"))

    results = await asyncio.wait_for(
        asyncio.gather(first, second, return_exceptions=True),
        timeout=1.0,
    )
    assert all(isinstance(result, RuntimeError) for result in results)
    assert all("incorrect result count" in str(result) for result in results)

    await batcher.close()


@pytest.mark.asyncio
async def test_unexpected_worker_cancellation_resolves_active_and_queued_submitters():
    batcher: MicroBatcher[str] = MicroBatcher(max_batch=2, max_wait_ms=1000)
    started = asyncio.Event()

    async def worker(items: list[str]) -> list[object]:
        started.set()
        await asyncio.Event().wait()
        return items

    await batcher.start(worker)
    submissions = [
        asyncio.create_task(batcher.submit("a")),
        asyncio.create_task(batcher.submit("b")),
        asyncio.create_task(batcher.submit("c")),
    ]
    await asyncio.wait_for(started.wait(), timeout=1.0)

    for _ in range(100):
        if len(batcher._q) == 1:
            break
        await asyncio.sleep(0)
    else:
        pytest.fail("third submission was not queued")

    assert batcher._task is not None
    batcher._task.cancel()
    await asyncio.gather(batcher._task, return_exceptions=True)

    outcomes = await asyncio.wait_for(
        asyncio.gather(*submissions, return_exceptions=True),
        timeout=1.0,
    )
    assert all(isinstance(outcome, asyncio.CancelledError) for outcome in outcomes)

    await batcher.close()
