"""SQLite connections must belong to a loop that outlives their callers.

The defect this file pins down, measured repeatedly: aiosqlite delivers a statement's
result with ``future.get_loop().call_soon_threadsafe(...)``. When the loop that issued the
statement has closed by then, that call raises *inside the worker thread*; aiosqlite then
tries to report the failure through the same call, which raises again, and the thread dies
with an unhandled exception::

    PytestUnhandledThreadExceptionWarning: Exception in thread Thread-331
    (_connection_worker_thread)
      File ".../aiosqlite/core.py", line 66, in _connection_worker_thread
        future.get_loop().call_soon_threadsafe(set_result, future, result)
      File "/usr/lib/python3.11/asyncio/base_events.py", line 806, in call_soon_threadsafe
        self._check_closed()
    RuntimeError: Event loop is closed

Three to six of those appeared in every full-suite run, and no amount of loop-awareness in
``_acquire``/``_release`` removed them, because the race is inherent in binding a
connection to whatever short-lived loop calls it: a test's ``asyncio.run`` (or a server
shutdown, or a cancelled request) can close the loop with a statement still in flight.

Fixed by giving SQLite a process-wide owner loop (``app/memory/sqlite.py``). Measured
after the fix: 0 warnings across three consecutive full-suite runs, and 0 stranded
``_connection_worker_thread``s after three iterations of each abuse scenario below.
"""

import asyncio
import gc
import threading
import time

import pytest

from app.memory.sqlite import _DB_EXECUTOR, SQLiteMemory


def _worker_threads() -> list[str]:
    return [t.name for t in threading.enumerate() if "_connection_worker_thread" in t.name]


def test_statements_survive_the_calling_loop_closing(tmp_path):
    """The original failure: one statement per fresh loop, then the loop dies."""
    path = str(tmp_path / "loops.db")

    for i in range(3):
        async def work(index: int):
            mem = SQLiteMemory(path)
            await mem.initialize()
            await mem.save_agent({"id": f"a{index}", "name": "n", "role": "r"})
            await mem.close()

        asyncio.run(work(i))          # a brand-new loop each iteration
        gc.collect()
        time.sleep(0.05)

    # The owner loop is still the only loop in play.
    assert _DB_EXECUTOR.loop.is_closed() is False
    asyncio.run(_read_back(path))


async def _read_back(path: str) -> None:
    async def query():
        mem = SQLiteMemory(path)
        async with mem.connect() as db:
            async with db.execute("SELECT COUNT(*) FROM agents") as cursor:
                row = await cursor.fetchone()
        await mem.close()
        return row[0]

    count = await query()
    assert count == 3, f"every write must have landed on disk, found {count} rows"


def test_a_cancelled_statement_is_not_abandoned(tmp_path):
    """Cancellation must be deferred until the statement settles on the owner loop.

    Abandoning an in-flight statement is what leaves aiosqlite posting into a loop that may
    be gone. The proxy waits for the statement, then re-raises the cancellation.
    """
    path = str(tmp_path / "cancel.db")
    # Relative to a baseline: other test files in the same session also hold connections,
    # so an absolute "no worker threads exist" assertion would be order-dependent.
    before = set(_worker_threads())

    async def work():
        mem = SQLiteMemory(path)
        await mem.initialize()

        async def writer():
            for i in range(40):
                await mem.save_agent({"id": f"c{i}", "name": "n", "role": "r"})

        task = asyncio.create_task(writer())
        await asyncio.sleep(0.001)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return mem

    mem = asyncio.run(work())
    asyncio.run(mem.close())
    gc.collect()
    time.sleep(0.05)
    leaked = set(_worker_threads()) - before
    assert not leaked, f"a cancelled statement left a worker behind: {leaked}"


def test_a_dropped_memory_instance_closes_its_connections():
    """Dropping an instance without ``close()`` must not leave aiosqlite to finalise it.

    aiosqlite's ``__del__`` warns "was deleted before being closed" and queues a stop
    sentinel whose future is created in the loop that happens to be current at collection
    time — which can be a closed one, producing the worker-thread exception. The
    ``weakref.finalize`` hook closes on the owner loop instead.
    """
    async def work(i: int):
        mem = SQLiteMemory(":memory:")
        await mem.initialize()
        await mem.save_agent({"id": f"w{i}", "name": "n", "role": "r"})
        # deliberately not closed

    before = set(_worker_threads())
    for i in range(3):
        asyncio.run(work(i))
        gc.collect()
        time.sleep(0.05)
    time.sleep(0.2)

    after = set(_worker_threads())
    assert not (after - before), f"abandoned instances left worker threads: {after - before}"


def test_all_statements_run_on_the_owner_loop(tmp_path):
    """The connection handle must never bind a future to the caller's loop."""
    path = str(tmp_path / "owner.db")

    async def work():
        mem = SQLiteMemory(path)
        await mem.initialize()
        owner = _DB_EXECUTOR.loop
        assert owner is not asyncio.get_running_loop(), "the caller loop is the owner loop"
        await mem.save_agent({"id": "o1", "name": "n", "role": "r"})
        async with mem.connect() as db:
            async with db.execute("SELECT COUNT(*) FROM agents") as cursor:
                assert (await cursor.fetchone())[0] == 1
        await mem.close()

    asyncio.run(work())


def test_a_closed_loop_does_not_break_the_next_one(tmp_path):
    """Sequential ``asyncio.run`` calls (the test-suite pattern) must all succeed."""
    path = str(tmp_path / "sequential.db")

    async def work(i: int) -> int:
        mem = SQLiteMemory(path)
        await mem.initialize()
        await mem.save_agent({"id": f"s{i}", "name": "n", "role": "r"})
        async with mem.connect() as db:
            async with db.execute("SELECT COUNT(*) FROM agents") as cursor:
                row = await cursor.fetchone()
        await mem.close()
        return int(row[0])

    counts = [asyncio.run(work(i)) for i in range(5)]
    assert counts == [1, 2, 3, 4, 5], f"writes were lost across loops: {counts}"


def test_the_executor_thread_is_a_daemon():
    """A non-daemon owner loop would hang interpreter exit (the original #37 lesson)."""
    thread = next(
        (t for t in threading.enumerate() if t.name == "sqlite-db-loop"), None
    )
    if thread is None:
        _DB_EXECUTOR.loop  # start it
        thread = next(t for t in threading.enumerate() if t.name == "sqlite-db-loop")
    assert thread.daemon is True
