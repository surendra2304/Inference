"""SQLite persistent memory store implementation using aiosqlite."""

import asyncio
import atexit

# ---------------------------------------------------------------------------
# A persistent event loop that owns every aiosqlite connection
# ---------------------------------------------------------------------------
# Why this exists (measured, twice):
#
# ``aiosqlite`` delivers a statement's result from its worker thread with
# ``future.get_loop().call_soon_threadsafe(...)``. If the loop that issued the statement
# has closed by then, that call raises *inside the worker thread*; aiosqlite then tries to
# report the failure through the same call, which raises again, and the thread dies with an
# unhandled exception. pytest surfaces it as
# ``PytestUnhandledThreadExceptionWarning: Exception in thread Thread-N
# (_connection_worker_thread) ... RuntimeError: Event loop is closed``. Three to six such
# warnings survived every earlier attempt to bound the damage, because the race is inherent
# in binding a connection to whatever short-lived loop happens to call it: a test's
# ``asyncio.run`` (or a server shutdown, or a cancelled request) can close the loop while a
# statement is still in flight.
#
# Binding connections to one process-wide loop removes the race by construction: the loop
# that owns the connections outlives every caller. Callers reach it through
# ``run_coroutine_threadsafe`` + ``asyncio.wrap_future``, whose cross-loop delivery checks
# ``dest_loop.is_closed()`` before posting, so a caller that has gone away is handled by
# asyncio itself rather than by an unhandled exception in a worker thread.
#
# Cancellation is also made safe: if the awaiting task is cancelled, the statement is *not*
# abandoned (abandoning it is what leaves the worker posting into a dead loop). The
# cancellation is deferred until the statement settles on the owner loop, then re-raised.
import concurrent.futures as _cf
import json
import os
import threading as _threading
import time
import weakref
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import aiosqlite

from app.core.config import settings
from app.memory.base import (
    BaseMemory,
    ExperimentRecord,
    MemoryRecord,
    MessageRecord,
    RunRecord,
    StrategyRecord,
    TaskRecord,
)
from app.utils.logger import logger


class _DbExecutor:
    """A daemon thread running one event loop that owns all aiosqlite connections."""

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: _threading.Thread | None = None
        self._lock = _threading.Lock()
        self._started_at: float | None = None

    def _ensure(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is not None and not self._loop.is_closed():
                return self._loop
            ready = _threading.Event()

            def _run() -> None:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                self._loop = loop
                ready.set()
                loop.run_forever()
                # run_forever returned: shutdown was requested.
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.close()

            self._thread = _threading.Thread(
                target=_run, name="sqlite-db-loop", daemon=True
            )
            self._thread.start()
            self._started_at = time.time()
            ready.wait(timeout=10.0)
            assert self._loop is not None
            return self._loop

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        return self._ensure()

    def submit(self, coro: Any) -> _cf.Future:
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    async def run(self, coro: Any) -> Any:
        """Await ``coro`` on the owner loop, deferring cancellation, never abandoning it."""
        future = self.submit(coro)
        try:
            return await asyncio.wrap_future(future)
        except asyncio.CancelledError:
            # The statement is still running on the owner loop. Abandoning it here would
            # leave aiosqlite posting the result into a loop that may already be gone, so
            # wait for it to settle and then propagate the cancellation.
            if not future.done():
                await asyncio.shield(asyncio.wrap_future(future))
            raise

    def shutdown(self, timeout: float = 5.0) -> None:
        loop, thread = self._loop, self._thread
        if loop is None or loop.is_closed():
            return
        loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            thread.join(timeout=timeout)


_DB_EXECUTOR = _DbExecutor()


class _CursorProxy:
    """A cursor whose every operation runs on the owner loop.

    aiosqlite's cursor methods are coroutines that create their result future with
    ``asyncio.get_event_loop().create_future()`` *at call time*, so awaiting them on the
    caller's loop would bind the future to that loop — reintroducing exactly the
    "Event loop is closed" delivery failure this module now avoids. Each method is
    therefore submitted to the owner loop, where the future is created.
    """

    __slots__ = ("_cursor", "_executor")

    def __init__(self, cursor: Any, executor: "_DbExecutor") -> None:
        self._cursor = cursor
        self._executor = executor

    async def fetchall(self) -> list[Any]:
        return list(await self._executor.run(self._cursor.fetchall()))

    async def fetchone(self) -> Any:
        return await self._executor.run(self._cursor.fetchone())

    async def fetchmany(self, size: int | None = None) -> list[Any]:
        if size is None:
            return list(await self._executor.run(self._cursor.fetchmany()))
        return list(await self._executor.run(self._cursor.fetchmany(size)))

    async def close(self) -> None:
        await self._executor.run(self._cursor.close())

    @property
    def lastrowid(self) -> int | None:
        return self._cursor.lastrowid

    @property
    def rowcount(self) -> int:
        return self._cursor.rowcount


class _ResultProxy:
    """Mirrors aiosqlite's ``Result``: awaitable *and* an async context manager.

    Both forms appear in this module (``cursor = await db.execute(...)`` and
    ``async with db.execute(...) as cursor:``), so the proxy supports both.
    """

    __slots__ = ("_conn", "_args", "_kwargs", "_cursor")

    def __init__(self, conn: "_ConnectionProxy", args: tuple, kwargs: dict) -> None:
        self._conn = conn
        self._args = args
        self._kwargs = kwargs
        self._cursor: Any = None

    async def _ensure(self) -> _CursorProxy:
        if self._cursor is None:
            self._cursor = await self._conn._cursor(*self._args, **self._kwargs)
        return self._cursor

    def __await__(self):
        return self._ensure().__await__()

    async def __aenter__(self) -> _CursorProxy:
        return await self._ensure()

    async def __aexit__(self, *_exc: Any) -> None:
        if self._cursor is not None:
            await self._cursor.close()


class _ConnectionProxy:
    """A connection handle that forwards every statement to the owner loop."""

    __slots__ = ("_conn", "_executor")

    def __init__(self, conn: Any, executor: "_DbExecutor") -> None:
        self._conn = conn
        self._executor = executor

    async def _cursor(self, sql: str, parameters: Any = None) -> _CursorProxy:
        async def _run() -> Any:
            if parameters is None:
                return await self._conn.execute(sql)
            return await self._conn.execute(sql, parameters)

        return _CursorProxy(await self._executor.run(_run()), self._executor)

    def execute(self, sql: str, parameters: Any = None) -> _ResultProxy:
        return _ResultProxy(self, (sql, parameters), {})

    async def executemany(self, sql: str, seq_of_parameters: Any) -> _CursorProxy:
        async def _run() -> Any:
            return await self._conn.executemany(sql, seq_of_parameters)

        return _CursorProxy(await self._executor.run(_run()), self._executor)

    async def executescript(self, script: str) -> Any:
        return await self._executor.run(self._conn.executescript(script))

    async def commit(self) -> None:
        await self._executor.run(self._conn.commit())

    async def rollback(self) -> None:
        await self._executor.run(self._conn.rollback())

    async def close(self) -> None:
        await self._executor.run(self._conn.close())

    async def __aenter__(self) -> "_ConnectionProxy":
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        return None

    @property
    def row_factory(self) -> Any:
        return self._conn.row_factory


async def _close_connection_on_owner_loop(conn: aiosqlite.Connection) -> None:
    """Close a connection *on the owner loop*.

    ``conn.close()`` creates its future with ``asyncio.get_event_loop()`` at call time, so
    awaiting it from a caller's loop binds the close to that loop. Three close sites did
    exactly that, and a close still in flight when the caller's loop went away produced the
    same "Event loop is closed" worker exception the owner loop exists to prevent.
    """
    try:
        await asyncio.wait_for(conn.close(), timeout=5.0)
    except Exception:  # noqa: BLE001 - teardown must not raise
        pass


def _schedule_owned_close(owned: "set[aiosqlite.Connection]") -> None:
    """Close every connection an abandoned ``SQLiteMemory`` still owned.

    Called from a ``weakref.finalize`` when the instance is garbage collected. Without it,
    a memory object that is dropped without ``close()`` leaves aiosqlite to finalise the
    connection itself: its ``__del__`` warns "was deleted before being closed" and queues a
    stop sentinel whose future is created with ``asyncio.get_event_loop()`` *at
    collection time* — so if collection happens while the final loop is closing (or after
    it closed), the worker thread raises ``RuntimeError: Event loop is closed`` inside
    ``call_soon_threadsafe``, which is the exact unhandled-thread-exception symptom.
    Closing on the owner loop is deterministic and cannot target a dead loop.
    """
    for conn in list(owned):
        try:
            _DB_EXECUTOR.submit(_close_connection_on_owner_loop(conn))
        except Exception:  # noqa: BLE001 - object finalisation must never raise
            pass
        _LIVE_CONNECTIONS.discard(conn)
    owned.clear()


async def _new_memory_connection() -> aiosqlite.Connection:
    """Create the shared ``:memory:`` connection (must run on the owner loop)."""
    return await _prepare_connection(await aiosqlite.connect(":memory:"))


def _parse_timestamp(value: str | None) -> datetime:
    """Parse a stored timestamp, always returning an aware UTC datetime.

    Rows written before timestamps were timezone-aware carry no offset; they
    were produced by datetime.utcnow() so they are UTC. Normalising on read
    keeps legacy and new rows comparable instead of raising
    "can't compare offset-naive and offset-aware datetimes".
    """
    if not value:
        raise ValueError("timestamp value is missing")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed



# ---------------------------------------------------------------------------
# Process-exit safety for pooled connections
# ---------------------------------------------------------------------------
# ``aiosqlite`` runs every connection on a dedicated **non-daemon** worker thread
# that only stops when the connection is closed. Two consequences were measured
# the hard way:
#
#   * CPython joins non-daemon threads in ``wait_for_thread_shutdown()`` *before*
#     it runs ``atexit`` handlers. An exit hook therefore cannot rescue a
#     connection that is still open — the interpreter hangs with no message. This
#     was observed directly: a pooled connection held by a module-level singleton
#     (``app/core/orchestrator.py``) hung the whole test run *after* pytest had
#     already printed "202 passed in 7.34s".
#   * The worker thread's ``daemon`` flag cannot be changed after ``start()``, so
#     the connection has to be created already-daemonised.
#
# ``aiosqlite`` only stops the worker when the connection is closed (or when
# ``Connection.__del__`` fires), so nothing may be left dangling. The registry and
# the loop-lifetime finaliser below make every pooled connection stop its worker as
# soon as the event loop that owns it goes away, which is exactly when the
# connection becomes unusable anyway.
_LIVE_CONNECTIONS: "weakref.WeakSet[aiosqlite.Connection]" = weakref.WeakSet()


def _stop_connection(conn: aiosqlite.Connection) -> None:
    """Stop a connection's worker thread synchronously, ignoring every failure.

    ``Connection.stop()`` merely enqueues the sentinel the worker is blocked on, so
    it is safe to call from a finaliser or a closed loop. The underlying sqlite3
    handle is finalised by its own destructor.
    """
    try:
        if getattr(conn, "_running", False):
            conn.stop()
    except Exception:  # noqa: BLE001 - shutdown paths must never raise
        pass


#: The event loop each connection was created on, held weakly on both sides so neither
#: the connection nor the loop is kept alive by this registry.
_CONNECTION_LOOPS: "weakref.WeakKeyDictionary[Any, weakref.ref]" = weakref.WeakKeyDictionary()


def _record_connection_loop(conn: aiosqlite.Connection, loop: Any) -> None:
    """Remember which loop owns ``conn`` so it is never used from another one."""
    try:
        _CONNECTION_LOOPS[conn] = weakref.ref(loop)
    except TypeError:
        pass


def _connection_usable_here(conn: aiosqlite.Connection, loop: Any) -> bool:
    """True when ``conn`` may be awaited on ``loop`` right now.

    aiosqlite's worker thread delivers every result with
    ``future.get_loop().call_soon_threadsafe(...)``. If that loop has closed, the worker
    raises ``RuntimeError: Event loop is closed`` *inside the thread* (an unhandled thread
    exception) and the future it was resolving is never settled — so the awaiting coroutine
    hangs forever. Measured: acquiring a connection and dropping it without releasing left
    two worker threads alive after ``asyncio.run`` returned, still bound to the dead loop.

    The previous code discarded the whole pool when the *pool's* loop changed, but a
    connection that was checked out at that moment was not in the queue, so it survived:
    the discard only drained connections that had already been returned.
    """
    # A connection whose worker thread has been stopped can never be awaited again:
    # aiosqlite raises "no active connection" on the next use. This is not theoretical —
    # a stopped connection left in the pool surfaced as a regression in
    # test_sqlite_pool_replaces_a_connection_that_raised, where the next acquire received
    # it and every statement failed. Stopped-ness is checked *first* because it is a
    # stronger statement than any liveness check on the loop.
    if not getattr(conn, "_running", True):
        return False
    owner_ref = _CONNECTION_LOOPS.get(conn)
    if owner_ref is None:
        return True  # provenance unknown (e.g. the :memory: connection)
    owner = owner_ref()
    if owner is None:
        return True  # the creating loop is gone; the worker cannot be waiting on it
    return owner is loop and not owner.is_closed()


def _release_worker_threads() -> None:
    """Stop any surviving connection worker threads (best-effort, at exit)."""
    for conn in list(_LIVE_CONNECTIONS):
        _stop_connection(conn)


async def _prepare_connection(conn: aiosqlite.Connection) -> aiosqlite.Connection:
    """Apply the standard PRAGMAs (and row factory) every connection needs."""
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA journal_mode=WAL;")
    await conn.execute("PRAGMA busy_timeout=30000;")
    return conn


def _make_aiosqlite_workers_daemonic() -> bool:
    """Make ``aiosqlite``'s per-connection worker threads daemon threads.

    ``aiosqlite`` starts one **non-daemon** ``Thread`` per connection and only stops it
    when that connection is closed. CPython joins non-daemon threads *before* running
    ``atexit`` handlers, so any connection still open at exit blocks the interpreter
    indefinitely — no error, no output, just a hung process. That is not hypothetical:
    pooled connections held by long-lived singletons took a test run that had already
    printed "202 passed in 7.34s" and hung it for as long as the process was allowed to
    live, and the same failure mode applies to any deployment that exits without a
    graceful shutdown.

    Closing every connection on every path is not achievable — a crashed worker, a
    ``SIGKILL``-adjacent shutdown, or an event loop that is torn down by a test harness
    all bypass orderly cleanup — so the threads must not be able to block exit at all.
    The flag cannot be set after ``Thread.start()``, hence the patch at the one place
    ``aiosqlite`` constructs its threads.

    SQLite is crash-safe by design (that is what the write-ahead log is for), so an
    abruptly terminated worker costs at most the transaction in flight — the same
    guarantee the database already gives on power loss. Blocking process exit is the
    strictly worse outcome.

    Returns ``True`` when the patch is in place.
    """
    try:
        import aiosqlite.core as _core

        existing = getattr(_core, "Thread", None)
        if existing is None or getattr(existing, "_inference_daemonic", False):
            return bool(getattr(existing, "_inference_daemonic", False))

        class _DaemonThread(existing):  # type: ignore[misc, valid-type]
            """``threading.Thread`` that always starts daemonised."""

            _inference_daemonic = True

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                kwargs["daemon"] = True
                super().__init__(*args, **kwargs)

        _core.Thread = _DaemonThread  # type: ignore[attr-defined,misc]
        logger.debug("aiosqlite worker threads will be daemonic")
        return True
    except Exception as exc:  # noqa: BLE001 - a failed patch must not stop the app
        logger.debug("Could not daemonise aiosqlite worker threads: %s", exc)
        return False


_AIOSQLITE_THREADS_ARE_DAEMONIC = _make_aiosqlite_workers_daemonic()


atexit.register(_release_worker_threads)


#: Connections owned by each event loop, so a loop that disappears takes its
#: connections' worker threads with it instead of leaving them to block interpreter
#: exit. Weak keys: the loop itself is not kept alive by this registry.
_LOOP_CONNECTIONS: "weakref.WeakKeyDictionary[Any, set]" = weakref.WeakKeyDictionary()


def _register_loop_cleanup(loop: Any, owned: set) -> None:
    """Arrange for ``owned`` connections to stop their workers with ``loop``."""

    def _finalize() -> None:
        for conn in list(owned):
            _stop_connection(conn)
            _LIVE_CONNECTIONS.discard(conn)
        owned.clear()

    try:
        weakref.finalize(loop, _finalize)
    except TypeError:
        # Not weak-referenceable (exotic event loop implementation): the atexit net
        # remains as a fallback.
        pass


class SQLiteMemory(BaseMemory):
    """Asynchronous SQLite storage implementation for agents, tasks, runs, messages, memories, strategies, and experiments."""

    def __init__(self, db_path: str | None = None) -> None:
        raw_path = db_path or settings.DATABASE_URL
        if raw_path.startswith("sqlite+aiosqlite:///"):
            self.db_path = raw_path[len("sqlite+aiosqlite:///"):]
        elif raw_path.startswith("sqlite:///"):
            self.db_path = raw_path[len("sqlite:///"):]
        else:
            self.db_path = raw_path

        self._memory_conn: aiosqlite.Connection | None = None
        #: Connections this instance created, so an instance dropped without ``close()``
        #: still has its connections closed deterministically (see ``_schedule_owned_close``).
        self._owned_connections: set[aiosqlite.Connection] = set()
        self._owned_finalizer = weakref.finalize(
            self, _schedule_owned_close, self._owned_connections
        )

        # ---- file-backed connection pool ------------------------------------
        # Every database method used to open a brand-new SQLite connection, run one
        # statement, and close it. SQLite in WAL mode keeps ``db``, ``db-wal`` and
        # ``db-shm`` open per connection, and aiosqlite additionally holds a pipe plus
        # a worker thread, so a single call costs roughly 37 file descriptors while it
        # is in flight. Measured on this machine: a 12-client load test drove the
        # process to 930 of the 1024 available descriptors (91%), i.e. the service's
        # real concurrency ceiling was the OS descriptor limit, not the model or the
        # CPU — and the ceiling was reached by a client count that a single operator
        # running a batch job could produce.
        #
        # A small persistent pool bounds that: at most ``settings.SQLITE_POOL_SIZE``
        # connections exist, so descriptors stay flat regardless of traffic, while
        # concurrent callers still get parallelism instead of queueing behind one
        # connection. Connections are reused rather than reopened, which also removes a
        # per-call ``PRAGMA`` round trip and lets SQLite's page cache stay warm.
        self._pool: asyncio.Queue[aiosqlite.Connection] | None = None
        self._pool_created: int = 0
        self._pool_lock = asyncio.Lock()
        # Weak: holding the loop strongly would keep it alive forever, which also
        # keeps its connections' worker threads alive and hangs interpreter exit.
        self._pool_loop_ref: weakref.ref | None = None
        # Set of connections currently owned by the pool, handed to the loop finaliser
        # so they can be stopped even if nobody ever calls close().
        self._pool_owner: set = set()
        self._pool_max: int = max(1, int(getattr(settings, "SQLITE_POOL_SIZE", 8)))

        #: Schema self-heal state (see ``ensure_schema``). ``_schema_ready`` is per instance:
        #: two components with their own ``SQLiteMemory`` over the same file each check once.
        self._schema_ready: bool = False
        self._schema_in_progress: bool = False

    async def _new_connection(self) -> aiosqlite.Connection:
        """Create one fully-configured file-backed connection."""
        db_dir = os.path.dirname(os.path.abspath(self.db_path))
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        conn = await aiosqlite.connect(self.db_path, timeout=30.0)
        conn = await _prepare_connection(conn)
        _LIVE_CONNECTIONS.add(conn)
        self._owned_connections.add(conn)
        _record_connection_loop(conn, asyncio.get_running_loop())
        owner = getattr(self, "_pool_owner", None)
        if owner is not None:
            owner.add(conn)
        return conn

    async def _discard_pool(self) -> None:
        """Close and drop every pooled connection (best effort).

        Closing is skipped when the pool belongs to a different (typically already
        closed) event loop: awaiting a close against a dead loop can never complete,
        it hangs the caller. Those connections are instead left to the module-level
        ``atexit`` net, which stops their worker threads synchronously.
        """
        pool = self._pool
        pool_loop = self._pool_loop_ref() if self._pool_loop_ref else None
        self._pool = None
        self._pool_created = 0
        self._pool_loop_ref = None
        if pool is None:
            return
        usable = pool_loop is not None and not pool_loop.is_closed()
        while not pool.empty():
            try:
                conn = pool.get_nowait()
            except asyncio.QueueEmpty:
                break
            if not usable:
                # The owning loop is gone, so this connection can never be used or
                # closed again; stopping its worker is the only way to release the
                # thread before interpreter exit.
                _stop_connection(conn)
                _LIVE_CONNECTIONS.discard(conn)
                continue
            try:
                await _DB_EXECUTOR.run(_close_connection_on_owner_loop(conn))
            except Exception:  # noqa: BLE001 - teardown must not raise
                pass
            finally:
                _LIVE_CONNECTIONS.discard(conn)
                owner = getattr(self, "_pool_owner", None)
                if owner is not None:
                    owner.discard(conn)

    def _reclaim_connections_stranded_by_dead_loops(self, loop: asyncio.AbstractEventLoop) -> int:
        """Stop connections whose owning loop is gone, including ones still checked out.

        A connection that was checked out when its loop died is never handed back, so
        ``_release`` cannot stop it and ``_discard_pool`` cannot see it (it is not in the
        queue). Its worker thread then stays alive for the life of the process, blocked on
        a queue nobody will feed, and any late delivery into the closed loop raises
        ``RuntimeError`` inside that thread. Measured before this sweep: two such threads
        survived ``asyncio.run`` returning and an explicit ``gc.collect()``.

        Called on every acquisition, so the next database operation on any loop reclaims
        the workers orphaned by the previous one. Only this instance's own connections are
        considered, and only those whose creating loop is closed, so a live loop's
        connections are never touched.
        """
        # Swept globally, not just for this instance: a connection stranded by one
        # SQLiteMemory can be inherited by another that opens the same database, which is
        # exactly what happened in the reproduction (a second SQLiteMemory found the
        # orphaned worker still alive because the orphan belonged to the first instance's
        # owner set). A connection whose creating loop has closed can never be awaited by
        # anyone again, so reclaiming it is unconditionally correct.
        reclaimed = 0
        for conn in list(_LIVE_CONNECTIONS):
            owner_ref = _CONNECTION_LOOPS.get(conn)
            if owner_ref is None:
                continue  # provenance unknown (e.g. the shared :memory: connection)
            conn_loop = owner_ref()
            if conn_loop is None or conn_loop is loop or not conn_loop.is_closed():
                continue
            _stop_connection(conn)
            _LIVE_CONNECTIONS.discard(conn)
            owner = getattr(self, "_pool_owner", None)
            if owner is not None:
                owner.discard(conn)
            self._pool_created = max(0, self._pool_created - 1)
            reclaimed += 1
        if reclaimed:
            logger.warning(
                "Reclaimed %d SQLite connection(s) stranded by a closed event loop", reclaimed
            )
        return reclaimed

    async def _acquire(self, loop: asyncio.AbstractEventLoop) -> aiosqlite.Connection:
        """Borrow a pooled connection, creating one while below the cap."""
        self._reclaim_connections_stranded_by_dead_loops(loop)
        async with self._pool_lock:
            current = self._pool_loop_ref() if self._pool_loop_ref else None
            if self._pool is None or current is not loop:
                # A pool belongs to the event loop that created it: aiosqlite calls back
                # into the loop that was running at connect time, so reusing connections
                # across loops (as test suites do) raises "Event loop is closed".
                await self._discard_pool()
                self._pool = asyncio.Queue()
                self._pool_loop_ref = weakref.ref(loop)
                self._pool_owner = set()
                _LOOP_CONNECTIONS[loop] = self._pool_owner
                _register_loop_cleanup(loop, self._pool_owner)
            assert self._pool is not None
            pool = self._pool
            # Reclaim connections stranded by a loop that has since closed. They are not
            # usable from here and holding them keeps their worker threads alive.
            stale: list[aiosqlite.Connection] = []
            keep: list[aiosqlite.Connection] = []
            while not pool.empty():
                try:
                    candidate = pool.get_nowait()
                except asyncio.QueueEmpty:
                    break
                (keep if _connection_usable_here(candidate, loop) else stale).append(candidate)
            for candidate in keep:
                pool.put_nowait(candidate)
            for candidate in stale:
                _stop_connection(candidate)
                _LIVE_CONNECTIONS.discard(candidate)
                self._pool_created = max(0, self._pool_created - 1)

            if keep:
                return keep.pop()
            if self._pool_created < self._pool_max:
                self._pool_created += 1
                try:
                    return await self._new_connection()
                except Exception:
                    self._pool_created -= 1
                    raise

        # Cap reached: wait for a connection to come back.
        while True:
            conn = await pool.get()
            if _connection_usable_here(conn, loop):
                return conn
            # A connection returned to the pool by a loop that has since died. Stop its
            # worker rather than awaiting into a closed loop, and try again.
            _stop_connection(conn)
            _LIVE_CONNECTIONS.discard(conn)
            async with self._pool_lock:
                self._pool_created = max(0, self._pool_created - 1)
                if self._pool_created < self._pool_max:
                    self._pool_created += 1
                    try:
                        return await self._new_connection()
                    except Exception:
                        self._pool_created -= 1
                        raise

    async def _release(
        self, conn: aiosqlite.Connection, *, broken: bool = False
    ) -> None:
        """Return a connection to the pool, replacing it if the operation failed."""
        pool = self._pool
        if pool is None or broken:
            self._pool_created = max(0, self._pool_created - 1)
            # Already running on the owner loop (called through ``executor.run``), so this
            # one can close directly.
            try:
                await _close_connection_on_owner_loop(conn)
            finally:
                _LIVE_CONNECTIONS.discard(conn)
            return
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        if current_loop is None or not _connection_usable_here(conn, current_loop):
            # This connection belongs to a loop that is gone (or to a different one).
            # Returning it to the pool would hand a dead connection to a future caller;
            # stop its worker now instead. Closing is skipped deliberately: awaiting a
            # close against a dead loop cannot complete, which would hang the caller.
            _stop_connection(conn)
            _LIVE_CONNECTIONS.discard(conn)
            owner = getattr(self, "_pool_owner", None)
            if owner is not None:
                owner.discard(conn)
            return
        pool.put_nowait(conn)

    @asynccontextmanager
    async def connect(self) -> AsyncIterator[_ConnectionProxy]:
        """Hand out a connection, creating the schema first if this instance has not yet.

        Self-healing, because the previous contract was "whoever constructs this must remember
        to call ``initialize()``" and that contract was already broken in production code:
        ``TradingConsultService`` builds its own ``SQLiteMemory`` and never initialises it, so
        ``/v1/trading/consult`` returned HTTP 500 ``no such table: tasks`` on a clean
        deployment (measured: it only ever worked on a machine where some *other* component
        had already created the tables — a hidden dependency on leftover state, which is why
        the failure appeared the moment the database was deleted). The guard is a boolean, so
        the cost on the hot path is one attribute read.
        """
        await self.ensure_schema()
        async with self._connect_raw() as proxy:
            yield proxy

    async def ensure_schema(self, force: bool = False) -> None:
        """Create missing tables/indexes once per instance (idempotent, cheap, never fatal).

        A failure is not swallowed silently: it is logged and left un-cached, so the next call
        retries instead of the process pretending the schema exists.
        """
        if self._schema_ready and not force:
            return
        if self._schema_in_progress and not force:
            # Another coroutine is creating it right now; give it a moment rather than
            # issuing a second burst of DDL against the same file.
            for _ in range(200):
                if self._schema_ready:
                    return
                await asyncio.sleep(0.01)
            if self._schema_ready:
                return
        self._schema_in_progress = True
        try:
            async with self._connect_raw() as db:
                await self._create_schema(db)
            self._schema_ready = True
        except Exception as exc:  # noqa: BLE001 - callers get the real error from their query
            logger.warning("Schema initialization failed for %s: %s", self.db_path, exc)
        finally:
            self._schema_in_progress = False

    @asynccontextmanager
    async def _connect_raw(self) -> AsyncIterator[_ConnectionProxy]:
        """Provides a connection handle whose statements run on the owner loop.

        Every statement is submitted to ``_DB_EXECUTOR`` — a process-wide daemon loop that
        outlives any caller — rather than executed on the caller's loop. That is what
        removes the "Event loop is closed" worker-thread exception documented at the top of
        this module: an aiosqlite result can only ever be delivered to a loop that is still
        running. Callers see the same interface (``execute`` returning an awaitable that is
        also an async context manager, plus ``commit``/``rollback``/``executescript``).

        File-backed databases borrow from a bounded pool; ``:memory:`` databases keep their
        single shared connection, since a second connection to ``:memory:`` would be a
        different, empty database.
        """
        executor = _DB_EXECUTOR
        owner_loop = executor.loop

        if self.db_path == ":memory:":
            current = self._memory_conn
            if current is not None and not _connection_usable_here(current, owner_loop):
                # The shared in-memory connection belongs to a loop that has since closed
                # (test suites run each test on a fresh loop). Stop the stale worker and
                # rebuild; awaiting it would fail with "no active connection".
                _stop_connection(current)
                _LIVE_CONNECTIONS.discard(current)
                self._memory_conn = None
                current = None
            if current is None:
                current = await executor.run(_new_memory_connection())
                _LIVE_CONNECTIONS.add(current)
                self._owned_connections.add(current)
                _record_connection_loop(current, owner_loop)
                self._memory_conn = current
            yield _ConnectionProxy(current, executor)
            return

        conn = await executor.run(self._acquire(owner_loop))
        broken = False
        try:
            yield _ConnectionProxy(conn, executor)
        except BaseException:
            # A connection that raised may be mid-transaction; do not hand it to the next
            # caller. Discard it and let the pool build a clean replacement.
            broken = True
            raise
        finally:
            await executor.run(self._release(conn, broken=broken))

    async def close(self) -> None:
        """Close any persistent connections if held."""
        if self._memory_conn:
            conn, self._memory_conn = self._memory_conn, None
            try:
                await _DB_EXECUTOR.run(_close_connection_on_owner_loop(conn))
            except Exception:  # noqa: BLE001 - teardown must not raise
                pass
            finally:
                _LIVE_CONNECTIONS.discard(conn)
        await self._discard_pool()

    async def initialize(self) -> None:
        """Create tables and indexes if they do not already exist (always re-checks)."""
        await self.ensure_schema(force=True)

    async def _create_schema(self, db: _ConnectionProxy) -> None:
        """The DDL itself, shared by ``initialize()`` and the ``connect()`` self-heal."""
        await db.execute("""
            CREATE TABLE IF NOT EXISTS agents (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                role TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                config_json TEXT
            )
        """)

        await db.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                question TEXT NOT NULL,
                mode TEXT NOT NULL,
                status TEXT NOT NULL,
                result TEXT,
                confidence REAL,
                created_at TEXT NOT NULL,
                completed_at TEXT,
                metadata_json TEXT
            )
        """)

        await db.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                stage TEXT NOT NULL,
                latency REAL NOT NULL DEFAULT 0.0,
                status TEXT NOT NULL DEFAULT 'completed',
                error TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (task_id) REFERENCES tasks(id)
            )
        """)

        await db.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                role TEXT NOT NULL,
                agent_id TEXT,
                content TEXT NOT NULL,
                stage TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (task_id) REFERENCES tasks(id)
            )
        """)

        await db.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL DEFAULT 'fact',
                importance REAL NOT NULL DEFAULT 0.5,
                tags TEXT,
                created_at TEXT NOT NULL
            )
        """)

        await db.execute("""
            CREATE TABLE IF NOT EXISTS strategies (
                id TEXT PRIMARY KEY,
                task_type TEXT NOT NULL UNIQUE,
                strategy TEXT NOT NULL,
                score REAL NOT NULL DEFAULT 0.0,
                sample_size INTEGER NOT NULL DEFAULT 1,
                recommended_agents TEXT,
                recommended_provider TEXT NOT NULL DEFAULT 'gemini',
                recommended_model TEXT NOT NULL DEFAULT 'gemini-3.8-flash',
                created_at TEXT NOT NULL,
                metadata_json TEXT
            )
        """)

        await db.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                hypothesis TEXT NOT NULL,
                configuration TEXT,
                status TEXT NOT NULL DEFAULT 'completed',
                result_json TEXT,
                created_at TEXT NOT NULL
            )
        """)

        await db.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                criterion TEXT NOT NULL,
                score REAL NOT NULL,
                feedback TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs(id)
            )
        """)

        # Fast query indexes
        await db.execute("CREATE INDEX IF NOT EXISTS idx_memories_agent ON memories(agent_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_runs_task ON runs(task_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_messages_task ON messages(task_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_evaluations_run ON evaluations(run_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_strategies_type ON strategies(task_type)")

        await db.commit()
        logger.info("SQLite database initialized at: %s", self.db_path)


    async def reconcile_orphaned_tasks(self, process_started_at: datetime) -> list[str]:
        """Mark tasks left ``running`` by a previous process as interrupted.

        Nothing runs a task outside a live process, so any task still marked ``running``
        whose creation predates this process's start cannot be in flight — it belongs to
        a process that died (crash, ``SIGKILL``, OOM kill, container eviction). Left
        alone, those rows are permanent lies in the audit trail: the operator sees work
        "in progress" that ended hours ago, dashboards show phantom load, and any
        "how many tasks are running?" query is wrong forever.

        Measured before this existed: two tasks sat in ``running`` from 14:00 while the
        newest completed task was 14:51 — every restart added more.

        The cutoff is this process's start time, which makes the operation safe for
        multi-worker deployments: a sibling worker started at the same moment cannot own
        a task created before the cutoff, so a genuine in-flight task is never touched.
        Returns the ids that were reconciled.
        """
        cutoff = process_started_at
        if cutoff.tzinfo is None:
            cutoff = cutoff.replace(tzinfo=timezone.utc)
        completed_at = datetime.now(timezone.utc).isoformat()
        reconciled: list[str] = []
        async with self.connect() as db:
            cursor = await db.execute(
                "SELECT id, metadata_json FROM tasks WHERE status = 'running' AND created_at < ?",
                (cutoff.isoformat(),),
            )
            rows = await cursor.fetchall()
            for row in rows:
                task_id = row["id"]
                try:
                    metadata = json.loads(row["metadata_json"] or "{}")
                    if not isinstance(metadata, dict):
                        metadata = {}
                except (TypeError, ValueError):
                    metadata = {}
                metadata["interrupted"] = (
                    "process ended while this task was in flight; no result was produced"
                )
                await db.execute(
                    "UPDATE tasks SET status = 'failed', completed_at = ?, metadata_json = ? "
                    "WHERE id = ? AND status = 'running'",
                    (completed_at, json.dumps(metadata), task_id),
                )
                reconciled.append(task_id)
            if reconciled:
                await db.commit()
        if reconciled:
            logger.warning(
                "Reconciled %d task(s) left running by a previous process: %s",
                len(reconciled), ", ".join(reconciled[:5]) + (" ..." if len(reconciled) > 5 else ""),
            )
        return reconciled

    async def save_agent(self, agent_data: dict[str, Any]) -> None:
        """Persist or update an agent configuration record."""
        agent_id = agent_data.get("id")
        name = agent_data.get("name", "")
        role = agent_data.get("role", "")
        provider = agent_data.get("model_provider", "gemini")
        model = agent_data.get("model_name", "gemini-3.8-flash")
        status = agent_data.get("status", "active")
        config_json = json.dumps(agent_data)

        async with self.connect() as db:
            await db.execute("""
                INSERT INTO agents (id, name, role, provider, model, status, config_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name=excluded.name,
                    role=excluded.role,
                    provider=excluded.provider,
                    model=excluded.model,
                    status=excluded.status,
                    config_json=excluded.config_json
            """, (agent_id, name, role, provider, model, status, config_json))
            await db.commit()

    async def save_task(self, task: TaskRecord) -> None:
        """Create or update a task record."""
        created_str = task.created_at.isoformat()
        completed_str = task.completed_at.isoformat() if task.completed_at else None
        meta_json = json.dumps(task.metadata)

        async with self.connect() as db:
            await db.execute("""
                INSERT INTO tasks (id, question, mode, status, result, confidence, created_at, completed_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    status=excluded.status,
                    result=excluded.result,
                    confidence=excluded.confidence,
                    completed_at=excluded.completed_at,
                    metadata_json=excluded.metadata_json
            """, (
                task.id,
                task.question,
                task.mode,
                task.status,
                task.result,
                task.confidence,
                created_str,
                completed_str,
                meta_json
            ))
            await db.commit()

    async def get_task(self, task_id: str) -> TaskRecord | None:
        """Retrieve a task record by its ID."""
        async with self.connect() as db:
            async with db.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)) as cursor:
                row = await cursor.fetchone()
                if not row:
                    return None

                created_at = _parse_timestamp(row["created_at"])
                completed_at = _parse_timestamp(row["completed_at"]) if row["completed_at"] else None
                meta = json.loads(row["metadata_json"]) if row["metadata_json"] else {}

                return TaskRecord(
                    id=row["id"],
                    question=row["question"],
                    mode=row["mode"],
                    status=row["status"],
                    result=row["result"],
                    confidence=row["confidence"],
                    created_at=created_at,
                    completed_at=completed_at,
                    metadata=meta
                )

    async def save_run(self, run: RunRecord) -> None:
        """Persist an execution run audit record."""
        created_str = run.created_at.isoformat()
        async with self.connect() as db:
            await db.execute("""
                INSERT INTO runs (id, task_id, agent_id, provider, model, stage, latency, status, error, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    latency=excluded.latency,
                    status=excluded.status,
                    error=excluded.error
            """, (
                run.id,
                run.task_id,
                run.agent_id,
                run.provider,
                run.model,
                run.stage,
                run.latency_seconds,
                run.status,
                run.error,
                created_str
            ))
            await db.commit()

    async def save_message(self, message: MessageRecord) -> None:
        """Persist a conversation or debate message."""
        created_str = message.created_at.isoformat()
        async with self.connect() as db:
            await db.execute("""
                INSERT INTO messages (id, run_id, task_id, role, agent_id, content, stage, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    content=excluded.content,
                    stage=excluded.stage
            """, (
                message.id,
                message.run_id,
                message.task_id,
                message.role,
                message.agent_id,
                message.content,
                message.stage,
                created_str
            ))
            await db.commit()

    async def get_task_messages(self, task_id: str) -> list[MessageRecord]:
        """Retrieve all messages associated with a task ID."""
        records: list[MessageRecord] = []
        async with self.connect() as db, db.execute(
            "SELECT * FROM messages WHERE task_id = ? ORDER BY created_at ASC",
            (task_id,)
        ) as cursor:
            rows = await cursor.fetchall()
            for row in rows:
                records.append(MessageRecord(
                    id=row["id"],
                    run_id=row["run_id"],
                    task_id=row["task_id"],
                    role=row["role"],
                    agent_id=row["agent_id"],
                    content=row["content"],
                    stage=row["stage"],
                    created_at=_parse_timestamp(row["created_at"])
                ))
        return records

    async def save_memory(self, memory: MemoryRecord) -> None:
        """Save a scoped persistent memory item."""
        tags_str = ",".join(memory.context_tags) if memory.context_tags else ""
        created_str = memory.created_at.isoformat()

        async with self.connect() as db:
            await db.execute("""
                INSERT INTO memories (id, agent_id, content, memory_type, importance, tags, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    content=excluded.content,
                    memory_type=excluded.memory_type,
                    importance=excluded.importance,
                    tags=excluded.tags
            """, (
                memory.id,
                memory.agent_id,
                memory.content,
                memory.memory_type,
                memory.importance,
                tags_str,
                created_str
            ))
            await db.commit()

    async def get_agent_memories(
        self,
        agent_id: str,
        limit: int = 10,
        memory_type: str | None = None
    ) -> list[MemoryRecord]:
        """Retrieve memories strictly scoped to a specific agent_id."""
        query = "SELECT * FROM memories WHERE agent_id = ?"
        params: list[Any] = [agent_id]

        if memory_type:
            query += " AND memory_type = ?"
            params.append(memory_type)

        query += " ORDER BY importance DESC, created_at DESC LIMIT ?"
        params.append(limit)

        records: list[MemoryRecord] = []
        async with self.connect() as db:
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                for row in rows:
                    tags = [t.strip() for t in row["tags"].split(",") if t.strip()] if row["tags"] else []
                    records.append(MemoryRecord(
                        id=row["id"],
                        agent_id=row["agent_id"],
                        content=row["content"],
                        memory_type=row["memory_type"],
                        importance=row["importance"],
                        context_tags=tags,
                        created_at=_parse_timestamp(row["created_at"])
                    ))
        return records

    async def search_memories(
        self,
        query: str,
        agent_id: str | None = None,
        limit: int = 5
    ) -> list[MemoryRecord]:
        """Search memory records matching text, optionally filtered by agent_id."""
        sql = "SELECT * FROM memories WHERE content LIKE ?"
        params: list[Any] = [f"%{query}%"]

        if agent_id:
            sql += " AND agent_id = ?"
            params.append(agent_id)

        sql += " ORDER BY importance DESC LIMIT ?"
        params.append(limit)

        records: list[MemoryRecord] = []
        async with self.connect() as db:
            async with db.execute(sql, params) as cursor:
                rows = await cursor.fetchall()
                for row in rows:
                    tags = [t.strip() for t in row["tags"].split(",") if t.strip()] if row["tags"] else []
                    records.append(MemoryRecord(
                        id=row["id"],
                        agent_id=row["agent_id"],
                        content=row["content"],
                        memory_type=row["memory_type"],
                        importance=row["importance"],
                        context_tags=tags,
                        created_at=_parse_timestamp(row["created_at"])
                    ))
        return records

    async def save_strategy(self, strategy: StrategyRecord) -> None:
        """Save or update a learned strategy record."""
        agents_str = ",".join(strategy.recommended_agents)
        meta_json = json.dumps(strategy.metadata)
        created_str = strategy.created_at.isoformat()

        async with self.connect() as db:
            await db.execute("""
                INSERT INTO strategies (
                    id, task_type, strategy, score, sample_size,
                    recommended_agents, recommended_provider, recommended_model,
                    created_at, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_type) DO UPDATE SET
                    strategy=excluded.strategy,
                    score=excluded.score,
                    sample_size=excluded.sample_size,
                    recommended_agents=excluded.recommended_agents,
                    recommended_provider=excluded.recommended_provider,
                    recommended_model=excluded.recommended_model,
                    metadata_json=excluded.metadata_json
            """, (
                strategy.id,
                strategy.task_type,
                strategy.strategy,
                strategy.score,
                strategy.sample_size,
                agents_str,
                strategy.recommended_provider,
                strategy.recommended_model,
                created_str,
                meta_json
            ))
            await db.commit()

    async def get_strategy(self, task_type: str) -> StrategyRecord | None:
        """Retrieve the best learned strategy for a specific task type."""
        async with self.connect() as db:
            async with db.execute(
                "SELECT * FROM strategies WHERE task_type = ?", (task_type,)
            ) as cursor:
                row = await cursor.fetchone()
                if not row:
                    return None

                agents = [a.strip() for a in row["recommended_agents"].split(",") if a.strip()] if row["recommended_agents"] else []
                meta = json.loads(row["metadata_json"]) if row["metadata_json"] else {}

                return StrategyRecord(
                    id=row["id"],
                    task_type=row["task_type"],
                    strategy=row["strategy"],
                    score=row["score"],
                    sample_size=row["sample_size"],
                    recommended_agents=agents,
                    recommended_provider=row["recommended_provider"],
                    recommended_model=row["recommended_model"],
                    created_at=_parse_timestamp(row["created_at"]),
                    metadata=meta
                )

    async def list_strategies(self) -> list[StrategyRecord]:
        """List all learned strategies."""
        records: list[StrategyRecord] = []
        async with self.connect() as db:
            async with db.execute("SELECT * FROM strategies ORDER BY score DESC") as cursor:
                rows = await cursor.fetchall()
                for row in rows:
                    agents = [a.strip() for a in row["recommended_agents"].split(",") if a.strip()] if row["recommended_agents"] else []
                    meta = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
                    records.append(StrategyRecord(
                        id=row["id"],
                        task_type=row["task_type"],
                        strategy=row["strategy"],
                        score=row["score"],
                        sample_size=row["sample_size"],
                        recommended_agents=agents,
                        recommended_provider=row["recommended_provider"],
                        recommended_model=row["recommended_model"],
                        created_at=_parse_timestamp(row["created_at"]),
                        metadata=meta
                    ))
        return records

    async def save_experiment(self, experiment: ExperimentRecord) -> None:
        """Save an experiment record."""
        config_json = json.dumps(experiment.configuration)
        res_json = json.dumps(experiment.result) if experiment.result else None
        created_str = experiment.created_at.isoformat()

        async with self.connect() as db:
            await db.execute("""
                INSERT INTO experiments (id, hypothesis, configuration, status, result_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    status=excluded.status,
                    result_json=excluded.result_json
            """, (
                experiment.id,
                experiment.hypothesis,
                config_json,
                experiment.status,
                res_json,
                created_str
            ))
            await db.commit()

    async def get_experiment(self, experiment_id: str) -> ExperimentRecord | None:
        """Retrieve experiment details by ID."""
        async with self.connect() as db:
            async with db.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,)) as cursor:
                row = await cursor.fetchone()
                if not row:
                    return None

                config = json.loads(row["configuration"]) if row["configuration"] else {}
                result = json.loads(row["result_json"]) if row["result_json"] else None

                return ExperimentRecord(
                    id=row["id"],
                    hypothesis=row["hypothesis"],
                    configuration=config,
                    status=row["status"],
                    result=result,
                    created_at=_parse_timestamp(row["created_at"])
                )
