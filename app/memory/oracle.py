"""Oracle Autonomous Database 26ai (23.26) implementation of :class:`BaseMemory`.

This module is the durable replacement for :mod:`app.memory.sqlite`. It preserves the
``SQLiteMemory`` public contract — the same methods, arguments, return types and data
classes — so callers keep working unchanged. Schema owner is the connecting Oracle user
(``INFERENCE`` in the Universe deployment), which is why every table name is unqualified.

Design decisions worth knowing before editing:

* **Synchronous SQLAlchemy Core, asynchronous façade.** ``oracledb`` and SQLAlchemy's sync
  engine are used together; every call is dispatched with ``asyncio.to_thread`` so the event
  loop is never blocked. No new dependencies are introduced (``sqlalchemy`` and ``oracledb``
  are already part of the runtime), and no ``asyncio`` primitives are held as instance state:
  a process-wide orchestrator is shared across event loops (tests run one loop per test), so
  loop-bound locks would be a latent ``RuntimeError`` exactly as they were for the SQLite pool.
* **Named binds only.** Oracle does not accept ``?``; every statement uses ``:name``.
* **Row limiting with** ``FETCH FIRST :n ROWS ONLY`` (Oracle has no ``LIMIT``).
* **No ``MERGE``-incompatible upserts.** ``INSERT OR REPLACE``/``ON CONFLICT`` do not exist;
  ``MERGE`` is used with a one-row ``SELECT ... FROM dual`` source. Conflict targets mirror
  the SQLite ``ON CONFLICT`` columns exactly, including which columns are *not* updated.
* **Reserved-word-safe columns.** Oracle reserves ``MODE``, ``LEVEL``, ``SIZE``, ``COMMENT``,
  ``SESSION``, ``ACCESS``, ``NUMBER``, ``DATE``, ``START`` and ``END`` as identifiers, so the
  SQLite ``tasks.mode`` column is stored here as ``tasks.task_mode``. The typed API is
  unaffected; :attr:`OracleMemory.backend_name` lets the one raw-SQL reader pick the right
  column without duplicating per-backend queries.
* **Timestamps** are ``TIMESTAMP(6)`` holding **UTC**, written as naive-UTC values and read
  back as aware UTC datetimes (which is what the SQLite store returns). Using a timezone-free
  column keeps comparisons independent of the database session timezone; the zone is UTC by
  convention and enforced on both the write and read paths here.
* **JSON and long text are CLOBs**, written with ``json.dumps`` and read back as text. CLOB
  binds are typed explicitly (``CLOB``) so payloads larger than 4000 bytes are inserted
  correctly, and ``oracledb.defaults.fetch_lobs`` is disabled so reads return ``str`` rather
  than LOB objects that would need a live cursor.
* **Empty string is NULL in Oracle.** Oracle converts ``''`` to ``NULL`` on write, so no text
  column is declared ``NOT NULL`` and the read path normalises ``NULL`` back to ``""`` for
  fields the contract declares as ``str`` (and keeps ``None`` for optional fields).
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import bindparam, create_engine, text
from sqlalchemy.engine import URL, Engine, make_url
from sqlalchemy.types import CLOB

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

#: SQLAlchemy driver name. ``oracle+oracledb`` selects the python-oracledb dialect in sync mode.
_SYNC_DRIVER = "oracle+oracledb"

#: Seconds a pooled Oracle connection may live before SQLAlchemy recycles it. Autonomous
#: Database closes idle sessions, and ``pool_pre_ping`` alone still pays a failed round trip.
_POOL_RECYCLE_SECONDS = 1800


class OracleMemoryError(RuntimeError):
    """Raised when the Oracle backend cannot be configured (missing driver / credentials)."""


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
# Oracle has no ``CREATE TABLE IF NOT EXISTS``; existence is checked against the data
# dictionary (``user_tables`` / ``user_indexes``) before each statement, which also makes
# ``initialize()`` idempotent across restarts.
_TABLE_DDL: tuple[tuple[str, str], ...] = (
    (
        "agents",
        """
        CREATE TABLE agents (
            id VARCHAR2(512) NOT NULL,
            name VARCHAR2(512),
            role VARCHAR2(1024),
            provider VARCHAR2(200),
            model VARCHAR2(400),
            status VARCHAR2(50) DEFAULT 'active',
            config_json CLOB,
            CONSTRAINT pk_agents PRIMARY KEY (id)
        )
        """,
    ),
    (
        "tasks",
        """
        CREATE TABLE tasks (
            id VARCHAR2(512) NOT NULL,
            question CLOB,
            task_mode VARCHAR2(50),
            status VARCHAR2(50),
            result CLOB,
            confidence BINARY_DOUBLE,
            created_at TIMESTAMP(6),
            completed_at TIMESTAMP(6),
            metadata_json CLOB,
            CONSTRAINT pk_tasks PRIMARY KEY (id)
        )
        """,
    ),
    (
        "runs",
        """
        CREATE TABLE runs (
            id VARCHAR2(512) NOT NULL,
            task_id VARCHAR2(512),
            agent_id VARCHAR2(512),
            provider VARCHAR2(200),
            model VARCHAR2(400),
            stage VARCHAR2(200),
            latency BINARY_DOUBLE DEFAULT 0.0,
            prompt_tokens NUMBER(19),
            completion_tokens NUMBER(19),
            status VARCHAR2(50) DEFAULT 'completed',
            error CLOB,
            created_at TIMESTAMP(6),
            CONSTRAINT pk_runs PRIMARY KEY (id)
        )
        """,
    ),
    (
        "messages",
        """
        CREATE TABLE messages (
            id VARCHAR2(512) NOT NULL,
            run_id VARCHAR2(512),
            task_id VARCHAR2(512),
            role VARCHAR2(50),
            agent_id VARCHAR2(512),
            content CLOB,
            stage VARCHAR2(200),
            created_at TIMESTAMP(6),
            CONSTRAINT pk_messages PRIMARY KEY (id)
        )
        """,
    ),
    (
        "memories",
        """
        CREATE TABLE memories (
            id VARCHAR2(512) NOT NULL,
            agent_id VARCHAR2(512),
            content CLOB,
            memory_type VARCHAR2(100),
            importance BINARY_DOUBLE DEFAULT 0.5,
            tags CLOB,
            created_at TIMESTAMP(6),
            CONSTRAINT pk_memories PRIMARY KEY (id)
        )
        """,
    ),
    (
        "strategies",
        """
        CREATE TABLE strategies (
            id VARCHAR2(512) NOT NULL,
            task_type VARCHAR2(400),
            strategy CLOB,
            score BINARY_DOUBLE DEFAULT 0.0,
            sample_size NUMBER(19) DEFAULT 1,
            recommended_agents CLOB,
            recommended_provider VARCHAR2(200) DEFAULT 'gemini',
            recommended_model VARCHAR2(400) DEFAULT 'gemini-3.8-flash',
            created_at TIMESTAMP(6),
            metadata_json CLOB,
            CONSTRAINT pk_strategies PRIMARY KEY (id),
            CONSTRAINT uq_strategies_task_type UNIQUE (task_type)
        )
        """,
    ),
    (
        "experiments",
        """
        CREATE TABLE experiments (
            id VARCHAR2(512) NOT NULL,
            hypothesis CLOB,
            configuration CLOB,
            status VARCHAR2(50) DEFAULT 'completed',
            result_json CLOB,
            created_at TIMESTAMP(6),
            CONSTRAINT pk_experiments PRIMARY KEY (id)
        )
        """,
    ),
    (
        "evaluations",
        """
        CREATE TABLE evaluations (
            id VARCHAR2(512) NOT NULL,
            run_id VARCHAR2(512),
            criterion VARCHAR2(400),
            score BINARY_DOUBLE,
            feedback CLOB,
            created_at TIMESTAMP(6),
            CONSTRAINT pk_evaluations PRIMARY KEY (id)
        )
        """,
    ),
)

#: ``strategies(task_type)`` has no dedicated index: the ``uq_strategies_task_type`` unique
#: constraint already created one, and a duplicate would be maintained on every upsert.
_INDEX_DDL: tuple[tuple[str, str], ...] = (
    ("idx_memories_agent", "CREATE INDEX idx_memories_agent ON memories (agent_id)"),
    ("idx_runs_task", "CREATE INDEX idx_runs_task ON runs (task_id)"),
    ("idx_messages_task", "CREATE INDEX idx_messages_task ON messages (task_id)"),
    ("idx_evaluations_run", "CREATE INDEX idx_evaluations_run ON evaluations (run_id)"),
    ("idx_tasks_status", "CREATE INDEX idx_tasks_status ON tasks (status)"),
)


# ---------------------------------------------------------------------------
# Statements
# ---------------------------------------------------------------------------
_MERGE_AGENT = """
MERGE INTO agents t
USING (SELECT :id AS id FROM dual) s
ON (t.id = s.id)
WHEN MATCHED THEN UPDATE SET
    t.name = :name,
    t.role = :role,
    t.provider = :provider,
    t.model = :model,
    t.status = :status,
    t.config_json = :config_json
WHEN NOT MATCHED THEN INSERT (id, name, role, provider, model, status, config_json)
    VALUES (:id, :name, :role, :provider, :model, :status, :config_json)
"""

_MERGE_TASK = """
MERGE INTO tasks t
USING (SELECT :id AS id FROM dual) s
ON (t.id = s.id)
WHEN MATCHED THEN UPDATE SET
    t.status = :status,
    t.result = :result,
    t.confidence = :confidence,
    t.completed_at = :completed_at,
    t.metadata_json = :metadata_json
WHEN NOT MATCHED THEN
    INSERT (id, question, task_mode, status, result, confidence, created_at, completed_at, metadata_json)
    VALUES (:id, :question, :task_mode, :status, :result, :confidence, :created_at, :completed_at, :metadata_json)
"""

_MERGE_RUN = """
MERGE INTO runs t
USING (SELECT :id AS id FROM dual) s
ON (t.id = s.id)
WHEN MATCHED THEN UPDATE SET
    t.latency = :latency,
    t.prompt_tokens = :prompt_tokens,
    t.completion_tokens = :completion_tokens,
    t.status = :status,
    t.error = :error
WHEN NOT MATCHED THEN
    INSERT (id, task_id, agent_id, provider, model, stage, latency,
            prompt_tokens, completion_tokens, status, error, created_at)
    VALUES (:id, :task_id, :agent_id, :provider, :model, :stage, :latency,
            :prompt_tokens, :completion_tokens, :status, :error, :created_at)
"""

_MERGE_MESSAGE = """
MERGE INTO messages t
USING (SELECT :id AS id FROM dual) s
ON (t.id = s.id)
WHEN MATCHED THEN UPDATE SET
    t.content = :content,
    t.stage = :stage
WHEN NOT MATCHED THEN
    INSERT (id, run_id, task_id, role, agent_id, content, stage, created_at)
    VALUES (:id, :run_id, :task_id, :role, :agent_id, :content, :stage, :created_at)
"""

_MERGE_MEMORY = """
MERGE INTO memories t
USING (SELECT :id AS id FROM dual) s
ON (t.id = s.id)
WHEN MATCHED THEN UPDATE SET
    t.content = :content,
    t.memory_type = :memory_type,
    t.importance = :importance,
    t.tags = :tags
WHEN NOT MATCHED THEN
    INSERT (id, agent_id, content, memory_type, importance, tags, created_at)
    VALUES (:id, :agent_id, :content, :memory_type, :importance, :tags, :created_at)
"""

_MERGE_STRATEGY = """
MERGE INTO strategies t
USING (SELECT :task_type AS task_type FROM dual) s
ON (t.task_type = s.task_type)
WHEN MATCHED THEN UPDATE SET
    t.strategy = :strategy,
    t.score = :score,
    t.sample_size = :sample_size,
    t.recommended_agents = :recommended_agents,
    t.recommended_provider = :recommended_provider,
    t.recommended_model = :recommended_model,
    t.metadata_json = :metadata_json
WHEN NOT MATCHED THEN
    INSERT (id, task_type, strategy, score, sample_size, recommended_agents,
            recommended_provider, recommended_model, created_at, metadata_json)
    VALUES (:id, :task_type, :strategy, :score, :sample_size, :recommended_agents,
            :recommended_provider, :recommended_model, :created_at, :metadata_json)
"""

_MERGE_EXPERIMENT = """
MERGE INTO experiments t
USING (SELECT :id AS id FROM dual) s
ON (t.id = s.id)
WHEN MATCHED THEN UPDATE SET
    t.status = :status,
    t.result_json = :result_json
WHEN NOT MATCHED THEN
    INSERT (id, hypothesis, configuration, status, result_json, created_at)
    VALUES (:id, :hypothesis, :configuration, :status, :result_json, :created_at)
"""

_SELECT_TASK = "SELECT * FROM tasks WHERE id = :id"
_SELECT_TASK_MESSAGES = "SELECT * FROM messages WHERE task_id = :task_id ORDER BY created_at ASC"
_SELECT_MEMORIES = "SELECT * FROM memories WHERE agent_id = :agent_id"
_SELECT_STRATEGY = "SELECT * FROM strategies WHERE task_type = :task_type"
_SELECT_STRATEGIES = "SELECT * FROM strategies ORDER BY score DESC"
_SELECT_EXPERIMENT = "SELECT * FROM experiments WHERE id = :id"
_SELECT_RUNNING_TASKS = (
    "SELECT id, metadata_json FROM tasks WHERE status = 'running' AND created_at < :cutoff"
)
_UPDATE_RECONCILED_TASK = (
    "UPDATE tasks SET status = 'failed', completed_at = :completed_at, metadata_json = :metadata_json "
    "WHERE id = :id AND status = 'running'"
)


# ---------------------------------------------------------------------------
# Value normalisation (Oracle ↔ contract)
# ---------------------------------------------------------------------------
def _statement(sql: str, clobs: Sequence[str] = ()):
    """Build a ``text()`` clause, typing CLOB binds so >4000-byte payloads fit."""
    clause = text(sql)
    if clobs:
        clause = clause.bindparams(*(bindparam(name, type_=CLOB) for name in clobs))
    return clause


def _text(value: Any) -> str:
    """Return a required text field: ``NULL``/``''`` both become ``""``."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    reader = getattr(value, "read", None)
    if callable(reader):
        # ``fetch_lobs`` is disabled on the engine, but a caller-supplied engine may not have
        # been configured by us; reading here keeps the contract instead of leaking a LOB.
        try:
            data = reader()
        except Exception:  # noqa: BLE001 - a LOB that cannot be read degrades to empty text
            return ""
        return data if isinstance(data, str) else str(data)
    return str(value)


def _optional_text(value: Any) -> str | None:
    """Return an optional text field: ``NULL``/``''`` both become ``None``."""
    rendered = _text(value)
    return rendered if rendered else None


def _naive_utc(value: datetime | None) -> datetime | None:
    """Convert to the naive-UTC form stored in ``TIMESTAMP(6)`` columns."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def _aware_utc(value: Any) -> datetime | None:
    """Convert a stored timestamp to aware UTC; tolerate ISO text from a hand-migrated row."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    raise TypeError(f"unsupported timestamp value: {type(value).__name__}")


def _required_timestamp(value: Any) -> datetime:
    parsed = _aware_utc(value)
    if parsed is None:
        raise ValueError("timestamp value is missing")
    return parsed


def _json_object(value: Any) -> dict[str, Any]:
    """Decode a JSON object column; absent or non-object payloads yield ``{}``."""
    rendered = _text(value)
    if not rendered:
        return {}
    parsed = json.loads(rendered)
    return parsed if isinstance(parsed, dict) else {}


def _json_optional(value: Any) -> dict[str, Any] | None:
    """Decode an optional JSON object column; absent payloads yield ``None``."""
    rendered = _text(value)
    if not rendered:
        return None
    parsed = json.loads(rendered)
    return parsed if isinstance(parsed, dict) else None


def _csv_list(value: Any) -> list[str]:
    return [item.strip() for item in _text(value).split(",") if item.strip()]


def _row_to_dict(row: Mapping[str, Any]) -> dict[str, Any]:
    """Lower-case Oracle's UPPERCASE result keys, as the contract expects them."""
    return {str(key).lower(): value for key, value in row.items()}


# ---------------------------------------------------------------------------
# Connection target resolution
# ---------------------------------------------------------------------------
def _normalise_oracle_url(raw: str) -> URL:
    """Parse an Oracle URL and force the sync ``oracle+oracledb`` driver.

    ``str(URL)`` hides the password (SQLAlchemy's safe default), so the URL object is returned
    as-is instead of being rendered back to a string and re-parsed.
    """
    url = make_url(raw)
    driver = url.drivername.lower()
    if driver not in {"oracle", "oracle+oracledb", "oracle+oracledb_async", "oracle+cx_oracle"}:
        raise OracleMemoryError(
            f"Unsupported Oracle driver '{url.drivername}'. Use oracle+oracledb "
            "(or a bare DSN / Easy Connect string)."
        )
    return url.set(drivername=_SYNC_DRIVER)


def _settings_easy_connect_dsn() -> str:
    """Easy Connect string for the ``ORACLE_*`` settings.

    Autonomous Database only accepts TCPS (mutual-TLS) connections, so ``tcps`` is the
    default protocol. ``ORACLE_PROTOCOL=tcp`` supports an on-premise listener.
    """
    protocol = (getattr(settings, "ORACLE_PROTOCOL", None) or "tcps").strip().lower()
    if protocol not in {"tcps", "tcp"}:
        raise OracleMemoryError(
            f"ORACLE_PROTOCOL must be 'tcps' or 'tcp', got {protocol!r}"
        )
    host = (settings.ORACLE_HOST or "").strip()
    service = (settings.ORACLE_SERVICE or "").strip()
    if not host or not service:
        raise OracleMemoryError(
            "Oracle connection is not configured: ORACLE_HOST and ORACLE_SERVICE are required "
            "(or set DATABASE_URL / ORACLE_CORE_URL to a full oracle+oracledb URL)."
        )
    return f"{protocol}://{host}:{int(settings.ORACLE_PORT)}/{service}"


def _oracle_credentials() -> tuple[str, str]:
    user = (settings.ORACLE_USER or "").strip()
    password = settings.ORACLE_PASSWORD or ""
    if not user or not password:
        raise OracleMemoryError(
            "Oracle credentials are missing: set ORACLE_USER and ORACLE_PASSWORD, or put the "
            "credentials in DATABASE_URL (oracle+oracledb://user:password@host:port?service_name=...)."
        )
    return user, password


def _with_credentials(url: URL) -> URL:
    """Fill in ``ORACLE_USER``/``ORACLE_PASSWORD`` when the URL carries no credentials."""
    if url.username:
        return url
    user, password = _oracle_credentials()
    return url.set(username=user, password=password)


def build_oracle_connection(raw_url: str | None = None) -> tuple[URL, dict[str, Any]]:
    """Resolve the configured Oracle target to a SQLAlchemy URL plus ``connect_args``.

    Accepts, in order of precedence:

    1. ``raw_url`` / ``DATABASE_URL`` when it starts with ``oracle`` — a full SQLAlchemy URL.
    2. ``ORACLE_CORE_URL`` / ``ORACLE_CORE_ASYNC_URL`` — same forms as (1), or a bare DSN/TNS
       alias / Easy Connect string (credentials come from ``ORACLE_USER``/``ORACLE_PASSWORD``).
    3. ``ORACLE_HOST``/``ORACLE_PORT``/``ORACLE_SERVICE``/``ORACLE_USER``/``ORACLE_PASSWORD``.
    """
    configured = (raw_url or "").strip() or (settings.DATABASE_URL or "").strip()
    if configured.lower().startswith("oracle"):
        return _with_credentials(_normalise_oracle_url(configured)), {}

    core_url = (settings.ORACLE_CORE_URL or settings.ORACLE_CORE_ASYNC_URL or "").strip()
    if core_url and core_url.lower().startswith("oracle"):
        return _with_credentials(_normalise_oracle_url(core_url)), {}

    user, password = _oracle_credentials()
    dsn = core_url or _settings_easy_connect_dsn()
    # ``URL.create`` without host/database yields ``dsn = url.host`` inside the dialect, so the
    # explicit ``dsn`` connect argument is what actually addresses the database.
    return URL.create(_SYNC_DRIVER, username=user, password=password), {"dsn": dsn}


def _describe_target(url: URL, connect_args: Mapping[str, Any]) -> str:
    """``user@host:port/service`` for logs, never including the password."""
    if connect_args.get("dsn"):
        target = str(connect_args["dsn"])
    else:
        port = f":{url.port}" if url.port else ""
        database = f"/{url.database}" if url.database else ""
        service = url.query.get("service_name")
        service_part = f"?service_name={service}" if service else ""
        target = f"{url.host or '?'}{port}{database}{service_part}"
    return f"{url.username or '?'}@{target}"


def _import_oracledb() -> Any:
    """Import python-oracledb on demand so the module imports on machines without it."""
    try:
        import oracledb  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on the deployment image
        raise OracleMemoryError(
            "The 'oracledb' driver is not installed. Install python-oracledb (26.x) or point "
            "DATABASE_URL at SQLite."
        ) from exc
    # Fetch CLOBs as ``str`` instead of LOB handles: a LOB is only readable while its cursor
    # is open, and the contract returns plain strings to callers.
    try:
        oracledb.defaults.fetch_lobs = False
    except Exception:  # noqa: BLE001 - older/newer driver defaults must not break the store
        logger.debug("Could not disable oracledb LOB fetching; reading LOBs explicitly")
    return oracledb


# ---------------------------------------------------------------------------
# ``connect()`` compatibility shim
# ---------------------------------------------------------------------------
class _OracleCursor:
    """Rows of one statement; mirrors the fetch surface of the SQLite cursor proxy."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    async def fetchall(self) -> list[dict[str, Any]]:
        return list(self._rows)

    async def fetchone(self) -> dict[str, Any] | None:
        return self._rows[0] if self._rows else None

    async def fetchmany(self, size: int | None = None) -> list[dict[str, Any]]:
        return list(self._rows if size is None else self._rows[:size])

    async def close(self) -> None:
        return None

    @property
    def rowcount(self) -> int:
        return len(self._rows)


class _OracleResult:
    """A statement that is both awaitable and an async context manager, like aiosqlite's."""

    def __init__(self, connection: "_OracleConnection", sql: str, parameters: Any) -> None:
        self._connection = connection
        self._sql = sql
        self._parameters = parameters
        self._cursor: _OracleCursor | None = None

    async def _ensure(self) -> _OracleCursor:
        if self._cursor is None:
            self._cursor = await self._connection._run(self._sql, self._parameters)
        return self._cursor

    def __await__(self):
        return self._ensure().__await__()

    async def __aenter__(self) -> _OracleCursor:
        return await self._ensure()

    async def __aexit__(self, *_exc: Any) -> None:
        return None


class _OracleConnection:
    """Minimal connection handle for callers that issue their own SQL.

    The statements must be valid Oracle SQL and use named binds (``:name``). This exists so
    that code written against ``memory.connect()`` keeps working; the typed methods above are
    the supported interface.
    """

    def __init__(self, store: "OracleMemory") -> None:
        self._store = store

    async def _run(self, sql: str, parameters: Any) -> _OracleCursor:
        params = self._coerce_parameters(parameters)
        verb = sql.strip().split(None, 1)[0].upper() if sql.strip() else ""
        if verb in {"SELECT", "WITH"}:
            return _OracleCursor(await self._store._read(sql, params))
        await self._store._write(sql, params)
        return _OracleCursor([])

    @staticmethod
    def _coerce_parameters(parameters: Any) -> dict[str, Any]:
        if parameters is None:
            return {}
        if isinstance(parameters, Mapping):
            return dict(parameters)
        raise TypeError(
            "Oracle statements use named binds (:name); positional parameters are not supported"
        )

    def execute(self, sql: str, parameters: Any = None) -> _OracleResult:
        return _OracleResult(self, sql, parameters)

    async def commit(self) -> None:
        # Every statement is committed by ``_write``; the call is kept for interface parity.
        return None

    async def rollback(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def __aenter__(self) -> "_OracleConnection":
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        return None


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------
class OracleMemory(BaseMemory):
    """Oracle Autonomous Database implementation of the persistent memory contract."""

    #: Lets callers pick backend-specific SQL without isinstance checks (see factory).
    backend_name = "oracle"

    def __init__(self, database_url: str | None = None, *, engine: Engine | None = None) -> None:
        self.database_url = (database_url or "").strip() or (settings.DATABASE_URL or "").strip()
        self._engine_obj: Engine | None = engine
        self._engine_override = engine
        self._engine_lock = threading.Lock()
        # Thread, not asyncio, primitives: one store can outlive an event loop.
        self._schema_lock = threading.Lock()
        self._schema_ready = False
        self._pool_size = max(1, int(getattr(settings, "ORACLE_POOL_SIZE", 2)))

    # -- engine lifecycle ---------------------------------------------------
    def describe(self) -> str:
        """Human-readable target, for startup logs and diagnostics."""
        try:
            url, connect_args = build_oracle_connection(self.database_url or None)
        except Exception:  # noqa: BLE001 - description must never raise
            return "oracle (unresolved)"
        return f"Oracle ({_describe_target(url, connect_args)})"

    def _create_engine(self) -> Engine:
        _import_oracledb()
        url, connect_args = build_oracle_connection(self.database_url or None)
        logger.info("Oracle memory store connecting as %s", _describe_target(url, connect_args))
        return create_engine(
            url,
            connect_args=connect_args,
            pool_pre_ping=True,
            pool_size=self._pool_size,
            max_overflow=self._pool_size * 2,
            pool_recycle=_POOL_RECYCLE_SECONDS,
            pool_timeout=30,
            future=True,
        )

    def _engine(self) -> Engine:
        engine = self._engine_obj
        if engine is None:
            with self._engine_lock:
                if self._engine_obj is None:
                    self._engine_obj = self._create_engine()
                engine = self._engine_obj
        return engine

    async def close(self) -> None:
        """Dispose the pool (no-op for a caller-supplied engine)."""
        with self._engine_lock:
            engine = self._engine_obj
            if engine is not self._engine_override:
                self._engine_obj = None
        self._schema_ready = False
        if engine is None or engine is self._engine_override:
            return
        try:
            await asyncio.to_thread(engine.dispose)
        except Exception as exc:  # noqa: BLE001 - teardown must not raise
            logger.debug("Oracle engine disposal failed: %s", exc)

    # -- schema -------------------------------------------------------------
    async def ensure_schema(self, force: bool = False) -> None:
        """Create missing tables/indexes once per instance (idempotent, never fatal)."""
        if self._schema_ready and not force:
            return
        try:
            await asyncio.to_thread(self._ensure_schema_sync, force)
        except Exception as exc:  # noqa: BLE001 - callers get the real error from their query
            logger.warning("Oracle schema initialization failed for %s: %s", self.describe(), exc)

    async def initialize(self) -> None:
        """Create tables and indexes if they do not already exist (always re-checks)."""
        await self.ensure_schema(force=True)

    def _ensure_schema_sync(self, force: bool) -> None:
        with self._schema_lock:
            if self._schema_ready and not force:
                return
            created: list[str] = []
            with self._engine().begin() as conn:
                for name, ddl in _TABLE_DDL:
                    if not self._object_exists(conn, "user_tables", "table_name", name):
                        conn.execute(text(ddl))
                        created.append(name)
                for name, ddl in _INDEX_DDL:
                    if not self._object_exists(conn, "user_indexes", "index_name", name):
                        conn.execute(text(ddl))
                        created.append(name)
            self._schema_ready = True
            if created:
                logger.info("Oracle memory schema initialised (created: %s)", ", ".join(created))
            else:
                logger.debug("Oracle memory schema already present")

    @staticmethod
    def _object_exists(conn: Any, dictionary: str, column: str, name: str) -> bool:
        # ``dictionary``/``column`` are module constants, never caller input.
        sql = text(f"SELECT COUNT(*) AS object_count FROM {dictionary} WHERE {column} = :name")
        count = conn.execute(sql, {"name": name.upper()}).scalar_one()
        return bool(count)

    @asynccontextmanager
    async def connect(self) -> AsyncIterator[_OracleConnection]:
        """Hand out a connection shim, creating the schema first if this instance has not yet."""
        await self.ensure_schema()
        yield _OracleConnection(self)

    # -- low-level statement execution -------------------------------------
    async def _read(
        self,
        sql: str,
        params: Mapping[str, Any] | None = None,
        *,
        clobs: Sequence[str] = (),
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._read_sync, sql, params, clobs)

    def _read_sync(
        self,
        sql: str,
        params: Mapping[str, Any] | None,
        clobs: Sequence[str],
    ) -> list[dict[str, Any]]:
        statement = _statement(sql, clobs)
        with self._engine().connect() as conn:
            rows = conn.execute(statement, dict(params or {})).mappings().all()
        return [_row_to_dict(row) for row in rows]

    async def _write(
        self,
        sql: str,
        params: Mapping[str, Any] | None = None,
        *,
        clobs: Sequence[str] = (),
    ) -> None:
        await asyncio.to_thread(self._write_sync, sql, params, clobs)

    def _write_sync(
        self,
        sql: str,
        params: Mapping[str, Any] | None,
        clobs: Sequence[str],
    ) -> None:
        statement = _statement(sql, clobs)
        with self._engine().begin() as conn:
            conn.execute(statement, dict(params or {}))

    # -- agents -------------------------------------------------------------
    async def save_agent(self, agent_data: dict[str, Any]) -> None:
        """Persist or update an agent configuration record."""
        await self._write(
            _MERGE_AGENT,
            {
                "id": agent_data.get("id"),
                "name": agent_data.get("name", ""),
                "role": agent_data.get("role", ""),
                "provider": agent_data.get("model_provider", "gemini"),
                "model": agent_data.get("model_name", "gemini-3.8-flash"),
                "status": agent_data.get("status", "active"),
                "config_json": json.dumps(agent_data),
            },
            clobs=("config_json",),
        )

    # -- tasks --------------------------------------------------------------
    async def save_task(self, task: TaskRecord) -> None:
        """Create or update a task record."""
        await self._write(
            _MERGE_TASK,
            {
                "id": task.id,
                "question": task.question,
                "task_mode": task.mode,
                "status": task.status,
                "result": task.result,
                "confidence": task.confidence,
                "created_at": _naive_utc(task.created_at),
                "completed_at": _naive_utc(task.completed_at),
                "metadata_json": json.dumps(task.metadata),
            },
            clobs=("question", "result", "metadata_json"),
        )

    async def get_task(self, task_id: str) -> TaskRecord | None:
        """Retrieve a task record by its ID."""
        rows = await self._read(_SELECT_TASK, {"id": task_id})
        if not rows:
            return None
        row = rows[0]
        return TaskRecord(
            id=_text(row.get("id")),
            question=_text(row.get("question")),
            mode=_text(row.get("task_mode")),
            status=_text(row.get("status")),
            result=_optional_text(row.get("result")),
            confidence=row.get("confidence"),
            created_at=_required_timestamp(row.get("created_at")),
            completed_at=_aware_utc(row.get("completed_at")),
            metadata=_json_object(row.get("metadata_json")),
        )

    async def reconcile_orphaned_tasks(self, process_started_at: datetime) -> list[str]:
        """Mark tasks left ``running`` by a previous process as interrupted.

        Same cutoff semantics as the SQLite store: anything created before this process
        started cannot be in flight here, so a sibling worker's genuine task is untouched.
        """
        cutoff = _naive_utc(process_started_at)
        if cutoff is None:
            raise ValueError("timestamp value is missing")
        completed_at = _naive_utc(datetime.now(timezone.utc))
        reconciled: list[str] = []
        rows = await asyncio.to_thread(self._reconcile_sync, cutoff, completed_at)
        reconciled.extend(rows)
        if reconciled:
            logger.warning(
                "Reconciled %d task(s) left running by a previous process: %s",
                len(reconciled),
                ", ".join(reconciled[:5]) + (" ..." if len(reconciled) > 5 else ""),
            )
        return reconciled

    def _reconcile_sync(self, cutoff: datetime, completed_at: datetime) -> list[str]:
        reconciled: list[str] = []
        with self._engine().begin() as conn:
            rows = conn.execute(_statement(_SELECT_RUNNING_TASKS), {"cutoff": cutoff}).mappings().all()
            update = _statement(_UPDATE_RECONCILED_TASK, ("metadata_json",))
            for row in rows:
                record = _row_to_dict(row)
                metadata = _json_object(record.get("metadata_json"))
                metadata["interrupted"] = (
                    "process ended while this task was in flight; no result was produced"
                )
                conn.execute(
                    update,
                    {
                        "completed_at": completed_at,
                        "metadata_json": json.dumps(metadata),
                        "id": record.get("id"),
                    },
                )
                reconciled.append(_text(record.get("id")))
        return reconciled

    # -- runs ---------------------------------------------------------------
    async def save_run(self, run: RunRecord) -> None:
        """Persist an execution run audit record."""
        await self._write(
            _MERGE_RUN,
            {
                "id": run.id,
                "task_id": run.task_id,
                "agent_id": run.agent_id,
                "provider": run.provider,
                "model": run.model,
                "stage": run.stage,
                "latency": run.latency_seconds,
                "prompt_tokens": run.prompt_tokens,
                "completion_tokens": run.completion_tokens,
                "status": run.status,
                "error": run.error,
                "created_at": _naive_utc(run.created_at),
            },
            clobs=("error",),
        )

        # Mirror to dokploy-postgres if available (parity with the SQLite store).
        try:
            from app.memory.postgres import postgres_memory

            await postgres_memory.save_run(run)
        except Exception as exc:  # noqa: BLE001 - the mirror is best effort
            logger.debug("PostgreSQL run mirror skipped: %s", exc)

    # -- messages -----------------------------------------------------------
    async def save_message(self, message: MessageRecord) -> None:
        """Persist a conversation or debate message."""
        await self._write(
            _MERGE_MESSAGE,
            {
                "id": message.id,
                "run_id": message.run_id,
                "task_id": message.task_id,
                "role": message.role,
                "agent_id": message.agent_id,
                "content": message.content,
                "stage": message.stage,
                "created_at": _naive_utc(message.created_at),
            },
            clobs=("content",),
        )

    async def get_task_messages(self, task_id: str) -> list[MessageRecord]:
        """Retrieve all messages associated with a task ID, oldest first."""
        rows = await self._read(_SELECT_TASK_MESSAGES, {"task_id": task_id})
        return [
            MessageRecord(
                id=_text(row.get("id")),
                run_id=_text(row.get("run_id")),
                task_id=_text(row.get("task_id")),
                role=_text(row.get("role")),
                agent_id=_optional_text(row.get("agent_id")),
                content=_text(row.get("content")),
                stage=_optional_text(row.get("stage")),
                created_at=_required_timestamp(row.get("created_at")),
            )
            for row in rows
        ]

    # -- memories -----------------------------------------------------------
    async def save_memory(self, memory: MemoryRecord) -> None:
        """Save a scoped persistent memory item."""
        tags = ",".join(memory.context_tags) if memory.context_tags else ""
        await self._write(
            _MERGE_MEMORY,
            {
                "id": memory.id,
                "agent_id": memory.agent_id,
                "content": memory.content,
                "memory_type": memory.memory_type,
                "importance": memory.importance,
                "tags": tags,
                "created_at": _naive_utc(memory.created_at),
            },
            clobs=("content", "tags"),
        )

    async def get_agent_memories(
        self,
        agent_id: str,
        limit: int = 10,
        memory_type: str | None = None,
    ) -> list[MemoryRecord]:
        """Retrieve memories strictly scoped to a specific agent_id."""
        sql = _SELECT_MEMORIES
        params: dict[str, Any] = {"agent_id": agent_id}
        if memory_type:
            sql += " AND memory_type = :memory_type"
            params["memory_type"] = memory_type
        sql += " ORDER BY importance DESC, created_at DESC"
        rows = await self._read(self._with_row_limit(sql, limit), {**params, **self._limit_params(limit)})
        return [self._memory_from_row(row) for row in rows]

    async def search_memories(
        self,
        query: str,
        agent_id: str | None = None,
        limit: int = 5,
    ) -> list[MemoryRecord]:
        """Search memory records matching text, optionally filtered by agent_id."""
        sql = "SELECT * FROM memories WHERE content LIKE :pattern"
        params: dict[str, Any] = {"pattern": f"%{query}%"}
        if agent_id:
            sql += " AND agent_id = :agent_id"
            params["agent_id"] = agent_id
        sql += " ORDER BY importance DESC"
        rows = await self._read(self._with_row_limit(sql, limit), {**params, **self._limit_params(limit)})
        return [self._memory_from_row(row) for row in rows]

    @staticmethod
    def _memory_from_row(row: Mapping[str, Any]) -> MemoryRecord:
        importance = row.get("importance")
        memory_type = row.get("memory_type")
        return MemoryRecord(
            id=_text(row.get("id")),
            agent_id=_text(row.get("agent_id")),
            content=_text(row.get("content")),
            memory_type=memory_type if memory_type else "fact",
            importance=float(importance) if importance is not None else 0.5,
            context_tags=_csv_list(row.get("tags")),
            created_at=_required_timestamp(row.get("created_at")),
        )

    @staticmethod
    def _with_row_limit(sql: str, limit: int | None) -> str:
        """Append Oracle row limiting; a non-positive limit means "no limit" (SQLite parity)."""
        if limit is None:
            return sql
        value = int(limit)
        if value <= 0:
            return sql
        return f"{sql} FETCH FIRST :row_limit ROWS ONLY"

    def _limit_params(self, limit: int | None) -> dict[str, Any]:
        value = int(limit) if limit is not None else 0
        return {"row_limit": value} if value > 0 else {}

    # -- strategies ---------------------------------------------------------
    async def save_strategy(self, strategy: StrategyRecord) -> None:
        """Save or update a learned strategy record (conflict key: ``task_type``)."""
        await self._write(
            _MERGE_STRATEGY,
            {
                "id": strategy.id,
                "task_type": strategy.task_type,
                "strategy": strategy.strategy,
                "score": strategy.score,
                "sample_size": strategy.sample_size,
                "recommended_agents": ",".join(strategy.recommended_agents),
                "recommended_provider": strategy.recommended_provider,
                "recommended_model": strategy.recommended_model,
                "created_at": _naive_utc(strategy.created_at),
                "metadata_json": json.dumps(strategy.metadata),
            },
            clobs=("strategy", "recommended_agents", "metadata_json"),
        )

        # Mirror to dokploy-postgres if available (parity with the SQLite store).
        try:
            from app.memory.postgres import postgres_memory

            await postgres_memory.save_strategy(strategy)
        except Exception as exc:  # noqa: BLE001 - the mirror is best effort
            logger.debug("PostgreSQL strategy mirror skipped: %s", exc)

    async def get_strategy(self, task_type: str) -> StrategyRecord | None:
        """Retrieve the learned strategy for a specific task type."""
        rows = await self._read(_SELECT_STRATEGY, {"task_type": task_type})
        return self._strategy_from_row(rows[0]) if rows else None

    async def list_strategies(self) -> list[StrategyRecord]:
        """List all learned strategies, best score first."""
        rows = await self._read(_SELECT_STRATEGIES)
        return [self._strategy_from_row(row) for row in rows]

    @staticmethod
    def _strategy_from_row(row: Mapping[str, Any]) -> StrategyRecord:
        score = row.get("score")
        sample_size = row.get("sample_size")
        provider = row.get("recommended_provider")
        model = row.get("recommended_model")
        return StrategyRecord(
            id=_text(row.get("id")),
            task_type=_text(row.get("task_type")),
            strategy=_text(row.get("strategy")),
            score=float(score) if score is not None else 0.0,
            sample_size=int(sample_size) if sample_size is not None else 1,
            recommended_agents=_csv_list(row.get("recommended_agents")),
            recommended_provider=provider if provider else "gemini",
            recommended_model=model if model else "gemini-3.8-flash",
            created_at=_required_timestamp(row.get("created_at")),
            metadata=_json_object(row.get("metadata_json")),
        )

    # -- experiments --------------------------------------------------------
    async def save_experiment(self, experiment: ExperimentRecord) -> None:
        """Save an experiment record."""
        await self._write(
            _MERGE_EXPERIMENT,
            {
                "id": experiment.id,
                "hypothesis": experiment.hypothesis,
                "configuration": json.dumps(experiment.configuration),
                "status": experiment.status,
                "result_json": json.dumps(experiment.result) if experiment.result else None,
                "created_at": _naive_utc(experiment.created_at),
            },
            clobs=("hypothesis", "configuration", "result_json"),
        )

    async def get_experiment(self, experiment_id: str) -> ExperimentRecord | None:
        """Retrieve experiment details by ID."""
        rows = await self._read(_SELECT_EXPERIMENT, {"id": experiment_id})
        if not rows:
            return None
        row = rows[0]
        return ExperimentRecord(
            id=_text(row.get("id")),
            hypothesis=_text(row.get("hypothesis")),
            configuration=_json_object(row.get("configuration")),
            status=_text(row.get("status")),
            result=_json_optional(row.get("result_json")),
            created_at=_required_timestamp(row.get("created_at")),
        )
