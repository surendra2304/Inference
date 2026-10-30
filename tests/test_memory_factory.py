"""Contract and factory tests for the memory backend split (Oracle / SQLite).

These tests run without an Oracle server and without the ``oracledb`` driver: they pin the
things that must hold *before* a connection exists — backend selection, fail-closed
configuration, and signature parity between the two stores.
"""

from __future__ import annotations

import inspect
import re
from datetime import datetime, timedelta, timezone

import pytest

from app.core.config import UNCONFIGURED_DATABASE_MESSAGE, settings
from app.memory.base import TaskRecord
from app.memory.factory import (
    DatabaseConfigurationError,
    create_memory_store,
    describe_memory_backend,
    task_mode_column,
)
from app.memory.oracle import OracleMemory, _aware_utc, _naive_utc, build_oracle_connection
from app.memory.sqlite import SQLiteMemory

ORACLE_URL = "oracle+oracledb://INFERENCE:secret@adb.example.com:1521?service_name=friday_medium"

#: The methods callers depend on. ``__init__`` is excluded: the SQLite store takes a path and
#: the Oracle store takes a URL (plus an optional injected engine) for testability.
CONTRACT_METHODS = (
    "initialize",
    "ensure_schema",
    "close",
    "connect",
    "reconcile_orphaned_tasks",
    "save_agent",
    "save_task",
    "get_task",
    "save_run",
    "save_message",
    "get_task_messages",
    "save_memory",
    "get_agent_memories",
    "search_memories",
    "save_strategy",
    "get_strategy",
    "list_strategies",
    "save_experiment",
    "get_experiment",
)


def _parameters(func) -> list[tuple[str, int, object]]:
    return [
        (parameter.name, parameter.kind, parameter.default)
        for parameter in inspect.signature(func).parameters.values()
    ]


# -- factory selection ------------------------------------------------------------


def test_factory_returns_sqlite_only_for_an_explicit_sqlite_url(tmp_path):
    store = create_memory_store(f"sqlite:///{tmp_path / 'explicit.db'}")
    assert isinstance(store, SQLiteMemory)
    assert store.db_path.endswith("explicit.db")


def test_factory_returns_the_oracle_store_for_an_oracle_url():
    store = create_memory_store(ORACLE_URL)
    assert isinstance(store, OracleMemory)
    assert store.backend_name == "oracle"


@pytest.mark.parametrize(
    "url",
    [
        "oracle://INFERENCE:secret@adb.example.com:1521?service_name=friday_medium",
        "oracle+oracledb_async://INFERENCE:secret@adb.example.com:1521?service_name=friday_medium",
    ],
)
def test_oracle_urls_are_normalised_to_the_sync_driver(url):
    parsed, connect_args = build_oracle_connection(url)
    assert parsed.drivername == "oracle+oracledb"
    assert parsed.password == "secret"
    assert connect_args == {}


def test_factory_fails_closed_when_nothing_is_configured(monkeypatch):
    """The old SQLite default silently wrote audit records to a local file; refuse instead."""
    monkeypatch.setattr(settings, "DATABASE_URL", "")
    monkeypatch.setattr(settings, "ORACLE_CORE_URL", None)
    monkeypatch.setattr(settings, "ORACLE_CORE_ASYNC_URL", None)
    monkeypatch.setattr(settings, "ORACLE_PASSWORD", None)

    with pytest.raises(DatabaseConfigurationError) as excinfo:
        create_memory_store()
    assert "No memory database is configured" in str(excinfo.value)
    assert excinfo.value.args[0] == UNCONFIGURED_DATABASE_MESSAGE


def test_factory_falls_back_to_oracle_settings_when_the_url_is_empty(monkeypatch):
    monkeypatch.setattr(settings, "DATABASE_URL", "")
    monkeypatch.setattr(settings, "ORACLE_CORE_URL", None)
    monkeypatch.setattr(settings, "ORACLE_CORE_ASYNC_URL", None)
    monkeypatch.setattr(settings, "ORACLE_HOST", "adb.example.com")
    monkeypatch.setattr(settings, "ORACLE_SERVICE", "friday_medium")
    monkeypatch.setattr(settings, "ORACLE_USER", "INFERENCE")
    monkeypatch.setattr(settings, "ORACLE_PASSWORD", "secret")

    assert settings.database_backend == "oracle"
    store = create_memory_store()
    assert isinstance(store, OracleMemory)
    # DSN building must not require a live database or the driver.
    url, connect_args = build_oracle_connection()
    assert url.username == "INFERENCE"
    assert connect_args["dsn"] == "tcps://adb.example.com:1521/friday_medium"


def test_factory_rejects_an_unsupported_scheme():
    with pytest.raises(DatabaseConfigurationError) as excinfo:
        create_memory_store("postgresql://inference:secret@localhost/inference_db")
    assert "Unsupported memory database URL scheme 'postgresql'" in str(excinfo.value)


def test_validate_database_configuration_fails_closed_and_reports_the_backend(monkeypatch):
    monkeypatch.setattr(settings, "DATABASE_URL", "")
    monkeypatch.setattr(settings, "ORACLE_CORE_URL", None)
    monkeypatch.setattr(settings, "ORACLE_CORE_ASYNC_URL", None)
    monkeypatch.setattr(settings, "ORACLE_PASSWORD", None)
    with pytest.raises(RuntimeError):
        settings.validate_database_configuration()

    monkeypatch.setattr(settings, "DATABASE_URL", "sqlite:///data/universe.db")
    assert settings.validate_database_configuration() == "sqlite"


def test_backend_description_names_the_real_backend():
    assert describe_memory_backend("sqlite:///data/universe.db").startswith("SQLite (sqlite:///data/universe.db")
    described = describe_memory_backend(ORACLE_URL)
    assert described.startswith("Oracle Autonomous Database 26ai")
    assert "secret" not in described  # never echo the password


# -- contract parity --------------------------------------------------------------


@pytest.mark.parametrize("method", CONTRACT_METHODS)
def test_oracle_store_exposes_the_sqlite_contract(method):
    sqlite_method = getattr(SQLiteMemory, method)
    oracle_method = getattr(OracleMemory, method)
    assert callable(oracle_method)
    assert _parameters(oracle_method) == _parameters(sqlite_method), (
        f"{method}() signature drifted between SQLiteMemory and OracleMemory"
    )


def test_oracle_store_implements_every_abstract_method():
    assert getattr(OracleMemory, "__abstractmethods__", frozenset()) == frozenset()


# -- backend-specific SQL ---------------------------------------------------------


def test_task_mode_column_reflects_the_reserved_word(monkeypatch):
    assert task_mode_column(SQLiteMemory(":memory:")) == "mode"
    assert task_mode_column(OracleMemory(ORACLE_URL)) == "task_mode"
    # Anything without a backend marker is treated as SQLite (the historical name).
    assert task_mode_column(object()) == "mode"


async def test_oracle_connect_shim_supports_both_cursor_styles():
    store = OracleMemory(ORACLE_URL)
    executed: list[str] = []

    async def no_schema(force: bool = False) -> None:
        return None

    async def fake_read(sql, params=None, *, clobs=()):
        executed.append(sql)
        return [{"id": "t1", "task_mode": "fast"}]

    store.ensure_schema = no_schema  # type: ignore[method-assign]
    store._read = fake_read  # type: ignore[method-assign]

    async with store.connect() as db:
        async with db.execute("SELECT * FROM tasks WHERE id = :id", {"id": "t1"}) as cursor:
            rows = await cursor.fetchall()
        awaited = await db.execute("SELECT * FROM tasks WHERE id = :id", {"id": "t1"})
        single = await awaited.fetchone()

    assert rows == [{"id": "t1", "task_mode": "fast"}]
    assert single == {"id": "t1", "task_mode": "fast"}
    assert executed == ["SELECT * FROM tasks WHERE id = :id"] * 2


async def test_oracle_connect_shim_rejects_positional_parameters():
    store = OracleMemory(ORACLE_URL)

    async def no_schema(force: bool = False) -> None:
        return None

    store.ensure_schema = no_schema  # type: ignore[method-assign]
    async with store.connect() as db:
        with pytest.raises(TypeError):
            await db.execute("SELECT * FROM tasks WHERE id = ?", ("t1",))


# -- value normalisation ----------------------------------------------------------


def test_timestamps_are_stored_as_naive_utc_and_read_back_as_aware_utc():
    ist = timezone(timedelta(hours=5, minutes=30))
    original = datetime(2026, 10, 27, 12, 0, tzinfo=ist)
    stored = _naive_utc(original)
    assert stored == datetime(2026, 10, 27, 6, 30)
    assert stored.tzinfo is None
    assert _aware_utc(stored) == original
    assert _aware_utc("2026-10-27T06:30:00") == original  # legacy ISO text
    assert _aware_utc(None) is None


async def test_saved_records_are_normalised_for_oracle_binds():
    """Empty strings are NULL in Oracle, so writes must carry the values the reader expects."""
    store = OracleMemory(ORACLE_URL)
    captured: list[tuple[str, dict]] = []

    async def fake_write(sql, params=None, *, clobs=()):
        captured.append((sql, dict(params or {})))

    async def fake_read(sql, params=None, *, clobs=()):
        return []

    store._write = fake_write  # type: ignore[method-assign]
    store._read = fake_read  # type: ignore[method-assign]

    await store.save_task(TaskRecord(id="t1", question="q", mode="trading_consult", status="pending"))
    assert captured[0][1]["task_mode"] == "trading_consult"
    assert captured[0][1]["created_at"].tzinfo is None

    # A row whose optional columns Oracle returned as NULL must still build a valid record.
    store_active = OracleMemory(ORACLE_URL)

    async def one_row(sql, params=None, *, clobs=()):
        return [
            {
                "id": "t1",
                "question": "q",
                "task_mode": "fast",
                "status": "pending",
                "result": None,
                "confidence": None,
                "created_at": datetime(2026, 10, 27, 6, 30),
                "completed_at": None,
                "metadata_json": None,
            }
        ]

    store_active._read = one_row  # type: ignore[method-assign]
    task = await store_active.get_task("t1")
    assert task is not None
    assert task.result is None and task.completed_at is None
    assert task.metadata == {}
    assert task.created_at.tzinfo is not None


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM memories WHERE content LIKE :pattern",
    ],
)
def test_sql_uses_named_binds_only(sql):
    from app.memory import oracle

    statements = [oracle._MERGE_TASK, oracle._MERGE_STRATEGY, oracle._MERGE_RUN, sql]
    for statement in statements:
        assert "?" not in statement, statement
        assert not re.search(r"\bLIMIT\b", statement, re.IGNORECASE), statement
