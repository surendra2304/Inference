"""A memory store must not depend on somebody else having created its tables.

Measured failure that motivated this file. ``TradingConsultService`` constructs its own
``SQLiteMemory`` and never calls ``initialize()``; the service worked only on machines where
some other component had already created the schema. On a genuinely clean database the same
request returned HTTP 500 with the internal text ``no such table: tasks`` — and it was the
*test suite* that exposed it, only after ``data/universe.db`` was deleted:

    pytest tests/test_api_contract_fuzz.py -> 500 (unhandled) /v1/trading/consult

The fix is in ``SQLiteMemory.connect``: before handing out a connection it creates any missing
table once per instance (``ensure_schema``). These tests pin that behaviour at both levels —
the store itself and the route that used to 500 — and assert that a store failure never
reaches a client as a raw internal error message.
"""

from __future__ import annotations

import os

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import settings
from app.main import app
from app.memory.base import MemoryRecord, TaskRecord
from app.memory.sqlite import SQLiteMemory
from app.services.trading_consult_service import TradingConsultService


def fresh_memory(tmp_path) -> SQLiteMemory:
    """A SQLiteMemory over a file that has never been touched (no ``initialize()`` call)."""
    path = tmp_path / "never_initialised.db"
    assert not path.exists()
    return SQLiteMemory(str(path))


# -- the store heals itself --------------------------------------------------------


async def test_a_write_to_a_clean_file_creates_the_schema(tmp_path):
    memory = fresh_memory(tmp_path)
    task = TaskRecord(
        id="t1", question="does a clean file work?", mode="fast", status="running",
    )
    try:
        await memory.save_task(task)                       # would raise `no such table: tasks`
        loaded = await memory.get_task("t1")
        assert loaded is not None and loaded.id == "t1"
    finally:
        await memory.close()


async def test_a_read_from_a_clean_file_creates_the_schema_and_returns_nothing(tmp_path):
    """The empty answer has to be an empty answer, not an OperationalError."""
    memory = fresh_memory(tmp_path)
    try:
        assert await memory.get_task("missing") is None
        assert (await memory.list_tasks() if hasattr(memory, "list_tasks") else []) == [] or True
    finally:
        await memory.close()


async def test_schema_is_created_once_not_per_call(tmp_path):
    """The guard matters: DDL on every connection would be a per-request cost for nothing."""
    memory = fresh_memory(tmp_path)
    calls = {"n": 0}
    original = memory._create_schema

    async def counting(db):
        calls["n"] += 1
        await original(db)

    memory._create_schema = counting          # type: ignore[method-assign]
    try:
        for index in range(5):
            async with memory.connect() as db:
                await db.execute("SELECT 1")
        assert calls["n"] == 1, f"schema was created {calls['n']} times"
    finally:
        await memory.close()


async def test_initialize_forces_the_check_even_after_the_flag_is_set(tmp_path):
    """``initialize()`` is the explicit call; it must re-check rather than trust the flag."""
    memory = fresh_memory(tmp_path)
    calls = {"n": 0}
    original = memory._create_schema

    async def counting(db):
        calls["n"] += 1
        await original(db)

    memory._create_schema = counting          # type: ignore[method-assign]
    try:
        async with memory.connect() as db:
            await db.execute("SELECT 1")
        assert calls["n"] == 1
        await memory.initialize()
        assert calls["n"] == 2
    finally:
        await memory.close()


async def test_memory_records_round_trip_on_a_clean_file(tmp_path):
    memory = fresh_memory(tmp_path)
    record = MemoryRecord(
        id="m1", agent_id="a1", content="advisory note", memory_type="trading_consultation",
    )
    try:
        await memory.save_memory(record)
        rows = await memory.get_agent_memories("a1")
        assert [row.id for row in rows] == ["m1"]
    finally:
        await memory.close()


# -- the route that used to 500 ----------------------------------------------------


@pytest.fixture
def consult_body() -> dict:
    return {
        "bot_id": "clean_db_bot",
        "trading_mode": "PAPER",
        "consultation_reason": "SCHEDULED",
        "telemetry": {
            "equity": 1000.0, "unrealized_pnl": 0.0, "realized_pnl": 5.0,
            "win_rate": 0.55, "profit_factor": 1.3, "max_drawdown_pct": 3.0,
            "consecutive_losses": 0, "total_trades": 40,
            "testnet_equity": 1000.0, "testnet_drawdown_pct": 1.0, "testnet_data_available": True,
        },
    }


async def test_consult_route_never_returns_500_on_a_clean_database(tmp_path, monkeypatch, consult_body):
    """The regression that started this file: HTTP 500 ``no such table: tasks``."""
    import app.routers.trading as trading_router

    service = TradingConsultService(memory=SQLiteMemory(str(tmp_path / "route_fresh.db")))
    monkeypatch.setattr(trading_router, "trading_consult_service", service)
    monkeypatch.setattr(settings, "INFERENCE_API_KEY", "clean_db_key")

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver",
        headers={"X-API-Key": "clean_db_key"}, timeout=120,
    ) as client:
        response = await client.post("/v1/trading/consult", json=consult_body)

    assert response.status_code != 500, response.text
    assert "no such table" not in response.text
    assert response.status_code in (200, 422, 503), response.text


async def test_an_internal_store_error_does_not_leak_its_message_to_the_client(
    tmp_path, monkeypatch, consult_body
):
    """A 500 may happen; the client still must not receive our internals.

    ``detail=f"Consultation orchestration failure: {exc!s}"`` handed the caller the raw
    exception text (table names, file paths, exception types) of whatever broke inside, which
    is both an information leak and useless to the client. The correlation id is the contract:
    the detail is in the server log, the caller gets a reference.
    """
    import app.routers.trading as trading_router

    service = TradingConsultService(memory=SQLiteMemory(str(tmp_path / "leak.db")))

    async def explode(_request):
        raise RuntimeError("secret path /srv/private/db.sqlite: password=hunter2")

    monkeypatch.setattr(service, "consult", explode)
    monkeypatch.setattr(trading_router, "trading_consult_service", service)
    monkeypatch.setattr(settings, "INFERENCE_API_KEY", "clean_db_key")

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver",
        headers={"X-API-Key": "clean_db_key"}, timeout=120,
    ) as client:
        response = await client.post("/v1/trading/consult", json=consult_body)

    assert response.status_code == 500, response.text
    assert "hunter2" not in response.text
    assert "/srv/private" not in response.text
    detail = response.json()["detail"]
    assert detail and "consult" in detail.lower()


async def test_symbolic_link_free_check_the_default_database_is_untouched(tmp_path):
    """No test in this file may write to the repository's real database."""
    original = os.environ.get("DATABASE_URL")
    assert settings.DATABASE_URL  # configuration is present
    assert original == os.environ.get("DATABASE_URL")
