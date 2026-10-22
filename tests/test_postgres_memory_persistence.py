"""Tests for PostgreSQL Dual-Persistence and Idempotent Schema Migration."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.config import Settings
from app.memory.base import DeliberationRecord, RunRecord, StrategyRecord
from app.memory.postgres import SCHEMA_DDL, PostgresMemory


def test_schema_ddl_contains_required_tables():
    assert "CREATE TABLE IF NOT EXISTS runs" in SCHEMA_DDL
    assert "CREATE TABLE IF NOT EXISTS strategies" in SCHEMA_DDL
    assert "CREATE TABLE IF NOT EXISTS deliberations" in SCHEMA_DDL
    assert "CREATE INDEX IF NOT EXISTS idx_deliberations_session_id" in SCHEMA_DDL
    assert "confidence DOUBLE PRECISION," in SCHEMA_DDL
    assert "ALTER TABLE deliberations ALTER COLUMN confidence DROP NOT NULL" in SCHEMA_DDL


@pytest.mark.asyncio
async def test_postgres_memory_fails_open_when_unreachable():
    mem = PostgresMemory()
    mem.host = "unreachable-non-existent-host-9999"

    # Should not crash; returns False/None gracefully
    assert await mem.initialize() is False
    run = RunRecord(
        id="run_test_01",
        task_id="task_01",
        agent_id="friday",
        provider="groq",
        model="llama-3.3-70b-versatile",
        stage="test",
        latency_seconds=0.1,
    )
    assert await mem.save_run(run) is False

    delib = DeliberationRecord(
        id="delib_01",
        session_id="session_01",
        topic="Architecture",
        participating_agents=["friday", "cortex"],
        consensus_strategy="adjudicator",
        final_decision="Approved",
    )
    assert await mem.save_deliberation(delib) is False
    assert await mem.get_deliberation("session_01") is None
    await mem.close()


@pytest.mark.asyncio
async def test_postgres_memory_saves_and_retrieves_records_mocked():
    mem = PostgresMemory()

    mock_conn = MagicMock()
    mock_conn.execute = AsyncMock()
    mock_conn.fetchrow = AsyncMock(return_value={
        "id": "delib_mock_01",
        "session_id": "session_mock_01",
        "topic": "Scaling Strategy",
        "participating_agents": "friday,stratex",
        "consensus_strategy": "adjudicator",
        "final_decision": "Execute immediately",
        "confidence": 0.95,
        "rounds": 2,
    })

    # Context manager mock for pool.acquire()
    acquire_cm = MagicMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=mock_conn)
    acquire_cm.__aexit__ = AsyncMock(return_value=None)

    mock_pool = MagicMock()
    mock_pool.acquire.return_value = acquire_cm
    mock_pool.close = AsyncMock()

    mem._pool = mock_pool
    mem._is_available = True

    # Test ensure_schema
    await mem._ensure_schema(mock_pool)
    mock_conn.execute.assert_called_with(SCHEMA_DDL)

    # Test save_run
    run = RunRecord(
        id="run_mock_01",
        task_id="task_mock_01",
        agent_id="cortex",
        provider="groq",
        model="llama-3.3-70b-versatile",
        stage="analysis",
        latency_seconds=0.25,
        prompt_tokens=150,
        completion_tokens=80,
    )
    res_run = await mem.save_run(run)
    assert res_run is True

    # Test save_strategy
    strat = StrategyRecord(
        id="strat_mock_01",
        task_type="code_review",
        strategy="spec_driven",
        score=0.92,
        recommended_agents=["forge", "cortex"],
        recommended_provider="openrouter",
        recommended_model="qwen/qwen-2.5-coder-32b-instruct",
    )
    res_strat = await mem.save_strategy(strat)
    assert res_strat is True

    # Test save_deliberation
    delib = DeliberationRecord(
        id="delib_mock_01",
        session_id="session_mock_01",
        topic="Scaling Strategy",
        participating_agents=["friday", "stratex"],
        consensus_strategy="adjudicator",
        final_decision="Execute immediately",
        confidence=0.95,
        rounds=2,
    )
    res_delib = await mem.save_deliberation(delib)
    assert res_delib is True

    # Test get_deliberation
    queried = await mem.get_deliberation("session_mock_01")
    assert queried is not None
    assert queried["id"] == "delib_mock_01"
    assert queried["final_decision"] == "Execute immediately"

    await mem.close()
    mock_pool.close.assert_called_once()


@pytest.mark.asyncio
async def test_injected_pool_works_when_asyncpg_is_not_installed(monkeypatch):
    """A supplied compatible pool must be usable without the optional asyncpg import."""
    mem = PostgresMemory()
    injected_pool = MagicMock()
    mem._pool = injected_pool
    monkeypatch.setattr("app.memory.postgres.HAS_ASYNCPG", False)

    assert await mem._get_pool() is injected_pool


@pytest.mark.asyncio
async def test_deliberation_without_observed_confidence_is_persisted_as_null():
    """Unknown confidence is stored as NULL, never coerced or fabricated to a number."""
    mem = PostgresMemory()
    mock_conn = MagicMock()
    mock_conn.execute = AsyncMock()

    acquire_cm = MagicMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=mock_conn)
    acquire_cm.__aexit__ = AsyncMock(return_value=None)

    mock_pool = MagicMock()
    mock_pool.acquire.return_value = acquire_cm
    mem._pool = mock_pool
    mem._is_available = True

    record = DeliberationRecord(
        id="delib_unknown_confidence",
        session_id="session_unknown_confidence",
        topic="Unknown confidence persistence",
        participating_agents=["friday"],
        final_decision="Insufficient evidence",
        confidence=None,
    )

    assert await mem.save_deliberation(record) is True
    assert mock_conn.execute.await_args.args[7] is None


@pytest.mark.asyncio
async def test_postgres_pool_is_not_created_without_explicit_password(monkeypatch):
    """PostgreSQL must not use an embedded or implicit credential."""
    mem = PostgresMemory()
    mem.password = None
    monkeypatch.setattr("app.memory.postgres.HAS_ASYNCPG", True)

    assert await mem.initialize() is False
    assert mem._pool is None
    assert mem._is_available is False


def test_postgres_password_has_no_hardcoded_default():
    assert Settings.model_fields["POSTGRES_PASSWORD"].default is None
