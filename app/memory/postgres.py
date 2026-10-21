"""PostgreSQL persistent memory store for dokploy-postgres (inference_db).

Provides dual persistence alongside SQLiteMemory for runs, strategies, tasks, and deliberations.
Fails open / gracefully if PostgreSQL is unreachable (e.g. offline tests or local host).
Automatically provisions and validates the database schema upon connection.
"""

from __future__ import annotations

import json
from typing import Any

from app.core.config import settings
from app.memory.base import DeliberationRecord, RunRecord, StrategyRecord
from app.utils.logger import logger

try:
    import asyncpg
    HAS_ASYNCPG = True
except ImportError:
    HAS_ASYNCPG = False
    asyncpg = None  # type: ignore[assignment]


SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    agent_id TEXT,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    stage TEXT,
    latency DOUBLE PRECISION NOT NULL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    status TEXT NOT NULL,
    error TEXT,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL
);

CREATE TABLE IF NOT EXISTS strategies (
    id TEXT PRIMARY KEY,
    task_type TEXT UNIQUE NOT NULL,
    strategy TEXT NOT NULL,
    score DOUBLE PRECISION NOT NULL,
    sample_size INTEGER NOT NULL DEFAULT 1,
    recommended_agents TEXT NOT NULL,
    recommended_provider TEXT NOT NULL,
    recommended_model TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS deliberations (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    topic TEXT NOT NULL,
    participating_agents TEXT NOT NULL,
    consensus_strategy TEXT NOT NULL,
    final_decision TEXT NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    rounds INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_runs_task_id ON runs (task_id);
CREATE INDEX IF NOT EXISTS idx_runs_provider ON runs (provider);
CREATE INDEX IF NOT EXISTS idx_deliberations_session_id ON deliberations (session_id);
"""


class PostgresMemory:
    """Async PostgreSQL client for inference_db runs, strategies, and deliberations."""

    def __init__(self) -> None:
        self.host = settings.POSTGRES_HOST
        self.port = settings.POSTGRES_PORT
        self.database = settings.POSTGRES_DB
        self.user = settings.POSTGRES_USER
        self.password = settings.POSTGRES_PASSWORD
        self._pool: Any = None
        self._is_available: bool | None = None

    async def _ensure_schema(self, pool: Any) -> None:
        """Applies idempotent DDL migration to guarantee inference_db tables and indexes exist."""
        try:
            async with pool.acquire() as conn:
                await conn.execute(SCHEMA_DDL)
            logger.info("PostgreSQL inference_db schema verified/migrated successfully.")
        except Exception as exc:
            logger.warning("PostgreSQL schema initialization encountered a non-fatal error: %s", exc)

    async def initialize(self) -> bool:
        """Pre-initializes the connection pool and provisions schema if PostgreSQL is reachable."""
        pool = await self._get_pool()
        return pool is not None

    async def _get_pool(self) -> Any:
        # Respect an explicitly injected pool before checking the optional driver.
        # This keeps adapter-level tests and externally supplied compatible pools
        # usable even when asyncpg is not installed in the current environment.
        if self._pool is not None:
            return self._pool
        if not HAS_ASYNCPG:
            return None
        if self._is_available is False:
            return None

        try:
            self._pool = await asyncpg.create_pool(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                min_size=1,
                max_size=5,
                command_timeout=5.0,
                timeout=3.0,
            )
            self._is_available = True
            await self._ensure_schema(self._pool)
            logger.info("Connected to PostgreSQL (inference_db) on %s:%s", self.host, self.port)
            return self._pool
        except Exception as exc:
            self._is_available = False
            logger.debug("PostgreSQL not reachable at %s:%s (%s); will fall back to SQLite.", self.host, self.port, exc)
            return None

    async def save_run(self, run: RunRecord) -> bool:
        """Persist model latency, token counts, and status directly into inference_db.runs."""
        pool = await self._get_pool()
        if not pool:
            return False

        query = """
            INSERT INTO runs (id, task_id, agent_id, provider, model, stage, latency, prompt_tokens, completion_tokens, status, error, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            ON CONFLICT (id) DO UPDATE SET
                latency = EXCLUDED.latency,
                prompt_tokens = EXCLUDED.prompt_tokens,
                completion_tokens = EXCLUDED.completion_tokens,
                status = EXCLUDED.status,
                error = EXCLUDED.error;
        """
        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    query,
                    run.id,
                    run.task_id,
                    run.agent_id,
                    run.provider,
                    run.model,
                    run.stage,
                    float(run.latency_seconds),
                    run.prompt_tokens,
                    run.completion_tokens,
                    run.status,
                    run.error,
                    run.created_at,
                )
            return True
        except Exception as exc:
            logger.warning("Failed to persist run '%s' to PostgreSQL: %s", run.id, exc)
            return False

    async def save_strategy(self, strategy: StrategyRecord) -> bool:
        """Persist task strategy, score, and recommended model directly into inference_db.strategies."""
        pool = await self._get_pool()
        if not pool:
            return False

        agents_str = ",".join(strategy.recommended_agents)
        meta_json = json.dumps(strategy.metadata)

        query = """
            INSERT INTO strategies (
                id, task_type, strategy, score, sample_size,
                recommended_agents, recommended_provider, recommended_model,
                created_at, metadata_json
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            ON CONFLICT (task_type) DO UPDATE SET
                strategy = EXCLUDED.strategy,
                score = EXCLUDED.score,
                sample_size = EXCLUDED.sample_size,
                recommended_agents = EXCLUDED.recommended_agents,
                recommended_provider = EXCLUDED.recommended_provider,
                recommended_model = EXCLUDED.recommended_model,
                metadata_json = EXCLUDED.metadata_json;
        """
        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    query,
                    strategy.id,
                    strategy.task_type,
                    strategy.strategy,
                    float(strategy.score),
                    strategy.sample_size,
                    agents_str,
                    strategy.recommended_provider,
                    strategy.recommended_model,
                    strategy.created_at,
                    meta_json,
                )
            return True
        except Exception as exc:
            logger.warning("Failed to persist strategy '%s' to PostgreSQL: %s", strategy.task_type, exc)
            return False

    async def save_deliberation(self, deliberation: DeliberationRecord) -> bool:
        """Persist a multi-agent deliberation outcome directly into inference_db.deliberations."""
        pool = await self._get_pool()
        if not pool:
            return False

        agents_str = ",".join(deliberation.participating_agents)
        meta_json = json.dumps(deliberation.metadata)

        query = """
            INSERT INTO deliberations (
                id, session_id, topic, participating_agents, consensus_strategy,
                final_decision, confidence, rounds, created_at, metadata_json
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            ON CONFLICT (id) DO UPDATE SET
                final_decision = EXCLUDED.final_decision,
                confidence = EXCLUDED.confidence,
                rounds = EXCLUDED.rounds,
                metadata_json = EXCLUDED.metadata_json;
        """
        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    query,
                    deliberation.id,
                    deliberation.session_id,
                    deliberation.topic,
                    agents_str,
                    deliberation.consensus_strategy,
                    deliberation.final_decision,
                    float(deliberation.confidence),
                    deliberation.rounds,
                    deliberation.created_at,
                    meta_json,
                )
            return True
        except Exception as exc:
            logger.warning("Failed to persist deliberation '%s' to PostgreSQL: %s", deliberation.id, exc)
            return False

    async def get_deliberation(self, session_id: str) -> dict[str, Any] | None:
        """Retrieve the latest deliberation outcome for a session ID."""
        pool = await self._get_pool()
        if not pool:
            return None

        query = "SELECT * FROM deliberations WHERE session_id = $1 ORDER BY created_at DESC LIMIT 1;"
        try:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(query, session_id)
                if not row:
                    return None
                return dict(row)
        except Exception as exc:
            logger.warning("Failed to query deliberation '%s' from PostgreSQL: %s", session_id, exc)
            return None

    async def close(self) -> None:
        if self._pool is not None:
            try:
                await self._pool.close()
            except Exception:
                pass
            self._pool = None


# Process-wide singleton
postgres_memory = PostgresMemory()
