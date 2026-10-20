"""PostgreSQL persistent memory store for dokploy-postgres (inference_db).

Provides dual persistence alongside SQLiteMemory for runs, strategies, and tasks.
Fails open / gracefully if PostgreSQL is unreachable (e.g. offline tests or local host).
"""

from __future__ import annotations

import json
from typing import Any

from app.core.config import settings
from app.memory.base import RunRecord, StrategyRecord
from app.utils.logger import logger

try:
    import asyncpg
    HAS_ASYNCPG = True
except ImportError:
    HAS_ASYNCPG = False
    asyncpg = None  # type: ignore[assignment]


class PostgresMemory:
    """Async PostgreSQL client for inference_db runs and strategies."""

    def __init__(self) -> None:
        self.host = settings.POSTGRES_HOST
        self.port = settings.POSTGRES_PORT
        self.database = settings.POSTGRES_DB
        self.user = settings.POSTGRES_USER
        self.password = settings.POSTGRES_PASSWORD
        self._pool: Any = None
        self._is_available: bool | None = None

    async def _get_pool(self) -> Any:
        if not HAS_ASYNCPG:
            return None
        if self._pool is not None:
            return self._pool
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

    async def close(self) -> None:
        if self._pool is not None:
            try:
                await self._pool.close()
            except Exception:
                pass
            self._pool = None


# Process-wide singleton
postgres_memory = PostgresMemory()
