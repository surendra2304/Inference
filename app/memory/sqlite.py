"""SQLite persistent memory store implementation using aiosqlite."""

import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
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

    @asynccontextmanager
    async def connect(self) -> AsyncIterator[aiosqlite.Connection]:
        """Provides an active aiosqlite connection with Row factory."""
        if self.db_path == ":memory:":
            if self._memory_conn is None:
                self._memory_conn = await aiosqlite.connect(":memory:")
                self._memory_conn.row_factory = aiosqlite.Row
            yield self._memory_conn
        else:
            db_dir = os.path.dirname(os.path.abspath(self.db_path))
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)
            async with aiosqlite.connect(self.db_path, timeout=30.0) as conn:
                conn.row_factory = aiosqlite.Row
                await conn.execute("PRAGMA journal_mode=WAL;")
                await conn.execute("PRAGMA busy_timeout=30000;")
                yield conn

    async def close(self) -> None:
        """Close any persistent connection if held."""
        if self._memory_conn:
            await self._memory_conn.close()
            self._memory_conn = None

    async def initialize(self) -> None:
        """Create tables and indexes if they do not already exist."""
        async with self.connect() as db:
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
                    recommended_model TEXT NOT NULL DEFAULT 'gemini-3.5-flash-lite',
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

    async def save_agent(self, agent_data: dict[str, Any]) -> None:
        """Persist or update an agent configuration record."""
        agent_id = agent_data.get("id")
        name = agent_data.get("name", "")
        role = agent_data.get("role", "")
        provider = agent_data.get("model_provider", "gemini")
        model = agent_data.get("model_name", "gemini-3.5-flash-lite")
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

                created_at = datetime.fromisoformat(row["created_at"])
                completed_at = datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None
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
                    created_at=datetime.fromisoformat(row["created_at"])
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
                        created_at=datetime.fromisoformat(row["created_at"])
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
                        created_at=datetime.fromisoformat(row["created_at"])
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
                    created_at=datetime.fromisoformat(row["created_at"]),
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
                        created_at=datetime.fromisoformat(row["created_at"]),
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
                    created_at=datetime.fromisoformat(row["created_at"])
                )
