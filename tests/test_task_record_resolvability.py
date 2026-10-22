"""Every task id the API hands out must be resolvable, and the cache must stay bounded.

Two live defects drove this file.

**#56 — unbounded recent-task cache.** ``Orchestrator._recent_tasks`` was a plain dict
written on every task and never pruned. Measured in-process: 400 ``POST /ask`` requests
left 400 entries and +10.9 MB RSS (~27.9 KB per task record). With the bound (1024
entries) the same driver showed 800 requests -> +14.4 MB, a further 800 -> +1.7 MB, and
``len(_recent_tasks) == 1024`` exactly: the per-task retention term is gone.

**#57 — an unresolvable task id.** ``POST /v1/debate`` (``app/api/v1_core_routes.py``)
drives the collaboration engine directly instead of ``Orchestrator.process_task``, so it
never persisted or registered anything. Measured against the running service:
``POST /v1/debate`` answered 200 with ``task_53017bc07162`` and
``GET /tasks/task_53017bc07162`` answered 404 "Task ... not found." The route now calls
``orchestrator.record_task`` on each of its three return paths.
"""


import pytest
from fastapi.testclient import TestClient

from app.core.orchestrator import RECENT_TASK_CACHE, Orchestrator
from app.memory.base import TaskRecord
from app.memory.sqlite import SQLiteMemory


def _record(i: int) -> TaskRecord:
    return TaskRecord(id=f"task_{i:05d}", question=f"question {i}", mode="fast", status="completed")


@pytest.mark.asyncio
async def test_recent_task_cache_is_bounded_and_says_so(tmp_path):
    """The cache caps, counts evictions, and dates the miss instead of hiding it."""
    orch = Orchestrator(memory=SQLiteMemory(str(tmp_path / "t.db")))
    await orch.memory.initialize()

    assert RECENT_TASK_CACHE <= 1024, "a lookup cache larger than this is a memory leak in slow motion"
    for i in range(RECENT_TASK_CACHE + 50):
        await orch.record_task(_record(i))

    assert len(orch._recent_tasks) == RECENT_TASK_CACHE
    described = orch.describe_recent_task_cache()
    assert described["max_entries"] == RECENT_TASK_CACHE
    assert described["evicted"] == 50

    # The oldest id was evicted from the cache but is still resolvable from SQLite...
    status = await orch.get_task_status("task_00000")
    assert status is not None, "an evicted-but-persisted task must still be readable"
    assert status["question"] == "question 0"

    # ...and a genuinely unknown id is described as such, not as an eviction.
    assert await orch.get_task_status("task_99999") is None
    detail = orch.recent_task_cache_miss_detail("task_99999")
    assert "No task record was recorded" in detail or "never" in detail.lower()

    # The evicted id is *known* to have been dropped, so the wording differs.
    evicted_detail = orch.recent_task_cache_miss_detail("task_00000")
    assert evicted_detail != detail, "eviction and absence must not read the same"
    await orch.memory.close()


@pytest.mark.asyncio
async def test_a_task_answered_by_the_collaboration_engine_is_persisted(tmp_path):
    """``record_task`` is the bridge for routes that bypass ``process_task``."""
    mem = SQLiteMemory(str(tmp_path / "t.db"))
    await mem.initialize()
    orch = Orchestrator(memory=mem)

    await orch.record_task(_record(7))
    # A fresh orchestrator over the same store must still resolve it: this is what makes
    # the id survive a restart, which the cache alone never did.
    other = Orchestrator(memory=mem)
    assert "task_00007" not in other._recent_tasks
    assert (await other.get_task_status("task_00007"))["question"] == "question 7"
    await mem.close()


def test_debate_endpoint_task_id_is_readable_back(auth):
    """End-to-end: the id from the answer is the id the read-back route accepts.

    This is the exact sequence that failed live: POST /v1/debate -> 200 task_53017bc07162,
    GET /tasks/task_53017bc07162 -> 404.
    """
    from app.main import app

    client = TestClient(app)
    resp = client.post(
        "/v1/debate",
        headers=auth,
        json={"topic": "Does WAL mode change read visibility across processes?"},
    )
    assert resp.status_code == 200, resp.text
    task_id = resp.json()["task_id"]
    assert task_id

    read_back = client.get(f"/tasks/{task_id}", headers=auth)
    assert read_back.status_code == 200, (
        f"the API handed out {task_id!r} and then could not find it: {read_back.text}"
    )
    body = read_back.json()
    assert body["id"] == task_id
    assert body["question"], "the persisted record must carry the question that was asked"
