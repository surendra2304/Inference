"""H8: every timestamp must be timezone-aware UTC.

The orchestrator stamped `completed_at` with datetime.now(timezone.utc) while
memory records defaulted `created_at` to datetime.utcnow() - naive. Comparing
or subtracting the two raises:

    TypeError: can't compare offset-naive and offset-aware datetimes
    TypeError: can't subtract offset-naive and offset-aware datetimes

which is exactly what any consumer computing task duration would hit.
"""

import ast
import pathlib
from datetime import datetime, timezone

import pytest

from app.memory.base import RunRecord, TaskRecord
from app.memory.sqlite import SQLiteMemory


@pytest.fixture
async def memory(tmp_path):
    m = SQLiteMemory(db_path=str(tmp_path / "h8.db"))
    await m.initialize()
    yield m


def test_task_record_timestamps_are_both_aware():
    task = TaskRecord(id="t1", question="q")
    assert task.created_at.tzinfo is not None, "created_at must not be naive"
    task.completed_at = datetime.now(timezone.utc)
    assert task.completed_at.tzinfo is not None

    # The exact operations that used to raise TypeError.
    assert task.created_at <= task.completed_at
    assert (task.completed_at - task.created_at).total_seconds() >= 0.0


def test_run_and_message_records_default_to_aware():
    run = RunRecord(id="r1", task_id="t1", agent_id="a", provider="p", model="m", stage="s")
    assert run.created_at.tzinfo is not None
    assert run.created_at.utcoffset().total_seconds() == 0, "must be UTC"


async def test_round_trip_through_sqlite_preserves_awareness(memory):
    task = TaskRecord(id="rt1", question="q", status="completed")
    task.completed_at = datetime.now(timezone.utc)
    await memory.save_task(task)

    loaded = await memory.get_task("rt1")
    assert loaded is not None
    assert loaded.created_at.tzinfo is not None, "created_at came back naive"
    assert loaded.completed_at is not None and loaded.completed_at.tzinfo is not None
    # Subtraction across the round trip must not raise.
    assert (loaded.completed_at - loaded.created_at).total_seconds() >= 0.0


async def test_legacy_naive_rows_are_normalised_on_read(memory):
    """Rows written before the fix carry no offset and were UTC."""
    legacy_id = "legacy1"
    naive_iso = "2025-01-02T03:04:05.123456"  # no offset, as datetime.utcnow() wrote it
    async with memory.connect() as db:
        await db.execute(
            """INSERT INTO tasks
               (id, question, mode, status, result, confidence, created_at, completed_at, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (legacy_id, "legacy question", "auto", "completed", None, None, naive_iso, naive_iso, "{}"),
        )
        await db.commit()

    loaded = await memory.get_task(legacy_id)
    assert loaded is not None
    assert loaded.created_at.tzinfo is not None, "legacy naive timestamp not normalised"
    assert loaded.created_at.utcoffset().total_seconds() == 0


async def test_legacy_row_comparable_with_new_aware_task(memory):
    """The TypeError scenario: naive created_at next to an aware completed_at."""
    legacy_id = "legacy2"
    async with memory.connect() as db:
        await db.execute(
            """INSERT INTO tasks
               (id, question, mode, status, result, confidence, created_at, completed_at, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (legacy_id, "q", "auto", "running", None, None,
             "2025-01-02T03:04:05.123456", None, "{}"),
        )
        await db.commit()

    legacy = await memory.get_task(legacy_id)
    assert legacy is not None
    legacy.completed_at = datetime.now(timezone.utc)

    fresh = TaskRecord(id="fresh1", question="q")
    fresh.completed_at = datetime.now(timezone.utc)

    # Mixed-age records must be comparable with each other and themselves.
    assert legacy.created_at <= legacy.completed_at
    assert (legacy.completed_at - legacy.created_at).total_seconds() >= 0.0
    assert sorted([legacy, fresh], key=lambda r: r.created_at)


def test_no_naive_utcnow_left_in_application_code():
    """Guard against reintroducing datetime.utcnow(), which returns naive times.

    Parsed with ast so the check matches real calls only - comments and
    docstrings that merely mention the API do not count.
    """
    app_root = pathlib.Path(__file__).resolve().parents[1] / "app"
    offenders = []
    for path in app_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "utcnow"
            ):
                offenders.append(f"{path.relative_to(app_root)}:{node.lineno}")
    assert not offenders, (
        "datetime.utcnow() returns a naive datetime and breaks comparisons "
        f"against datetime.now(timezone.utc): {offenders}"
    )
