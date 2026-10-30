"""Memory-store factory: Oracle Autonomous Database when configured, SQLite otherwise.

Callers must not choose a backend themselves. ``create_memory_store()`` reads
``DATABASE_URL`` (or an explicit URL) and returns:

* :class:`app.memory.oracle.OracleMemory` when the URL starts with ``oracle`` or when the
  URL is empty but the ``ORACLE_*`` settings are complete;
* :class:`app.memory.sqlite.SQLiteMemory` only when a ``sqlite`` URL is explicitly configured;
* a hard :class:`DatabaseConfigurationError` when nothing configures a database.

The last case is deliberate: the previous default (``sqlite:///data/universe.db``) meant a
process with no database configuration silently served from a local file that lost every
audit record on redeploy. Failing closed at startup is the cheaper failure.
"""

from __future__ import annotations

import threading
from typing import Any

from app.core.config import UNCONFIGURED_DATABASE_MESSAGE, settings
from app.memory.base import BaseMemory

__all__ = [
    "DatabaseConfigurationError",
    "create_memory_store",
    "describe_memory_backend",
    "get_memory_store",
    "reset_memory_store",
    "task_mode_column",
]


class DatabaseConfigurationError(RuntimeError):
    """No usable database backend is configured (or the configured one is unsupported)."""


def _configured_url(database_url: str | None) -> str:
    return (database_url or "").strip() or (settings.DATABASE_URL or "").strip()


def backend_for(database_url: str | None = None) -> str:
    """Return ``"oracle"``, ``"sqlite"`` or ``"unconfigured"`` for the given URL."""
    raw = _configured_url(database_url)
    lowered = raw.lower()
    if lowered.startswith("oracle"):
        return "oracle"
    if lowered.startswith("sqlite"):
        return "sqlite"
    if not raw and settings.oracle_configured:
        return "oracle"
    return "unconfigured"


def create_memory_store(database_url: str | None = None) -> BaseMemory:
    """Build the memory store for the configured database.

    ``database_url`` overrides ``settings.DATABASE_URL`` (same precedence as
    ``SQLiteMemory(db_path)``). Raises :class:`DatabaseConfigurationError` when no database is
    configured, so a misconfigured deployment stops instead of writing to an implicit file.
    """
    raw = _configured_url(database_url)
    backend = backend_for(raw)

    if backend == "oracle":
        from app.memory.oracle import OracleMemory

        return OracleMemory(raw or None)
    if backend == "sqlite":
        from app.memory.sqlite import SQLiteMemory

        return SQLiteMemory(raw)

    scheme = raw.split(":", 1)[0] if raw else ""
    if scheme:
        raise DatabaseConfigurationError(
            f"Unsupported memory database URL scheme '{scheme}'. Configure an "
            "'oracle+oracledb://' or 'sqlite:///' DATABASE_URL."
        )
    raise DatabaseConfigurationError(UNCONFIGURED_DATABASE_MESSAGE)


def describe_memory_backend(database_url: str | None = None) -> str:
    """Describe the active backend for status endpoints (never raises).

    Mirrors :meth:`app.core.config.Settings.memory_backend_description` but honours an
    explicit URL, so a test or a CLI override is reported truthfully.
    """
    backend = backend_for(database_url)
    if backend == "oracle":
        raw = _configured_url(database_url)
        target = raw or f"{settings.ORACLE_SERVICE} on {settings.ORACLE_HOST}:{settings.ORACLE_PORT}"
        if raw and "@" in raw:
            target = raw.split("@", 1)[1]
        return f"Oracle Autonomous Database 26ai ({target})"
    if backend == "sqlite":
        return f"SQLite ({_configured_url(database_url)} with WAL mode)"
    return "unconfigured (no DATABASE_URL and no Oracle settings)"


# ---------------------------------------------------------------------------
# Process-wide store for callers that cannot own a lifecycle
# ---------------------------------------------------------------------------
_store: BaseMemory | None = None
_store_lock = threading.Lock()


def get_memory_store() -> BaseMemory:
    """Return a lazily-created process-wide store.

    For fire-and-forget writers (outcome telemetry) that used to build a throwaway
    ``SQLiteMemory`` per record. Building a store per record would open a new Oracle pool per
    call, so the shared instance is created once. The store is not closed by callers; it lives
    for the process, exactly like the module-level orchestrator.
    """
    global _store
    store = _store
    if store is None:
        with _store_lock:
            if _store is None:
                _store = create_memory_store()
            store = _store
    return store


def reset_memory_store(store: BaseMemory | None = None) -> None:
    """Replace (or clear) the process-wide store. Test hook; not used by the application."""
    global _store
    with _store_lock:
        _store = store


def task_mode_column(memory: Any) -> str:
    """Name of the ``tasks`` column holding the task mode for this backend.

    Oracle reserves ``MODE`` as an identifier, so the Oracle schema stores it as
    ``task_mode`` while SQLite keeps ``mode``. Raw-SQL readers must ask instead of hardcoding
    one name; the value returned is always a controlled identifier, never caller input.
    """
    return "task_mode" if getattr(memory, "backend_name", "sqlite") == "oracle" else "mode"
