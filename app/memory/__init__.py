"""Persistent memory package.

Import ``create_memory_store`` from :mod:`app.memory.factory` to obtain the configured
backend (Oracle Autonomous Database or SQLite). ``SQLiteMemory`` remains importable from
:mod:`app.memory.sqlite` for tests and explicit local use.

The factory is re-exported here for convenience; the store implementations are imported
lazily inside the factory so that importing this package stays cheap and works even when the
Oracle driver is not installed.
"""

from app.memory.factory import (
    DatabaseConfigurationError,
    create_memory_store,
    describe_memory_backend,
    get_memory_store,
    reset_memory_store,
    task_mode_column,
)

__all__ = [
    "DatabaseConfigurationError",
    "create_memory_store",
    "describe_memory_backend",
    "get_memory_store",
    "reset_memory_store",
    "task_mode_column",
]
