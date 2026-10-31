"""Shared fixtures for the Inference test suite.

The API is fail-closed: every route that executes models, spends budget or
exposes audit data requires `X-INFERENCE-API-KEY` (see
`app.core.security.require_inference_api_key`). Tests that exercise those routes
authenticate through the `auth` fixture, which points the server at a known
non-secret test credential and returns the matching request headers.
"""

import os

# Tests never open a live database. A placeholder Oracle URL keeps the fail-closed
# settings validator satisfied; individual tests monkeypatch settings or pass explicit
# test paths to SQLiteMemory.
os.environ.setdefault(
    "DATABASE_URL",
    "oracle+oracledb://INFERENCE:placeholder@adb.example.com:1522/?service_name=TEST",
)

import pytest

from app.core.config import settings

# Not a credential: it exists only inside the test process.
TEST_API_KEY = "test_inference_api_key_not_a_secret"


class _ProbeCursor:
    async def fetchone(self):
        return (1,)

    async def fetchall(self):
        return [(1,)]


class _ProbeConnection:
    async def execute(self, sql, params=None):
        return _ProbeCursor()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _ProbeMemory:
    """Minimal healthy store for app-level probes; no database is contacted."""

    backend_name = "oracle"

    async def initialize(self):
        return None

    async def close(self):
        return None

    def connect(self):
        return _ProbeConnection()

    def __getattr__(self, name: str):
        """Any other store call is a no-op: probes must not need a live database."""
        async def _noop(*args, **kwargs):
            return None

        return _noop


@pytest.fixture(autouse=True)
def healthy_placeholder_database(monkeypatch):
    """Keep app-level probes independent of a live database.

    The process is configured with a non-routable placeholder Oracle DSN so the
    fail-closed settings validator is satisfied. Tests that build their own store or
    patch ``connect`` are untouched.
    """
    from app.core import orchestrator as orchestrator_module

    memory = getattr(orchestrator_module.orchestrator, "memory", None)
    if memory is not None and "adb.example.com" in getattr(memory, "database_url", ""):
        monkeypatch.setattr(orchestrator_module.orchestrator, "memory", _ProbeMemory())


@pytest.fixture(autouse=True)
def default_server_key(monkeypatch):
    """Ensure the test server has a configured key by default so routes can authenticate."""
    monkeypatch.setattr(settings, "INFERENCE_API_KEY", TEST_API_KEY)
    monkeypatch.setattr(settings, "INSECURE_DEV_AUTH", False)


@pytest.fixture
def auth(monkeypatch):
    """Configure a server-side key and yield the headers a caller must send."""
    monkeypatch.setattr(settings, "INFERENCE_API_KEY", TEST_API_KEY)
    monkeypatch.setattr(settings, "inference_api_KEY", None)
    monkeypatch.setattr(settings, "FRIDAY_UNIVERSE_API_KEY", None)
    monkeypatch.setattr(settings, "X_FRIDAY_API_KEY", None)
    monkeypatch.setattr(settings, "FRIDAY_API_KEY", None)
    monkeypatch.setattr(settings, "INSECURE_DEV_AUTH", False)
    return {"X-INFERENCE-API-KEY": TEST_API_KEY}


@pytest.fixture
def no_server_key(monkeypatch):
    """Simulate a server with no configured credential (fail-closed 503 path)."""
    monkeypatch.setattr(settings, "INFERENCE_API_KEY", None)
    monkeypatch.setattr(settings, "inference_api_KEY", None)
    monkeypatch.setattr(settings, "FRIDAY_UNIVERSE_API_KEY", None)
    monkeypatch.setattr(settings, "X_FRIDAY_API_KEY", None)
    monkeypatch.setattr(settings, "FRIDAY_API_KEY", None)
    monkeypatch.setattr(settings, "INSECURE_DEV_AUTH", False)
    return None
