"""Shared fixtures for the Inference test suite.

The API is fail-closed: every route that executes models, spends budget or
exposes audit data requires `X-INFERENCE-API-KEY` (see
`app.core.security.require_inference_api_key`). Tests that exercise those routes
authenticate through the `auth` fixture, which points the server at a known
non-secret test credential and returns the matching request headers.
"""

import pytest

from app.core.config import settings

# Not a credential: it exists only inside the test process.
TEST_API_KEY = "test_inference_api_key_not_a_secret"


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
