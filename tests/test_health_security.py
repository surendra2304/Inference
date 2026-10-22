"""Regression tests for health counts and API-key configuration."""

import asyncio

from app.agents.registry import agent_registry
from app.core.config import settings
from app.health import basic_health, detailed_health, readiness_check
from app.security.api_security import APISecurityManager


def test_health_counts_follow_agent_registry(monkeypatch):
    agents = [object() for _ in range(19)]
    monkeypatch.setattr(agent_registry, "list_agents", lambda: agents)

    basic = asyncio.run(basic_health())
    ready = asyncio.run(readiness_check())
    detailed = asyncio.run(detailed_health())

    assert basic["active_specialist_agents"] == 19
    assert ready["active_specialist_agents"] == 19
    assert ready["ready"] is True
    assert detailed["active_specialist_agents"] == 19


def test_health_readiness_reports_incomplete_registry_below_minimum(monkeypatch):
    monkeypatch.setattr(agent_registry, "list_agents", lambda: [object() for _ in range(9)])

    ready = asyncio.run(readiness_check())

    assert ready["active_specialist_agents"] == 9
    assert ready["ready"] is False
    # Status labels the actual evidence: registry configuration only, not
    # end-to-end provider/memory/database readiness.
    assert ready["status"] == "registry_incomplete"
    assert ready["evidence_class"] == "agent_registry_configuration"


def test_security_manager_loads_only_configured_keys(monkeypatch):
    for field in (
        "INFERENCE_API_KEY",
        "inference_api_KEY",
        "FRIDAY_UNIVERSE_API_KEY",
        "X_FRIDAY_API_KEY",
        "FRIDAY_API_KEY",
    ):
        monkeypatch.setattr(settings, field, None)

    assert APISecurityManager()._valid_api_keys == set()


def test_security_manager_accepts_configured_bearer_key(monkeypatch):
    monkeypatch.setattr(settings, "INFERENCE_API_KEY", "configured-test-key")
    manager = APISecurityManager()

    assert manager.validate_api_key("configured-test-key") is True
    assert manager.validate_api_key("Bearer configured-test-key") is True
    assert manager.validate_api_key("different-key") is False


def test_security_manager_supports_explicit_test_key_injection():
    manager = APISecurityManager(valid_api_keys={"injected-test-key"})

    assert manager.validate_api_key("injected-test-key") is True
    assert manager.validate_api_key(None) is False
