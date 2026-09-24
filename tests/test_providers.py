"""Unit tests for providers and model configurations."""

import pytest
from app.providers.gemini import GeminiProvider
from app.providers.groq import GroqProvider
from app.providers.nvidia import NvidiaProvider
from app.providers.openrouter import OpenRouterProvider
from app.providers.unified_manager import unified_provider_manager
from app.agents.roles import get_all_specialist_agents
from app.agents.software_specialists import get_software_specialist_agents


def test_gemini_defaults():
    provider = GeminiProvider()
    assert provider.default_model == "gemini-3.8-flash"
    assert "gemini-3.8-flash" in provider.SUPPORTED_MODELS
    assert "gemini-3.6-flash" in provider.SUPPORTED_MODELS


def test_groq_defaults():
    provider = GroqProvider()
    assert provider.default_model == "openai/gpt-oss-120b"
    assert "openai/gpt-oss-120b" in provider.supported_models
    assert "qwen/qwen3.8-27b" in provider.supported_models


def test_nvidia_defaults():
    provider = NvidiaProvider()
    assert provider.default_model == "nvidia/nemotron-3-super-120b-a12b"
    assert "nvidia/nemotron-3-super-120b-a12b" in provider.supported_models


def test_openrouter_defaults():
    provider = OpenRouterProvider()
    assert provider.default_model == "liquid/lfm-2.5-2.6b:free"
    assert "liquid/lfm-2.5-2.6b:free" in provider.supported_models


def test_unified_manager_roles():
    mapping = unified_provider_manager.ROLE_PROVIDER_MAPPING
    assert mapping["requirements_analyst"] == ("gemini", "gemini-3.8-flash")
    assert mapping["code_reviewer"] == ("gemini", "gemini-3.8-flash")
    assert mapping["test_generator"] == ("gemini", "gemini-3.8-flash")
    assert mapping["documentation_writer"] == ("gemini", "gemini-3.8-flash")
    assert mapping["researcher"] == ("gemini", "gemini-3.8-flash")
    assert mapping["system_architect"] == ("nvidia", "nvidia/nemotron-3-super-120b-a12b")
    assert mapping["trading_analyst"] == ("groq", "openai/gpt-oss-120b")


def test_all_specialist_agents_configured():
    specialists = get_all_specialist_agents()
    assert len(specialists) == 12
    researcher = next(a for a in specialists if a.id == "researcher")
    assert researcher.model_name == "gemini-3.8-flash"

    forge_specialists = get_software_specialist_agents()
    req_analyst = next(a for a in forge_specialists if a.id == "requirements_analyst")
    assert req_analyst.model_name == "gemini-3.8-flash"
