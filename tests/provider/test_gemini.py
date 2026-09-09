"""Unit and mock tests for Google Gemini Provider adapter."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.providers import get_provider
from app.providers.base import ProviderMessage, ProviderRequest
from app.providers.gemini import GeminiProvider


@pytest.fixture
def gemini_provider():
    return GeminiProvider(api_key="test_dummy_key_12345", timeout=5.0)


def test_gemini_capabilities(gemini_provider):
    caps = gemini_provider.capabilities()
    assert caps.provider_name == "gemini"
    assert "gemini-3.7-flash" in caps.supported_models
    assert caps.supports_streaming is True
    assert caps.supports_structured_output is True


def test_gemini_estimate_usage(gemini_provider):
    req = ProviderRequest(
        messages=[ProviderMessage(role="user", content="Explain quantum computing in 50 words.")],
        system_instruction="You are a physics professor."
    )
    estimate = gemini_provider.estimate_usage(req)
    assert estimate.estimated_prompt_tokens > 0
    assert estimate.estimated_completion_tokens > 0
    assert estimate.estimated_cost_usd >= 0.0


def test_gemini_factory():
    provider = get_provider("gemini")
    assert isinstance(provider, GeminiProvider)
    assert provider.provider_name == "gemini"

    with pytest.raises(ValueError, match="Unsupported provider"):
        get_provider("unknown_provider")


@pytest.mark.asyncio
async def test_gemini_generate_success(gemini_provider):
    mock_response_data = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "Quantum computing uses qubits and superposition."}],
                    "role": "model"
                },
                "finishReason": "STOP"
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 15,
            "candidatesTokenCount": 8,
            "totalTokenCount": 23
        }
    }

    req = ProviderRequest(
        messages=[ProviderMessage(role="user", content="What is quantum computing?")]
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_response_data
        mock_post.return_value = mock_resp

        result = await gemini_provider.generate(req)

        assert result.content == "Quantum computing uses qubits and superposition."
        assert result.prompt_tokens == 15
        assert result.completion_tokens == 8
        assert result.total_tokens == 23
        assert result.model == gemini_provider.default_model
        assert result.provider == "gemini"


@pytest.mark.asyncio
async def test_gemini_generate_rate_limit_429(gemini_provider):
    req = ProviderRequest(
        messages=[ProviderMessage(role="user", content="Test message")]
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.text = "Resource exhausted"
        mock_post.return_value = mock_resp

        with pytest.raises(RuntimeError, match="rate limit exceeded"):
            await gemini_provider.generate(req)


@pytest.mark.asyncio
async def test_gemini_generate_timeout_handling(gemini_provider):
    req = ProviderRequest(
        messages=[ProviderMessage(role="user", content="Test timeout")]
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = httpx.TimeoutException("Read timed out")

        with pytest.raises(TimeoutError, match="timed out"):
            await gemini_provider.generate(req)


@pytest.mark.asyncio
async def test_gemini_health_check(gemini_provider):
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp

        healthy = await gemini_provider.health()
        assert healthy is True

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = httpx.RequestError("Network error")

        healthy = await gemini_provider.health()
        assert healthy is False
