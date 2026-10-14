"""Verification of OpenAI-compatible gateway endpoints (/v1/chat/completions, /v1/models, /stream)."""

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_list_models(auth):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver", headers=auth) as client:
        res = await client.get("/v1/models")
    assert res.status_code == 200
    data = res.json()
    assert data["object"] == "list"
    assert len(data["data"]) >= 5
    model_ids = [m["id"] for m in data["data"]]
    assert "gemini-3.8-flash" in model_ids
    assert "openai/gpt-oss-120b" in model_ids


@pytest.mark.asyncio
async def test_chat_completions_non_streaming(auth, monkeypatch):
    # Mock unified_provider_manager execution
    from app.providers.unified_manager import UnifiedExecutionResponse, unified_provider_manager

    async def mock_execute(req):
        return UnifiedExecutionResponse(
            content="Hello from test model!",
            provider_used="groq",
            model_used="openai/gpt-oss-120b",
            agent_role="system_architect",
            latency_ms=120.0,
            timestamp=1729000000.0,
            token_usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            finish_reason="stop",
        )

    monkeypatch.setattr(unified_provider_manager, "execute", mock_execute)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver", headers=auth) as client:
        res = await client.post(
            "/v1/chat/completions",
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False,
            },
        )
    assert res.status_code == 200
    payload = res.json()
    assert payload["object"] == "chat.completion"
    assert payload["choices"][0]["message"]["content"] == "Hello from test model!"
    assert payload["choices"][0]["message"]["role"] == "assistant"
    assert payload["usage"]["total_tokens"] == 15


@pytest.mark.asyncio
async def test_chat_completions_streaming(auth, monkeypatch):
    from app.providers.gateway import ModelGateway

    async def mock_stream(*args, **kwargs):
        yield "Hello"
        yield " world"

    monkeypatch.setattr(ModelGateway, "stream", mock_stream)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver", headers=auth) as client:
        res = await client.post(
            "/v1/chat/completions",
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
        )
    assert res.status_code == 200
    assert "text/event-stream" in res.headers["content-type"]
    assert "data: [DONE]" in res.text
    assert "Hello" in res.text


@pytest.mark.asyncio
async def test_top_level_stream(auth, monkeypatch):
    from app.providers.gateway import ModelGateway

    async def mock_stream(*args, **kwargs):
        yield "Token 1"
        yield " Token 2"

    monkeypatch.setattr(ModelGateway, "stream", mock_stream)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver", headers=auth) as client:
        res = await client.post("/stream", json={"prompt": "Stream test"})
    assert res.status_code == 200
    assert "text/event-stream" in res.headers["content-type"]
    assert "Token 1" in res.text
    assert '"done": true' in res.text.lower()
