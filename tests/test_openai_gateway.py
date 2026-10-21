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
@pytest.mark.parametrize(
    ("failure", "error_code"),
    [
        (pytest.param("timeout", "provider_timeout", id="timeout")),
        (pytest.param("provider", "provider_error", id="provider-error")),
    ],
)
async def test_chat_completions_streaming_reports_failure_without_success(auth, monkeypatch, failure, error_code):
    from app.providers.errors import GatewayError
    from app.providers.errors import TimeoutError as ProviderTimeoutError
    from app.providers.gateway import ModelGateway

    async def mock_stream(*args, **kwargs):
        error = ProviderTimeoutError("secret provider detail") if failure == "timeout" else GatewayError("secret provider detail")
        raise error
        yield "unreachable"

    monkeypatch.setattr(ModelGateway, "stream", mock_stream)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver", headers=auth) as client:
        res = await client.post(
            "/v1/chat/completions",
            json={"model": "auto", "messages": [{"role": "user", "content": "Hello"}], "stream": True},
        )

    assert res.status_code == 200
    assert f'"code": "{error_code}"' in res.text
    assert "secret provider detail" not in res.text
    assert '"finish_reason": "stop"' not in res.text
    assert "data: [DONE]" in res.text


@pytest.mark.asyncio
async def test_chat_completions_streaming_midstream_provider_error_preserves_partial_content(monkeypatch):
    from app.api.v1_core_routes import ChatCompletionRequest, chat_completions
    from app.providers.errors import GatewayError
    from app.providers.gateway import ModelGateway

    async def mock_stream(*args, **kwargs):
        yield "partial answer"
        raise GatewayError("secret provider detail")

    monkeypatch.setattr(ModelGateway, "stream", mock_stream)
    response = await chat_completions(
        ChatCompletionRequest(model="auto", messages=[{"role": "user", "content": "Hello"}], stream=True)
    )
    events = [event async for event in response.body_iterator]

    assert any('"content": "partial answer"' in event for event in events)
    assert any('"code": "provider_error"' in event for event in events)
    assert all("secret provider detail" not in event for event in events)
    assert all('"finish_reason": "stop"' not in event for event in events)
    assert events[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_chat_completions_streaming_midstream_timeout_preserves_partial_content(monkeypatch):
    from app.api.v1_core_routes import ChatCompletionRequest, chat_completions
    from app.providers.errors import TimeoutError as ProviderTimeoutError
    from app.providers.gateway import ModelGateway

    async def mock_stream(*args, **kwargs):
        yield "partial answer"
        raise ProviderTimeoutError("secret provider detail")

    monkeypatch.setattr(ModelGateway, "stream", mock_stream)
    response = await chat_completions(
        ChatCompletionRequest(model="auto", messages=[{"role": "user", "content": "Hello"}], stream=True)
    )
    events = [event async for event in response.body_iterator]

    assert any('"content": "partial answer"' in event for event in events)
    assert any('"code": "provider_timeout"' in event for event in events)
    assert all("secret provider detail" not in event for event in events)
    assert all('"finish_reason": "stop"' not in event for event in events)
    assert events[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_chat_completions_streaming_preserves_cancellation(monkeypatch):
    import asyncio

    from app.api.v1_core_routes import ChatCompletionRequest, chat_completions
    from app.providers.gateway import ModelGateway

    async def mock_stream(*args, **kwargs):
        raise asyncio.CancelledError
        yield "unreachable"

    monkeypatch.setattr(ModelGateway, "stream", mock_stream)
    response = await chat_completions(
        ChatCompletionRequest(model="auto", messages=[{"role": "user", "content": "Hello"}], stream=True)
    )

    with pytest.raises(asyncio.CancelledError):
        async for _event in response.body_iterator:
            pass


@pytest.mark.asyncio
async def test_chat_completions_streaming_midstream_cancellation_does_not_emit_success(monkeypatch):
    import asyncio

    from app.api.v1_core_routes import ChatCompletionRequest, chat_completions
    from app.providers.gateway import ModelGateway

    async def mock_stream(*args, **kwargs):
        yield "partial answer"
        raise asyncio.CancelledError

    monkeypatch.setattr(ModelGateway, "stream", mock_stream)
    response = await chat_completions(
        ChatCompletionRequest(model="auto", messages=[{"role": "user", "content": "Hello"}], stream=True)
    )
    events = []

    with pytest.raises(asyncio.CancelledError):
        async for event in response.body_iterator:
            events.append(event)

    assert any('"content": "partial answer"' in event for event in events)
    assert all('"finish_reason": "stop"' not in event for event in events)
    assert all("data: [DONE]" not in event for event in events)


@pytest.mark.asyncio
async def test_friday_stream_error_event_is_safe_and_typed(monkeypatch):
    from app.api.friday_routes import FridayRequest, friday_stream
    from app.providers.errors import TimeoutError as ProviderTimeoutError
    from app.providers.gateway import ModelGateway

    async def mock_stream(*args, **kwargs):
        raise ProviderTimeoutError("secret provider detail")
        yield "unreachable"

    monkeypatch.setattr(ModelGateway, "stream", mock_stream)
    response = await friday_stream(FridayRequest(question="Test stream"))
    events = [event async for event in response.body_iterator]
    payload = events[-1]

    assert '"code": "provider_timeout"' in payload
    assert '"done": true' in payload
    assert "secret provider detail" not in payload


@pytest.mark.asyncio
async def test_friday_stream_preserves_cancellation_without_terminal_event(monkeypatch):
    import asyncio

    from app.api.friday_routes import FridayRequest, friday_stream
    from app.providers.gateway import ModelGateway

    async def mock_stream(*args, **kwargs):
        raise asyncio.CancelledError
        yield "unreachable"

    monkeypatch.setattr(ModelGateway, "stream", mock_stream)
    response = await friday_stream(FridayRequest(question="Test stream"))
    events = []

    with pytest.raises(asyncio.CancelledError):
        async for event in response.body_iterator:
            events.append(event)

    assert any('"event": "start"' in event for event in events)
    assert all('"done": true' not in event and '"error"' not in event for event in events)


@pytest.mark.asyncio
async def test_friday_stream_midstream_cancellation_does_not_cache_partial_answer(monkeypatch):
    import asyncio

    from app.api.friday_routes import FridayRequest, friday_stream
    from app.performance_cache import perf_cache
    from app.providers.gateway import ModelGateway

    cache_writes = []

    async def mock_stream(*args, **kwargs):
        yield "partial answer"
        raise asyncio.CancelledError

    monkeypatch.setattr(ModelGateway, "stream", mock_stream)
    monkeypatch.setattr(perf_cache, "set_query", lambda **kwargs: cache_writes.append(kwargs))
    response = await friday_stream(FridayRequest(question="Midstream cancellation cache regression"))
    events = []

    with pytest.raises(asyncio.CancelledError):
        async for event in response.body_iterator:
            events.append(event)

    assert any('"token": "partial answer"' in event for event in events)
    assert all('"done": true' not in event and '"error"' not in event for event in events)
    assert cache_writes == []


@pytest.mark.asyncio
async def test_agent_stream_error_event_is_safe_and_typed(monkeypatch):
    from app.api.agent_routes import AgentAssistRequest, agent_stream_endpoint
    from app.providers.errors import GatewayError
    from app.providers.gateway import ModelGateway

    async def mock_stream(*args, **kwargs):
        raise GatewayError("secret provider detail")
        yield "unreachable"

    monkeypatch.setattr(ModelGateway, "stream", mock_stream)
    response = await agent_stream_endpoint(AgentAssistRequest(prompt="Test stream"))
    events = [event async for event in response.body_iterator]
    payload = events[-1]

    assert '"code": "provider_error"' in payload
    assert '"done": true' in payload
    assert "secret provider detail" not in payload


@pytest.mark.asyncio
async def test_agent_stream_preserves_cancellation_without_terminal_event(monkeypatch):
    import asyncio

    from app.api.agent_routes import AgentAssistRequest, agent_stream_endpoint
    from app.providers.gateway import ModelGateway

    async def mock_stream(*args, **kwargs):
        raise asyncio.CancelledError
        yield "unreachable"

    monkeypatch.setattr(ModelGateway, "stream", mock_stream)
    response = await agent_stream_endpoint(AgentAssistRequest(prompt="Test stream"))
    events = []

    with pytest.raises(asyncio.CancelledError):
        async for event in response.body_iterator:
            events.append(event)

    assert events == []


@pytest.mark.asyncio
async def test_agent_stream_midstream_cancellation_does_not_start_fallback(monkeypatch):
    import asyncio

    from app.api.agent_routes import AgentAssistRequest, agent_stream_endpoint
    from app.providers.gateway import ModelGateway

    providers = []

    async def mock_stream(_gateway, provider, *args, **kwargs):
        providers.append(provider)
        if provider == "groq":
            yield "partial answer"
            raise asyncio.CancelledError
        yield "unexpected fallback"

    monkeypatch.setattr(ModelGateway, "stream", mock_stream)
    response = await agent_stream_endpoint(AgentAssistRequest(prompt="Midstream cancellation fallback regression"))
    events = []

    with pytest.raises(asyncio.CancelledError):
        async for event in response.body_iterator:
            events.append(event)

    assert any('"token": "partial answer"' in event for event in events)
    assert all('"done": true' not in event and '"error"' not in event for event in events)
    assert providers == ["groq"]


@pytest.mark.asyncio
async def test_agent_stream_midstream_provider_error_does_not_duplicate_partial_response(monkeypatch):
    from app.api.agent_routes import AgentAssistRequest, agent_stream_endpoint
    from app.providers.errors import GatewayError
    from app.providers.gateway import ModelGateway

    providers = []

    async def mock_stream(_gateway, provider, *args, **kwargs):
        providers.append(provider)
        yield "partial answer"
        raise GatewayError("secret provider detail")

    monkeypatch.setattr(ModelGateway, "stream", mock_stream)
    response = await agent_stream_endpoint(AgentAssistRequest(prompt="Midstream provider error fallback regression"))
    events = [event async for event in response.body_iterator]

    assert any('"token": "partial answer"' in event for event in events)
    assert '"code": "provider_error"' in events[-1]
    assert '"done": true' in events[-1]
    assert "secret provider detail" not in "".join(events)
    assert all('"done": true' not in event for event in events[:-1])
    assert providers == ["groq"]


@pytest.mark.asyncio
async def test_agent_stream_falls_back_before_first_token(monkeypatch):
    from app.api.agent_routes import AgentAssistRequest, agent_stream_endpoint
    from app.providers.errors import GatewayError
    from app.providers.gateway import ModelGateway

    providers = []

    async def mock_stream(_gateway, provider, *args, **kwargs):
        providers.append(provider)
        if provider == "groq":
            raise GatewayError("secret provider detail")
            yield "unreachable"
        yield "fallback answer"

    monkeypatch.setattr(ModelGateway, "stream", mock_stream)
    response = await agent_stream_endpoint(AgentAssistRequest(prompt="Initial provider failure fallback regression"))
    events = [event async for event in response.body_iterator]

    assert any('"token": "fallback answer"' in event for event in events)
    terminal_events = [event for event in events if '"done": true' in event]
    assert terminal_events == ['data: {"token": "", "done": true}\n\n']
    assert all('"error"' not in event for event in events)
    assert providers == ["groq", "gemini"]


@pytest.mark.asyncio
async def test_agent_stream_fallback_failure_emits_single_safe_terminal_error(monkeypatch):
    from app.api.agent_routes import AgentAssistRequest, agent_stream_endpoint
    from app.providers.errors import GatewayError
    from app.providers.errors import TimeoutError as ProviderTimeoutError
    from app.providers.gateway import ModelGateway

    providers = []

    async def mock_stream(_gateway, provider, *args, **kwargs):
        providers.append(provider)
        if provider == "groq":
            raise GatewayError("secret initial provider detail")
            yield "unreachable"
        raise ProviderTimeoutError("secret fallback provider detail")
        yield "unreachable"

    monkeypatch.setattr(ModelGateway, "stream", mock_stream)
    response = await agent_stream_endpoint(AgentAssistRequest(prompt="Fallback terminal error regression"))
    events = [event async for event in response.body_iterator]

    assert not any('"token":' in event for event in events)
    terminal_events = [event for event in events if '"done": true' in event]
    assert terminal_events == [
        'data: {"error": "The provider could not complete the stream.", "code": "provider_timeout", "done": true}\n\n'
    ]
    assert "secret" not in "".join(events)
    assert providers == ["groq", "gemini"]


@pytest.mark.asyncio
async def test_agent_stream_fallback_provider_failure_emits_single_safe_terminal_error(monkeypatch):
    from app.api.agent_routes import AgentAssistRequest, agent_stream_endpoint
    from app.providers.errors import GatewayError
    from app.providers.gateway import ModelGateway

    providers = []

    async def mock_stream(_gateway, provider, *args, **kwargs):
        providers.append(provider)
        detail = "secret initial provider detail" if provider == "groq" else "secret fallback provider detail"
        raise GatewayError(detail)
        yield "unreachable"

    monkeypatch.setattr(ModelGateway, "stream", mock_stream)
    response = await agent_stream_endpoint(AgentAssistRequest(prompt="Fallback provider error regression"))
    events = [event async for event in response.body_iterator]

    assert not any('"token":' in event for event in events)
    terminal_events = [event for event in events if '"done": true' in event]
    assert terminal_events == [
        'data: {"error": "The provider could not complete the stream.", "code": "provider_error", "done": true}\n\n'
    ]
    assert "secret" not in "".join(events)
    assert providers == ["groq", "gemini"]


@pytest.mark.asyncio
async def test_agent_stream_fallback_provider_error_preserves_partial_content(monkeypatch):
    from app.api.agent_routes import AgentAssistRequest, agent_stream_endpoint
    from app.providers.errors import GatewayError
    from app.providers.gateway import ModelGateway

    providers = []

    async def mock_stream(_gateway, provider, *args, **kwargs):
        providers.append(provider)
        if provider == "groq":
            raise GatewayError("secret initial provider detail")
            yield "unreachable"
        yield "partial fallback answer"
        raise GatewayError("secret fallback provider detail")

    monkeypatch.setattr(ModelGateway, "stream", mock_stream)
    response = await agent_stream_endpoint(AgentAssistRequest(prompt="Fallback midstream provider error regression"))
    events = [event async for event in response.body_iterator]
    terminal_events = [event for event in events if '"done": true' in event]

    assert any('"token": "partial fallback answer"' in event for event in events)
    assert terminal_events == [
        'data: {"error": "The provider could not complete the stream.", "code": "provider_error", "done": true}\n\n'
    ]
    assert all('"token": "", "done": true' not in event for event in events)
    assert "secret" not in "".join(events)
    assert providers == ["groq", "gemini"]


@pytest.mark.asyncio
async def test_agent_stream_fallback_timeout_preserves_partial_content(monkeypatch):
    from app.api.agent_routes import AgentAssistRequest, agent_stream_endpoint
    from app.providers.errors import GatewayError
    from app.providers.errors import TimeoutError as ProviderTimeoutError
    from app.providers.gateway import ModelGateway

    providers = []

    async def mock_stream(_gateway, provider, *args, **kwargs):
        providers.append(provider)
        if provider == "groq":
            raise GatewayError("initial provider detail")
            yield "unreachable"
        yield "partial fallback answer"
        raise ProviderTimeoutError("secret fallback provider detail")

    monkeypatch.setattr(ModelGateway, "stream", mock_stream)
    response = await agent_stream_endpoint(AgentAssistRequest(prompt="Fallback midstream timeout regression"))
    events = [event async for event in response.body_iterator]
    terminal_events = [event for event in events if '"done": true' in event]

    assert any('"token": "partial fallback answer"' in event for event in events)
    assert terminal_events == [
        'data: {"error": "The provider could not complete the stream.", "code": "provider_timeout", "done": true}\n\n'
    ]
    assert all('"token": "", "done": true' not in event for event in events)
    assert "secret fallback provider detail" not in "".join(events)
    assert providers == ["groq", "gemini"]


@pytest.mark.asyncio
async def test_agent_stream_fallback_cancellation_preserves_cancellation(monkeypatch):
    import asyncio

    from app.api.agent_routes import AgentAssistRequest, agent_stream_endpoint
    from app.providers.errors import GatewayError
    from app.providers.gateway import ModelGateway

    providers = []

    async def mock_stream(_gateway, provider, *args, **kwargs):
        providers.append(provider)
        if provider == "groq":
            raise GatewayError("initial provider detail")
            yield "unreachable"
        yield "partial fallback answer"
        raise asyncio.CancelledError

    monkeypatch.setattr(ModelGateway, "stream", mock_stream)
    response = await agent_stream_endpoint(AgentAssistRequest(prompt="Fallback cancellation regression"))
    events = []

    with pytest.raises(asyncio.CancelledError):
        async for event in response.body_iterator:
            events.append(event)

    assert any('"token": "partial fallback answer"' in event for event in events)
    assert all('"done": true' not in event and '"error"' not in event for event in events)
    assert providers == ["groq", "gemini"]


@pytest.mark.asyncio
async def test_agent_stream_fallback_cancellation_before_first_token_preserves_cancellation(monkeypatch):
    import asyncio

    from app.api.agent_routes import AgentAssistRequest, agent_stream_endpoint
    from app.providers.errors import GatewayError
    from app.providers.gateway import ModelGateway

    providers = []

    async def mock_stream(_gateway, provider, *args, **kwargs):
        providers.append(provider)
        if provider == "groq":
            raise GatewayError("initial provider detail")
            yield "unreachable"
        raise asyncio.CancelledError
        yield "unreachable"

    monkeypatch.setattr(ModelGateway, "stream", mock_stream)
    response = await agent_stream_endpoint(AgentAssistRequest(prompt="Pre-token fallback cancellation regression"))
    events = []

    with pytest.raises(asyncio.CancelledError):
        async for event in response.body_iterator:
            events.append(event)

    assert events == []
    assert providers == ["groq", "gemini"]


@pytest.mark.asyncio
async def test_agent_stream_empty_primary_completion_does_not_start_fallback(monkeypatch):
    from app.api.agent_routes import AgentAssistRequest, agent_stream_endpoint
    from app.providers.gateway import ModelGateway

    providers = []

    async def mock_stream(_gateway, provider, *args, **kwargs):
        providers.append(provider)
        if provider == "gemini":
            yield "unexpected fallback answer"

    monkeypatch.setattr(ModelGateway, "stream", mock_stream)
    response = await agent_stream_endpoint(AgentAssistRequest(prompt="Empty primary completion regression"))
    events = [event async for event in response.body_iterator]

    assert events == ['data: {"token": "", "done": true}\n\n']
    assert providers == ["groq"]


@pytest.mark.asyncio
async def test_agent_stream_empty_fallback_completion_emits_normal_terminal_event(monkeypatch):
    from app.api.agent_routes import AgentAssistRequest, agent_stream_endpoint
    from app.providers.errors import GatewayError
    from app.providers.gateway import ModelGateway

    providers = []

    async def mock_stream(_gateway, provider, *args, **kwargs):
        providers.append(provider)
        if provider == "groq":
            raise GatewayError("initial provider failure")
            yield "unreachable"

    monkeypatch.setattr(ModelGateway, "stream", mock_stream)
    response = await agent_stream_endpoint(AgentAssistRequest(prompt="Empty fallback completion regression"))
    events = [event async for event in response.body_iterator]

    assert events == ['data: {"token": "", "done": true}\n\n']
    assert providers == ["groq", "gemini"]


@pytest.mark.asyncio
async def test_agent_assist_returns_degraded_result_without_caching(monkeypatch):
    from app.api.agent_routes import AgentAssistRequest, agent_assist_endpoint
    from app.performance_cache import perf_cache
    from app.providers.unified_manager import UnifiedExecutionResponse, unified_provider_manager

    cache_writes = []

    async def mock_execute(_req):
        return UnifiedExecutionResponse(
            content="",
            provider_used="groq",
            model_used="openai/gpt-oss-120b",
            agent_role="code_generator",
            latency_ms=25.0,
            timestamp=1729000000.0,
            status="degraded",
            degraded=True,
            error="The provider could not complete the request.",
        )

    monkeypatch.setattr(perf_cache, "get_query", lambda *args, **kwargs: None)
    monkeypatch.setattr(perf_cache, "set_query", lambda **kwargs: cache_writes.append(kwargs))
    monkeypatch.setattr(unified_provider_manager, "execute", mock_execute)

    response = await agent_assist_endpoint(
        AgentAssistRequest(caller_agent="forge", task_type="code", prompt="Degraded cache regression")
    )

    assert response.status == "degraded"
    assert response.error == "The provider could not complete the request."
    assert response.response == ""
    assert cache_writes == []


@pytest.mark.asyncio
async def test_agent_assist_cache_hit_skips_provider_and_cache_write(monkeypatch):
    from app.api.agent_routes import AgentAssistRequest, agent_assist_endpoint
    from app.performance_cache import perf_cache
    from app.providers.unified_manager import unified_provider_manager

    provider_executions = []
    cache_writes = []

    def mock_execute(req):
        provider_executions.append(req)
        raise AssertionError("provider execution must not run for a cache hit")

    monkeypatch.setattr(
        perf_cache,
        "get_query",
        lambda *args, **kwargs: ("Cached answer", {"tokens": 37}),
    )
    monkeypatch.setattr(perf_cache, "set_query", lambda **kwargs: cache_writes.append(kwargs))
    monkeypatch.setattr(unified_provider_manager, "execute", mock_execute)

    response = await agent_assist_endpoint(
        AgentAssistRequest(caller_agent="forge", task_type="code", prompt="Cached answer regression")
    )

    assert response.status == "cache_hit"
    assert response.response == "Cached answer"
    assert response.cache_hit is True
    assert response.token_usage == {"total_tokens": 37}
    assert provider_executions == []
    assert cache_writes == []


@pytest.mark.asyncio
async def test_agent_assist_cache_read_is_scoped_to_caller_and_task(monkeypatch):
    from app.api.agent_routes import AgentAssistRequest, agent_assist_endpoint
    from app.performance_cache import perf_cache
    from app.providers.unified_manager import UnifiedExecutionResponse, unified_provider_manager

    cache_reads = []
    cache_writes = []
    provider_executions = []
    expected_cache_extra = {
        "caller_agent": "forge",
        "task_type": "code",
        "context": {},
        "max_tokens": 75,
        "temperature": 0.2,
        "speculative": False,
    }

    def mock_get_query(*args, **kwargs):
        cache_reads.append((args, kwargs))
        return None

    def mock_set_query(**kwargs):
        cache_writes.append(kwargs)

    async def mock_execute(req):
        provider_executions.append(req)
        return UnifiedExecutionResponse(
            content="Scoped cache response",
            provider_used="groq",
            model_used="openai/gpt-oss-120b",
            agent_role="code_generator",
            latency_ms=25.0,
            timestamp=1729000000.0,
            status="success",
        )

    monkeypatch.setattr(perf_cache, "get_query", mock_get_query)
    monkeypatch.setattr(perf_cache, "set_query", mock_set_query)
    monkeypatch.setattr(unified_provider_manager, "execute", mock_execute)

    response = await agent_assist_endpoint(
        AgentAssistRequest(
            caller_agent="forge",
            task_type="code",
            prompt="Scoped cache lookup regression",
        )
    )

    assert response.status == "success"
    assert response.response == "Scoped cache response"
    assert cache_reads == [
        (
            ("Scoped cache lookup regression",),
            {
                "mode": "agent_assist_forge_code",
                "caller_id": "forge",
                "extra": expected_cache_extra,
                "namespace": "agent.assist",
            },
        )
    ]
    assert len(provider_executions) == 1
    assert len(cache_writes) == 1
    assert cache_writes[0]["mode"] == "agent_assist_forge_code"
    assert cache_writes[0]["caller_id"] == "forge"
    assert cache_writes[0]["extra"] == expected_cache_extra
    assert cache_writes[0]["namespace"] == "agent.assist"
    assert cache_writes[0]["ttl"] == 600.0


@pytest.mark.asyncio
async def test_agent_assist_cache_does_not_cross_callers_or_contexts(monkeypatch):
    from app.api.agent_routes import AgentAssistRequest, agent_assist_endpoint
    from app.performance_cache import MultiLevelCache, perf_cache
    from app.providers.unified_manager import UnifiedExecutionResponse, unified_provider_manager

    isolated_cache = MultiLevelCache()
    provider_executions = []

    async def mock_execute(req):
        provider_executions.append(req)
        return UnifiedExecutionResponse(
            content=f"Provider response {len(provider_executions)}",
            provider_used="groq",
            model_used="openai/gpt-oss-120b",
            agent_role=req.agent_role,
            latency_ms=25.0,
            timestamp=1729000000.0,
        )

    monkeypatch.setattr(perf_cache, "get_query", isolated_cache.get_query)
    monkeypatch.setattr(perf_cache, "set_query", isolated_cache.set_query)
    monkeypatch.setattr(unified_provider_manager, "execute", mock_execute)

    forge_alpha = AgentAssistRequest(
        caller_agent="forge",
        task_type="code",
        prompt="Cache isolation regression",
        context={"repository": "alpha"},
        max_tokens=60,
    )
    first = await agent_assist_endpoint(forge_alpha)
    repeated = await agent_assist_endpoint(forge_alpha)
    forge_review = await agent_assist_endpoint(
        AgentAssistRequest(
            caller_agent="forge",
            task_type="review",
            prompt="Cache isolation regression",
            context={"repository": "alpha"},
            max_tokens=60,
        )
    )
    different_temperature = await agent_assist_endpoint(
        AgentAssistRequest(
            caller_agent="forge",
            task_type="code",
            prompt="Cache isolation regression",
            context={"repository": "alpha"},
            max_tokens=60,
            temperature=0.7,
        )
    )
    different_max_tokens = await agent_assist_endpoint(
        AgentAssistRequest(
            caller_agent="forge",
            task_type="code",
            prompt="Cache isolation regression",
            context={"repository": "alpha"},
            max_tokens=75,
        )
    )
    speculative = await agent_assist_endpoint(
        AgentAssistRequest(
            caller_agent="forge",
            task_type="code",
            prompt="Cache isolation regression",
            context={"repository": "alpha"},
            max_tokens=60,
            speculative=True,
        )
    )
    forge_beta = await agent_assist_endpoint(
        AgentAssistRequest(
            caller_agent="forge",
            task_type="code",
            prompt="Cache isolation regression",
            context={"repository": "beta"},
        )
    )
    friday_alpha = await agent_assist_endpoint(
        AgentAssistRequest(
            caller_agent="friday",
            task_type="code",
            prompt="Cache isolation regression",
            context={"repository": "alpha"},
        )
    )

    assert first.status == "success"
    assert first.response == "Provider response 1"
    assert repeated.status == "cache_hit"
    assert repeated.response == first.response
    assert forge_review.status == "success"
    assert forge_review.response == "Provider response 2"
    assert forge_review.agent_role == "code_reviewer"
    assert different_temperature.status == "success"
    assert different_temperature.response == "Provider response 3"
    assert different_max_tokens.status == "success"
    assert different_max_tokens.response == "Provider response 4"
    assert speculative.status == "success"
    assert speculative.response == "Provider response 5"
    assert forge_beta.status == "success"
    assert forge_beta.response == "Provider response 6"
    assert friday_alpha.status == "success"
    assert friday_alpha.response == "Provider response 7"
    assert len(provider_executions) == 7
    assert [execution.agent_role for execution in provider_executions] == [
        "code_generator",
        "code_reviewer",
        "code_generator",
        "code_generator",
        "code_generator",
        "code_generator",
        "system_architect",
    ]
    assert provider_executions[1].prompt == "Cache isolation regression"
    assert provider_executions[1].context == {"repository": "alpha"}
    assert [execution.max_tokens for execution in provider_executions] == [60, 60, 60, 75, 60, 75, 75]
    assert [execution.temperature for execution in provider_executions] == [
        0.2, 0.2, 0.7, 0.2, 0.2, 0.2, 0.2
    ]
    assert [execution.speculative for execution in provider_executions] == [
        False, False, False, False, True, False, False
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_field",
    [
        pytest.param({"task_type": "unsupported"}, id="unsupported-task-type"),
        pytest.param({"max_tokens": 49}, id="max-tokens-below-minimum"),
        pytest.param({"max_tokens": 8193}, id="max-tokens-above-maximum"),
        pytest.param({"temperature": -0.01}, id="temperature-below-minimum"),
        pytest.param({"temperature": 2.01}, id="temperature-above-maximum"),
    ],
)
async def test_agent_assist_rejects_malformed_input_before_provider_execution(auth, monkeypatch, invalid_field):
    from app.providers.unified_manager import unified_provider_manager

    provider_executions = []

    async def unexpected_execute(_request):
        provider_executions.append(_request)
        raise AssertionError("Invalid agent-assist input must not reach provider execution")

    monkeypatch.setattr(unified_provider_manager, "execute", unexpected_execute)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver", headers=auth) as client:
        response = await client.post(
            "/v1/agent/assist",
            json={"prompt": "Malformed request regression", **invalid_field},
        )

    assert response.status_code == 422
    assert provider_executions == []


@pytest.mark.asyncio
async def test_agent_assist_rejects_unsupported_caller_before_provider_execution(auth, monkeypatch):
    from app.providers.unified_manager import unified_provider_manager

    provider_executions = []

    async def unexpected_execute(_request):
        provider_executions.append(_request)
        raise AssertionError("Unsupported caller_agent must not reach provider execution")

    monkeypatch.setattr(unified_provider_manager, "execute", unexpected_execute)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver", headers=auth) as client:
        response = await client.post(
            "/v1/agent/assist",
            json={"caller_agent": "unsupported", "prompt": "Caller validation regression"},
        )

    assert response.status_code == 422
    assert provider_executions == []


@pytest.mark.asyncio
@pytest.mark.parametrize("prompt", ["", " \t\n"])
async def test_agent_assist_rejects_blank_prompt_before_provider_execution(auth, monkeypatch, prompt):
    from app.providers.unified_manager import unified_provider_manager

    provider_executions = []

    async def unexpected_execute(_request):
        provider_executions.append(_request)
        raise AssertionError("A blank prompt must not reach provider execution")

    monkeypatch.setattr(unified_provider_manager, "execute", unexpected_execute)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver", headers=auth) as client:
        response = await client.post(
            "/v1/agent/assist",
            json={"caller_agent": "forge", "prompt": prompt},
        )

    assert response.status_code == 422
    assert provider_executions == []


@pytest.mark.asyncio
async def test_agent_assist_rejects_null_prompt_before_provider_execution(auth, monkeypatch):
    from app.providers.unified_manager import unified_provider_manager

    provider_executions = []

    async def unexpected_execute(_request):
        provider_executions.append(_request)
        raise AssertionError("A null prompt must not reach provider execution")

    monkeypatch.setattr(unified_provider_manager, "execute", unexpected_execute)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver", headers=auth) as client:
        response = await client.post(
            "/v1/agent/assist",
            json={"caller_agent": "forge", "prompt": None},
        )

    assert response.status_code == 422
    assert provider_executions == []


@pytest.mark.asyncio
async def test_agent_stream_rejects_blank_prompt_before_provider_stream(auth, monkeypatch):
    from app.providers.gateway import ModelGateway

    provider_streams = []

    async def unexpected_stream(*_args, **_kwargs):
        provider_streams.append(True)
        raise AssertionError("A blank prompt must not start a provider stream")
        yield "unreachable"

    monkeypatch.setattr(ModelGateway, "stream", unexpected_stream)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver", headers=auth) as client:
        response = await client.post(
            "/v1/agent/stream",
            json={"caller_agent": "forge", "prompt": " \t\n"},
        )

    assert response.status_code == 422
    assert provider_streams == []


@pytest.mark.asyncio
async def test_agent_stream_forwards_request_and_streams_tokens(auth, monkeypatch):
    from app.providers.gateway import ModelGateway

    stream_calls = []

    async def mock_stream(_gateway, provider, request, *, stage_name):
        stream_calls.append((provider, request, stage_name))
        yield "First token"
        yield " second token"

    monkeypatch.setattr(ModelGateway, "stream", mock_stream)
    prompt = "Implement a safe retry policy"
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver", headers=auth) as client:
        response = await client.post(
            "/v1/agent/stream",
            json={
                "caller_agent": "forge",
                "task_type": "code",
                "prompt": prompt,
                "temperature": 0.65,
                "max_tokens": 321,
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.text.splitlines() == [
        'data: {"token": "First token", "done": false}',
        "",
        'data: {"token": " second token", "done": false}',
        "",
        'data: {"token": "", "done": true}',
        "",
    ]
    assert len(stream_calls) == 1
    provider, request, stage_name = stream_calls[0]
    assert provider == "groq"
    assert request.model == "openai/gpt-oss-120b"
    assert request.messages[0].content == prompt
    assert request.temperature == 0.65
    assert request.max_tokens == 321
    assert stage_name == "agent_stream_forge"


@pytest.mark.asyncio
async def test_agent_stream_falls_back_before_first_chunk(auth, monkeypatch):
    from app.providers.errors import GatewayError
    from app.providers.gateway import ModelGateway

    stream_calls = []

    async def mock_stream(_gateway, provider, request, *, stage_name):
        stream_calls.append((provider, request.model, request.messages[0].content, request.temperature, request.max_tokens, stage_name))
        if provider == "groq":
            raise GatewayError("private provider transport detail", provider="groq", model=request.model)
        yield "fallback answer"
        yield " continues"

    monkeypatch.setattr(ModelGateway, "stream", mock_stream)
    prompt = "Summarize the deployment failure"
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver", headers=auth) as client:
        response = await client.post(
            "/v1/agent/stream",
            json={
                "caller_agent": "forge",
                "task_type": "code",
                "prompt": prompt,
                "temperature": 0.4,
                "max_tokens": 222,
            },
        )

    assert response.status_code == 200
    assert 'fallback answer' in response.text
    assert ' continues' in response.text
    assert response.text.endswith('data: {"token": "", "done": true}\n\n')
    assert "private provider transport detail" not in response.text
    assert stream_calls == [
        ("groq", "openai/gpt-oss-120b", prompt, 0.4, 222, "agent_stream_forge"),
        ("gemini", "gemini-3.6-flash", prompt, 0.4, 222, "agent_stream_forge_fb"),
    ]


@pytest.mark.asyncio
async def test_agent_stream_does_not_restart_fallback_after_partial_primary_output(auth, monkeypatch):
    from app.providers.errors import GatewayError
    from app.providers.gateway import ModelGateway

    stream_calls = []

    async def mock_stream(_gateway, provider, request, *, stage_name):
        stream_calls.append(provider)
        if provider == "groq":
            yield "partial primary"
            raise GatewayError("private mid-stream failure", provider="groq", model=request.model)
        yield "unexpected fallback"

    monkeypatch.setattr(ModelGateway, "stream", mock_stream)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver", headers=auth) as client:
        response = await client.post(
            "/v1/agent/stream",
            json={"caller_agent": "forge", "prompt": "Test partial stream failure"},
        )

    assert response.status_code == 200
    assert "partial primary" in response.text
    assert "provider_error" in response.text
    assert "private mid-stream failure" not in response.text
    assert "unexpected fallback" not in response.text
    assert stream_calls == ["groq"]


@pytest.mark.asyncio
async def test_agent_assist_no_cache_bypasses_cache_reads_and_writes(monkeypatch):
    from app.api.agent_routes import AgentAssistRequest, agent_assist_endpoint
    from app.performance_cache import perf_cache
    from app.providers.unified_manager import UnifiedExecutionResponse, unified_provider_manager

    cache_reads = []
    cache_writes = []
    executions = []

    def mock_get_query(*args, **kwargs):
        cache_reads.append((args, kwargs))
        return None

    def mock_set_query(**kwargs):
        cache_writes.append(kwargs)

    async def mock_execute(req):
        executions.append(req)
        return UnifiedExecutionResponse(
            content="Successful uncached response",
            provider_used="groq",
            model_used="openai/gpt-oss-120b",
            agent_role="code_generator",
            latency_ms=25.0,
            timestamp=1729000000.0,
            status="success",
            degraded=False,
        )

    monkeypatch.setattr(perf_cache, "get_query", mock_get_query)
    monkeypatch.setattr(perf_cache, "set_query", mock_set_query)
    monkeypatch.setattr(unified_provider_manager, "execute", mock_execute)

    response = await agent_assist_endpoint(
        AgentAssistRequest(
            caller_agent="forge",
            task_type="code",
            prompt="No-cache behavior regression",
            no_cache=True,
        )
    )

    assert response.status == "success"
    assert response.response == "Successful uncached response"
    assert response.cache_hit is False
    assert cache_reads == []
    assert cache_writes == []
    assert len(executions) == 1


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
