import asyncio

import pytest

import app.providers
from app.providers.base import ProviderMessage, ProviderRequest
from app.providers.errors import GatewayError, ProviderUnconfiguredError, TimeoutError
from app.providers.gateway import KeyPool, ModelGateway


class _Limiter:
    def __init__(self):
        self.acquired = 0
        self.released = 0

    async def acquire(self):
        self.acquired += 1

    def release(self):
        self.released += 1


class _Health:
    def __init__(self):
        self.successes = []
        self.failures = []

    def record_success(self, provider, latency):
        self.successes.append((provider, latency))

    def record_failure(self, provider, error, **kwargs):
        self.failures.append((provider, error, kwargs))


def _gateway():
    gateway = ModelGateway.__new__(ModelGateway)
    gateway.key_pools = {"test-provider": KeyPool("test-provider", ["test-key"])}
    gateway.rate_limiters = {"test-provider": _Limiter()}
    gateway.health_tracker = _Health()
    return gateway


def _request(timeout=1.0):
    return ProviderRequest(
        messages=[ProviderMessage(role="user", content="hello")],
        model="test-model",
        extra_params={"timeout": timeout},
    )


def _patch_provider(monkeypatch, provider):
    monkeypatch.setattr("app.providers.gateway.provider_has_credentials", lambda _provider: True)
    monkeypatch.setattr(app.providers, "get_provider", lambda *_args, **_kwargs: provider)


@pytest.mark.asyncio
async def test_stream_completes_and_releases_limiter(monkeypatch):
    gateway = _gateway()
    started = []

    class _Provider:
        async def stream(self, _request):
            started.append(True)
            yield "hello"
            yield " world"

    _patch_provider(monkeypatch, _Provider())
    chunks = [chunk async for chunk in gateway.stream("test-provider", _request())]

    assert chunks == ["hello", " world"]
    assert started == [True]
    assert gateway.rate_limiters["test-provider"].acquired == 1
    assert gateway.rate_limiters["test-provider"].released == 1
    assert [item[0] for item in gateway.health_tracker.successes] == ["test-provider"]
    assert gateway.health_tracker.failures == []


@pytest.mark.asyncio
async def test_stream_total_deadline_produces_typed_timeout_and_closes(monkeypatch):
    gateway = _gateway()
    closed = asyncio.Event()

    class _Provider:
        async def stream(self, _request):
            try:
                yield "first"
                await asyncio.sleep(1)
                yield "never"
            finally:
                closed.set()

    _patch_provider(monkeypatch, _Provider())
    chunks = []

    with pytest.raises(TimeoutError, match="Stream deadline exceeded"):
        async for chunk in gateway.stream("test-provider", _request(timeout=0.05)):
            chunks.append(chunk)

    assert chunks == ["first"]
    assert closed.is_set()
    assert gateway.rate_limiters["test-provider"].released == 1
    assert gateway.health_tracker.failures[0][2]["is_503"] is True


@pytest.mark.asyncio
async def test_stream_cancellation_closes_iterator_and_releases_limiter(monkeypatch):
    gateway = _gateway()
    started = asyncio.Event()
    closed = asyncio.Event()

    class _Provider:
        async def stream(self, _request):
            try:
                started.set()
                await asyncio.sleep(10)
                yield "never"
            finally:
                closed.set()

    _patch_provider(monkeypatch, _Provider())

    async def consume():
        async for _chunk in gateway.stream("test-provider", _request(timeout=20)):
            pass

    task = asyncio.create_task(consume())
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert closed.is_set()
    assert gateway.rate_limiters["test-provider"].released == 1
    assert gateway.health_tracker.failures == []


@pytest.mark.asyncio
async def test_stream_rejects_unconfigured_provider_before_acquiring(monkeypatch):
    gateway = _gateway()
    monkeypatch.setattr("app.providers.gateway.provider_has_credentials", lambda _provider: False)

    with pytest.raises(ProviderUnconfiguredError):
        async for _chunk in gateway.stream("test-provider", _request()):
            pass

    assert gateway.rate_limiters["test-provider"].acquired == 0
    assert gateway.rate_limiters["test-provider"].released == 0
    assert gateway.health_tracker.failures == []


@pytest.mark.asyncio
async def test_stream_normalizes_provider_failure(monkeypatch):
    gateway = _gateway()

    class _Provider:
        async def stream(self, _request):
            raise RuntimeError("503 service unavailable")
            yield "unreachable"

    _patch_provider(monkeypatch, _Provider())

    with pytest.raises(GatewayError) as caught:
        async for _chunk in gateway.stream("test-provider", _request()):
            pass

    assert caught.value.provider == "test-provider"
    assert gateway.rate_limiters["test-provider"].released == 1
    assert gateway.health_tracker.failures[0][2]["is_503"] is True

@pytest.mark.asyncio
async def test_stream_timeout_records_service_unavailable_not_rate_limit(monkeypatch):
    gateway = _gateway()
    closed = asyncio.Event()

    class _Provider:
        async def stream(self, _request):
            try:
                await asyncio.sleep(1)
                yield "never"
            finally:
                closed.set()

    _patch_provider(monkeypatch, _Provider())

    with pytest.raises(TimeoutError, match="Stream deadline exceeded"):
        async for _chunk in gateway.stream("test-provider", _request(timeout=0.02)):
            pass

    assert closed.is_set()
    assert gateway.health_tracker.failures
    _, _, flags = gateway.health_tracker.failures[0]
    assert flags["is_503"] is True
    assert flags["is_429"] is False
    assert gateway.rate_limiters["test-provider"].released == 1
