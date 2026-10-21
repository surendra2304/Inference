from __future__ import annotations

import asyncio
import time

import pytest

from app.inference_runtime.budget import BudgetLedger
from app.inference_runtime.circuit import CircuitState
from app.inference_runtime.config import RuntimeConfig
from app.inference_runtime.contracts import (
    Capability,
    CompletionRequest,
    CompletionResult,
    Message,
    ProviderEndpoint,
)
from app.inference_runtime.errors import SchemaViolation
from app.inference_runtime.gateway import HardenedGateway


class _Transport:
    def __init__(self, handler):
        self.handler = handler
        self.calls: list[CompletionRequest] = []

    async def complete(self, request, endpoint, api_key):
        self.calls.append(request)
        return await self.handler(request, endpoint, api_key)

    async def health(self):
        return True


def _endpoint() -> ProviderEndpoint:
    return ProviderEndpoint(
        provider="test-provider",
        model="test-model",
        capabilities=frozenset({Capability.CHAT, Capability.JSON}),
        cost_per_1k_input=0.01,
        cost_per_1k_output=0.01,
    )


def _request(**overrides) -> CompletionRequest:
    values = {
        "model": "test-model",
        "messages": (Message(role="user", content="hello"),),
        "timeout_seconds": 5.0,
        "tenant_id": "tenant-a",
        "request_id": "request-1",
    }
    values.update(overrides)
    return CompletionRequest(**values)


def _gateway(transport: _Transport) -> HardenedGateway:
    gateway = HardenedGateway(
        RuntimeConfig(max_retries=0, request_timeout_seconds=2.0, default_budget_usd=10.0)
    )
    gateway.register(_endpoint(), transport)
    return gateway


def _result(text: str, endpoint: ProviderEndpoint) -> CompletionResult:
    return CompletionResult(
        text=text,
        provider=endpoint.provider,
        model=endpoint.model,
        latency_seconds=0.01,
        total_tokens=2,
        cost_usd=0.001,
    )


@pytest.mark.asyncio
async def test_cache_hit_is_returned_without_reserving_budget_twice():
    async def respond(_request, endpoint, _key):
        return _result('{"ok": true}', endpoint)

    transport = _Transport(respond)
    gateway = _gateway(transport)
    request = _request(cacheable=True, response_schema={"type": "object", "required": ["ok"]})

    first = await gateway.complete(request)
    remaining_after_first = gateway.budget.remaining(request.tenant_id)
    second = await gateway.complete(request)

    assert second.cached is True
    assert second.text == first.text
    assert len(transport.calls) == 1
    assert gateway.budget.remaining(request.tenant_id) == pytest.approx(remaining_after_first)


@pytest.mark.asyncio
async def test_cache_hit_remains_available_when_provider_is_unhealthy():
    async def respond(_request, endpoint, _key):
        return _result('{"ok": true}', endpoint)

    transport = _Transport(respond)
    gateway = _gateway(transport)
    request = _request(cacheable=True, response_schema={"type": "object", "required": ["ok"]})

    await gateway.complete(request)
    for _ in range(3):
        gateway.health.failure("test-provider", "provider unavailable")

    cached = await gateway.complete(request)

    assert cached.cached is True
    assert len(transport.calls) == 1


@pytest.mark.asyncio
async def test_invalid_schema_response_is_not_cached_but_provider_cost_is_reconciled():
    responses = iter(("not-json", '{"ok": true}'))

    async def respond(_request, endpoint, _key):
        return _result(next(responses), endpoint)

    transport = _Transport(respond)
    gateway = _gateway(transport)
    request = _request(
        cacheable=True,
        response_schema={"type": "object", "required": ["ok"], "properties": {"ok": {"type": "boolean"}}},
    )
    starting_budget = gateway.budget.remaining(request.tenant_id)

    with pytest.raises(SchemaViolation):
        await gateway.complete(request)

    # The response cannot be cached, but real provider usage still costs money.
    assert gateway.budget.remaining(request.tenant_id) == pytest.approx(starting_budget - 0.001)

    second = await gateway.complete(request)
    assert second.text == '{"ok": true}'
    assert second.cached is False
    assert len(transport.calls) == 2


@pytest.mark.asyncio
async def test_cancelled_inference_releases_its_budget_reservation():
    started = asyncio.Event()

    async def respond(_request, _endpoint, _key):
        started.set()
        await asyncio.sleep(30)

    transport = _Transport(respond)
    gateway = _gateway(transport)
    request = _request()
    starting_budget = gateway.budget.remaining(request.tenant_id)

    task = asyncio.create_task(gateway.complete(request))
    await asyncio.wait_for(started.wait(), timeout=1.0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert gateway.budget.remaining(request.tenant_id) == pytest.approx(starting_budget)


@pytest.mark.asyncio
async def test_gateway_cancellation_releases_half_open_circuit_probe():
    started = asyncio.Event()

    async def respond(_request, _endpoint, _key):
        started.set()
        await asyncio.sleep(30)

    transport = _Transport(respond)
    gateway = _gateway(transport)
    breaker = gateway.breakers["test-provider"]

    for _ in range(breaker.threshold):
        breaker.failure()
    breaker.opened_at = time.monotonic() - breaker.recovery_seconds - 1.0

    task = asyncio.create_task(gateway.complete(_request()))
    await asyncio.wait_for(started.wait(), timeout=1.0)
    assert breaker.state is CircuitState.HALF_OPEN
    assert breaker._probe_in_flight is True

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # The shared executor task handles cancellation on its own event-loop turn.
    for _ in range(50):
        if breaker.state is CircuitState.OPEN and not breaker._probe_in_flight:
            break
        await asyncio.sleep(0)

    assert breaker.state is CircuitState.OPEN
    assert breaker._probe_in_flight is False


@pytest.mark.asyncio
async def test_reused_request_id_cannot_dedupe_across_tenants():
    async def respond(request, endpoint, _key):
        await asyncio.sleep(0.02)
        return _result(request.tenant_id, endpoint)

    transport = _Transport(respond)
    gateway = _gateway(transport)
    tenant_a = _request(tenant_id="tenant-a", request_id="same-id")
    tenant_b = _request(tenant_id="tenant-b", request_id="same-id")

    result_a, result_b = await asyncio.gather(
        gateway.complete(tenant_a),
        gateway.complete(tenant_b),
    )

    assert result_a.text == "tenant-a"
    assert result_b.text == "tenant-b"
    assert len(transport.calls) == 2


@pytest.mark.asyncio
async def test_cache_key_includes_extra_generation_parameters():
    async def respond(request, endpoint, _key):
        return _result(str(request.extra["top_p"]), endpoint)

    transport = _Transport(respond)
    gateway = _gateway(transport)
    first_request = _request(cacheable=True, extra={"top_p": 0.5})
    second_request = _request(cacheable=True, extra={"top_p": 0.9})

    first = await gateway.complete(first_request)
    second = await gateway.complete(second_request)

    assert first.text == "0.5"
    assert second.text == "0.9"
    assert len(transport.calls) == 2


def test_budget_ledger_starts_with_no_reserved_or_spent_amount():
    ledger = BudgetLedger(default_budget=2.0)
    assert ledger.remaining("tenant-a") == pytest.approx(2.0)
