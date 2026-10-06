"""CRITICAL-4: governance Prometheus endpoint must never fabricate telemetry.

Before the fix this endpoint returned a JSON envelope containing seven
hardcoded series that never changed:

    inference_requests_total{service="nexus",status="200"} 412
    inference_requests_total{service="forge",status="200"} 1240
    inference_requests_total{service="trading_consult",status="200"} 88
    inference_provider_health{provider="gemini"} 1   (and 6 more, all 1)

Real counters, real circuit state and real provider health are asserted instead.
"""

import re

import pytest
from httpx import ASGITransport, AsyncClient

from app.governance.circuit_breaker import circuit_breaker_manager
from app.main import app
from app.monitoring import monitor

SERIAL = re.compile(r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?P<labels>\{[^}]*\})? (?P<value>[-+0-9.eE]+|NaN)$")

FABRICATED_MARKERS = (
    "412",
    "1240",
    'service="nexus"',
    'service="forge"',
    'service="trading_consult"',
    "quantile=\"0.50\"} 0.045",
)


@pytest.fixture(autouse=True)
def _isolate_observability_state():
    """Snapshot the process-global metrics this endpoint reads."""
    saved_latencies = list(monitor.request_latencies)
    saved_totals = (monitor.total_requests, monitor.failed_requests)
    saved_provider_stats = {k: dict(v) for k, v in monitor.provider_stats.items()}
    saved_circuits = {p: (b.state, b.consecutive_failures) for p, b in circuit_breaker_manager.breakers.items()}
    yield
    monitor.request_latencies[:] = saved_latencies
    monitor.total_requests, monitor.failed_requests = saved_totals
    monitor.provider_stats.clear()
    monitor.provider_stats.update(saved_provider_stats)
    for p, (state, failures) in saved_circuits.items():
        breaker = circuit_breaker_manager.breakers[p]
        breaker.state = state
        breaker.consecutive_failures = failures


@pytest.fixture
def metrics_client(auth):
    return AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
        headers=auth,
    )


async def test_endpoint_returns_prometheus_text_not_json(metrics_client):
    r = await metrics_client.get("/v1/governance/prometheus-metrics")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain"), r.headers["content-type"]
    assert r.text.lstrip().startswith("#"), "expected Prometheus exposition text"
    with pytest.raises(ValueError):
        import json

        json.loads(r.text)  # the old implementation returned a JSON object


async def test_no_hardcoded_sample_values(metrics_client):
    r = await metrics_client.get("/v1/governance/prometheus-metrics")
    for marker in FABRICATED_MARKERS:
        assert marker not in r.text, f"fabricated sample still present: {marker}"


async def test_every_series_is_well_formed(metrics_client):
    r = await metrics_client.get("/v1/governance/prometheus-metrics")
    samples = [
        ln for ln in r.text.splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
    assert samples, "expected at least one sample line"
    for line in samples:
        assert SERIAL.match(line), f"malformed exposition line: {line!r}"


async def test_request_counter_reflects_recorded_traffic(metrics_client):
    before = monitor.total_requests
    monitor.record_request(0.05, success=True)
    monitor.record_request(0.15, success=True)
    monitor.record_request(0.25, success=True)

    r = await metrics_client.get("/v1/governance/prometheus-metrics")
    m = re.search(r"^inference_consultation_requests_total (\S+)$", r.text, re.M)
    assert m, "counter series missing"
    assert float(m.group(1)) == float(before + 3), (
        f"expected {before + 3} recorded requests, endpoint reported {m.group(1)}"
    )


async def test_failed_counter_reflects_real_failures(metrics_client):
    before = monitor.failed_requests
    monitor.record_request(0.4, success=False)
    monitor.record_request(0.6, success=False)

    r = await metrics_client.get("/v1/governance/prometheus-metrics")
    m = re.search(r"^inference_consultation_failed_total (\S+)$", r.text, re.M)
    assert m, "failed counter series missing"
    assert float(m.group(1)) == float(before + 2), (
        f"expected {before + 2} failures, endpoint reported {m.group(1)}"
    )


async def test_circuit_metrics_reflect_real_breaker_state(metrics_client):
    r_ok = await metrics_client.get("/v1/governance/prometheus-metrics")
    m = re.search(r'^inference_provider_circuit_available\{provider="groq"\} (\S+)$', r_ok.text, re.M)
    assert m, "groq circuit series missing"
    assert m.group(1) == "1", "groq circuit should start available"

    # Trip the breaker and confirm the endpoint reports the new truth.
    breaker = circuit_breaker_manager.breakers["groq"]
    original_state = breaker.state
    for _ in range(10):
        circuit_breaker_manager.record_failure("groq")
    try:
        r_trip = await metrics_client.get("/v1/governance/prometheus-metrics")
        m_avail = re.search(r'^inference_provider_circuit_available\{provider="groq"\} (\S+)$', r_trip.text, re.M)
        m_state = re.search(r'^inference_provider_circuit_state\{provider="groq"\} (\S+)$', r_trip.text, re.M)
        assert m_avail and m_state, "groq circuit series missing after trip"
        assert m_avail.group(1) == "0", "tripped breaker must report unavailable"
        assert m_state.group(1) == "2", f"tripped breaker must report OPEN, got {m_state.group(1)}"
    finally:
        breaker.state = original_state
        breaker.consecutive_failures = 0


async def test_provider_success_ratio_appears_only_with_observed_calls(metrics_client):
    """No data -> no series. Fabricating a value is exactly what we fixed."""
    assert monitor.provider_stats == {}
    r_empty = await metrics_client.get("/v1/governance/prometheus-metrics")
    assert "inference_provider_success_ratio" not in r_empty.text

    monitor.record_provider_call("groq", 0.2, success=True)
    monitor.record_provider_call("groq", 0.4, success=True)
    r_data = await metrics_client.get("/v1/governance/prometheus-metrics")
    m = re.search(r'^inference_provider_success_ratio\{provider="groq"\} (\S+)$', r_data.text, re.M)
    assert m, "series should appear once real calls exist"
    assert float(m.group(1)) == 1.0, "two successes out of two calls must report 1.0"


async def test_latency_quantiles_track_recorded_latencies(metrics_client):
    monitor.record_request(1.5, success=True)
    r = await metrics_client.get("/v1/governance/prometheus-metrics")
    m = re.search(r'^inference_request_duration_seconds\{quantile="0.99"\} (\S+)$', r.text, re.M)
    assert m, "p99 series missing"
    assert float(m.group(1)) >= 1.5, "p99 must reflect the 1.5s sample that was recorded"


async def test_public_metrics_endpoint_still_reports_live_counters():
    """The dashboard's /metrics (app.health) must stay anonymous and real."""
    before = monitor.total_requests
    monitor.record_request(0.1, success=True)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as c:
        r = await c.get("/metrics")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    m = re.search(r"^inference_requests_total (\S+)$", r.text, re.M)
    assert m and float(m.group(1)) == float(before + 1)
