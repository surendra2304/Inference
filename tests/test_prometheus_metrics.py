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
    """Snapshot the process-global metrics this endpoint reads.

    State is cleared *before* each test as well as restored afterwards. Restoration
    alone left these tests dependent on execution order: ``test_provider_success_ratio_
    appears_only_with_observed_calls`` asserts ``monitor.provider_stats == {}``, which
    only holds if no earlier test in the same process ever recorded a provider call. Any
    test that drives a route reaching a provider (the contract-fuzz suite does) broke it.
    Starting from a known-empty monitor makes the assertions deterministic; restoring the
    snapshot afterwards still leaves the process exactly as it was found, so this file
    leaks nothing into its neighbours either. The quantile test is strictly stronger this
    way: it can no longer pass because of a latency sample recorded by another test.
    """
    saved_latencies = list(monitor.request_latencies)
    saved_totals = (monitor.total_requests, monitor.failed_requests)
    saved_provider_stats = {k: dict(v) for k, v in monitor.provider_stats.items()}
    saved_circuits = {p: (b.state, b.consecutive_failures) for p, b in circuit_breaker_manager.breakers.items()}

    monitor.request_latencies.clear()
    monitor.total_requests, monitor.failed_requests = 0, 0
    monitor.provider_stats.clear()
    for breaker in circuit_breaker_manager.breakers.values():
        breaker.state = "closed"
        breaker.consecutive_failures = 0

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


async def test_absent_metrics_are_omitted_not_rendered_as_none(metrics_client, monkeypatch):
    """A metric with no samples must vanish, not print the Python word "None".

    Measured before the fix: ``inference_error_rate_percent None`` — a malformed exposition
    line that a Prometheus scraper rejects, produced by an honest "no samples" value passing
    through an exporter that assumed a number.
    """
    empty_api = {
        "total_requests": 0,
        "samples": 0,
        "error_rate_pct": None,
        "p50_latency_sec": None,
        "p95_latency_sec": None,
        "p99_latency_sec": None,
        "avg_latency_sec": None,
        "uptime_seconds": 1.0,
    }
    monkeypatch.setattr(monitor, "get_api_metrics", lambda: empty_api)
    monkeypatch.setattr(monitor, "request_latencies", [])

    r = await metrics_client.get("/v1/governance/prometheus-metrics")
    assert "None" not in r.text, "an unknown value leaked into the exposition format"
    assert "inference_error_rate_percent" not in r.text
    assert "inference_request_duration_seconds" not in r.text
    for line in r.text.splitlines():
        if line.strip() and not line.startswith("#"):
            assert SERIAL.match(line), f"malformed exposition line: {line!r}"


async def test_latency_sample_count_is_exported(metrics_client):
    """The window size travels with the percentiles, so a p99 over 3 samples is visible."""
    monitor.record_request(0.01, success=True)
    monitor.record_request(0.02, success=True)
    monitor.record_request(0.03, success=True)
    # The scrape's own request is recorded by the middleware *after* the body is built, so
    # the exported count is the count at scrape time.
    expected = len(monitor.request_latencies)

    r = await metrics_client.get("/v1/governance/prometheus-metrics")
    m = re.search(r"^inference_latency_samples (\S+)$", r.text, re.M)
    assert m, "sample-count series missing"
    assert float(m.group(1)) == float(expected), (
        f"exported {m.group(1)} samples, {expected} were recorded before the scrape"
    )
    assert expected >= 3


async def test_request_counter_reflects_recorded_traffic(metrics_client):
    before = monitor.total_requests
    monitor.record_request(0.05, success=True)
    monitor.record_request(0.15, success=True)
    monitor.record_request(0.25, success=True)

    r = await metrics_client.get("/v1/governance/prometheus-metrics")
    m = re.search(r"^inference_requests_total (\S+)$", r.text, re.M)
    assert m, "counter series missing"
    assert float(m.group(1)) == float(before + 3), (
        f"expected {before + 3} recorded requests, endpoint reported {m.group(1)}"
    )


async def test_failed_counter_reflects_real_failures(metrics_client):
    before = monitor.failed_requests
    monitor.record_request(0.4, success=False)
    monitor.record_request(0.6, success=False)

    r = await metrics_client.get("/v1/governance/prometheus-metrics")
    m = re.search(r"^inference_requests_failed_total (\S+)$", r.text, re.M)
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


async def test_both_metrics_endpoints_publish_the_same_names_and_values(metrics_client):
    """One window of traffic, two endpoints, one set of numbers.

    Measured before the fix: ``/metrics`` published ``inference_requests_total`` while
    ``/v1/governance/prometheus-metrics`` published ``inference_api_requests_total`` for the
    same counter, and percentiles were ``inference_latency_p50_seconds`` in one and
    ``inference_request_duration_seconds{quantile="0.50"}`` in the other. A dashboard could
    not graph both, and a fix to the exposition rules reached only one of them (#69). They now
    render through ``app.observability.prometheus.render_prometheus``.

    This test drives real traffic and then asserts the *values* agree, so a future divergence
    cannot pass by renaming both sides.
    """
    monitor.record_request(0.02, success=True)
    monitor.record_request(0.04, success=True)
    monitor.record_request(0.06, success=False)

    governed = await metrics_client.get("/v1/governance/prometheus-metrics")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as c:
        public = await c.get("/metrics")
    assert public.status_code == 200

    def samples(text: str) -> dict[str, str]:
        out = {}
        for line in text.splitlines():
            if line.strip() and not line.startswith("#"):
                name, value = line.rsplit(" ", 1)
                out[name] = value
        return out

    governed_samples = samples(governed.text)
    public_samples = samples(public.text)
    core = (
        "inference_requests_total",
        "inference_requests_failed_total",
        "inference_latency_samples",
        "inference_error_rate_percent",
        "inference_uptime_seconds",
    )
    for name in core:
        assert name in public_samples, f"{name} missing from the public /metrics endpoint"
        assert name in governed_samples, f"{name} missing from the governance endpoint"
    # Failures are the same counter in both renderings, and neither scrape failed, so the
    # values must match exactly.
    assert (
        governed_samples["inference_requests_failed_total"]
        == public_samples["inference_requests_failed_total"]
    ), "the failure counter diverges between the two exporters"
    # The request counter differs only by the scrapes themselves: the middleware records a
    # scrape after its body was already rendered, so the second endpoint observed one more
    # request than the first. A divergence larger than the scrape traffic means the two
    # endpoints are no longer reading the same counter.
    delta = int(public_samples["inference_requests_total"]) - int(governed_samples["inference_requests_total"])
    assert 0 <= delta <= 1, (
        f"request counters diverged by {delta}: "
        f"governance={governed_samples['inference_requests_total']} "
        f"public={public_samples['inference_requests_total']}"
    )
    # Percentiles are one family with labels, in both.
    for endpoint, text in (("governance", governed.text), ("public", public.text)):
        assert 'inference_request_duration_seconds{quantile="0.99"}' in text, (
            f"{endpoint} endpoint lost the p99 series"
        )


async def test_public_metrics_hides_provider_identity(metrics_client):
    """``/metrics`` is unauthenticated, so it must not report which providers exist or fail.

    Provider inventory and circuit state are operational detail about the deployment. The
    authenticated governance endpoint publishes them; the anonymous one must not.
    """
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as c:
        public = await c.get("/metrics")
    assert public.status_code == 200
    for family in (
        "inference_provider_",
        "inference_tracked_tenants",
    ):
        assert family not in public.text, f"{family} must not be exposed anonymously"

    governed = await metrics_client.get("/v1/governance/prometheus-metrics")
    assert "inference_provider_circuit_available" in governed.text
    assert "inference_tracked_tenants" in governed.text


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
