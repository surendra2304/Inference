"""Provider status stays scoped to timestamped observations from this process."""

from datetime import datetime

from app.monitoring import PerformanceMonitor


def test_provider_health_is_unknown_without_observations():
    monitor = PerformanceMonitor()
    monitor.provider_stats["unobserved"]

    report = monitor.get_provider_health()["unobserved"]

    assert report["total_calls"] == 0
    assert report["success_rate_pct"] is None
    assert report["status"] == "unknown"
    assert report["evidence_class"] == "observed_provider_calls"
    assert report["observed_at"] is None


def test_provider_health_has_server_observation_time_and_avoids_healthy_claim():
    monitor = PerformanceMonitor()
    monitor.record_provider_call("example", latency_sec=0.125, success=True)

    report = monitor.get_provider_health()["example"]

    assert report["total_calls"] == 1
    assert report["success_rate_pct"] == 100.0
    assert report["status"] == "observed_ok"
    assert report["status"] != "healthy"
    assert report["evidence_class"] == "observed_provider_calls"
    assert datetime.fromisoformat(report["observed_at"])
