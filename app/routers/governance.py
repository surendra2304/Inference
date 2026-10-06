"""FastAPI Router for Tenant Key Governance, Multi-Tenant Budgets, Prometheus Metrics, and Degradation."""

from fastapi import APIRouter, HTTPException, Path, Response, status
from pydantic import BaseModel

from app.governance.circuit_breaker import circuit_breaker_manager
from app.governance.tenant_manager import tenant_manager
from app.monitoring import monitor
from app.optimization import concurrency_controller, telemetry_cache
from app.providers.health import provider_health_tracker

governance_router = APIRouter(prefix="/v1/governance", tags=["API Governance & Multi-Tenancy"])


class RotateKeyRequest(BaseModel):
    old_key: str


@governance_router.get("/tenants/{tenant_id}", status_code=status.HTTP_200_OK)
async def get_tenant_policy(tenant_id: str = Path(..., description="Tenant ID")):
    """Returns rate limit, budget ceiling, and key policy for a specific tenant."""
    policy = tenant_manager.tenants.get(tenant_id)
    if not policy:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Tenant '{tenant_id}' not found.")
    return policy.model_dump()


@governance_router.post("/tenants/{tenant_id}/rotate-key", status_code=status.HTTP_200_OK)
async def rotate_tenant_api_key(tenant_id: str, req: RotateKeyRequest):
    """Securely rotates an API key for a tenant."""
    try:
        new_key = tenant_manager.rotate_tenant_key(tenant_id, req.old_key)
        return {"status": "ROTATED", "tenant_id": tenant_id, "new_key": new_key}
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@governance_router.get("/circuits", status_code=status.HTTP_200_OK)
async def get_circuit_breaker_statuses():
    """Returns circuit breaker states across all 7 cloud providers."""
    return circuit_breaker_manager.get_circuit_statuses()


@governance_router.get("/prometheus-metrics", status_code=status.HTTP_200_OK)
async def get_prometheus_formatted_metrics() -> Response:
    """Returns Prometheus text exposition format built from live runtime state.

    Every series is computed from an instrumented source at request time. This
    endpoint previously returned a JSON envelope containing seven hardcoded
    sample series (412/1240/88 requests, all providers health=1), which was
    fabricated data served under a Prometheus content type and therefore
    impossible to distinguish from real telemetry.

    Series are emitted only when their backing source holds data: an absent
    metric is honest, a fabricated one is not.
    """
    api = monitor.get_api_metrics()
    circuits = circuit_breaker_manager.get_circuit_statuses()
    provider_stats = monitor.get_provider_health()
    health_reports = provider_health_tracker.get_all_health()

    lines: list[str] = []

    def counter(name: str, help_text: str, value: float) -> None:
        lines.extend([f"# HELP {name} {help_text}", f"# TYPE {name} counter", f"{name} {value}", ""])

    def gauge(name: str, help_text: str, value: float, labels: str | None = None) -> None:
        suffix = f"{{{labels}}}" if labels else ""
        lines.extend([f"# HELP {name} {help_text}", f"# TYPE {name} gauge", f"{name}{suffix} {value}", ""])

    # Consultation traffic (app.monitoring only records trading consultations;
    # the metric name says so rather than implying all HTTP traffic).
    counter("inference_consultation_requests_total", "Consultation requests handled", api["total_requests"])
    counter("inference_consultation_failed_total", "Consultation requests that failed", monitor.failed_requests)
    gauge("inference_error_rate_percent", "Percentage of consultation requests that failed", api["error_rate_pct"])
    gauge("inference_request_duration_seconds", "Consultation latency, 0.50 quantile", api["p50_latency_sec"], 'quantile="0.50"')
    gauge("inference_request_duration_seconds", "Consultation latency, 0.95 quantile", api["p95_latency_sec"], 'quantile="0.95"')
    gauge("inference_request_duration_seconds", "Consultation latency, 0.99 quantile", api["p99_latency_sec"], 'quantile="0.99"')
    gauge("inference_uptime_seconds", "Process uptime", api["uptime_seconds"])

    gauge("inference_active_requests", "Requests currently in flight", concurrency_controller.active_count)
    gauge("inference_cache_hit_rate_percent", "Telemetry cache hit rate", telemetry_cache.get_hit_rate())
    gauge("inference_tracked_tenants", "Tenants registered with the governance store", len(tenant_manager.tenants))

    # Circuit breakers: the authoritative traffic-admission decision per provider.
    if circuits:
        lines.extend([
            "# HELP inference_provider_circuit_available Whether the provider circuit admits traffic (1=yes, 0=no)",
            "# TYPE inference_provider_circuit_available gauge",
        ])
        for provider, status_info in sorted(circuits.items()):
            lines.append(
                f'inference_provider_circuit_available{{provider="{provider}"}} '
                f"{1 if status_info.get('is_available') else 0}"
            )
        lines.append("")
        lines.extend([
            "# HELP inference_provider_circuit_state Circuit breaker state (0=CLOSED, 1=HALF_OPEN, 2=OPEN)",
            "# TYPE inference_provider_circuit_state gauge",
        ])
        order = {"CLOSED": 0, "HALF_OPEN": 1, "OPEN": 2}
        for provider, status_info in sorted(circuits.items()):
            state = str(status_info.get("state", "CLOSED")).upper()
            lines.append(
                f'inference_provider_circuit_state{{provider="{provider}"}} {order.get(state, 3)}'
            )
        lines.append("")

    # Observed provider performance: only once real calls have been recorded.
    if provider_stats:
        lines.extend([
            "# HELP inference_provider_success_ratio Observed provider success rate (0-1)",
            "# TYPE inference_provider_success_ratio gauge",
        ])
        for provider, stats in sorted(provider_stats.items()):
            total = stats.get("total_calls", 0)
            rate = stats.get("success_rate_pct")
            if total and rate is not None:
                lines.append(
                    f'inference_provider_success_ratio{{provider="{provider}"}} {round(rate / 100.0, 4)}'
                )
        lines.append("")

    # Health tracker: emitted only for providers it has actually observed.
    if health_reports:
        lines.extend([
            "# HELP inference_provider_health_score Normalised provider health score (0-1)",
            "# TYPE inference_provider_health_score gauge",
        ])
        for provider, report in sorted(health_reports.items()):
            lines.append(
                f'inference_provider_health_score{{provider="{provider}"}} {round(report.health_score, 4)}'
            )
        lines.append("")
        lines.extend([
            "# HELP inference_provider_rate_limited_keys Number of API keys quarantined for rate limiting",
            "# TYPE inference_provider_rate_limited_keys gauge",
        ])
        for provider, report in sorted(health_reports.items()):
            lines.append(
                f'inference_provider_rate_limited_keys{{provider="{provider}"}} {report.quarantined_keys_count}'
            )
        lines.append("")

    return Response(content="\n".join(lines).rstrip() + "\n", media_type="text/plain; version=0.0.4")
