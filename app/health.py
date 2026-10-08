"""Production Health and Prometheus Metrics Endpoints."""

from datetime import UTC, datetime

from fastapi import APIRouter, Response, status

from app.agents.registry import agent_registry
from app.config_production import production_config
from app.monitoring import monitor
from app.observability.prometheus import render_prometheus
from app.optimization import concurrency_controller, telemetry_cache
from app.version import VERSION

health_router = APIRouter(tags=["Health & Monitoring"])


def _active_specialist_count() -> int:
    """Return the current registry size used by readiness and status routes."""
    return len(agent_registry.list_agents())


@health_router.get("/health", status_code=status.HTTP_200_OK)
@health_router.head("/health", status_code=status.HTTP_200_OK)
async def basic_health():
    """Report process liveness only; no dependency readiness is implied."""
    return {
        "status": "responding",
        "evidence_class": "process_liveness",
        "evidence_scope": "This Inference process answered /health; providers, memory, and database readiness are not checked.",
        "observed_at": datetime.now(UTC).isoformat(),
        "service": "inference-api",
        "version": VERSION,
        "active_specialist_agents": _active_specialist_count(),
    }


@health_router.get("/health/ready", status_code=status.HTTP_200_OK)
async def readiness_check():
    """Report registry configuration only, not end-to-end inference readiness."""
    agents = agent_registry.list_agents()
    ready = len(agents) >= 10
    return {
        "status": "registry_configured" if ready else "registry_incomplete",
        "evidence_class": "agent_registry_configuration",
        "evidence_scope": "Checks only the configured specialist registry count; provider, memory, and database readiness are not checked.",
        "observed_at": datetime.now(UTC).isoformat(),
        "service": "inference-api",
        "version": VERSION,
        "ready": ready,
        "active_specialist_agents": len(agents)
    }


@health_router.get("/health/detailed", status_code=status.HTTP_200_OK)
async def detailed_health():
    """Return a timestamped snapshot of this process, not a provider/dependency probe."""
    return {
        "status": "responding",
        "evidence_class": "process_runtime_snapshot",
        "evidence_scope": "This Inference process answered /health/detailed; external provider health is not checked here.",
        "observed_at": datetime.now(UTC).isoformat(),
        "service": "inference-api",
        "version": VERSION,
        "app_env": production_config.APP_ENV,
        "active_specialist_agents": _active_specialist_count(),
        "performance": monitor.get_api_metrics(),
        "cache": {
            "enabled": production_config.CACHE_ENABLED,
            "hit_rate_pct": telemetry_cache.get_hit_rate(),
            "total_cached_entries": len(telemetry_cache.cache)
        },
        "concurrency": {
            "active_requests": concurrency_controller.active_count,
            "max_allowed": production_config.MAX_CONCURRENT_REQUESTS
        }
    }


@health_router.get("/health/providers", status_code=status.HTTP_200_OK)
async def provider_health():
    """Provider-specific health, success rates, latency, and key pool status."""
    from app.providers.gateway import model_gateway

    active_keys = {
        prov: {
            "total_keys": pool.total_keys_count,
            "active_keys": pool.get_active_keys_count(),
            "quarantined_keys": pool.get_quarantined_keys_count(),
        }
        for prov, pool in model_gateway.key_pools.items()
    }
    return {
        "evidence_class": "process_observed_provider_calls",
        "evidence_scope": "Provider outcomes recorded by this Inference process; no direct provider probes are performed.",
        "observed_at": datetime.now(UTC).isoformat(),
        "providers": monitor.get_provider_health(),
        "priority_chain": production_config.PROVIDER_PRIORITY,
        "key_pools": active_keys,
    }


@health_router.get("/status", status_code=status.HTTP_200_OK)
async def system_status():
    """System capabilities, active agents, and operational modes."""
    agents = agent_registry.list_agents()
    return {
        "system": "Inference",
        "version": VERSION,
        "status": "operational",
        "advisory_only": True,
        "capabilities": [
            "trading_consultation",
            "ab_testing_framework",
            "testnet_risk_evaluator",
            "multi_agent_adversarial_debate"
        ],
        "active_specialists_count": len(agents),
        "agents": [a.role for a in agents]
    }


@health_router.get("/metrics", status_code=status.HTTP_200_OK)
async def prometheus_metrics() -> Response:
    """Exposes Prometheus-formatted metrics for this process.

    Rendered by :func:`app.observability.prometheus.render_prometheus`, the same function the
    authenticated ``/v1/governance/prometheus-metrics`` endpoint uses, so the two can no
    longer disagree about a metric's name or about how absent data is represented. See that
    module for the two measured defects (malformed ``None`` samples; counters named after
    consultation traffic while they counted every route) and how they are fixed.

    Provider-level detail is deliberately excluded here: this endpoint is mounted outside
    authentication so a scraper can reach it when a dependency is down, and which providers
    are configured or failing is infrastructure detail, not a public metric.
    """
    return Response(
        content=render_prometheus(include_provider_detail=False),
        media_type="text/plain; version=0.0.4",
    )
