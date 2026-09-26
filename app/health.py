"""Production Health and Prometheus Metrics Endpoints."""

from fastapi import APIRouter, Response, status

from app.agents.registry import agent_registry
from app.config_production import production_config
from app.monitoring import monitor
from app.optimization import concurrency_controller, telemetry_cache
from app.version import VERSION

health_router = APIRouter(tags=["Health & Monitoring"])


def _active_specialist_count() -> int:
    """Return the current registry size used by readiness and status routes."""
    return len(agent_registry.list_agents())


@health_router.get("/health", status_code=status.HTTP_200_OK)
@health_router.head("/health", status_code=status.HTTP_200_OK)
async def basic_health():
    """Basic liveness health check."""
    return {
        "status": "healthy",
        "service": "inference-api",
        "version": VERSION,
        "active_specialist_agents": _active_specialist_count(),
    }


@health_router.get("/health/ready", status_code=status.HTTP_200_OK)
async def readiness_check():
    """Readiness probe checking memory and agent registry readiness."""
    agents = agent_registry.list_agents()
    ready = len(agents) >= 10
    return {
        "status": "ready" if ready else "degraded",
        "service": "inference-api",
        "version": VERSION,
        "ready": ready,
        "active_specialist_agents": len(agents)
    }


@health_router.get("/health/detailed", status_code=status.HTTP_200_OK)
async def detailed_health():
    """Detailed health status with live API metrics, cache performance, and concurrency."""
    return {
        "status": "healthy",
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
async def prometheus_metrics():
    """Exposes Prometheus-formatted metrics."""
    metrics_data = monitor.get_api_metrics()
    cache_rate = telemetry_cache.get_hit_rate()

    lines = [
        "# HELP inference_requests_total Total number of consultation requests",
        "# TYPE inference_requests_total counter",
        f"inference_requests_total {metrics_data['total_requests']}",
        "",
        "# HELP inference_latency_p50_seconds P50 response latency in seconds",
        "# TYPE inference_latency_p50_seconds gauge",
        f"inference_latency_p50_seconds {metrics_data['p50_latency_sec']}",
        "",
        "# HELP inference_latency_p95_seconds P95 response latency in seconds",
        "# TYPE inference_latency_p95_seconds gauge",
        f"inference_latency_p95_seconds {metrics_data['p95_latency_sec']}",
        "",
        "# HELP inference_latency_p99_seconds P99 response latency in seconds",
        "# TYPE inference_latency_p99_seconds gauge",
        f"inference_latency_p99_seconds {metrics_data['p99_latency_sec']}",
        "",
        "# HELP inference_error_rate_percent Percentage of failed requests",
        "# TYPE inference_error_rate_percent gauge",
        f"inference_error_rate_percent {metrics_data['error_rate_pct']}",
        "",
        "# HELP inference_cache_hit_rate_percent Telemetry cache hit rate percentage",
        "# TYPE inference_cache_hit_rate_percent gauge",
        f"inference_cache_hit_rate_percent {cache_rate}",
        "",
        "# HELP inference_active_requests Current in-flight consultations",
        "# TYPE inference_active_requests gauge",
        f"inference_active_requests {concurrency_controller.active_count}"
    ]

    return Response(content="\n".join(lines), media_type="text/plain; version=0.0.4")
