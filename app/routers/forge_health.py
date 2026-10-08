"""FastAPI Router for FORGE Health, Capabilities, and Admin Telemetry."""

import time

from fastapi import APIRouter, Depends, Query, status

from app.core.security import require_inference_api_key
from app.providers.pool_optimizer import provider_pool_optimizer
from app.routing.consumer_router import ConsumerType, consumer_router

# Auth is attached per-route rather than to the whole router because this module mixes
# two very different kinds of endpoint:
#   * liveness/capability polls (``/v1/forge/health``, ``/v1/forge/capabilities``) that a
#     monitoring system or a service-discovery probe must be able to call without
#     holding an inference credential;
#   * back-office telemetry (``/v1/admin/*``) that exposes per-consumer call volume,
#     token spend and estimated cost, and the provider pool's latency and health.
# The second class is operational data about paying consumers, so it requires the same
# credential as every other data-exposing router in ``app/main.py``.
#
# Before this, the router was mounted with no dependencies at all, so both admin
# endpoints answered ``200`` to an anonymous request: the usage and performance of every
# configured consumer were readable by anyone who could reach the port. Verified with an
# unauthenticated request returning the full consumer breakdown.
_ADMIN_AUTH = [Depends(require_inference_api_key)]

forge_health_router = APIRouter(tags=["FORGE Health & Admin"])


@forge_health_router.get("/v1/forge/health", status_code=status.HTTP_200_OK)
async def get_forge_health():
    """Quick health poll endpoint for FORGE autonomous engine."""
    perf = provider_pool_optimizer.get_performance_report()
    demoted_count = len(perf["active_demotions"])
    status_str = "healthy" if demoted_count <= 2 else ("degraded" if demoted_count <= 4 else "unhealthy")

    return {
        "status": status_str,
        "providers_available": 7 - demoted_count,
        "queue_depth": 0,
        "avg_latency_ms": 28.5,
        "timestamp": time.time()
    }


@forge_health_router.get("/v1/forge/capabilities", status_code=status.HTTP_200_OK)
async def get_forge_capabilities():
    """Lists intelligence services and capacity available for FORGE."""
    return {
        "services": [
            "code_generation (/v1/forge/generate-code)",
            "architecture_planning (/v1/forge/plan-architecture)",
            "code_review (/v1/forge/review-code)",
            "debugging (/v1/forge/debug)",
            "test_generation (/v1/forge/generate-tests)",
            "batch_generation (/v1/forge/batch-generate)"
        ],
        "supported_file_types": ["python", "html", "css", "js", "json", "markdown", "sql"],
        "supported_test_frameworks": ["pytest", "jest", "playwright"],
        "estimated_throughput_rps": 120,
        "active_specialist_agents": 17
    }


@forge_health_router.get(
    "/v1/admin/usage", status_code=status.HTTP_200_OK, dependencies=_ADMIN_AUTH
)
async def get_consumer_usage(consumer: ConsumerType | None = Query(default=None, description="forge, trading_bot, friday, human")):
    """Returns token and call usage metrics per consumer."""
    return consumer_router.get_usage(consumer)


@forge_health_router.get(
    "/v1/admin/providers/performance", status_code=status.HTTP_200_OK, dependencies=_ADMIN_AUTH
)
async def get_providers_performance():
    """Returns latency and health metrics across the provider pool."""
    return provider_pool_optimizer.get_performance_report()
