"""FastAPI Router for Tenant Key Governance, Multi-Tenant Budgets, Prometheus Metrics, and Degradation."""

from fastapi import APIRouter, HTTPException, Path, Response, status
from pydantic import BaseModel

from app.governance.circuit_breaker import circuit_breaker_manager
from app.governance.tenant_manager import tenant_manager
from app.observability.prometheus import render_prometheus

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

    This endpoint previously returned a JSON envelope containing seven hardcoded sample
    series (412/1240/88 requests, every provider health=1): fabricated telemetry served under
    a Prometheus content type, impossible to tell apart from real data.

    It now renders through :func:`app.observability.prometheus.render_prometheus` — the same
    implementation as the public ``/metrics`` endpoint — and adds provider/circuit/health
    families, which stay here because this route is authenticated while ``/metrics`` is not.
    """
    return Response(
        content=render_prometheus(include_provider_detail=True),
        media_type="text/plain; version=0.0.4",
    )
