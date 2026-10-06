"""Security utilities and authentication dependencies for Inference."""

import hmac
import ipaddress

from fastapi import Header, HTTPException, status

from app.core.config import settings
from app.utils.logger import logger


def _is_trusted_hop(host: str) -> bool:
    """True when the direct TCP peer is us or a reverse proxy we operate."""
    if host in ("localhost", "testclient"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    if ip.is_loopback:
        return True
    for cidr in settings.TRUSTED_PROXY_CIDRS:
        try:
            if ip in ipaddress.ip_network(cidr, strict=False):
                return True
        except ValueError:
            continue
    return False


def resolve_client_ip(request) -> str:
    """Return the real client IP for rate limiting and audit logging.

    Rate limiting keyed on `request.client.host` alone collapses every caller
    behind a reverse proxy into a single bucket (and, when the proxy sits on
    loopback, can hand every request the localhost bypass). Forwarded headers
    are only trusted when the direct peer is loopback or an explicitly
    configured proxy network, so a client connecting from the public internet
    cannot forge X-Forwarded-For to escape its own bucket.

    With one trusted proxy the last X-Forwarded-For entry is the address that
    proxy actually accepted the connection from: a header the client sent is
    prepended-to, not trusted, so spoofing through the proxy fails.
    """
    peer = request.client.host if getattr(request, "client", None) else None
    if not peer:
        return "unknown"
    if not settings.TRUST_PROXY_HEADERS or not _is_trusted_hop(peer):
        return peer

    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        hops = [h.strip() for h in forwarded.split(",") if h.strip()]
        if hops:
            return hops[-1]
    real_ip = request.headers.get("x-real-ip")
    if real_ip and real_ip.strip():
        return real_ip.strip()
    return peer


def _extract_provided_key(
    x_inference_api_key: str | None,
    x_api_key: str | None,
    x_friday_api_key: str | None,
    authorization: str | None,
) -> str | None:
    """Pull the presented credential out of any of the accepted header forms."""
    provided = x_inference_api_key or x_api_key or x_friday_api_key
    if not provided and authorization:
        provided = (
            authorization[7:].strip()
            if authorization.startswith("Bearer ")
            else authorization.strip()
        )
    return provided


def _configured_keys() -> list[str]:
    """Every credential accepted for this service (any of them authenticates)."""
    return [
        k
        for k in (
            settings.INFERENCE_API_KEY,
            settings.inference_api_KEY,
            settings.FRIDAY_UNIVERSE_API_KEY,
            settings.X_FRIDAY_API_KEY,
            settings.FRIDAY_API_KEY,
        )
        if k
    ]


async def _authenticate(provided_key: str | None, source: str) -> str:
    """Shared fail-closed authentication core.

      * INSECURE_DEV_AUTH explicitly enabled -> allow, loudly logged.
      * No credential configured on server   -> 503 (never anonymous access).
      * Credential required but absent       -> 401.
      * Credential present but wrong         -> 403 (constant-time compare).
    """
    if settings.INSECURE_DEV_AUTH:
        logger.warning(
            "INSECURE_DEV_AUTH is enabled (%s); accepting unauthenticated request "
            "without an integration key. NEVER enable this outside local development.",
            source,
        )
        return "insecure_dev_key"

    valid_keys = _configured_keys()
    if not valid_keys:
        logger.error("INFERENCE_API_KEY is not configured on the server.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Server security configuration error: no integration key is "
                "configured, so this endpoint cannot authenticate callers and is "
                "refusing to operate without authentication. Set INFERENCE_API_KEY "
                "or, for local development only, INSECURE_DEV_AUTH=true."
            ),
        )

    if not provided_key:
        logger.warning("Unauthorized access attempt (%s): missing authentication header.", source)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "Unauthorized: missing API key. Provide X-INFERENCE-API-KEY, "
                "X-API-Key, X-FRIDAY-API-Key, or 'Authorization: Bearer <key>'."
            ),
        )

    candidate = provided_key.encode("utf-8")
    # Constant-time comparison across every configured key: no short-circuit leak.
    if not any(hmac.compare_digest(candidate, vk.encode("utf-8")) for vk in valid_keys):
        logger.warning("Forbidden access attempt (%s): invalid API key.", source)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: invalid API key.",
        )

    return provided_key


async def require_inference_api_key(
    x_inference_api_key: str | None = Header(None, alias="X-INFERENCE-API-KEY"),
    x_api_key: str | None = Header(None, alias="X-API-Key"),
    x_friday_api_key: str | None = Header(None, alias="X-FRIDAY-API-Key"),
    authorization: str | None = Header(None, alias="Authorization"),
) -> str:
    """Fail-closed authentication for every state-changing or model-executing route.

    Applied router-wide in app/main.py. Health probes and the dashboard's
    read-only routes intentionally stay unauthenticated.
    """
    provided = _extract_provided_key(x_inference_api_key, x_api_key, x_friday_api_key, authorization)
    return await _authenticate(provided, source="require_inference_api_key")


async def verify_friday_api_key(
    x_inference_api_key: str | None = Header(None, alias="X-INFERENCE-API-KEY"),
    x_api_key: str | None = Header(None, alias="X-API-Key"),
    x_friday_api_key: str | None = Header(None, alias="X-FRIDAY-API-Key"),
    authorization: str | None = Header(None, alias="Authorization"),
) -> str:
    """FRIDAY boundary check — same policy as `require_inference_api_key`.

    Historically this was a second, divergent implementation that answered 500
    (instead of 503) when no key was configured. It now delegates so both
    boundaries share one status-code contract.
    """
    return await require_inference_api_key(
        x_inference_api_key=x_inference_api_key,
        x_api_key=x_api_key,
        x_friday_api_key=x_friday_api_key,
        authorization=authorization,
    )
