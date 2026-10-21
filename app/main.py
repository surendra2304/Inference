import asyncio
import json
import os
import signal
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.agents.software_specialists import register_software_specialists
from app.api.agent_routes import agent_router
from app.api.deliberate_stream_routes import deliberate_stream_router
from app.api.friday_routes import friday_router
from app.api.instant_routes import instant_router
from app.api.routes import router as api_router
from app.api.universal_task_routes import task_router
from app.api.v1_core_routes import v1_router
from app.config_production import production_config
from app.core.config import settings
from app.core.orchestrator import orchestrator
from app.core.security import require_inference_api_key
from app.health import health_router
from app.middleware.rate_limiter import EnhancedRateLimiterMiddleware
from app.middleware.request_metrics import RequestMetricsMiddleware
from app.providers.base import ProviderMessage, ProviderRequest
from app.providers.gateway import model_gateway
from app.providers.http_client import http_client_pool
from app.routers.admin_analytics import analytics_router
from app.routers.batch import batch_router
from app.routers.debate_trace import debate_router
from app.routers.ecosystem import ecosystem_router
from app.routers.enhanced_trading import enhanced_router
from app.routers.evolution_intel import evolution_router
from app.routers.experiment_routes import experiment_router
from app.routers.forge_health import forge_health_router
from app.routers.forge_services import forge_router
from app.routers.futuris import futuris_router
from app.routers.governance import governance_router
from app.routers.intelx import intelx_router
from app.routers.live_intelligence import live_router
from app.routers.multi_market import multi_market_router
from app.routers.multimodal import multimodal_router
from app.routers.nexus import nexus_router
from app.routers.operational import operational_router
from app.routers.predictions import predictions_router
from app.routers.providers import providers_router
from app.routers.sentinel import sentinel_router
from app.routers.trading import router as trading_router
from app.security.api_security import ProductionSecurityMiddleware
from app.ui.dashboard import get_dashboard_html
from app.utils.json_safe import make_safe_response_class, validation_errors_payload
from app.utils.logger import logger, setup_logger
from app.utils.memory_guard import idle_trim_watchdog
from app.version import VERSION

#: When this process started. Tasks created before it belong to a previous process and
#: cannot be in flight here, which is what makes orphan reconciliation safe.
_PROCESS_STARTED_AT = datetime.now(timezone.utc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle events: startup and shutdown."""
    setup_logger(name="inference", log_level=production_config.LOG_LEVEL)
    logger.info(
        "Starting %s in %s environment on %s:%d",
        production_config.APP_NAME,
        production_config.APP_ENV,
        production_config.HOST,
        production_config.PORT
    )
    env_info = settings.validate_environment()
    logger.info(
        "Startup environment validated: env=%s, active_providers=%s",
        env_info["app_env"],
        ", ".join(env_info["configured_providers"]) or "none (degraded)",
    )
    # Register software engineering specialists for FORGE
    register_software_specialists()
    # Initialize persistent SQLite memory database
    await orchestrator.memory.initialize()
    # An abrupt exit (crash, SIGKILL, OOM kill, container eviction) leaves whatever was
    # in flight marked ``running`` forever, so the audit trail keeps claiming work is in
    # progress long after it stopped. Reconcile before serving any traffic.
    try:
        reconciler = getattr(orchestrator.memory, "reconcile_orphaned_tasks", None)
        if reconciler is not None:
            await reconciler(_PROCESS_STARTED_AT)
    except Exception as exc:  # noqa: BLE001 - startup must not fail on bookkeeping
        logger.warning("Orphaned-task reconciliation skipped: %s", exc)
    # Initialize shared HTTP connection pool and pre-warm primary endpoints
    await http_client_pool.get_client()
    asyncio.create_task(
        http_client_pool.prewarm([
            "https://api.groq.com",
            "https://generativelanguage.googleapis.com",
            "https://api.mistral.ai",
            "https://openrouter.ai",
        ])
    )
    # Memory hygiene for the quiet tail: while traffic flows, the middleware considers a trim
    # on every completed request; once it stops, this loop (guard policy still applies) returns
    # freed arenas to the OS instead of leaving the high-water mark in place until the next
    # burst. Cancelled on shutdown.
    watchdog = asyncio.create_task(
        idle_trim_watchdog(
            interval_seconds=float(getattr(production_config, "MEMORY_TRIM_WATCHDOG_SECONDS", 30.0))
        )
    )
    loop = asyncio.get_running_loop()

    def _on_signal(sig_name: str) -> None:
        logger.info("Received %s signal, initiating clean shutdown sequence...", sig_name)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, lambda s=sig: _on_signal(s.name))
        except (NotImplementedError, RuntimeError, AttributeError):
            pass

    yield

    logger.info("Initiating graceful shutdown of %s within 5s...", production_config.APP_NAME)
    try:
        async with asyncio.timeout(5.0):
            watchdog.cancel()
            try:
                await watchdog
            except asyncio.CancelledError:
                pass
            # Cleanly close pooled HTTP connections and streaming clients
            await http_client_pool.close()
    except Exception as exc:
        logger.warning("Shutdown cleanup encountered notice: %s", exc)
    logger.info("Shutdown of %s complete", production_config.APP_NAME)


# Multi-consumer dynamic rate limiter with burst allowance and retry-after headers
# Every JSON response is encoded through ``_SafeJSONResponse``, which converts non-finite
# floats to their names first. Starlette encodes with ``allow_nan=False``, and Pydantic
# accepts ``NaN``/``inf`` for a plain ``float`` field, so without this a body such as
# {"portfolio_equity": NaN} passed validation and then made the *response* unencodable —
# an unhandled 500 caused entirely by the shape of the reply.
_SafeJSONResponse = make_safe_response_class()

app = FastAPI(
    title=production_config.APP_NAME,
    description="Local-first, provider-agnostic multi-agent intelligence platform with structured adversarial debate.",
    version=VERSION,
    lifespan=lifespan,
    default_response_class=_SafeJSONResponse,
)

# Multi-consumer dynamic rate limiter with burst allowance and retry-after headers
app.add_middleware(EnhancedRateLimiterMiddleware)

# Request-level metrics for every route (see app/middleware/request_metrics.py): the only
# previous caller of monitor.record_request was the trading-consult service, so the latency
# window and error rate did not describe the process that /health reported them for.
app.add_middleware(RequestMetricsMiddleware)

# Production security middleware (headers, rate limiting, payload bounding)
app.add_middleware(ProductionSecurityMiddleware)

# GZip response compression middleware for production efficiency
app.add_middleware(GZipMiddleware, minimum_size=500)

cors_origins = settings.CORS_ALLOWED_ORIGINS if settings.CORS_ALLOWED_ORIGINS else [
    "https://friday-universe.duckdns.org",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:8004",
    "http://127.0.0.1:8004",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
    allow_headers=["*"],
)


STATUS_TO_ERROR_CODE: dict[int, str] = {
    400: "BAD_REQUEST",
    401: "UNAUTHORIZED",
    403: "FORBIDDEN",
    404: "NOT_FOUND",
    405: "METHOD_NOT_ALLOWED",
    408: "REQUEST_TIMEOUT",
    413: "PAYLOAD_TOO_LARGE",
    422: "VALIDATION_ERROR",
    429: "RATE_LIMIT_EXCEEDED",
    500: "INTERNAL_SERVER_ERROR",
    502: "BAD_GATEWAY",
    503: "SERVICE_UNAVAILABLE",
    504: "GATEWAY_TIMEOUT",
}


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    correlation_id = (
        request.headers.get("X-Correlation-ID")
        or request.headers.get("X-Request-ID")
        or str(uuid.uuid4())
    )
    detail = exc.detail
    if isinstance(detail, dict):
        error_code = (
            detail.get("error")
            or detail.get("error_code")
            or STATUS_TO_ERROR_CODE.get(exc.status_code, f"HTTP_{exc.status_code}")
        )
        message = str(detail.get("message") or detail.get("detail") or "HTTP exception occurred.")
    else:
        error_code = STATUS_TO_ERROR_CODE.get(exc.status_code, f"HTTP_{exc.status_code}")
        message = str(detail) if detail is not None else "HTTP exception occurred."

    response_headers = dict(exc.headers) if getattr(exc, "headers", None) else {}
    response_headers["X-Correlation-ID"] = correlation_id

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": error_code,
            "message": message,
            "detail": detail,
            "success": False,
            "correlation_id": correlation_id,
        },
        headers=response_headers,
    )


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return a 422 that can actually be serialised."""
    correlation_id = (
        request.headers.get("X-Correlation-ID")
        or request.headers.get("X-Request-ID")
        or str(uuid.uuid4())
    )
    logger.warning(
        "Request validation failed [correlation_id=%s] on %s %s: %s",
        correlation_id, request.method, request.url.path,
        "; ".join(
            f"{'.'.join(str(p) for p in e.get('loc', []))}: {e.get('msg', '')}"
            for e in exc.errors()[:5]
        ),
    )
    details = validation_errors_payload(exc.errors())
    msg_parts = [
        f"{'.'.join(str(p) for p in e.get('loc', []))}: {e.get('msg', '')}"
        for e in exc.errors()[:3]
    ]
    message = f"Request validation failed: {'; '.join(msg_parts)}" if msg_parts else "Request validation failed."
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "VALIDATION_ERROR",
            "message": message,
            "detail": details,
            "success": False,
            "correlation_id": correlation_id,
        },
        headers={"X-Correlation-ID": correlation_id},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    correlation_id = (
        request.headers.get("X-Correlation-ID")
        or request.headers.get("X-Request-ID")
        or str(uuid.uuid4())
    )
    logger.error(
        "Unhandled global exception [correlation_id=%s] on %s %s: %s",
        correlation_id, request.method, request.url.path, str(exc), exc_info=True
    )
    message = "An internal server error occurred."
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "INTERNAL_SERVER_ERROR",
            "message": message,
            "detail": f"{message} (correlation id {correlation_id})",
            "success": False,
            "correlation_id": correlation_id,
        },
        headers={"X-Correlation-ID": correlation_id},
    )


# Mount API routes
# ── Authentication boundary ────────────────────────────────────────────────────
# Every router that executes models, spends budget, mutates state, or exposes
# audit data requires the configured INFERENCE_API_KEY (fail-closed).
# Health probes and the dashboard's read-only routes stay unauthenticated so
# liveness checks and /ui keep working without a credential.
app.include_router(health_router)
app.include_router(forge_health_router)
app.include_router(operational_router, dependencies=[Depends(require_inference_api_key)])
app.include_router(api_router)  # per-route auth applied in app/api/routes.py
app.include_router(friday_router)  # already carries its own verify_friday_api_key
app.include_router(task_router, dependencies=[Depends(require_inference_api_key)])
app.include_router(agent_router, dependencies=[Depends(require_inference_api_key)])
app.include_router(instant_router, dependencies=[Depends(require_inference_api_key)])
app.include_router(v1_router, dependencies=[Depends(require_inference_api_key)])
app.include_router(trading_router, dependencies=[Depends(require_inference_api_key)])
app.include_router(enhanced_router, dependencies=[Depends(require_inference_api_key)])
app.include_router(live_router, dependencies=[Depends(require_inference_api_key)])
app.include_router(multi_market_router, dependencies=[Depends(require_inference_api_key)])
app.include_router(evolution_router, dependencies=[Depends(require_inference_api_key)])
app.include_router(predictions_router, dependencies=[Depends(require_inference_api_key)])
app.include_router(ecosystem_router, dependencies=[Depends(require_inference_api_key)])
app.include_router(providers_router, dependencies=[Depends(require_inference_api_key)])
app.include_router(forge_router, dependencies=[Depends(require_inference_api_key)])
app.include_router(batch_router, dependencies=[Depends(require_inference_api_key)])
app.include_router(analytics_router, dependencies=[Depends(require_inference_api_key)])
app.include_router(nexus_router, dependencies=[Depends(require_inference_api_key)])
app.include_router(debate_router, dependencies=[Depends(require_inference_api_key)])
app.include_router(governance_router, dependencies=[Depends(require_inference_api_key)])
app.include_router(multimodal_router, dependencies=[Depends(require_inference_api_key)])
app.include_router(experiment_router, dependencies=[Depends(require_inference_api_key)])
app.include_router(sentinel_router, dependencies=[Depends(require_inference_api_key)])
app.include_router(intelx_router, dependencies=[Depends(require_inference_api_key)])
app.include_router(futuris_router, dependencies=[Depends(require_inference_api_key)])
app.include_router(deliberate_stream_router)


@app.get("/ui", response_class=HTMLResponse)
async def ui_dashboard():
    """Interactive Web Dashboard for Inference 2.0."""
    return HTMLResponse(content=get_dashboard_html(), status_code=200)


@app.get("/")
@app.head("/")
async def root(request: Request):
    """Root metadata and web dashboard endpoint."""
    accept = request.headers.get("accept", "").lower()
    # If accessed from a browser (text/html) and not requesting JSON explicitly:
    if "text/html" in accept and "application/json" not in accept:
        return HTMLResponse(content=get_dashboard_html(), status_code=200)

    return {
        "name": production_config.APP_NAME,
        "status": "online",
        "version": VERSION,
        "env": production_config.APP_ENV,
        "description": "Provider-agnostic multi-agent intelligence platform"
    }


class DirectStreamRequest(BaseModel):
    prompt: str = ""
    question: str = ""
    model: str = "auto"
    max_tokens: int = Field(default=1024, ge=1, le=128000)


@app.post("/stream", dependencies=[Depends(require_inference_api_key)])
async def top_level_stream(req: DirectStreamRequest):
    """Direct SSE token streaming gateway endpoint."""
    text = req.prompt or req.question or "Hello"
    tokens = max(1, min(int(req.max_tokens or 1024), 128000))
    prov_req = ProviderRequest(
        messages=[ProviderMessage(role="user", content=text)],
        system_instruction="You are Inference, the centralized LLM gateway for the FRIDAY Universe.",
        model="openai/gpt-oss-120b",
        temperature=0.3,
        max_tokens=tokens,
    )

    async def _token_generator():
        try:
            async for chunk in model_gateway.stream(provider="groq", request=prov_req, stage_name="top_level_stream"):
                yield f"data: {json.dumps({'token': chunk, 'done': False})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc), 'done': True})}\n\n"
            return
        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(
        _token_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", getattr(settings, "PORT", 8000)))
    host = os.environ.get("HOST", getattr(settings, "HOST", "0.0.0.0"))
    uvicorn.run("app.main:app", host=host, port=port, workers=int(os.environ.get("WEB_CONCURRENCY", 1)))


