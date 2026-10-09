import asyncio
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from app.agents.software_specialists import register_software_specialists
from app.api.agent_routes import agent_router
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
    yield
    watchdog.cancel()
    try:
        await watchdog
    except asyncio.CancelledError:
        pass
    # Cleanly close pooled HTTP connections
    try:
        await http_client_pool.close()
    except Exception:
        pass
    logger.info("Shutting down %s", production_config.APP_NAME)


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

# CORS middleware with explicit allowlist from configuration
cors_origins = settings.CORS_ALLOWED_ORIGINS if settings.CORS_ALLOWED_ORIGINS else ["http://localhost:3000", "http://127.0.0.1:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return a 422 that can actually be serialised.

    FastAPI's default handler echoes the offending input back to the caller. Python's
    ``json`` parses the non-standard literals ``NaN``/``Infinity``, so a body such as
    ``{"question": "q", "max_agents": 1e309}`` reaches Pydantic as ``inf``; the validator
    correctly rejects it ("Input should be a finite number"), the error payload carries
    ``'input': inf``, and Starlette's encoder — which sets ``allow_nan=False`` — raises
    ``ValueError: Out of range float values are not JSON compliant``. That escaped into
    the global handler and turned a 422 into a 500 on every validating route: the client
    was told the server had broken when it had merely sent a bad number.
    """
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
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": validation_errors_payload(exc.errors())},
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
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error_code": "INTERNAL_SERVER_ERROR",
            "message": "An internal server error occurred.",
            "correlation_id": correlation_id
        },
        headers={"X-Correlation-ID": correlation_id}
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
