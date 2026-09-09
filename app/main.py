import asyncio
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from app.agents.software_specialists import register_software_specialists
from app.api.agent_routes import agent_router
from app.api.friday_routes import friday_router
from app.api.instant_routes import instant_router
from app.api.routes import router as api_router
from app.config_production import production_config
from app.core.config import settings
from app.core.orchestrator import orchestrator
from app.health import health_router
from app.middleware.rate_limiter import EnhancedRateLimiterMiddleware
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
from app.utils.logger import logger, setup_logger
from app.version import VERSION


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
    yield
    # Cleanly close pooled HTTP connections
    try:
        await http_client_pool.close()
    except Exception:
        pass
    logger.info("Shutting down %s", production_config.APP_NAME)


# Multi-consumer dynamic rate limiter with burst allowance and retry-after headers
app = FastAPI(
    title=production_config.APP_NAME,
    description="Local-first, provider-agnostic multi-agent intelligence platform with structured adversarial debate.",
    version=VERSION,
    lifespan=lifespan
)

# Multi-consumer dynamic rate limiter with burst allowance and retry-after headers
app.add_middleware(EnhancedRateLimiterMiddleware)

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
app.include_router(health_router)
app.include_router(operational_router)
app.include_router(api_router)
app.include_router(friday_router)
app.include_router(agent_router)
app.include_router(instant_router)
app.include_router(trading_router)
app.include_router(enhanced_router)
app.include_router(live_router)
app.include_router(multi_market_router)
app.include_router(evolution_router)
app.include_router(predictions_router)
app.include_router(ecosystem_router)
app.include_router(providers_router)
app.include_router(forge_router)
app.include_router(batch_router)
app.include_router(forge_health_router)
app.include_router(analytics_router)
app.include_router(nexus_router)
app.include_router(debate_router)
app.include_router(governance_router)
app.include_router(multimodal_router)
app.include_router(experiment_router)
app.include_router(sentinel_router)
app.include_router(intelx_router)
app.include_router(futuris_router)


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
