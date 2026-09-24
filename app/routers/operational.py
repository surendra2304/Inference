"""Operational and telemetry endpoints for Inference runtime."""

from typing import Any

from fastapi import APIRouter

from app.providers.gateway import model_gateway
from app.providers.health import provider_health_tracker
from app.version import __version__

operational_router = APIRouter(tags=["Operational"])


@operational_router.get("/health/providers")
async def get_providers_health() -> dict[str, Any]:
    """Returns runtime health status and latency snapshots for all registered providers without credentials."""
    snapshots = provider_health_tracker.get_all_health()
    active_keys = {
        prov: {
            "total_keys": pool.total_keys_count,
            "active_keys": pool.get_active_keys_count(),
            "quarantined_keys": pool.get_quarantined_keys_count(),
        }
        for prov, pool in model_gateway.key_pools.items()
    }
    return {
        "status": "healthy",
        "version": __version__,
        "providers": snapshots,
        "key_pools": active_keys,
    }


@operational_router.get("/models")
async def list_models() -> dict[str, Any]:
    """Returns available model registry capabilities and supported configurations."""
    all_models = [
        {"provider": "gemini", "model": "gemini-3.7-flash", "capabilities": ["chat", "stream", "json", "tools", "vision", "thinking"], "context_window": 2097152},
        {"provider": "gemini", "model": "gemini-3.6-flash", "capabilities": ["chat", "stream", "json", "tools", "vision"], "context_window": 2097152},
        {"provider": "gemini", "model": "gemini-3.5-flash", "capabilities": ["chat", "stream", "json", "tools", "vision"], "context_window": 1048576},
        {"provider": "groq", "model": "openai/gpt-oss-120b", "capabilities": ["chat", "stream", "json", "tools"], "context_window": 131072},
        {"provider": "groq", "model": "qwen/qwen3.8-27b", "capabilities": ["chat", "stream", "json", "tools"], "context_window": 131072},
        {"provider": "groq", "model": "openai/gpt-oss-20b", "capabilities": ["chat", "stream", "json", "tools"], "context_window": 131072},
        {"provider": "nvidia", "model": "nvidia/nemotron-3-super-120b-a12b", "capabilities": ["chat", "stream", "json"], "context_window": 131072},
        {"provider": "nvidia", "model": "nvidia/nemotron-3-ultra-550b-a55b", "capabilities": ["chat", "stream", "json"], "context_window": 131072},
        {"provider": "nvidia", "model": "z-ai/glm-5.3", "capabilities": ["chat", "stream", "json"], "context_window": 131072},
        {"provider": "openrouter", "model": "liquid/lfm-2.5-2.6b:free", "capabilities": ["chat", "stream"], "context_window": 32768},
        {"provider": "openrouter", "model": "qwen/qwen3.8-27b:free", "capabilities": ["chat", "stream", "json"], "context_window": 131072},
        {"provider": "openrouter", "model": "nvidia/nemotron-3.5-lightning:free", "capabilities": ["chat", "stream", "json", "tools"], "context_window": 131072},
        {"provider": "vllm", "model": "local-model", "capabilities": ["chat", "stream", "json", "tools"], "context_window": 32768},
        {"provider": "sglang", "model": "local-sglang-model", "capabilities": ["chat", "stream", "json"], "context_window": 32768},
        {"provider": "llamacpp", "model": "local-llama", "capabilities": ["chat", "stream", "json"], "context_window": 8192},
    ]
    return {
        "models": all_models,
        "count": len(all_models),
    }


@operational_router.get("/metrics/runtime")
async def get_runtime_metrics() -> dict[str, Any]:
    """Returns runtime telemetry metrics, spend budgets, and circuit breaker states."""
    breakers = {
        prov: {
            "threshold": getattr(limiter, "max_concurrency", 4),
            "rate_rps": getattr(limiter, "rate", 5.0),
        }
        for prov, limiter in model_gateway.rate_limiters.items()
    }
    return {
        "metrics": {
            "providers_tracked": len(model_gateway.rate_limiters),
            "rate_limiters": breakers,
            "system_version": __version__,
        }
    }
