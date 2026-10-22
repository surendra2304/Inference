"""LiteLLM selection/fallback helpers for Inference's ModelGateway."""

from __future__ import annotations

import json
from typing import Any

from app.core.config import settings
from app.providers.base import ProviderRequest, ProviderResponse
from app.providers.litellm_adapter import LiteLLMProvider


def litellm_enabled() -> bool:
    return bool(settings.INFERENCE_LITELLM_ENABLED)


def resolve_model(model: str, aliases_json: str | None = None) -> str:
    raw = aliases_json if aliases_json is not None else settings.LITELLM_MODEL_ALIASES_JSON
    try:
        aliases: dict[str, str] = json.loads(raw or "{}")
    except json.JSONDecodeError:
        aliases = {}
    return aliases.get(model, model)


def sanitize_litellm_params(extra_params: dict[str, Any]) -> dict[str, Any]:
    """Prevent Inference-only control fields leaking into provider kwargs."""
    blocked = {
        "timeout",
        "retry_policy",
        "fallback_policy",
        "routing_reason",
        "task_id",
        "agent_id",
    }
    if settings.LITELLM_DROP_PARAMS:
        return {k: v for k, v in extra_params.items() if k not in blocked}
    return dict(extra_params)


async def execute_via_litellm(
    request: ProviderRequest,
    api_key: str | None = None,
) -> ProviderResponse:
    if not litellm_enabled():
        raise RuntimeError("LiteLLM integration is disabled.")
    resolved = request.model or ""
    resolved = resolve_model(resolved)
    sanitized_extra = sanitize_litellm_params(request.extra_params)
    updated = request.model_copy(
        update={
            "model": resolved,
            "extra_params": {
                **sanitized_extra,
                "timeout": request.extra_params.get("timeout", settings.LITELLM_DEFAULT_TIMEOUT),
            },
        }
    )
    provider = LiteLLMProvider(api_key=api_key)
    return await provider.generate(updated)
