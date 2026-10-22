"""LiteLLM-backed provider adapter for Inference.

This adapter keeps Inference's internal ProviderRequest/ProviderResponse contracts
while delegating normalized LLM transport to LiteLLM. It intentionally does not
replace Inference's gateway policies (health, budgets, telemetry, fallback).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

from app.providers.base import (
    BaseLLMProvider,
    ProviderCapabilities,
    ProviderRequest,
    ProviderResponse,
    UsageEstimate,
)
from app.providers.errors import normalize_provider_exception

PROTECTED_FIELDS = {
    "model",
    "messages",
    "temperature",
    "max_tokens",
    "stream",
    "response_format",
    "api_key",
    "timeout",
}


class LiteLLMProvider(BaseLLMProvider):
    """Provider adapter using LiteLLM's unified async completion interface."""

    def __init__(
        self,
        provider_name: str = "litellm",
        api_key: str | None = None,
    ) -> None:
        self._provider_name = provider_name
        self._api_key = api_key

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @staticmethod
    def _build_messages(request: ProviderRequest) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if request.system_instruction:
            messages.append({"role": "system", "content": request.system_instruction})
        messages.extend(
            {"role": m.role, "content": m.content, **({"name": m.name} if m.name else {})} for m in request.messages
        )
        return messages

    async def generate(self, request: ProviderRequest) -> ProviderResponse:
        if not request.model:
            raise ValueError("LiteLLMProvider.generate requires ProviderRequest.model.")

        try:
            import litellm  # type: ignore[import-not-found,import-untyped]
        except ImportError as exc:
            raise RuntimeError("LiteLLM integration is enabled but the 'litellm' package is not installed.") from exc

        started = time.perf_counter()
        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": self._build_messages(request),
            "temperature": request.temperature,
        }
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        if request.response_schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "inference_response",
                    "schema": request.response_schema,
                    "strict": True,
                },
            }
        safe_extra = {k: v for k, v in request.extra_params.items() if k not in PROTECTED_FIELDS}
        kwargs.update(safe_extra)

        if self._api_key:
            kwargs["api_key"] = self._api_key

        timeout_sec = max(1.0, float(request.extra_params.get("timeout", 60.0)))

        try:
            response = await asyncio.wait_for(
                litellm.acompletion(**kwargs),
                timeout=timeout_sec,
            )
        except Exception as exc:
            raise normalize_provider_exception(exc, provider=self._provider_name, model=request.model) from exc

        latency = time.perf_counter() - started

        choice = response.choices[0]
        content = choice.message.content or ""
        usage = getattr(response, "usage", None)

        return ProviderResponse(
            content=content,
            model=str(getattr(response, "model", request.model)),
            provider=self._provider_name,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
            total_tokens=getattr(usage, "total_tokens", 0) if usage else 0,
            latency_seconds=latency,
            finish_reason=getattr(choice, "finish_reason", "stop"),
            raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
        )

    async def stream(self, request: ProviderRequest) -> AsyncIterator[str]:
        if not request.model:
            raise ValueError("LiteLLMProvider.stream requires ProviderRequest.model.")

        try:
            import litellm  # type: ignore[import-not-found,import-untyped]
        except ImportError as exc:
            raise RuntimeError("LiteLLM integration is enabled but the 'litellm' package is not installed.") from exc

        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": self._build_messages(request),
            "temperature": request.temperature,
            "stream": True,
        }
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        if self._api_key:
            kwargs["api_key"] = self._api_key
        safe_extra = {k: v for k, v in request.extra_params.items() if k not in PROTECTED_FIELDS}
        kwargs.update(safe_extra)

        try:
            response = await litellm.acompletion(**kwargs)
            async for chunk in response:
                try:
                    text = chunk.choices[0].delta.content
                except (AttributeError, IndexError):
                    text = None
                if text:
                    yield text
        except Exception as exc:
            raise normalize_provider_exception(exc, provider=self._provider_name, model=request.model) from exc

    def estimate_usage(self, request: ProviderRequest) -> UsageEstimate:
        text = "\n".join([request.system_instruction or ""] + [m.content for m in request.messages])
        estimated_prompt = max(1, len(text) // 4)
        completion = request.max_tokens or 1024
        return UsageEstimate(
            estimated_prompt_tokens=estimated_prompt,
            estimated_completion_tokens=completion,
            estimated_total_tokens=estimated_prompt + completion,
        )

    def capabilities(self, model: str | None = None) -> ProviderCapabilities:
        ctx = 128000
        supports_tools = True
        supports_json = True
        if model:
            m = model.lower()
            if "llama-3" in m:
                ctx = 8192 if "8b" in m else 131072
            elif "claude" in m:
                ctx = 200000
            elif "gpt-4" in m or "gpt-4o" in m:
                ctx = 128000
            elif "deepseek" in m:
                ctx = 64000
                supports_tools = "tool" in m or "chat" in m
            elif "embedding" in m:
                ctx = 8192
                supports_tools = False
                supports_json = False
        return ProviderCapabilities(
            provider_name=self._provider_name,
            supported_models=[model] if model else [],
            supports_streaming=True,
            supports_structured_output=supports_json,
            supports_system_instructions=True,
            supports_tool_calling=supports_tools,
            max_context_window=ctx,
            rate_limits={},
        )

    async def health(self) -> bool:
        from app.providers.health import provider_health_tracker

        snapshot = provider_health_tracker.get_provider_health(self._provider_name)
        if snapshot:
            return bool(snapshot.is_healthy)
        return True
