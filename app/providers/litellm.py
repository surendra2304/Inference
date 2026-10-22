from __future__ import annotations

import asyncio
import time

from app.inference_runtime.contracts import CompletionRequest, ProviderEndpoint

from .normalization import normalize

PROTECTED = {"model", "messages", "temperature", "max_tokens", "stream", "response_format"}


class LiteLLMTransport:
    async def complete(self, request: CompletionRequest, endpoint: ProviderEndpoint, api_key: str | None = None):
        try:
            import litellm  # type: ignore[import-not-found,import-untyped]
        except ImportError as exc:
            raise RuntimeError("litellm package is not installed") from exc
        kwargs = {
            "model": request.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "temperature": request.temperature,
        }
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        if request.response_schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "inference_response", "schema": dict(request.response_schema), "strict": True},
            }
        kwargs.update({k: v for k, v in request.extra.items() if k not in PROTECTED})
        if api_key:
            kwargs["api_key"] = api_key
        started = time.perf_counter()
        response = await asyncio.wait_for(litellm.acompletion(**kwargs), timeout=request.timeout_seconds)
        return normalize(response, request, endpoint, time.perf_counter() - started)

    async def health(self) -> bool:
        return True
