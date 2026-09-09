"""Google Gemini LLM Provider Adapter."""

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.core.config import settings
from app.providers.base import (
    BaseLLMProvider,
    ProviderCapabilities,
    ProviderMessage,
    ProviderRequest,
    ProviderResponse,
    UsageEstimate,
)
from app.providers.http_client import get_shared_client
from app.utils.logger import logger


class GeminiProvider(BaseLLMProvider):
    """Adapter for Google Gemini API via async HTTP."""

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
    DEFAULT_MODEL = "gemini-3.5-flash-lite"
    SUPPORTED_MODELS = [
        "gemini-3.5-flash-lite",
        "gemini-3.6-flash",
        "gemini-3.7-flash",
        "gemini-3.1-flash-lite",
        "gemini-3.5-flash",
    ]

    def __init__(self, api_key: str | None = None, default_model: str | None = None, timeout: float = 60.0) -> None:
        self.api_keys = settings.get_provider_keys("gemini")
        if api_key:
            self.api_keys = [k.strip() for k in api_key.split(",") if k.strip()]
        self._key_index = 0
        self.default_model = default_model or self.DEFAULT_MODEL
        self.timeout = timeout

    @property
    def api_key(self) -> str | None:
        if not self.api_keys:
            return None
        key = self.api_keys[self._key_index % len(self.api_keys)]
        self._key_index += 1
        return key

    @property
    def provider_name(self) -> str:
        return "gemini"

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            provider_name="gemini",
            supported_models=self.SUPPORTED_MODELS,
            supports_streaming=True,
            supports_structured_output=True,
            supports_system_instructions=True,
            supports_tool_calling=True,
            max_context_window=1000000,
            rate_limits={"rpm": 15, "rpd": 1500, "tpm": 1000000},
        )

    def estimate_usage(self, request: ProviderRequest) -> UsageEstimate:
        """Heuristic estimation of tokens and compute cost."""
        total_chars = sum(len(m.content) for m in request.messages)
        if request.system_instruction:
            total_chars += len(request.system_instruction)

        # Rule of thumb: ~4 characters per token
        prompt_tokens = max(1, total_chars // 4)
        max_completion = request.max_tokens or 1000

        # Approximate Gemini Flash pricing ($0.075 / 1M input, $0.30 / 1M output)
        cost = (prompt_tokens * 0.000000075) + (max_completion * 0.00000030)

        return UsageEstimate(
            estimated_prompt_tokens=prompt_tokens,
            estimated_completion_tokens=max_completion,
            estimated_total_tokens=prompt_tokens + max_completion,
            estimated_cost_usd=round(cost, 6),
        )

    def _convert_messages(self, messages: list[ProviderMessage]) -> list[dict[str, Any]]:
        """Converts standardized ProviderMessages to Gemini contents structure."""
        contents: list[dict[str, Any]] = []
        for msg in messages:
            role = "model" if msg.role in ("assistant", "model") else "user"
            contents.append({"role": role, "parts": [{"text": msg.content}]})
        return contents

    def _build_payload(self, request: ProviderRequest) -> dict[str, Any]:
        """Constructs the JSON payload for the Gemini generateContent API."""
        contents = self._convert_messages(request.messages)

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": request.temperature,
            },
        }

        if request.max_tokens:
            payload["generationConfig"]["maxOutputTokens"] = request.max_tokens

        if request.response_schema:
            payload["generationConfig"]["responseMimeType"] = "application/json"

        if request.system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": request.system_instruction}]}

        return payload

    async def generate(self, request: ProviderRequest) -> ProviderResponse:
        """Generate full completion using Gemini REST API."""
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not configured.")

        model = request.model or self.default_model
        url = f"{self.BASE_URL}/models/{model}:generateContent"
        headers = {"Content-Type": "application/json", "x-goog-api-key": self.api_key}
        payload = self._build_payload(request)

        start_time = time.perf_counter()
        try:
            client = await get_shared_client()
            response = await client.post(url, headers=headers, json=payload, timeout=self.timeout)
            latency = time.perf_counter() - start_time

            if response.status_code in (429, 503):
                logger.warning(
                    "Gemini transient error (%d) encountered on model %s; cooling down for 2.0s",
                    response.status_code,
                    model,
                )
                await asyncio.sleep(2.0)
                if response.status_code == 429:
                    raise RuntimeError("Gemini API rate limit exceeded (HTTP 429).")
                else:
                    raise RuntimeError(f"Gemini API temporarily unavailable (HTTP 503): {response.text}")
            elif response.status_code != 200:
                error_msg = response.text
                logger.error("Gemini API error (%d): %s", response.status_code, error_msg)
                raise RuntimeError(f"Gemini API returned HTTP {response.status_code}: {error_msg}")

            data = response.json()

            # Extract text from response candidates
            candidates = data.get("candidates", [])
            if not candidates:
                return ProviderResponse(
                    content="",
                    model=model,
                    provider=self.provider_name,
                    latency_seconds=latency,
                    finish_reason="empty",
                    raw_response=data,
                )

            content_parts = candidates[0].get("content", {}).get("parts", [])
            generated_text = "".join(part.get("text", "") for part in content_parts)
            finish_reason = candidates[0].get("finishReason", "stop")

            # Extract token usage metadata
            usage = data.get("usageMetadata", {})
            prompt_tokens = usage.get("promptTokenCount", 0)
            completion_tokens = usage.get("candidatesTokenCount", 0)
            total_tokens = usage.get("totalTokenCount", prompt_tokens + completion_tokens)

            return ProviderResponse(
                content=generated_text,
                model=model,
                provider=self.provider_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                latency_seconds=round(latency, 4),
                finish_reason=finish_reason,
                raw_response=data,
            )

        except httpx.TimeoutException as exc:
            logger.error("Gemini request timed out after %.1fs", self.timeout)
            raise TimeoutError(f"Gemini API request timed out after {self.timeout}s") from exc
        except httpx.RequestError as exc:
            logger.error("Gemini network request failure: %s", type(exc).__name__)
            raise RuntimeError(f"Gemini network connection error: {type(exc).__name__}") from exc

    async def stream(self, request: ProviderRequest) -> AsyncIterator[str]:
        """Stream generated chunks using Gemini Server-Sent Events (SSE)."""
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not configured.")

        model = request.model or self.default_model
        url = f"{self.BASE_URL}/models/{model}:streamGenerateContent?alt=sse"
        headers = {"Content-Type": "application/json", "x-goog-api-key": self.api_key}
        payload = self._build_payload(request)

        try:
            client = await get_shared_client()
            async with client.stream("POST", url, headers=headers, json=payload, timeout=self.timeout) as stream_resp:
                    if stream_resp.status_code != 200:
                        error_body = await stream_resp.aread()
                        raise RuntimeError(
                            f"Gemini streaming failed (HTTP {stream_resp.status_code}): {error_body.decode('utf-8')}"
                        )

                    async for line in stream_resp.aiter_lines():
                        if line.startswith("data: "):
                            raw_json = line[6:].strip()
                            if raw_json == "[DONE]":
                                break
                            try:
                                chunk_data = json.loads(raw_json)
                                candidates = chunk_data.get("candidates", [])
                                if candidates:
                                    parts = candidates[0].get("content", {}).get("parts", [])
                                    for part in parts:
                                        text_chunk = part.get("text", "")
                                        if text_chunk:
                                            yield text_chunk
                            except json.JSONDecodeError:
                                continue
        except httpx.RequestError as exc:
            logger.error("Gemini streaming network error: %s", type(exc).__name__)
            raise RuntimeError(f"Gemini streaming connection error: {type(exc).__name__}") from exc

    async def health(self) -> bool:
        """Check provider health and key validity with a minimal probe."""
        if not self.api_key:
            return False

        url = f"{self.BASE_URL}/models/{self.default_model}"
        headers = {"x-goog-api-key": self.api_key}

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url, headers=headers)
                return resp.status_code == 200
        except Exception as exc:
            logger.warning("Gemini health check failed: %s", str(exc))
            return False
