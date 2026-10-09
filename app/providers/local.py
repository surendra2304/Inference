"""Local (self-hosted) LLM Provider Adapter — the zero-cost, zero-quota tier.

This adapter talks to any OpenAI-compatible inference server reachable from the
Inference process:

* ``llama-server``   (llama.cpp)
* ``vllm serve``      (vLLM)
* ``python -m sglang.launch_server`` (SGLang)
* ``ollama serve``    (Ollama, which exposes ``/v1`` since 0.1.30)
* LM Studio, text-generation-webui, KoboldCpp, TabbyAP, or
* the bundled development rig at ``tools/local_model_server.py``.

Why this module exists
----------------------
The repository already ships ``app/providers/llamacpp.py``, ``app/providers/vllm.py``
and ``app/providers/sglang.py``. All three are real classes, all three subclass
``OpenAICompatibleTransport``, and **none of them is registered in
``app/providers/__init__.py`` or reachable from ``ModelGateway``** — so "local
inference" was advertised by the file tree and unavailable in practice. This
module is the one that is actually wired in, under the provider name ``local``.

Two deliberate departures from the cloud adapters
-------------------------------------------------
1. **Keyless by default.** Cloud adapters raise
   ``ValueError("<PROV>_API_KEY is not configured.")`` when no credential is
   present. A local server is protected by the network boundary, not by a bearer
   token, so making a key mandatory would leave the only genuinely free,
   private, unlimited tier unreachable out of the box. ``LOCAL_API_KEY`` is
   still honoured when set (reverse-proxied deployments sometimes do
   authenticate), and when it is set the gateway's normal key-pool rotation
   applies to it like any other provider.
2. **Cost is exactly zero.** The base class estimates a notional per-token price
   so the budget guardrails can reason about cloud spend. Local compute has no
   marginal cost, so ``estimate_usage`` reports ``0.0`` usd while still counting
   tokens — the budget ceiling then never trips on local calls, which is the
   point of having a free tier at all.

Operational contract
--------------------
``LOCAL_ENABLED`` gates whether the gateway will *route* to this provider or use
it as the final rung of the fallback ladder. It defaults to ``False`` so that no
deployment starts paying a connection-refused penalty for a server that is not
running. Set ``LOCAL_ENABLED=true`` plus ``LOCAL_BASE_URL`` and
``LOCAL_MODEL`` to activate it.
"""

from __future__ import annotations

from typing import Any

import httpx

from app.core.config import settings
from app.providers.base import ProviderCapabilities, ProviderRequest, UsageEstimate
from app.providers.openai_compatible import OpenAICompatibleProvider
from app.utils.logger import logger

#: Models reported when the local server's /v1/models endpoint is unavailable.
#: ``local-model`` is the deliberate placeholder Ollama uses for an unspecified
#: tag, so a misconfigured deployment produces a clear 404 from the server rather
#: than a confusing KeyError here.
FALLBACK_MODEL_LIST: list[str] = ["local-model"]


class LocalProvider(OpenAICompatibleProvider):
    """Adapter for a self-hosted OpenAI-compatible inference endpoint.

    Keyless, costless, and the terminal tier of the fallback ladder.
    """

    #: Local servers do not require a bearer token by default.
    requires_api_key = False

    #: The cloud adapters sleep 2.0s inside ``generate`` before raising a
    #: transient error, and then the gateway sleeps again before retrying with
    #: the next key. For a local server under load that double sleep is pure
    #: added latency: the server is often merely busy and recovers in
    #: milliseconds. The gateway owns backoff and key quarantine, so this
    #: adapter defers to it.
    transient_cooldown_seconds = 0.0

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> None:
        resolved_base_url = (base_url or settings.LOCAL_BASE_URL or "").strip()
        if not resolved_base_url:
            raise ValueError(
                "LOCAL_BASE_URL is not configured. Set it to the OpenAI-compatible "
                "base of your local server, e.g. http://127.0.0.1:11434/v1 (Ollama) "
                "or http://127.0.0.1:8080/v1 (llama.cpp llama-server)."
            )
        resolved_model = (default_model or settings.LOCAL_MODEL or "").strip() or "local-model"
        super().__init__(
            provider_name="local",
            base_url=resolved_base_url,
            api_key=api_key if api_key is not None else settings.LOCAL_API_KEY,
            default_model=resolved_model,
            supported_models=[resolved_model],
            timeout=timeout if timeout is not None else float(settings.REQUEST_TIMEOUT or 60.0),
        )
        logger.debug("LocalProvider configured: base_url=%s model=%s", self.base_url, resolved_model)

    def capabilities(self) -> ProviderCapabilities:
        """Report local-server capabilities.

        Context window and tool-calling support are read from settings because
        they are properties of the *model you loaded*, which Inference cannot
        discover reliably; guessing 128k here (as the cloud base class does)
        would let the router build prompts a local model cannot accept.
        """
        return ProviderCapabilities(
            provider_name="local",
            supported_models=self.supported_models,
            supports_streaming=True,
            supports_structured_output=True,
            supports_system_instructions=True,
            supports_tool_calling=bool(settings.LOCAL_SUPPORTS_TOOL_CALLING),
            max_context_window=int(settings.LOCAL_CONTEXT_WINDOW),
            rate_limits={"tier": "self_hosted", "quota": "unlimited"},
        )

    def estimate_usage(self, request: ProviderRequest) -> UsageEstimate:
        """Token counts are estimated as for cloud providers; **cost is zero**.

        A local GPU/CPU cycle has no marginal dollar cost, so charging a notional
        price would make the budget guardrails degrade a request that is in fact
        free.
        """
        total_chars = sum(len(m.content) for m in request.messages)
        if request.system_instruction:
            total_chars += len(request.system_instruction)
        prompt_tokens = max(1, total_chars // 4)
        max_completion = request.max_tokens or 1000
        return UsageEstimate(
            estimated_prompt_tokens=prompt_tokens,
            estimated_completion_tokens=max_completion,
            estimated_total_tokens=prompt_tokens + max_completion,
            estimated_cost_usd=0.0,
        )

    async def health(self) -> bool:
        """Probe ``GET /models`` and report whether the local server answers.

        Returns ``False`` rather than raising so that ``/health/providers`` can
        include local in its report on a machine where no server is running —
        which is the normal state of a laptop that has not started Ollama.
        """
        url = f"{self.base_url}/models"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(url, headers=self._get_headers())
            return response.status_code == 200
        except Exception as exc:  # noqa: BLE001 - health probes never raise
            logger.debug("LocalProvider health probe failed: %s", type(exc).__name__)
            return False

    async def discover_models(self) -> list[str]:
        """Return the model ids the local server actually serves.

        Used by tooling (and by ``/v1/providers/models``) to tell an operator that
        ``LOCAL_MODEL`` does not match anything the server has loaded — the single
        most common cause of a confusing local 404.
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/models", headers=self._get_headers())
            if response.status_code != 200:
                return list(FALLBACK_MODEL_LIST)
            payload: dict[str, Any] = response.json()
            models = [
                str(entry.get("id"))
                for entry in payload.get("data", [])
                if isinstance(entry, dict) and entry.get("id")
            ]
            return models or list(FALLBACK_MODEL_LIST)
        except Exception as exc:  # noqa: BLE001 - discovery is best-effort
            logger.debug("LocalProvider model discovery failed: %s", type(exc).__name__)
            return list(FALLBACK_MODEL_LIST)


__all__ = ["FALLBACK_MODEL_LIST", "LocalProvider"]
