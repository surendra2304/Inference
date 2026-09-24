"""OpenRouter LLM Provider Adapter with Dynamic Capability-Based Model Discovery."""

import time

import httpx

from app.core.config import settings
from app.providers.base import ProviderRequest, ProviderResponse
from app.providers.openai_compatible import OpenAICompatibleProvider
from app.utils.logger import logger

# Capability-to-model preference mapping for OpenRouter free tier
CAPABILITY_KEYWORDS: dict[str, list[str]] = {
    "coding": ["code", "coder", "python", "qwen", "deepseek", "dev", "starcoder", "codellama"],
    "reasoning": ["nemotron", "r1", "reasoning", "qwen", "llama-3.3", "llama-3.1", "instruct"],
    "research": ["llama", "mistral", "nemotron", "gemma", "phi"],
    "analysis": ["nemotron", "llama-3.3", "qwen", "mistral"],
    "security": ["mistral", "llama-3.3", "nemotron", "guard"],
    "general": ["nemotron-3.5-lightning:free", "llama", "mistral"],
}

FALLBACK_FREE_MODELS: dict[str, str] = {
    "coding": "liquid/lfm-2.5-2.6b:free",
    "reasoning": "nvidia/nemotron-3.5-lightning:free",
    "research": "liquid/lfm-2.5-2.6b:free",
    "analysis": "qwen/qwen3.8-27b:free",
    "security": "nvidia/nemotron-3.5-lightning:free",
    "general": "liquid/lfm-2.5-2.6b:free",
}


class OpenRouterProvider(OpenAICompatibleProvider):
    """Adapter for OpenRouter multi-model aggregation API with dynamic capability matching."""

    BASE_URL = "https://openrouter.ai/api/v1"
    DEFAULT_MODEL = "liquid/lfm-2.5-2.6b:free"
    SUPPORTED_MODELS = [
        "liquid/lfm-2.5-2.6b:free",
        "qwen/qwen3.8-27b:free",
        "nvidia/nemotron-3.5-lightning:free",
        "z-ai/glm-5.2:free",
        "poolside/laguna-s-2.1:free",
        "meta-llama/llama-3.3-70b-instruct",
    ]

    def __init__(self, api_key: str | None = None, default_model: str | None = None, timeout: float = 60.0) -> None:
        super().__init__(
            provider_name="openrouter",
            base_url=self.BASE_URL,
            api_key=api_key or settings.OPENROUTER_API_KEY,
            default_model=default_model or self.DEFAULT_MODEL,
            supported_models=self.SUPPORTED_MODELS,
            timeout=timeout,
            extra_headers={"HTTP-Referer": "https://github.com/surendra2304/Inference", "X-Title": "Inference"},
        )
        self._cached_free_models: list[str] = []
        self._cache_timestamp: float = 0.0
        self._cache_ttl_seconds: float = 300.0  # 5 minutes

    async def fetch_available_free_models(self) -> list[str]:
        """Fetch all currently available :free models directly from OpenRouter API."""
        now = time.time()
        if self._cached_free_models and (now - self._cache_timestamp) < self._cache_ttl_seconds:
            return self._cached_free_models

        if not self.api_key:
            return list(self.SUPPORTED_MODELS)

        url = f"{self.base_url}/models"
        headers = self._get_headers()
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url, headers=headers)
                if resp.status_code == 200:
                    data = resp.json().get("data", [])
                    free_models = [
                        m["id"]
                        for m in data
                        if m.get("id", "").endswith(":free")
                        or (m.get("pricing", {}).get("prompt") == "0" and m.get("pricing", {}).get("completion") == "0")
                    ]
                    if free_models:
                        self._cached_free_models = free_models
                        self._cache_timestamp = now
                        logger.info("Discovered %d active free models on OpenRouter", len(free_models))
                        return free_models
        except Exception as exc:
            logger.debug("Failed to query live OpenRouter models: %s", str(exc))

        return list(self.SUPPORTED_MODELS)

    async def get_best_free_model(self, capability: str = "general") -> str:
        """
        Dynamically selects the best available free model on OpenRouter for a given capability.
        Queries OpenRouter API (/api/v1/models) or uses cache, filtering for free models
        and matching against the requested capability tag.
        """
        return await self.find_model_by_capability(capability)

    async def find_model_by_capability(self, capability: str = "general") -> str:
        """
        Dynamically finds the best available free model matching a capability tag
        (e.g., 'coding', 'reasoning', 'research', 'analysis', 'security', 'general').
        """
        cap_lower = capability.lower().strip()
        models = await self.fetch_available_free_models()

        keywords = CAPABILITY_KEYWORDS.get(cap_lower, CAPABILITY_KEYWORDS["general"])
        for kw in keywords:
            for model_id in models:
                if kw in model_id.lower() and (model_id.endswith(":free") or ":free" in model_id):
                    logger.info("Dynamically selected OpenRouter model '%s' for capability '%s'", model_id, capability)
                    return model_id

        # Fallback to default free model
        fallback = FALLBACK_FREE_MODELS.get(cap_lower, self.DEFAULT_MODEL)
        return fallback

    async def generate_with_capability(self, request: ProviderRequest, capability: str = "general") -> ProviderResponse:
        """Dynamically resolves model via capability tag and generates response."""
        if not request.model or request.model == self.DEFAULT_MODEL:
            resolved_model = await self.get_best_free_model(capability)
            request.model = resolved_model
        return await self.generate(request)
