"""OpenAI LLM Provider Adapter."""

from app.core.config import settings
from app.providers.openai_compatible import OpenAICompatibleProvider


class OpenAIProvider(OpenAICompatibleProvider):
    """Adapter for OpenAI cloud API."""

    BASE_URL = "https://api.openai.com/v1"
    DEFAULT_MODEL = "gpt-4o"
    SUPPORTED_MODELS = [
        "gpt-4o",
        "gpt-4o-mini",
        "o1",
        "o3-mini",
    ]

    def __init__(self, api_key: str | None = None, default_model: str | None = None, timeout: float = 60.0) -> None:
        key = api_key or settings.OPENAI_API_KEY
        if not key:
            keys = settings.get_provider_keys("openai")
            key = keys[0] if keys else None
        super().__init__(
            provider_name="openai",
            base_url=self.BASE_URL,
            api_key=key,
            default_model=default_model or self.DEFAULT_MODEL,
            supported_models=self.SUPPORTED_MODELS,
            timeout=timeout,
        )
