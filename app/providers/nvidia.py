"""NVIDIA NIM LLM Provider Adapter."""

from app.core.config import settings
from app.providers.openai_compatible import OpenAICompatibleProvider

NVIDIA_DEFAULT_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"

NVIDIA_SUPPORTED_MODELS: list[str] = [
    "nvidia/nemotron-3-ultra-550b-a55b",
    "nvidia/nemotron-3.5-lightning-30b-a3b",
    "nvidia/llama-3.1-nemotron-70b-instruct",
    "mistralai/mistral-large-2-instruct",
    "nvidia/nemotron-3.5-content-safety",
]


class NvidiaProvider(OpenAICompatibleProvider):
    """NVIDIA NIM inference adapter inheriting from OpenAICompatibleProvider."""

    def __init__(
        self, api_key: str | None = None, default_model: str = NVIDIA_DEFAULT_MODEL, timeout: float = 60.0
    ) -> None:
        key = api_key or settings.NVIDIA_API_KEY
        super().__init__(
            provider_name="nvidia",
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=key,
            default_model=default_model,
            supported_models=NVIDIA_SUPPORTED_MODELS,
            timeout=timeout,
        )
