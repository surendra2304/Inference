"""HuggingFace Inference API LLM Provider Adapter."""

from app.core.config import settings
from app.providers.openai_compatible import OpenAICompatibleProvider

HUGGINGFACE_DEFAULT_MODEL = "meta-llama/llama-3.1-8b-instruct"

HUGGINGFACE_SUPPORTED_MODELS: list[str] = [
    "meta-llama/llama-3.1-8b-instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen2.5-72B-Instruct",
]


class HuggingFaceProvider(OpenAICompatibleProvider):
    """HuggingFace Serverless Inference adapter via high-performance router."""

    def __init__(
        self, api_key: str | None = None, default_model: str = HUGGINGFACE_DEFAULT_MODEL, timeout: float = 60.0
    ) -> None:
        key = api_key or settings.HUGGINGFACE_API_KEY
        super().__init__(
            provider_name="huggingface",
            base_url="https://router.huggingface.co/novita/v3/openai",
            api_key=key,
            default_model=default_model,
            supported_models=HUGGINGFACE_SUPPORTED_MODELS,
            timeout=timeout,
        )
