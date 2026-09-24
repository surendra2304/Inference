"""Application configuration settings for Inference."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Application settings
    APP_NAME: str = "Inference"
    APP_VERSION: str = "2.0.0"
    APP_ENV: str = "production"
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"
    CORS_ALLOWED_ORIGINS: list[str] = Field(default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:8000", "http://127.0.0.1:8000"])

    # Security & Auth Configuration
    INSECURE_DEV_AUTH: bool = Field(default=False, description="Explicitly enable unauthenticated development mode (NEVER for production)")

    # Database
    DATABASE_URL: str = "sqlite:///data/universe.db"

    # 7 Active Cloud Provider API Keys (Supports single or comma-separated lists)
    GEMINI_API_KEY: str | None = Field(default=None)
    GEMINI_API_KEYS: str | None = Field(default=None)

    GROQ_API_KEY: str | None = Field(default=None)
    GROQ_API_KEYS: str | None = Field(default=None)

    MISTRAL_API_KEY: str | None = Field(default=None)
    MISTRAL_API_KEYS: str | None = Field(default=None)

    OPENROUTER_API_KEY: str | None = Field(default=None)
    OPENROUTER_API_KEYS: str | None = Field(default=None)

    COHERE_API_KEY: str | None = Field(default=None)
    COHERE_API_KEYS: str | None = Field(default=None)

    HUGGINGFACE_API_KEY: str | None = Field(default=None)
    HUGGINGFACE_API_KEYS: str | None = Field(default=None)

    NVIDIA_API_KEY: str | None = Field(default=None)
    NVIDIA_API_KEYS: str | None = Field(default=None)

    THIRD_PARTY_GEMINI_KEY: str | None = Field(default=None)
    THIRD_PARTY_GEMINI_KEYS: str | None = Field(default=None)
    GEMINI_FALLBACK_KEY: str | None = Field(default=None)
    GEMINI_FALLBACK_KEYS: str | None = Field(default=None)
    THIRD_PARTY_GROQ_KEY: str | None = Field(default=None)
    THIRD_PARTY_GROQ_KEYS: str | None = Field(default=None)
    GROQ_FALLBACK_KEY: str | None = Field(default=None)
    GROQ_FALLBACK_KEYS: str | None = Field(default=None)
    THIRD_PARTY_OPENROUTER_KEY: str | None = Field(default=None)
    THIRD_PARTY_OPENROUTER_KEYS: str | None = Field(default=None)
    OPENROUTER_FALLBACK_KEY: str | None = Field(default=None)
    OPENROUTER_FALLBACK_KEYS: str | None = Field(default=None)
    THIRD_PARTY_NVIDIA_KEY: str | None = Field(default=None)
    THIRD_PARTY_NVIDIA_KEYS: str | None = Field(default=None)
    THIRD_PARTY_FALLBACK_API_KEY: str | None = Field(default=None)

    # Integration Keys (Strict: No hardcoded fallback credentials)
    INFERENCE_API_KEY: str | None = Field(default=None)
    inference_api_KEY: str | None = Field(default=None)
    FRIDAY_UNIVERSE_API_KEY: str | None = Field(default=None)
    X_FRIDAY_API_KEY: str | None = Field(default=None)
    FRIDAY_API_KEY: str | None = Field(default=None)

    def get_friday_api_key(self) -> str | None:
        return self.INFERENCE_API_KEY or self.inference_api_KEY or self.FRIDAY_UNIVERSE_API_KEY or self.X_FRIDAY_API_KEY or self.FRIDAY_API_KEY

    # Operational Budgets & Limits
    MAX_BUDGET: float = Field(default=999999.0, description="Legacy spend limit threshold")
    DEFAULT_BUDGET_USD: float = Field(default=10.0, description="Explicit hard budget ceiling per tenant/period in USD")
    REQUEST_TIMEOUT: float = Field(default=60.0, description="Default timeout in seconds for provider calls")
    ALLOW_DEV_RATE_LIMIT_BYPASS: bool = Field(default=False, description="Explicit flag required to bypass rate limits on localhost/testclient")

    # LiteLLM Integration (Optional Transport & Fallback Layer)
    INFERENCE_LITELLM_ENABLED: bool = Field(default=False, description="Enable LiteLLM unified model transport layer")
    INFERENCE_LITELLM_FALLBACK_ENABLED: bool = Field(default=True, description="Allow falling back to LiteLLM when native routes fail")
    LITELLM_DEFAULT_TIMEOUT: float = Field(default=60.0, description="Default LiteLLM request timeout")
    LITELLM_DROP_PARAMS: bool = Field(default=True, description="Drop non-standard parameters in LiteLLM calls")
    LITELLM_SUCCESS_CALLBACKS: str = Field(default="", description="LiteLLM success callback handlers")
    LITELLM_FAILURE_CALLBACKS: str = Field(default="", description="LiteLLM failure callback handlers")
    LITELLM_MODEL_ALIASES_JSON: str = Field(default="{}", description="JSON mapping for LiteLLM model aliases")

    def get_provider_keys(self, provider_name: str) -> list[str]:
        """
        Returns a deduplicated list of non-empty API keys for the specified provider.
        Checks both plural (e.g. GEMINI_API_KEYS) and singular (e.g. GEMINI_API_KEY) variables.
        Supports comma-separated strings in both.
        """
        prov = provider_name.upper().strip()
        keys: list[str] = []

        singular_val = getattr(self, f"{prov}_API_KEY", None)
        plural_val = getattr(self, f"{prov}_API_KEYS", None)
        tp_singular_val = getattr(self, f"THIRD_PARTY_{prov}_KEY", None)
        tp_plural_val = getattr(self, f"THIRD_PARTY_{prov}_KEYS", None)
        fb_singular_val = getattr(self, f"{prov}_FALLBACK_KEY", None)
        fb_plural_val = getattr(self, f"{prov}_FALLBACK_KEYS", None)
        legacy_val = self.THIRD_PARTY_FALLBACK_API_KEY if prov == "GEMINI" else None

        for raw_val in [plural_val, singular_val, tp_singular_val, tp_plural_val, fb_singular_val, fb_plural_val, legacy_val]:
            if raw_val:
                for k in raw_val.split(","):
                    cleaned = k.strip().strip("'\"")
                    if cleaned and cleaned not in keys:
                        keys.append(cleaned)

        return keys


settings = Settings()
