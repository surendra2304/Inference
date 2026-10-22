"""Production Configuration for Inference."""

from pydantic_settings import BaseSettings


class ProductionConfig(BaseSettings):
    """Production runtime settings for high throughput & resilience."""
    APP_NAME: str = "Inference"
    APP_ENV: str = "production"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "WARNING"  # WARNING for general logs, ERROR for critical

    # Performance & Concurrency
    MAX_CONCURRENT_REQUESTS: int = 100
    CONSULTATION_TIMEOUT_SECONDS: float = 180.0
    AGENT_INVOCATION_TIMEOUT_SECONDS: float = 3.5  # Fast agent reasoning timeout to ensure p95 < 30s
    PROVIDER_CONNECTION_POOL_SIZE: int = 50

    # Caching
    CACHE_ENABLED: bool = True
    CACHE_TTL_SECONDS: int = 86400  # 24 hours
    CACHE_MAX_ENTRIES: int = 5000

    # Rate Limiting & Safety
    RATE_LIMIT_WINDOW_SECONDS: float = 3600.0
    RATE_LIMIT_MAX_REQUESTS_PER_BOT: int = 20
    MAX_REQUEST_SIZE_BYTES: int = 1024 * 1024  # 1MB

    # Provider Fallback Chain Optimization (Free-Tier Prioritized)
    PROVIDER_PRIORITY: list[str] = [
        "gemini",
        "groq",
        "cerebras",
        "mistral",
        "openrouter",
        "huggingface",
        "nvidia",
        "cohere",
        "ollama"
    ]

    # Circuit Breaker Configuration
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = 3
    CIRCUIT_BREAKER_RECOVERY_TIME_SECONDS: float = 60.0


production_config = ProductionConfig()
