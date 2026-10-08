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
    CORS_ALLOWED_ORIGINS: list[str] = Field(default_factory=lambda: [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "https://inference-h7bn.onrender.com",
    ])
    INFERENCE_URL: str = Field(default="https://inference-h7bn.onrender.com", description="Canonical live deployment URL for Inference")

    # Security & Auth Configuration
    INSECURE_DEV_AUTH: bool = Field(default=False, description="Explicitly enable unauthenticated development mode (NEVER for production)")

    # Database
    DATABASE_URL: str = "sqlite:///data/universe.db"

    #: Ceiling for the per-tenant request-id deduplication window. Each entry retains a
    #: full response payload; a deduplicated response is an optimisation (a miss is
    #: recomputed), so this trades memory for occasional recomputation.
    DEDUP_CACHE_MAX_ENTRIES: int = 1000

    #: Enables GET /v1/operational/memory/diagnostics (tracemalloc attribution of live
    #: allocations). Off by default: the output names internal file paths and line numbers.
    ENABLE_MEMORY_DIAGNOSTICS: bool = False

    #: Allocator high-water guard. Measured (1,600 /ask requests, concurrency 6, tracemalloc
    #: off): live Python blocks return to their baseline exactly, but RSS stays 1.3 MB higher
    #: because glibc keeps freed arenas mapped. Left alone, every burst ratchets the resident
    #: set up until a small container OOM-kills the process. The guard calls glibc's
    #: ``malloc_trim`` once the resident set is above the threshold and no request is in
    #: flight; it returns free memory only, so it cannot disturb live data.
    MEMORY_TRIM_ENABLED: bool = True
    MEMORY_TRIM_THRESHOLD_MB: float = 400.0
    MEMORY_TRIM_COOLDOWN_SECONDS: float = 60.0

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
    TRUST_PROXY_HEADERS: bool = Field(
        default=True,
        description=(
            "Honour X-Forwarded-For / X-Real-IP when determining the client IP for "
            "rate limiting. Headers are ONLY consulted when the direct TCP peer is "
            "loopback or listed in TRUSTED_PROXY_CIDRS, so a public client cannot "
            "forge them. Disable if the service is reached directly with no proxy."
        ),
    )
    TRUSTED_PROXY_CIDRS: list[str] = Field(
        default_factory=list,
        description="Extra CIDR ranges treated as trusted reverse proxies (e.g. 10.0.0.0/8)",
    )

    # LiteLLM Integration (Optional Transport & Fallback Layer)
    INFERENCE_LITELLM_ENABLED: bool = Field(default=False, description="Enable LiteLLM unified model transport layer")
    INFERENCE_LITELLM_FALLBACK_ENABLED: bool = Field(default=True, description="Allow falling back to LiteLLM when native routes fail")
    LITELLM_DEFAULT_TIMEOUT: float = Field(default=60.0, description="Default LiteLLM request timeout")
    LITELLM_DROP_PARAMS: bool = Field(default=True, description="Drop non-standard parameters in LiteLLM calls")
    LITELLM_SUCCESS_CALLBACKS: str = Field(default="", description="LiteLLM success callback handlers")
    LITELLM_FAILURE_CALLBACKS: str = Field(default="", description="LiteLLM failure callback handlers")
    LITELLM_MODEL_ALIASES_JSON: str = Field(default="{}", description="JSON mapping for LiteLLM model aliases")

    # ------------------------------------------------------------------
    # Local (self-hosted) inference — the zero-cost, zero-quota tier.
    #
    # Disabled by default so that a deployment with no local server running does
    # not pay a connection-refused penalty on every fallback. Enabling it adds
    # `local` to the provider pool, to the routing candidates, and as the
    # terminal rung of the fallback ladder.
    # ------------------------------------------------------------------
    LOCAL_ENABLED: bool = Field(
        default=False,
        description=(
            "Enable the self-hosted OpenAI-compatible provider ('local'). Requires "
            "LOCAL_BASE_URL to point at a running server (Ollama, llama.cpp, vLLM, SGLang)."
        ),
    )
    LOCAL_BASE_URL: str = Field(
        default="http://127.0.0.1:11434/v1",
        description=(
            "Base URL of the local OpenAI-compatible server, including the /v1 suffix. "
            "Ollama 127.0.0.1:11434/v1, llama.cpp llama-server 127.0.0.1:8080/v1, vLLM 127.0.0.1:8000/v1."
        ),
    )
    LOCAL_MODEL: str = Field(
        default="local-model",
        description=(
            "Model id to request from the local server. Must match a tag the server has loaded; "
            "run 'python -m app.core.key_inventory --probe-local' or GET /v1/providers/models to list them."
        ),
    )
    LOCAL_API_KEY: str | None = Field(
        default=None,
        description=(
            "Optional bearer token for the local server. Local servers normally need none; set this "
            "only when the endpoint sits behind an authenticating reverse proxy."
        ),
    )
    LOCAL_CONTEXT_WINDOW: int = Field(
        default=16384,
        description=(
            "Context window of the locally loaded model. Deliberately explicit rather than assumed: "
            "the cloud adapters advertise 128k, and letting the router believe a 4k local model has "
            "128k would produce prompts the server rejects."
        ),
    )
    LOCAL_SUPPORTS_TOOL_CALLING: bool = Field(
        default=False,
        description="Whether the locally loaded model supports OpenAI-style tool calling.",
    )
    HUMAN_RATE_LIMIT_PER_HOUR: int = Field(
        default=600,
        ge=1,
        description=(
            "Hourly request allowance for the interactive 'human' consumer profile, "
            "enforced as a token bucket, so bursts are allowed and the sustained rate is "
            "this value divided by 3600 seconds. The previous hardcoded value was 50/hour, "
            "which contradicted the request-level limiter (120/minute) by ~72x and locked a "
            "human operator out of the product after 50 requests. Default 600/hour = 10 "
            "requests per minute sustained, which still leaves the other limiter as the "
            "binding constraint for scripted traffic."
        ),
    )
    SQLITE_POOL_SIZE: int = Field(
        default=8,
        ge=1,
        le=64,
        description=(
            "Maximum concurrent SQLite connections per memory store. Each in-flight "
            "connection holds ~37 file descriptors (db + -wal + -shm + pipe + thread), "
            "so an unbounded connection-per-call pattern makes the OS descriptor limit "
            "the service's real concurrency ceiling."
        ),
    )
    PEER_COVERAGE_MAX_ATTEMPTS: int = Field(
        default=3,
        ge=0,
        description=(
            "How many peer specialists may be tried to cover one failed specialist. "
            "Unbounded peer coverage is a stampede: every registered agent becomes an "
            "extra sequential model call, so a single provider outage turns one request "
            "into a minute of retries. Measured before this cap: 18 peer attempts x ~3s "
            "= 55s for a request whose budget was 30s."
        ),
    )
    PEER_COVERAGE_BUDGET_SECONDS: float = Field(
        default=20.0,
        gt=0.0,
        description=(
            "Wall-clock ceiling for the whole round-1 specialist phase, including peer "
            "coverage. This is the backstop that keeps a request bounded when many "
            "specialists fail at once: without it, the panel's runtime is proportional "
            "to the number of registered agents rather than to its own deadline."
        ),
    )
    LOCAL_MAX_CONCURRENCY: int = Field(
        default=16,
        ge=1,
        description=(
            "Maximum simultaneous requests to the self-hosted server. This — not a rate — is the real "
            "constraint on local inference: the host has finite compute slots, but no quota."
        ),
    )
    LOCAL_MAX_RPS: float = Field(
        default=1000.0,
        gt=0.0,
        description=(
            "Token-bucket refill rate for the self-hosted tier. Deliberately very high: a local server "
            "has no vendor quota, and the panel fans one request out to 8+ model calls, so a low rate "
            "silently ceilings throughput for the whole agent. Measured effect of the previous 10 req/s: "
            "latency stepped from 0.11s to 0.80s once the bucket drained, and stayed there. Use "
            "LOCAL_MAX_CONCURRENCY to protect the host instead."
        ),
    )
    LOCAL_PREFERRED: bool = Field(
        default=False,
        description=(
            "Air-gapped mode: answer every request from the self-hosted tier first and attempt no "
            "outbound provider call. Use for offline development or when prompts must not leave the "
            "machine. If the local tier fails, the request falls through to the cloud providers with a "
            "loud warning — set LOCAL_ENABLED and leave LOCAL_PREFERRED off if you need cloud-first "
            "behaviour instead."
        ),
    )

    # ------------------------------------------------------------------
    # Gateway deadline policy
    # ------------------------------------------------------------------
    PROVIDER_HEALTH_RECOVERY_SECONDS: float = Field(
        default=30.0,
        ge=0.0,
        description=(
            "Seconds after a provider's most recent failure before it is offered a half-open "
            "probe. Callers skip providers reported unhealthy, and only a successful call can "
            "clear an unhealthy verdict — so without this cooldown a provider that accumulated "
            "consecutive failures could never be retried and would stay disabled for the life of "
            "the process. Set to 0 for immediate probing."
        ),
    )
    PRIMARY_ATTEMPT_FRACTION: float = Field(
        default=0.6,
        ge=0.1,
        le=1.0,
        description=(
            "Fraction of the total request deadline a primary provider attempt may consume before the "
            "gateway escalates to the fallback ladder. The remainder is reserved for fallbacks. Without "
            "this reservation a primary that times out exhausts the whole budget and the ladder collapses "
            "to re-raising the original error — i.e. the resilience design becomes unreachable exactly "
            "when it is needed."
        ),
    )

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
