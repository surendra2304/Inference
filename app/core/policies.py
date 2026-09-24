"""System policies, execution guardrails, and provider switching rules across 7 Free Cloud Providers."""

from enum import Enum

from pydantic import BaseModel

from app.utils.logger import logger


class SwitchReason(str, Enum):
    QUOTA = "quota"
    LATENCY = "latency"
    CAPABILITY = "capability"
    EXPERIMENT = "experiment"
    TIMEOUT = "timeout"


class FallbackRoute(BaseModel):
    """Fallback mapping for a given primary provider."""
    primary_provider: str
    fallback_provider: str
    fallback_model: str


# Explicit provider fallback configuration across active verified providers
PROVIDER_FALLBACK_MATRIX: dict[str, FallbackRoute] = {
    "gemini": FallbackRoute(
        primary_provider="gemini",
        fallback_provider="groq",
        fallback_model="openai/gpt-oss-120b"
    ),
    "groq": FallbackRoute(
        primary_provider="groq",
        fallback_provider="nvidia",
        fallback_model="nvidia/nemotron-3-super-120b-a12b"
    ),
    "nvidia": FallbackRoute(
        primary_provider="nvidia",
        fallback_provider="gemini",
        fallback_model="gemini-3.8-flash"
    ),
    "openrouter": FallbackRoute(
        primary_provider="openrouter",
        fallback_provider="gemini",
        fallback_model="gemini-3.8-flash"
    ),
    "mistral": FallbackRoute(
        primary_provider="mistral",
        fallback_provider="groq",
        fallback_model="openai/gpt-oss-120b"
    ),
    "cohere": FallbackRoute(
        primary_provider="cohere",
        fallback_provider="gemini",
        fallback_model="gemini-3.8-flash"
    ),
    "huggingface": FallbackRoute(
        primary_provider="huggingface",
        fallback_provider="gemini",
        fallback_model="gemini-3.8-flash"
    ),
}


class ProviderSwitchingPolicy:
    """
    Governs provider selection and failover.
    Core Invariant: Never silently switch models in the middle of a consequential debate stage.
    All fallback events must be explicitly logged and auditable.
    """

    @staticmethod
    def get_fallback_provider(
        primary_provider: str,
        reason: SwitchReason,
        stage: str | None = None
    ) -> FallbackRoute | None:
        """
        Determines appropriate fallback provider upon error/quota exhaustion.
        Logs audit record with reason.
        """
        route = PROVIDER_FALLBACK_MATRIX.get(primary_provider.lower())
        if route:
            logger.warning(
                "PROVIDER SWITCH EVENT: Switching from '%s' to '%s' (Model: %s) at stage '%s'. Reason: %s",
                primary_provider,
                route.fallback_provider,
                route.fallback_model,
                stage or "general_execution",
                reason.value
            )
            return route
        return None

    @staticmethod
    def can_switch_in_stage(stage: str, allow_mid_stage: bool = False) -> bool:
        """
        Enforces policy: Consequential debate stages (e.g. cross_review, synthesis)
        must not silently switch unless explicitly allowed with auditable provenance.
        """
        consequential_stages = ["cross_review_critique", "consensus_synthesis", "evidence_check"]
        if stage in consequential_stages and not allow_mid_stage:
            logger.info("Mid-stage model switch restricted for consequential stage '%s'", stage)
            return False
        return True


class SystemPolicies:
    """Execution bounds, budget thresholds, timeouts, and safety policies."""
    DEFAULT_MAX_BUDGET_USD: float = 999999.0
    DEFAULT_MAX_LATENCY_SECONDS: float = 60.0
    FAST_MODE_LATENCY_THRESHOLD_SECONDS: float = 3.0
    REVIEW_MODE_LATENCY_THRESHOLD_SECONDS: float = 10.0
    FAST_MODE_BUDGET_THRESHOLD_USD: float = 0.005
    REVIEW_MODE_BUDGET_THRESHOLD_USD: float = 0.02

    # Allowlist of safe operations
    ALLOWED_TOOLS: list[str] = ["read_memory", "search_knowledge", "calculate"]
