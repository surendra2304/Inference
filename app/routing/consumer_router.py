"""Multi-Consumer Router and Usage Attribution for Trading Bot, FORGE, FRIDAY, and Human users."""

from typing import Any, Literal

from pydantic import BaseModel

from app.core.config import settings

ConsumerType = Literal["trading_bot", "forge", "friday", "human", "nexus", "sentinel", "intelx", "futuris"]


class ConsumerProfile(BaseModel):
    name: ConsumerType
    rate_limit_per_hour: int
    priority: int  # Higher is higher priority
    mode: str
    description: str


class ConsumerUsageRecord(BaseModel):
    """Per-consumer usage, with unknown kept separate from zero.

    ``total_tokens`` used to start at 0 and every caller passed a *fixed* number — nexus 650,
    intelx 600, sentinel 550, futuris 500 — none of which came from a provider (none of those
    services calls one). The admin view then presented the sum as measured usage and derived
    ``estimated_cost_usd`` from it. A consumer whose tokens were never measured now reports
    ``None`` and ``unmeasured_calls``, not a plausible-looking number.
    """

    consumer: ConsumerType
    total_calls: int = 0
    #: Sum of measured tokens, or ``None`` when no call reported usage.
    total_tokens: int | None = None
    #: Number of calls that reported token usage, so a total is readable next to its coverage.
    tokens_measured_calls: int = 0
    total_latency_seconds: float = 0.0
    latency_measured_calls: int = 0
    #: Derived from measured tokens only.
    estimated_cost_usd: float | None = None
    unmeasured_calls: int = 0


class MultiConsumerRouter:
    """Manages consumer identity, rate-limit policies, priority queues, and usage accounting."""

    PROFILES: dict[ConsumerType, ConsumerProfile] = {
        "futuris": ConsumerProfile(
            name="futuris",
            rate_limit_per_hour=150,
            priority=2,
            mode="statistical_grounding",
            description="Futuris predictive forecasting and statistical grounding engine."
        ),
        "intelx": ConsumerProfile(
            name="intelx",
            rate_limit_per_hour=200,
            priority=2,
            mode="research_reasoning",
            description="IntelX autonomous deep research and evidence verification engine."
        ),
        "sentinel": ConsumerProfile(
            name="sentinel",
            rate_limit_per_hour=100,
            priority=2,
            mode="security_intelligence",
            description="Autonomous cybersecurity posture & threat analysis engine."
        ),
        "nexus": ConsumerProfile(
            name="nexus",
            rate_limit_per_hour=200,
            priority=2,
            mode="intelligence_routing",
            description="Nexus high-throughput intelligence decision engine."
        ),
        "forge": ConsumerProfile(
            name="forge",
            rate_limit_per_hour=200,
            priority=1,
            mode="code_generation",
            description="Autonomous software engineering engine with heavy per-file code generation."
        ),
        "trading_bot": ConsumerProfile(
            name="trading_bot",
            rate_limit_per_hour=20,
            priority=2,
            mode="conservative_advisory",
            description="Algorithmic trading consultation with isolated queue and strict bounds."
        ),
        "friday": ConsumerProfile(
            name="friday",
            rate_limit_per_hour=100,
            priority=3,
            mode="assistant",
            description="General purpose intelligence assistant."
        ),
        "human": ConsumerProfile(
            name="human",
            # Settings-driven, not hardcoded. Two independent limiters guard this API and
            # their limits disagreed by ~72x: this profile allowed 50 requests per hour
            # while ``ProductionSecurityMiddleware`` allowed 120 per *minute*, so the
            # tighter one silently governed and a human operator driving the agent hit the
            # ceiling after 50 requests, then waited ~72 seconds per request for the rest
            # of the hour. A quota that blocks the product's own operator from using it is
            # a defect, and a hardcoded quota cannot be tuned per deployment.
            rate_limit_per_hour=int(settings.HUMAN_RATE_LIMIT_PER_HOUR),
            priority=4,
            mode="interactive",
            description="Direct human developer queries with detailed explanations."
        )
    }

    def __init__(self) -> None:
        self.usage_records: dict[ConsumerType, ConsumerUsageRecord] = {
            k: ConsumerUsageRecord(consumer=k) for k in self.PROFILES.keys()
        }

    def identify_consumer(self, api_key_or_header: str | None) -> ConsumerType:
        """Determines the consumer from API key, header, or default fallback."""
        if not api_key_or_header:
            return "forge"  # Default for forge service paths
        val = api_key_or_header.lower()
        if "nexus" in val:
            return "nexus"
        elif "forge" in val:
            return "forge"
        elif "trading" in val or "bot" in val:
            return "trading_bot"
        elif "friday" in val:
            return "friday"
        return "human"

    #: $0.0005 per 1k tokens, applied only to measured tokens.
    TOKEN_COST_PER_1K_USD = 0.0005

    def record_usage(
        self,
        consumer: ConsumerType,
        tokens: int | None,
        latency_sec: float | None,
    ) -> None:
        """Accumulates usage metrics, keeping "not measured" out of the totals."""
        rec = self.usage_records.get(consumer)
        if not rec:
            return
        rec.total_calls += 1
        if tokens is None:
            rec.unmeasured_calls += 1
        else:
            rec.total_tokens = (rec.total_tokens or 0) + tokens
            rec.tokens_measured_calls += 1
            rec.estimated_cost_usd = round(
                (rec.estimated_cost_usd or 0.0) + (tokens / 1000.0) * self.TOKEN_COST_PER_1K_USD, 6
            )
        if latency_sec is not None:
            rec.total_latency_seconds = round(rec.total_latency_seconds + latency_sec, 6)
            rec.latency_measured_calls += 1

    def get_usage(self, consumer: ConsumerType | None = None) -> dict[str, Any]:
        """Returns usage stats."""
        if consumer and consumer in self.usage_records:
            return self.usage_records[consumer].model_dump()
        return {k: v.model_dump() for k, v in self.usage_records.items()}


consumer_router = MultiConsumerRouter()
