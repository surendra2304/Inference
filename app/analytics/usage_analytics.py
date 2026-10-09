"""Usage Analytics Engine for Multi-Consumer Tracking (FORGE, Trading Bot, FRIDAY, Human)."""

import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.utils.bounded_store import DEFAULT_MAX_RECORDS, BoundedList


class RequestAnalyticsRecord(BaseModel):
    r"""One observed request.

    Every field that can be unknown is optional, and unknown is *not* zero. The previous
    shape gave ``provider`` no default (so callers invented one), and defaulted the token
    counts to 0 with ``cost_usd`` derived from them — a request that never touched a model
    was indistinguishable from one that used a provider and spent nothing. [FACT] all four
    services that logged into this table passed ``provider="gemini"`` and fixed token counts
    while none of them calls a provider at all (``grep -c "model_gateway\." `` = 0 for
    ``nexus_intelligence``/``intelx_intelligence``/``sentinel_intelligence``/
    ``futuris_enhancement``), so the analytics table reported provider attribution and spend
    for traffic that had neither.
    """

    timestamp: float = Field(default_factory=time.time)
    consumer: str
    service: str
    #: Provider that actually served the request; ``None`` when no provider was involved or
    #: the caller did not report one. Never a guess.
    provider: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    total_tokens: int | None = None
    latency_ms: float | None = None
    success: bool = True
    #: None when the caller reported no confidence. The old default of 0.90 put a
    #: measured-looking confidence on requests that never had one.
    confidence: float | None = None
    #: ``fixture`` for the seeded demonstration rows, ``observed`` for a real request.
    source: Literal["observed", "fixture"] = "observed"
    #: Derived from measured tokens only; ``None`` means "not measurable" rather than free.
    cost_usd: float | None = None
    #: Names the fields the caller could not measure, so a consumer of this record can state
    #: what is missing instead of summing zeros and calling it a total.
    unmeasured_fields: list[str] = Field(default_factory=list)


class UsageAnalyticsEngine:
    """Aggregates per-request tokens, latency, cost attribution, and daily ceiling budgets."""

    def __init__(self) -> None:
        self.records: BoundedList[RequestAnalyticsRecord] = BoundedList(
            "usage_analytics.records",
            max_entries=DEFAULT_MAX_RECORDS,
            initial=[
                RequestAnalyticsRecord(
                    consumer="forge", service="generate-code", provider="groq",
                    tokens_in=500, tokens_out=800, total_tokens=1300, latency_ms=32.4,
                    success=True, confidence=0.92, cost_usd=0.00065, source="fixture",
                ),
                RequestAnalyticsRecord(
                    consumer="forge", service="plan-architecture", provider="nvidia",
                    tokens_in=800, tokens_out=1200, total_tokens=2000, latency_ms=85.1,
                    success=True, confidence=0.95, cost_usd=0.001, source="fixture",
                ),
                RequestAnalyticsRecord(
                    consumer="trading_bot", service="trading_consult", provider="groq",
                    tokens_in=400, tokens_out=600, total_tokens=1000, latency_ms=45.0,
                    success=True, confidence=0.88, cost_usd=0.0005, source="fixture",
                ),
                RequestAnalyticsRecord(
                    consumer="friday", service="general_query", provider="gemini",
                    tokens_in=200, tokens_out=300, total_tokens=500, latency_ms=28.0,
                    success=True, confidence=0.94, cost_usd=0.00025, source="fixture",
                ),
            ],
        )
        #: The four ``initial`` records above are demonstration rows, labelled
        #: ``source="fixture"`` and excluded from every aggregate by ``_observed()``.
        self.daily_budget_usd = 10.0
        self.alert_threshold_pct = 0.80

    #: $0.0005 per 1k tokens — a proxy used only to turn *measured* tokens into a comparable
    #: cost signal. It is never applied to an unknown token count.
    TOKEN_COST_PER_1K_USD = 0.0005

    def log_request(
        self,
        consumer: str,
        service: str,
        provider: str | None = None,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
        latency_ms: float | None = None,
        success: bool = True,
        confidence: float | None = None,
    ) -> RequestAnalyticsRecord:
        """Record one request. Pass ``None`` for anything the caller did not measure.

        ``provider`` and the token counts used to be required positional fields, which is why
        callers filled them with ``"gemini"`` and made-up counts. They are now optional: the
        record stores "not measured", the derived cost stays ``None``, and
        :meth:`get_overview` reports how many records lack each field.
        """
        total_tokens: int | None = None
        if tokens_in is not None and tokens_out is not None:
            total_tokens = tokens_in + tokens_out
        cost_usd: float | None = None
        if total_tokens is not None:
            cost_usd = round((total_tokens / 1000.0) * self.TOKEN_COST_PER_1K_USD, 6)

        missing = [
            name
            for name, value in (
                ("provider", provider),
                ("tokens_in", tokens_in),
                ("tokens_out", tokens_out),
                ("latency_ms", latency_ms),
                ("confidence", confidence),
            )
            if value is None
        ]

        record = RequestAnalyticsRecord(
            consumer=consumer,
            service=service,
            provider=provider,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            success=success,
            confidence=confidence,
            cost_usd=cost_usd,
            unmeasured_fields=missing,
        )
        self.records.append(record)
        return record

    def _observed(self) -> list[RequestAnalyticsRecord]:
        """Only requests that actually happened; seeded demonstration rows are excluded."""
        return [r for r in self.records if r.source == "observed"]

    def _coverage(self, observed: list[RequestAnalyticsRecord]) -> dict[str, int]:
        """How many observed records actually carry each measurement.

        Published next to every aggregate so a total over 40% of the traffic cannot be read as
        a total over all of it.
        """
        return {
            field: sum(1 for r in observed if getattr(r, field) is not None)
            for field in ("provider", "tokens_in", "tokens_out", "latency_ms", "confidence", "cost_usd")
        }

    def get_overview(self) -> dict[str, Any]:
        observed = self._observed()
        total_calls = len(observed)
        token_records = [r for r in observed if r.total_tokens is not None]
        cost_records = [r for r in observed if r.cost_usd is not None]
        latency_records = [r.latency_ms for r in observed if r.latency_ms is not None]

        total_tokens = sum(r.total_tokens or 0 for r in token_records)
        total_cost = sum(r.cost_usd or 0.0 for r in cost_records)
        avg_latency = (sum(latency_records) / len(latency_records)) if latency_records else None

        # A budget can only be reported as "used" when spend was measurable. Previously the
        # sum over records with an unknown cost contributed 0.0, so unmeasured traffic read as
        # "no spend" — the opposite of the truth when a provider was not reporting tokens.
        measured_cost = bool(cost_records)
        budget_used_pct = (
            round((total_cost / max(0.1, self.daily_budget_usd)) * 100.0, 2)
            if measured_cost
            else None
        )

        return {
            "total_calls": total_calls,
            "total_tokens": total_tokens if token_records else None,
            "total_cost_usd": round(total_cost, 6) if measured_cost else None,
            "average_latency_ms": round(avg_latency, 2) if avg_latency is not None else None,
            "measurement_coverage": self._coverage(observed),
            "unmeasured_calls": {
                "tokens": total_calls - len(token_records),
                "cost": total_calls - len(cost_records),
                "latency": total_calls - len(latency_records),
            },
            "fixture_rows_excluded": len(self.records) - total_calls,
            "records_dropped_for_capacity": self.records.dropped,
            "daily_budget_usd": self.daily_budget_usd,
            "budget_used_pct": budget_used_pct,
            "ceiling_alert_active": (
                budget_used_pct >= self.alert_threshold_pct if budget_used_pct is not None else None
            ),
            "budget_basis": (
                "derived from measured token counts"
                if measured_cost
                else "no observed request reported token usage; spend cannot be stated"
            ),
        }

    def get_consumer_breakdown(self, consumer_id: str) -> dict[str, Any]:
        c_records = [r for r in self._observed() if r.consumer.lower() == consumer_id.lower()]
        total_calls = len(c_records)
        if total_calls == 0:
            # ``max(1, 0)`` used to turn an absence into ``success_rate_pct: 0.0`` — a
            # measured-looking claim that every call failed, when there were no calls.
            return {
                "consumer": consumer_id,
                "calls": 0,
                "tokens": 0,
                "cost_usd": 0.0,
                "success_rate_pct": None,
                "note": f"no observed request for consumer {consumer_id!r}",
            }
        measured_tokens = [r.total_tokens for r in c_records if r.total_tokens is not None]
        measured_cost = [r.cost_usd for r in c_records if r.cost_usd is not None]
        return {
            "consumer": consumer_id,
            "calls": total_calls,
            "tokens": sum(measured_tokens) if measured_tokens else None,
            "tokens_measured_calls": len(measured_tokens),
            "cost_usd": round(sum(measured_cost), 6) if measured_cost else None,
            "cost_measured_calls": len(measured_cost),
            "success_rate_pct": round((sum(1 for r in c_records if r.success) / total_calls) * 100.0, 1)
        }

    def get_service_breakdown(self, service_name: str) -> dict[str, Any]:
        s_records = [r for r in self._observed() if r.service.lower() == service_name.lower()]
        total_calls = len(s_records)
        measured_tokens = [r.total_tokens for r in s_records if r.total_tokens is not None]
        latency_records = [r.latency_ms for r in s_records if r.latency_ms is not None]
        return {
            "service": service_name,
            "calls": total_calls,
            "tokens": sum(measured_tokens) if measured_tokens else None,
            "tokens_measured_calls": len(measured_tokens),
            "avg_latency_ms": (
                round(sum(latency_records) / len(latency_records), 2) if latency_records else None
            ),
            "avg_latency_basis": f"{len(latency_records)} of {total_calls} calls reported latency",
        }

    def get_providers_comparison(self) -> dict[str, Any]:
        """Per-provider totals, keyed by the provider that actually served each request.

        Only providers with observed calls appear. The previous version returned a row for all
        seven configured providers with ``calls: 0, tokens: 0, cost_usd: 0.0``, which renders in
        a dashboard exactly like a provider that served traffic and spent nothing.
        """
        grouped: dict[str, list[RequestAnalyticsRecord]] = {}
        for record in self._observed():
            grouped.setdefault(record.provider or "unattributed", []).append(record)

        res: dict[str, Any] = {}
        for provider, records in sorted(grouped.items()):
            measured_tokens = [r.total_tokens for r in records if r.total_tokens is not None]
            measured_cost = [r.cost_usd for r in records if r.cost_usd is not None]
            latency_records = [r.latency_ms for r in records if r.latency_ms is not None]
            res[provider] = {
                "calls": len(records),
                "tokens": sum(measured_tokens) if measured_tokens else None,
                "tokens_measured_calls": len(measured_tokens),
                "cost_usd": round(sum(measured_cost), 6) if measured_cost else None,
                "avg_latency_ms": (
                    round(sum(latency_records) / len(latency_records), 2) if latency_records else None
                ),
            }
        return res


usage_analytics = UsageAnalyticsEngine()
