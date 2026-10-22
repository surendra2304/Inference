"""Provider Real-Time Cost Tracking, Cost Efficiency, Anomaly Detection & Adaptive Leaderboard."""

import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.utils.bounded_store import DEFAULT_MAX_RECORDS, BoundedList
from app.utils.logger import logger


class CostRecord(BaseModel):
    timestamp: float = Field(default_factory=time.time)
    provider: str
    consumer: str
    task_type: str
    cost_usd: float
    is_success: bool = True
    #: ``fixture`` for the seeded examples, ``observed`` for a real logged call.
    source: Literal["observed", "fixture"] = "observed"
    #: Where the money number came from. A provider that returns token usage but no price
    #: list can only be costed from a *declared* rate, and a report that does not say which
    #: of the two it used presents an estimate as a bill.
    price_basis: Literal["provider_reported", "declared_estimate", "unavailable"] = "unavailable"
    is_estimate: bool = True


class ProviderCostTracker:
    """Tracks cost per successful outcome ($/successful_outcome), daily spikes (>3x average), and adaptive leaderboards."""

    #: Retention for the cost ledger (bounded: appended per provider call).
    MAX_RECORDS = DEFAULT_MAX_RECORDS

    def __init__(self) -> None:
        # Seeded examples, labelled ``fixture`` so the report can exclude them from spend.
        #: Count of anomalies actually raised by :meth:`log_cost_event`. The report used to
        #: publish ``anomaly_detected: False`` unconditionally, so the field could not
        #: disagree with reality even when a spike had just been logged.
        self.anomalies_detected = 0
        self.records: BoundedList[CostRecord] = BoundedList(
            "cost_tracking.records",
            max_entries=self.MAX_RECORDS,
            initial=[
                CostRecord(provider="groq", consumer="forge", task_type="code_generation", cost_usd=0.00045, is_success=True, source="fixture"),
                CostRecord(provider="gemini", consumer="nexus", task_type="lead_qualification", cost_usd=0.00072, is_success=True, source="fixture"),
                CostRecord(provider="nvidia", consumer="forge", task_type="architecture", cost_usd=0.00120, is_success=True, source="fixture"),
                CostRecord(provider="groq", consumer="trading_bot", task_type="trading_consult", cost_usd=0.00050, is_success=True, source="fixture"),
            ],
        )

    def daily_average_cost_usd(self) -> float | None:
        """Mean observed cost per call, computed from the ledger.

        This used to be the attribute ``self.daily_average_cost_usd = 0.50`` — a literal
        that no code ever recomputed, used to decide whether a request cost is anomalous
        ("$x is >3x daily average"). An anomaly alert based on an invented average is a
        number about nothing. ``None`` when the ledger holds no observed call.
        """
        observed = [r for r in self.records if getattr(r, "source", "observed") == "observed"]
        if not observed:
            return None
        return sum(r.cost_usd for r in observed) / len(observed)

    def log_cost_event(
        self,
        provider: str,
        consumer: str,
        task_type: str,
        cost_usd: float,
        is_success: bool = True,
        price_basis: Literal["provider_reported", "declared_estimate", "unavailable"] = "declared_estimate",
        is_estimate: bool = True,
    ) -> CostRecord:
        """Record one measured call's cost, saying where the number came from."""
        record = CostRecord(
            provider=provider,
            consumer=consumer,
            task_type=task_type,
            cost_usd=cost_usd,
            is_success=is_success,
            price_basis=price_basis,
            is_estimate=is_estimate,
        )
        self.records.append(record)
        average = self.daily_average_cost_usd()
        if average is not None and cost_usd > (average * 3.0):
            self.anomalies_detected += 1
            logger.warning(
                "[COST ANOMALY ALERT] Request cost ($%0.4f) is >3x the observed average ($%0.6f).",
                cost_usd, average,
            )
        return record

    def get_cost_report(self) -> dict[str, Any]:
        """Calculates cost per successful outcome and month-end spend projections."""
        observed = [r for r in self.records if getattr(r, "source", "observed") == "observed"]
        total_cost = sum(r.cost_usd for r in observed)
        successful_recs = [r for r in observed if r.is_success]
        cost_per_success = total_cost / len(successful_recs) if successful_recs else None

        # Leaderboard based on cost efficiency. Providers with no observed call are
        # reported as such: they used to receive an invented ``p_cost = 0.001`` with one
        # invented success, which produced a concrete $/success figure and a
        # "PROMOTE_PRIMARY" recommendation for a provider that had never run.
        providers = ["groq", "gemini", "openrouter", "mistral", "nvidia", "cohere", "huggingface"]
        leaderboard: list[dict[str, Any]] = []
        for p in providers:
            p_recs = [r for r in observed if r.provider == p]
            if not p_recs:
                leaderboard.append({
                    "provider": p,
                    "cost_per_successful_outcome_usd": None,
                    "total_spend_usd": 0.0,
                    "samples": 0,
                    "recommendation": "NO_DATA",
                })
                continue
            p_cost = sum(r.cost_usd for r in p_recs)
            p_succ = sum(1 for r in p_recs if r.is_success)
            cost_eff = p_cost / p_succ if p_succ else None
            leaderboard.append({
                "provider": p,
                "cost_per_successful_outcome_usd": round(cost_eff, 6) if cost_eff is not None else None,
                "total_spend_usd": round(p_cost, 4),
                "samples": len(p_recs),
                "recommendation": (
                    "NO_SUCCESSFUL_OUTCOME" if cost_eff is None
                    else ("PROMOTE_PRIMARY" if cost_eff < 0.0008 else "EXPLORE_OR_SECONDARY")
                ),
            })

        leaderboard.sort(
            key=lambda x: (
                x["cost_per_successful_outcome_usd"] is None,
                x["cost_per_successful_outcome_usd"] or 0.0,
            )
        )

        # A "monthly projection" needs an observation window. ``total_cost * 30`` assumed the
        # ledger held exactly one day of data — with one call recorded that produced a
        # month-end spend figure from a single sample, and with no calls it printed $0.00,
        # which reads like "this deployment is free".
        span_seconds = None
        if observed:
            span_seconds = max(r.timestamp for r in observed) - min(r.timestamp for r in observed)
        projected = None
        projection_basis = "no observed call; a projection would be a guess"
        if observed and span_seconds and span_seconds >= 60.0:
            projected = round(total_cost * (30.0 * 86400.0 / span_seconds), 2)
            projection_basis = (
                f"extrapolated from {len(observed)} observed call(s) spanning "
                f"{span_seconds / 3600.0:.2f}h of wall clock"
            )
        elif observed:
            projection_basis = (
                f"observation window is {span_seconds or 0.0:.0f}s, too short to extrapolate "
                f"to a month; {len(observed)} call(s) recorded"
            )

        bases = sorted({r.price_basis for r in observed})
        return {
            "total_spend_usd": round(total_cost, 4),
            "spend_basis": bases or ["no observed call"],
            "all_observed_costs_are_estimates": all(r.is_estimate for r in observed) if observed else None,
            "cost_per_successful_outcome_usd": round(cost_per_success, 6) if cost_per_success is not None else None,
            "projected_monthly_spend_usd": projected,
            "projection_basis": projection_basis,
            "observed_calls": len(observed),
            "fixture_rows_excluded": len(self.records) - len(observed),
            "records_dropped_for_capacity": self.records.dropped,
            "anomaly_detected": self.anomalies_detected > 0,
            "anomalies_detected_count": self.anomalies_detected,
            "provider_leaderboard": leaderboard
        }


provider_cost_tracker = ProviderCostTracker()
