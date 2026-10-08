"""Consumer Outcome Tracking for FORGE and Trading Bot."""

import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.utils.bounded_store import DEFAULT_MAX_RECORDS, BoundedList


class OutcomeReportRequest(BaseModel):
    """A downstream outcome as the consumer reported it.

    Every optional field defaults to ``None`` ("not reported") rather than to a concrete
    value. The previous defaults — ``detail="verification_passed"``,
    ``provider_used="gemini"``, ``service="code_generation"`` — were counted as facts by
    the summary below, so a report that named nothing was still credited to gemini.
    """

    consumer: Literal["forge", "trading_bot", "friday", "human"]
    request_id: str
    outcome: Literal["success", "partial", "failure"]
    detail: str | None = Field(default=None, description="verification_passed, verification_failed, build_error, profit_gained, drawdown_mitigated")
    provider_used: str | None = Field(default=None)
    service: str | None = Field(default=None)


class ConsumerOutcomeTracker:
    """Records real-world downstream effectiveness of Inference generations and advice."""

    #: Retention for the legacy ledger (see app/utils/bounded_store.py for the soak
    #: measurements that motivated bounding these stores).
    MAX_HISTORY = DEFAULT_MAX_RECORDS

    def __init__(self) -> None:
        # The three entries below are demonstrations, labelled as such: they were
        # previously indistinguishable from real reports and fed the pass-rate summary.
        self.outcome_history: BoundedList[dict[str, Any]] = BoundedList(
            "outcomes.consumer_outcome_history",
            max_entries=self.MAX_HISTORY,
            initial=[
                {"consumer": "forge", "request_id": "req-001", "outcome": "success", "detail": "verification_passed", "provider_used": "gemini", "service": "code_generation", "source": "fixture", "timestamp": time.time() - 3600},
                {"consumer": "forge", "request_id": "req-002", "outcome": "success", "detail": "verification_passed", "provider_used": "groq", "service": "code_generation", "source": "fixture", "timestamp": time.time() - 2400},
                {"consumer": "trading_bot", "request_id": "req-003", "outcome": "success", "detail": "drawdown_mitigated", "provider_used": "groq", "service": "trading_consult", "source": "fixture", "timestamp": time.time() - 1200},
            ],
        )

    def record_outcome(self, req: OutcomeReportRequest) -> dict[str, Any]:
        entry = req.model_dump()
        entry["timestamp"] = time.time()
        entry["source"] = "observed"
        self.outcome_history.append(entry)
        return {"status": "RECORDED", "request_id": req.request_id, "outcome": req.outcome}

    def get_outcome_summary(self) -> dict[str, Any]:
        """Downstream success rates, computed from observed reports.

        The loop below used to end with ``prov_quality[p] = 90.0  # Default initial prior``
        for every provider with no records, and that number was then consumed by
        ``SelfOptimizingRouter.adapt_weights_from_outcomes()`` to set live routing weights:
        a provider that had never served a request was routed to as if it passed 90% of
        verification. A provider with no data now returns ``None`` plus a reason, and the
        router skips it. Seeded demonstration rows are excluded from the computation and
        counted separately.
        """
        observed = [o for o in self.outcome_history if o.get("source") == "observed"]
        total = len(observed)
        successes = sum(1 for o in observed if o["outcome"] == "success")
        success_pct = round((successes / total) * 100.0, 1) if total else None

        prov_quality: dict[str, Any] = {}
        for p in ["gemini", "groq", "nvidia", "mistral", "openrouter"]:
            p_outcomes = [o for o in observed if o.get("provider_used") == p]
            if p_outcomes:
                p_success = sum(1 for o in p_outcomes if o["outcome"] == "success")
                prov_quality[p] = {
                    "verification_pass_rate_pct": round((p_success / len(p_outcomes)) * 100.0, 1),
                    "samples": len(p_outcomes),
                }
            else:
                prov_quality[p] = {
                    "verification_pass_rate_pct": None,
                    "samples": 0,
                    "insufficient_data": f"no observed outcome names {p}",
                }

        return {
            "total_outcomes_reported": total,
            "fixture_rows_excluded": len(self.outcome_history) - total,
            "retention_limit": self.MAX_HISTORY,
            "records_dropped_for_capacity": self.outcome_history.dropped,
            "overall_downstream_success_rate_pct": success_pct,
            "provider_verification_pass_rates": prov_quality,
            "basis": (
                "observed reports only; providers without data carry null, not a prior"
            ),
        }


consumer_outcome_tracker = ConsumerOutcomeTracker()
