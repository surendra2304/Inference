"""Outcome Feedback Loop, Cross-Consumer Learning, Confidence Calibration & Strategy Bank."""

import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.routing.self_optimizer import self_optimizing_router
from app.utils.bounded_store import DEFAULT_MAX_RECORDS, BoundedList
from app.utils.logger import logger

ConsumerName = Literal["trading_bot", "forge", "nexus", "sentinel", "intelx", "futuris", "friday", "human"]
OutcomeStatus = Literal["success", "partial", "failure"]


class DetailedOutcomeReport(BaseModel):
    """One observed downstream outcome.

    Defaults here are deliberately the *absence* of a measurement. They used to assert
    values nobody reported — ``provider_used="gemini"``, ``task_type="code_generation"``,
    ``agent_composition=["strategist","critic"]``, ``stated_confidence=0.85``,
    ``detail="verification_passed"`` — and those defaults flowed straight into the
    performance statistics, so a report that said only "something succeeded" was counted
    as a gemini-backed, two-agent, 0.85-confidence success. ``None`` means "not reported"
    and the statistics skip it rather than crediting it.

    ``source`` separates a genuinely observed outcome from the seeded demonstrations that
    ship with the process. The seeds are useful as shape examples but they are not
    measurements, and statistics that mix them are reporting a fixture, not a result.
    """

    consumer: ConsumerName
    request_id: str
    outcome: OutcomeStatus
    detail: str | None = None
    measured_metrics: dict[str, Any] = Field(default_factory=dict)
    task_type: str | None = None
    provider_used: str | None = None
    agent_composition: list[str] | None = None
    stated_confidence: float | None = None
    source: Literal["observed", "fixture"] = "observed"
    timestamp: float = Field(default_factory=time.time)


class StrategyBankEntry(BaseModel):
    pattern_id: str
    task_type: str
    context_summary: str
    #: None when the recorded outcome carried no recommendation. A placeholder sentence
    #: here would be indistinguishable from advice someone actually gave.
    recommendation: str | None = None
    outcome_summary: str
    #: None when no rate was measured. A recorded outcome carries an outcome *status*,
    #: not a rate, and inventing 0.90 for it (which this engine did) made every success
    #: look like a 90%-success pattern regardless of what was actually measured.
    success_rate: float | None = None
    usage_count: int = 1
    #: Where this entry came from. ``observed`` means a real outcome report produced it;
    #: ``illustrative_fixture`` means it ships with the code and describes a scenario that
    #: was never run. The distinction is load-bearing: nexus cites these as evidence, and
    #: the two seeded entries below were previously served as "Similar past situation ...
    #: (92% success)" for any unrelated goal, because query_strategy_bank fell back to the
    #: first two records when nothing matched. A sales-funnel pattern was cited as
    #: precedent for a database-performance incident.
    source: Literal["observed", "illustrative_fixture"] = "observed"
    created_at: float = Field(default_factory=time.time)
    expires_at: float = Field(default_factory=lambda: time.time() + (90 * 86400))  # 90 days retention


class OutcomeLearningEngine:
    """Tracks downstream success, rolling exponential success rates, and cross-consumer insights."""

    #: Retained outcome reports. Bounded because this list is appended on every reported
    #: outcome and previously grew forever (measured +2.42 MB/1k requests on the analytics
    #: route, retained after gc). ``records_dropped`` says how many fell off, so a rate
    #: can be described as "over the last N retained records".
    MAX_OUTCOME_RECORDS = DEFAULT_MAX_RECORDS
    MAX_STRATEGY_ENTRIES = 2048

    def __init__(self) -> None:
        self._pattern_seq = 0
        self.outcome_records: BoundedList[DetailedOutcomeReport] = BoundedList(
            "outcome_learning.outcome_records", max_entries=self.MAX_OUTCOME_RECORDS
        )
        for seed in self._seed_outcome_records():
            self.outcome_records.append(seed)
        self.strategy_bank: BoundedList[StrategyBankEntry] = BoundedList(
            "outcome_learning.strategy_bank",
            max_entries=self.MAX_STRATEGY_ENTRIES,
            initial=self._seed_strategy_entries(),
        )

    def _seed_outcome_records(self) -> list[DetailedOutcomeReport]:
        """The demo outcomes that ship with the process, explicitly labelled.

        These four records were previously indistinguishable from real reports, so the
        computed performance statistics (which now read them) would have published
        fixture telemetry — including "conversion_delta_pct: 14.5" — as an observed
        result. ``source="fixture"`` keeps the shape examples without claiming they
        happened.
        """
        return [
            DetailedOutcomeReport(
                consumer="forge",
                request_id="init-forge-01",
                outcome="success",
                detail="build_passed",
                measured_metrics={"build_time_s": 4.2},
                task_type="code_generation",
                provider_used="gemini",
                agent_composition=["coder", "critic"],
                stated_confidence=0.90,
                source="fixture",
                timestamp=time.time() - 7200
            ),
            DetailedOutcomeReport(
                consumer="forge",
                request_id="init-forge-02",
                outcome="success",
                detail="build_passed",
                measured_metrics={"build_time_s": 3.8},
                task_type="code_generation",
                provider_used="groq",
                agent_composition=["coder"],
                stated_confidence=0.85,
                source="fixture",
                timestamp=time.time() - 3600
            ),
            DetailedOutcomeReport(
                consumer="nexus",
                request_id="init-nexus-01",
                outcome="success",
                detail="conversion_boosted",
                measured_metrics={"conversion_delta_pct": 14.5},
                task_type="lead_qualification",
                provider_used="gemini",
                agent_composition=["strategist", "data_analyst", "critic"],
                stated_confidence=0.92,
                source="fixture",
                timestamp=time.time() - 1800
            ),
            DetailedOutcomeReport(
                consumer="trading_bot",
                request_id="init-bot-01",
                outcome="success",
                detail="drawdown_mitigated",
                measured_metrics={"drawdown_reduced_pct": 3.5},
                task_type="trading_consult",
                provider_used="groq",
                agent_composition=["strategist", "critic"],
                stated_confidence=0.88,
                source="fixture",
                timestamp=time.time() - 900
            )
        ]

    def _seed_strategy_entries(self) -> list[StrategyBankEntry]:
        # Illustrative fixtures, explicitly labelled. They demonstrate the shape of a
        # strategy record; they are not results of any experiment, so anything citing them
        # must say so rather than presenting them as precedent.
        return [
            StrategyBankEntry(
                pattern_id="PAT-001",
                task_type="lead_qualification",
                context_summary="Enterprise inbound lead with verified ARR > $50k",
                recommendation="Accelerated SDR routing with tailored security compliance briefing",
                outcome_summary="+14.5% conversion to closed-won within 30 days",
                success_rate=0.92,
                source="illustrative_fixture",
            ),
            StrategyBankEntry(
                pattern_id="PAT-002",
                task_type="trading_consult",
                context_summary="High market volatility with sudden ATR expansion > 2.0",
                recommendation="Reduce max position leverage by 50% and widen stop-loss bound",
                outcome_summary="Mitigated portfolio drawdown during severe liquidity squeeze",
                success_rate=0.88,
                source="illustrative_fixture",
            )
        ]

    def record_outcome(self, report: DetailedOutcomeReport) -> dict[str, Any]:
        """Ingests an outcome report and feeds the router's adaptation.

        Three things were invented here and are not any more:

        * ``success_rate=0.90`` — a recorded outcome says *whether* it succeeded, not how
          often that pattern succeeds. The rate comes from ``measured_metrics`` when the
          reporter measured one and is ``None`` otherwise. Publishing 0.90 for every
          success made the bank look like a validated study.
        * ``recommendation=f"Recommended actions for {task_type}"`` — a placeholder sentence
          wearing the label "recommendation". The report carries none, so the entry records
          none.
        * ``pattern_id=f"PAT-{int(time.time())}"`` — every outcome recorded in the same
          second produced the same id, so a busy process emitted duplicate keys.
        """
        self.outcome_records.append(report)
        logger.info("[OUTCOME] Recorded from %s (task: %s): %s", report.consumer, report.task_type, report.outcome)

        # Index successes into the strategy bank, with only what was reported.
        if report.outcome == "success":
            self._pattern_seq += 1
            measured_rate: float | None = None
            for key in ("success_rate", "success_rate_pct", "closed_won_rate"):
                raw = report.measured_metrics.get(key)
                if isinstance(raw, (int, float)):
                    measured_rate = float(raw)
                    if key.endswith("_pct") or measured_rate > 1.0:
                        measured_rate = measured_rate / 100.0
                    break
            self.strategy_bank.append(
                StrategyBankEntry(
                    pattern_id=f"PAT-{int(time.time() * 1000)}-{self._pattern_seq}",
                    task_type=report.task_type or "unspecified",
                    context_summary=(
                        f"{report.consumer} reported outcome for "
                        f"{report.task_type or 'an unspecified task'}: "
                        f"{report.detail or 'no detail supplied'}"
                    ),
                    recommendation=None,
                    outcome_summary=str(report.measured_metrics or report.detail or report.outcome),
                    success_rate=measured_rate,
                    source="observed",
                )
            )

        # Auto-trigger router adaptation
        self_optimizing_router.adapt_weights_from_outcomes()
        return {"status": "RECORDED", "request_id": report.request_id, "outcome": report.outcome}

    def compute_provider_performance(self) -> dict[str, Any]:
        """Rolling success rates with 3x recent weighting, over observed outcomes only.

        Two corrections. A provider with no records was reported at ``90.0`` with
        ``total_samples: 0`` — a fabricated pass rate for a provider that had never run,
        published next to a real sample count. And the seeded demonstration records were
        counted alongside real reports; they are now excluded and reported separately.
        """
        now = time.time()
        providers = ["gemini", "groq", "nvidia", "mistral", "openrouter", "cohere", "huggingface"]
        stats: dict[str, dict[str, Any]] = {}

        observed = [r for r in self.outcome_records if r.source == "observed"]
        fixtures = len(self.outcome_records) - len(observed)

        for p in providers:
            p_recs = [r for r in observed if (r.provider_used or "").lower() == p]
            if not p_recs:
                stats[p] = {
                    "success_rate_pct": None,
                    "total_samples": 0,
                    "insufficient_data": (
                        f"no observed outcome names {p}; a rate cannot be stated"
                    ),
                }
                continue

            weighted_success = 0.0
            total_weight = 0.0
            for r in p_recs:
                age_hours = (now - r.timestamp) / 3600.0
                weight = 3.0 if age_hours <= 24.0 else 1.0
                total_weight += weight
                if r.outcome == "success":
                    weighted_success += weight
                elif r.outcome == "partial":
                    weighted_success += (0.5 * weight)

            success_rate = (weighted_success / max(1.0, total_weight)) * 100.0
            stats[p] = {
                "success_rate_pct": round(success_rate, 1),
                "total_samples": len(p_recs),
                "recent_samples": sum(
                    1 for r in p_recs if (now - r.timestamp) / 3600.0 <= 24.0
                ),
            }
        return {
            "providers": stats,
            "records_observed": len(observed),
            "records_fixture_excluded": fixtures,
            "records_retained": len(self.outcome_records),
            "records_dropped_for_capacity": self.outcome_records.dropped,
            "weighting": "3x for outcomes reported within 24h, 1x otherwise",
        }

    def compute_agent_composition_performance(self) -> dict[str, Any]:
        """Compare solo-agent runs against multi-agent runs using recorded outcomes only.

        The previous implementation returned four hardcoded percentages — "solo 78.4",
        "debate 91.2", plus three invented compositions with invented rates — while
        ``outcome_records`` (the real store) was never consulted. That is a performance
        study with no study behind it, and it invites an architecture decision ("panels
        beat solo agents by 12.8 points") that the data does not support. This computes the
        comparison from ``agent_composition`` / ``outcome`` on the recorded reports, and
        reports ``insufficient_data`` with the observed sample size when there is not
        enough signal to state a rate.
        """
        def rate(records: list[DetailedOutcomeReport]) -> float | None:
            if not records:
                return None
            wins = sum(1 for r in records if r.outcome == "success")
            return round(100.0 * wins / len(records), 1)

        observed = [r for r in self.outcome_records if r.source == "observed"]
        # A report that never stated its composition is not evidence about composition:
        # it is counted separately rather than defaulted into either arm.
        solo = [r for r in observed if r.agent_composition is not None and len(r.agent_composition) <= 1]
        multi = [r for r in observed if r.agent_composition is not None and len(r.agent_composition) > 1]
        unstated = [r for r in observed if r.agent_composition is None]
        solo_rate, multi_rate = rate(solo), rate(multi)

        composition_stats: dict[str, dict[str, Any]] = {}
        for report in multi:
            key = ", ".join(sorted(report.agent_composition or []))
            bucket = composition_stats.setdefault(key, {"runs": 0, "successes": 0})
            bucket["runs"] += 1
            if report.outcome == "success":
                bucket["successes"] += 1
        top = sorted(
            (
                {
                    "composition": key.split(", "),
                    "success_rate_pct": round(100.0 * v["successes"] / v["runs"], 1),
                    "sample_size": v["runs"],
                }
                for key, v in composition_stats.items()
            ),
            key=lambda item: (-item["success_rate_pct"], -item["sample_size"]),
        )

        MIN = 20
        result: dict[str, Any] = {
            "solo_agent_success_rate_pct": solo_rate,
            "debate_composition_success_rate_pct": multi_rate,
            "solo_sample_size": len(solo),
            "debate_sample_size": len(multi),
            "top_performing_compositions": top,
            "unstated_composition_runs": len(unstated),
            "records_observed": len(observed),
            "records_fixture_excluded": len(self.outcome_records) - len(observed),
            "basis": "observed outcome reports only (source='observed')",
        }
        if solo_rate is None or multi_rate is None or min(len(solo), len(multi)) < MIN:
            result["insufficient_data"] = (
                f"solo={len(solo)} run(s), multi-agent={len(multi)} run(s); at least {MIN} of "
                "each are required before reporting a comparison."
            )
            result["diversity_lift_pct"] = None
        else:
            result["diversity_lift_pct"] = round(multi_rate - solo_rate, 1)
        return result

    #: Minimum observed reports before a cross-consumer claim is stated.
    MIN_PATTERN_SAMPLES = 20

    def get_cross_consumer_insights(self) -> dict[str, Any]:
        """Cross-consumer quality, computed from observed reports only.

        The previous body published four headline "patterns" — "verification improves
        accuracy 23% over single-model across ALL consumers", "research synthesis succeeds
        91% but attack path reasoning only 71% without multi-round debate", and so on — that
        appear nowhere in this codebase as measurements. A consumer with **zero** recorded
        evaluations was also reported at 92%/94%/95% depending on which consumer it was
        (``total_evaluations: 0`` beside ``success_rate_pct: 92.0``). Both are the same
        defect: a number that was written down rather than measured. Patterns are now
        derived from the recorded outcomes, each carrying the sample it rests on, and are
        withheld entirely until there is enough data to state them.
        """
        consumer_stats: dict[str, dict[str, Any]] = {}
        observed = [r for r in self.outcome_records if r.source == "observed"]
        for c in ["trading_bot", "forge", "nexus", "sentinel", "intelx", "futuris", "friday"]:
            c_recs = [r for r in observed if r.consumer == c]
            total = len(c_recs)
            success = sum(1 for r in c_recs if r.outcome == "success")
            if total == 0:
                consumer_stats[c] = {
                    "total_evaluations": 0,
                    "success_rate_pct": None,
                    "alert_status": "NO_DATA",
                    "note": "no observed outcome was reported by this consumer",
                }
                continue
            rate = (success / total) * 100.0
            consumer_stats[c] = {
                "total_evaluations": total,
                "success_rate_pct": round(rate, 1),
                "alert_status": "NORMAL" if rate >= 60.0 else "ALERT_DROPPED_BELOW_60PCT",
            }

        # Derived patterns: only what the observed records support, each with its sample.
        by_task: dict[str, dict[str, int]] = {}
        for r in observed:
            if not r.task_type:
                continue
            bucket = by_task.setdefault(r.task_type, {"runs": 0, "successes": 0})
            bucket["runs"] += 1
            bucket["successes"] += 1 if r.outcome == "success" else 0

        patterns = []
        for task_type, bucket in sorted(by_task.items()):
            if bucket["runs"] >= self.MIN_PATTERN_SAMPLES:
                patterns.append(
                    f"{task_type}: {round(100.0 * bucket['successes'] / bucket['runs'], 1)}% success "
                    f"over {bucket['runs']} observed report(s)."
                )

        result: dict[str, Any] = {
            "cross_consumer_patterns": patterns,
            "consumer_quality_metrics": consumer_stats,
            "provider_performance_by_task": self.compute_provider_performance(),
            "agent_composition_performance": self.compute_agent_composition_performance(),
            "records_observed": len(observed),
            "records_fixture_excluded": len(self.outcome_records) - len(observed),
            "records_dropped_for_capacity": self.outcome_records.dropped,
        }
        if not patterns:
            result["patterns_insufficient_data"] = (
                f"no task type has {self.MIN_PATTERN_SAMPLES} observed reports yet; "
                f"{len(observed)} observed report(s) in total. No cross-consumer pattern "
                "is stated rather than stating one the data does not support."
            )
        return result

    #: Minimum observed outcomes in a confidence bin before a calibration claim is made.
    MIN_CALIBRATION_SAMPLES = 10

    def get_confidence_calibration(self) -> dict[str, Any]:
        """Stated confidence vs observed success, computed from reported outcomes.

        The previous curve was four literal rows — stated 0.94 / empirical 0.92, stated
        0.84 / empirical 0.82, and so on — each labelled ``HONEST_CALIBRATED``. Nothing
        computed them; they were the same in a fresh process as in a busy one, so the label
        "measured" attached to numbers that were typed. Bins are now built from reports
        that actually carried a ``stated_confidence`` and a known outcome, and a bin with
        too few samples says so instead of borrowing a plausible-looking number.
        """
        bins = [
            ("0.90 - 1.00", 0.90, 1.01),
            ("0.80 - 0.89", 0.80, 0.90),
            ("0.70 - 0.79", 0.70, 0.80),
            ("< 0.70", 0.0, 0.70),
        ]
        observed = [
            r for r in self.outcome_records
            if r.source == "observed" and r.stated_confidence is not None
        ]
        curve: list[dict[str, Any]] = []
        for label, low, high in bins:
            rows = [r for r in observed if low <= (r.stated_confidence or 0.0) < high]
            if len(rows) < self.MIN_CALIBRATION_SAMPLES:
                curve.append({
                    "confidence_bin": label,
                    "stated_avg": None,
                    "empirical_success_rate": None,
                    "sample_size": len(rows),
                    "status": "INSUFFICIENT_DATA",
                })
                continue
            stated_avg = sum(r.stated_confidence or 0.0 for r in rows) / len(rows)
            empirical = sum(1 for r in rows if r.outcome == "success") / len(rows)
            delta = empirical - stated_avg
            curve.append({
                "confidence_bin": label,
                "stated_avg": round(stated_avg, 3),
                "empirical_success_rate": round(empirical, 3),
                "sample_size": len(rows),
                "delta": round(delta, 3),
                "status": (
                    "CALIBRATED" if abs(delta) <= 0.15
                    else ("OVERCONFIDENT" if delta < 0 else "UNDERCONFIDENT")
                ),
            })
        return {
            "calibration_curve": curve,
            "reports_used": len(observed),
            "reports_without_stated_confidence": sum(
                1 for r in self.outcome_records
                if r.source == "observed" and r.stated_confidence is None
            ),
            "min_samples_per_bin": self.MIN_CALIBRATION_SAMPLES,
            "recalibration_policy": (
                "A bin whose empirical rate is more than 0.15 below its stated confidence is "
                "reported OVERCONFIDENT; consumers may apply a 0.85x confidence multiplier."
            ),
        }

    def query_strategy_bank(self, task_type: str, context_query: str) -> list[dict[str, Any]]:
        """Queries matching past successful strategy patterns."""
        now = time.time()
        # Expiry is applied at read time. The previous form reassigned
        # ``self.strategy_bank = [e for e in ...]``, which replaced the store with a plain
        # list and silently removed its capacity bound — the leak would have come back the
        # first time this method ran on a bounded store.
        matches = [
            e.model_dump() for e in self.strategy_bank
            if e.expires_at > now and (e.task_type == task_type or task_type in e.task_type)
        ]
        # No fallback to "the first two records". Returning an unrelated record is worse
        # than returning nothing: the caller treats it as precedent for this task, and the
        # citation enters the response as supporting evidence. Measured before this change:
        # a Postgres/SQLite architecture question was answered with
        # "[STRATEGY_BANK] Similar past situation for strategic_decision: Accelerated SDR
        # routing ... (+14.5% conversion to closed-won ...)".
        return matches


outcome_learning_engine = OutcomeLearningEngine()
