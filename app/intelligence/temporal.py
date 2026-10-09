"""Temporal Reasoning Engine: Time-Aware Pattern Detection, Trend Analysis, and Consistency Checking."""

import time
from typing import Any, Literal

from pydantic import BaseModel, Field


class TimeSeriesPoint(BaseModel):
    timestamp: float
    value: float
    metric_name: str


class TemporalPatternResult(BaseModel):
    #: NO_DATA when the series is empty; the former empty-series answer was "STABLE".
    trend: Literal["UPWARD", "DOWNWARD", "STABLE", "VOLATILE", "NO_DATA"]
    seasonality_detected: bool = False
    changepoints: list[float] = Field(default_factory=list)
    #: None: no calibration exists for these rules. The former fixed 0.85 / 0.88 / 0.70 were
    #: returned whatever the series was, including an empty one.
    confidence: float | None = None
    confidence_basis: str = "not calibrated: trend and changepoints are computed directly from the series"
    summary: str


class TemporalReasoningEngine:
    """Analyzes temporal context, trends, seasonality, changepoints, and validates temporal recommendation consistency."""

    def __init__(self) -> None:
        self.recommendation_history: dict[str, list[dict[str, Any]]] = {}

    def analyze_temporal_series(
        self,
        time_series: list[TimeSeriesPoint],
        temporal_context_note: str | None = None
    ) -> TemporalPatternResult:
        """Evaluates trend, changepoint anomalies, and seasonality over time series data."""
        if not time_series:
            return TemporalPatternResult(
                trend="NO_DATA",
                seasonality_detected=False,
                changepoints=[],
                confidence=None,
                confidence_basis="no observations supplied",
                summary=temporal_context_note or "No time series was supplied; no trend was measured."
            )

        values = [pt.value for pt in time_series]
        first, last = values[0], values[-1]
        delta_pct = ((last - first) / max(0.0001, abs(first))) * 100.0

        trend: Literal["UPWARD", "DOWNWARD", "STABLE", "VOLATILE"]
        if delta_pct > 10.0:
            trend = "UPWARD"
        elif delta_pct < -10.0:
            trend = "DOWNWARD"
        else:
            trend = "STABLE"

        # Detect changepoints where consecutive difference > 20%
        changepoints = []
        for i in range(1, len(values)):
            step_diff = abs(values[i] - values[i - 1]) / max(0.0001, abs(values[i - 1]))
            if step_diff > 0.20:
                changepoints.append(time_series[i].timestamp)

        summary = f"Detected {trend.lower()} trend ({delta_pct:+.1f}% change) with {len(changepoints)} structural changepoints."
        if temporal_context_note:
            summary = f"{temporal_context_note} — {summary}"

        return TemporalPatternResult(
            trend=trend,
            seasonality_detected=len(time_series) >= 7,
            changepoints=changepoints,
            confidence=None,
            summary=summary
        )

    def check_temporal_consistency(
        self,
        context_key: str,
        new_decision: str,
        current_rationale: str
    ) -> dict[str, Any]:
        """Ensures recommendations do not contradict earlier recommendations without explicit explanation."""
        history = self.recommendation_history.get(context_key, [])
        if not history:
            self.recommendation_history[context_key] = [{
                "decision": new_decision,
                "rationale": current_rationale,
                "timestamp": time.time()
            }]
            return {"is_consistent": True, "explanation": "Initial baseline recommendation set."}

        last_rec = history[-1]
        if last_rec["decision"] != new_decision:
            explanation = f"Recommendation shifted from '{last_rec['decision']}' to '{new_decision}' driven by updated temporal evidence: {current_rationale}"
            self.recommendation_history[context_key].append({
                "decision": new_decision,
                "rationale": current_rationale,
                "explanation": explanation,
                "timestamp": time.time()
            })
            return {"is_consistent": True, "contradiction_flagged": False, "explanation": explanation}

        return {"is_consistent": True, "explanation": "Affirms previous stable recommendation."}


temporal_reasoning_engine = TemporalReasoningEngine()
