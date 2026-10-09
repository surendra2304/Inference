"""Prediction Performance and Calibration Tracking Engine.

This module is on a **trading** surface (``/v1/intelligence/accuracy``,
``/v1/predict/{asset}/history``), so every number it publishes must be traceable to an
observation. It previously was not:

* ``history`` was a hardcoded list of four invented BTCUSDT/ETHUSDT calls presented as
  "historical prediction log and out-of-sample accuracy verification" (three CORRECT, one
  INCORRECT => 75% accuracy for a model that never ran);
* ``get_source_accuracy_report()`` published ``lstm_transformer_model: 78.4``,
  ``news_sentiment_nlp: 71.2``, ``onchain_whale_signals: 81.5``, ``technical_momentum:
  69.8`` and ``calibration_score_0_to_1: 0.88`` with ``accuracy_status:
  "HIGH_CONFIDENCE_PRODUCTION_READY"`` — all constants, none measured;
* with an empty history it fell back to ``accuracy_pct = 75.0``;
* and the engine had **no way to record a prediction at all**, so no observation could
  ever replace the constants.

Now: observed predictions are recorded and resolved through :meth:`record_prediction` /
:meth:`resolve_prediction`; scores are computed only from resolved observations; the four
demonstration rows are labelled ``source: "fixture"`` and excluded from every statistic;
and a score that cannot be supported by enough observations is reported as ``None`` with
the reason, never as a plausible constant.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from app.utils.bounded_store import BoundedList

#: Resolved observations required before a directional accuracy is published. Below this
#: the interval around an accuracy estimate is wider than any decision it could inform, so
#: the honest answer is "not enough data" rather than a number.
MIN_OBSERVATIONS_FOR_SCORE = 20

#: Directional calls are compared against a coin-flip baseline, which is the only baseline
#: this module can defend without a backtest of the underlying market.
BASELINE_ACCURACY_PCT = 50.0


class PredictionTrackingEngine:
    """Tracks observed directional predictions and their resolved outcomes."""

    def __init__(self, max_history: int = 2000) -> None:
        # The rows below are demonstrations, and are labelled as such: they were previously
        # indistinguishable from real evaluated predictions and were what the published
        # accuracy was computed from.
        self.history: BoundedList[dict[str, Any]] = BoundedList(
            "ml.prediction_history",
            max_entries=max_history,
            initial=[
                {"id": "pred-fixture-001", "timestamp": time.time() - 86400 * 3, "symbol": "BTCUSDT",
                 "direction": "BULLISH", "confidence": 0.82, "source": "fixture",
                 "evidence_class": "demonstration", "actual_outcome": "CORRECT",
                 "resolved_at": time.time() - 86400 * 2},
                {"id": "pred-fixture-002", "timestamp": time.time() - 86400 * 2, "symbol": "BTCUSDT",
                 "direction": "BULLISH", "confidence": 0.78, "source": "fixture",
                 "evidence_class": "demonstration", "actual_outcome": "CORRECT",
                 "resolved_at": time.time() - 86400 * 1},
                {"id": "pred-fixture-003", "timestamp": time.time() - 86400, "symbol": "ETHUSDT",
                 "direction": "BEARISH", "confidence": 0.71, "source": "fixture",
                 "evidence_class": "demonstration", "actual_outcome": "INCORRECT",
                 "resolved_at": time.time() - 3600 * 12},
                {"id": "pred-fixture-004", "timestamp": time.time() - 3600 * 12, "symbol": "BTCUSDT",
                 "direction": "BULLISH", "confidence": 0.85, "source": "fixture",
                 "evidence_class": "demonstration", "actual_outcome": "CORRECT",
                 "resolved_at": time.time() - 3600},
            ],
        )

    # -- ingestion -----------------------------------------------------------
    def record_prediction(
        self,
        symbol: str,
        direction: str,
        confidence: float,
        source: str = "unspecified_source",
        horizon_hours: int | None = None,
    ) -> dict[str, Any]:
        """Record an observed directional prediction; returns the stored entry.

        ``source`` names the sub-model/strategy the call came from, which is what makes a
        per-source accuracy breakdown possible at all. Before this method existed the
        engine had no ingestion path, so its published per-model accuracies could only have
        come from constants.
        """
        entry = {
            "id": f"pred_{uuid.uuid4().hex[:12]}",
            "timestamp": time.time(),
            "symbol": symbol.upper(),
            "direction": direction.upper(),
            "confidence": float(confidence),
            "source": source,
            "horizon_hours": horizon_hours,
            "evidence_class": "observed_prediction",
            "actual_outcome": None,
            "resolved_at": None,
        }
        self.history.append(entry)
        return entry

    def resolve_prediction(self, prediction_id: str, actual_outcome: str) -> dict[str, Any] | None:
        """Attach the realised outcome (``CORRECT`` / ``INCORRECT`` / ``FLAT``) to a call."""
        outcome = actual_outcome.upper()
        if outcome not in ("CORRECT", "INCORRECT", "FLAT"):
            raise ValueError("actual_outcome must be CORRECT, INCORRECT or FLAT")
        for entry in self.history:
            if entry.get("id") == prediction_id:
                entry["actual_outcome"] = outcome
                entry["resolved_at"] = time.time()
                return entry
        return None

    # -- reporting -----------------------------------------------------------
    def _observed_resolved(self) -> list[dict[str, Any]]:
        return [
            e for e in self.history
            if e.get("evidence_class") == "observed_prediction"
            and e.get("actual_outcome") in ("CORRECT", "INCORRECT")
        ]

    def describe_history(self) -> dict[str, Any]:
        """Retention window and fixture split of the stored history."""
        described = self.history.describe()
        fixtures = sum(1 for e in self.history if e.get("source") == "fixture")
        described.update({"fixture_rows": fixtures, "observed_rows": len(self.history) - fixtures})
        return described

    def get_source_accuracy_report(self) -> dict[str, Any]:
        """Directional accuracy and calibration computed from resolved observations only.

        Nothing in this report is a constant: each field is either derived from the
        observations listed in ``history``, or ``None``/``INSUFFICIENT_DATA`` with the
        reason stated in ``basis``.
        """
        fixtures = sum(1 for e in self.history if e.get("source") == "fixture")
        observed_all = [e for e in self.history if e.get("evidence_class") == "observed_prediction"]
        resolved = self._observed_resolved()

        report: dict[str, Any] = {
            "total_predictions_evaluated": len(resolved),
            "observed_predictions_recorded": len(observed_all),
            "pending_resolution": len(observed_all) - len(resolved),
            "fixture_rows_excluded": fixtures,
            "overall_directional_accuracy_pct": None,
            "sub_model_accuracies": {},
            "calibration_score_0_to_1": None,
            "calibration_basis": "withheld: no resolved observations to calibrate against",
            "accuracy_status": "INSUFFICIENT_DATA",
            "min_observations_required": MIN_OBSERVATIONS_FOR_SCORE,
            "baseline_accuracy_pct": BASELINE_ACCURACY_PCT,
            "evidence_class": "no_observations" if not resolved else "observed_resolved_predictions",
            "basis": (
                f"computed from {len(resolved)} resolved observed predictions; "
                f"{fixtures} demonstration rows excluded; "
                f"{MIN_OBSERVATIONS_FOR_SCORE} resolved predictions are required before an "
                "accuracy figure is published"
            ),
        }

        if len(resolved) < MIN_OBSERVATIONS_FOR_SCORE:
            return report

        correct = sum(1 for e in resolved if e["actual_outcome"] == "CORRECT")
        accuracy = round(correct / len(resolved) * 100.0, 1)
        report["overall_directional_accuracy_pct"] = accuracy
        report["accuracy_status"] = (
            "OBSERVED_ACCURACY_ABOVE_BASELINE"
            if accuracy > BASELINE_ACCURACY_PCT
            else "OBSERVED_ACCURACY_AT_OR_BELOW_BASELINE"
        )

        # Per-source accuracy, only where that source has enough resolved observations.
        by_source: dict[str, list[dict[str, Any]]] = {}
        for entry in resolved:
            by_source.setdefault(str(entry.get("source") or "unspecified_source"), []).append(entry)
        for source, rows in sorted(by_source.items()):
            hits = sum(1 for r in rows if r["actual_outcome"] == "CORRECT")
            if len(rows) < MIN_OBSERVATIONS_FOR_SCORE:
                report["sub_model_accuracies"][source] = {
                    "accuracy_pct": None,
                    "sample_size": len(rows),
                    "status": "insufficient_sample",
                }
            else:
                report["sub_model_accuracies"][source] = {
                    "accuracy_pct": round(hits / len(rows) * 100.0, 1),
                    "sample_size": len(rows),
                    "status": "observed",
                }

        # Calibration: how far stated confidence sits from the realised hit rate.
        # Brier-style mean squared error over resolved calls, expressed as a 0..1 score.
        errors = []
        for entry in resolved:
            stated = float(entry.get("confidence") or 0.0)
            hit = 1.0 if entry["actual_outcome"] == "CORRECT" else 0.0
            errors.append((stated - hit) ** 2)
        if errors:
            brier = sum(errors) / len(errors)
            report["calibration_score_0_to_1"] = round(1.0 - brier, 3)
            report["calibration_basis"] = (
                f"1 - mean((stated_confidence - realised_hit)^2) over {len(errors)} resolved "
                f"observed predictions (Brier score {round(brier, 3)})"
            )
        return report


prediction_tracker = PredictionTrackingEngine()
