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

import math
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
        # Validate at ingestion so invalid observations cannot poison accuracy or
        # Brier-style calibration results later. Do not silently coerce strings/bools
        # into numeric confidence values.
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("symbol must be a non-empty string")
        if not isinstance(direction, str) or not direction.strip():
            raise ValueError("direction must be a non-empty string")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("source must be a non-empty string")
        symbol_value = symbol.strip()
        direction_value = direction.strip()
        source_value = source.strip()
        for label, value, max_length in (
            ("symbol", symbol_value, 64),
            ("direction", direction_value, 32),
            ("source", source_value, 128),
        ):
            if len(value) > max_length:
                raise ValueError(f"{label} must not exceed {max_length} characters")
            if any(ord(char) < 32 or ord(char) == 127 for char in value):
                raise ValueError(f"{label} must not contain control characters")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError("confidence must be a finite number between 0 and 1")
        try:
            confidence_value = float(confidence)
        except (OverflowError, ValueError):
            raise ValueError("confidence must be a finite number between 0 and 1") from None
        if not math.isfinite(confidence_value) or not 0.0 <= confidence_value <= 1.0:
            raise ValueError("confidence must be a finite number between 0 and 1")
        if horizon_hours is not None and (
            isinstance(horizon_hours, bool)
            or not isinstance(horizon_hours, int)
            or horizon_hours <= 0
        ):
            raise ValueError("horizon_hours must be a positive integer when provided")

        entry = {
            "id": f"pred_{uuid.uuid4().hex[:12]}",
            "timestamp": time.time(),
            "symbol": symbol_value.upper(),
            "direction": direction_value.upper(),
            "confidence": confidence_value,
            "source": source_value,
            "horizon_hours": horizon_hours,
            "evidence_class": "observed_prediction",
            "actual_outcome": None,
            "resolved_at": None,
        }
        self.history.append(entry)
        return entry

    def resolve_prediction(self, prediction_id: str, actual_outcome: str) -> dict[str, Any] | None:
        """Record one outcome for an observed prediction; demo fixtures stay immutable.

        FLAT is a terminal observation but is deliberately excluded from directional
        accuracy/Brier scoring. Unknown ids and demonstration fixtures return None;
        attempting to resolve a previously resolved observed prediction raises ValueError
        instead of silently rewriting its audit history.
        """
        if not isinstance(prediction_id, str) or not prediction_id.strip():
            raise ValueError("prediction_id must be a non-empty string")
        if not isinstance(actual_outcome, str):
            raise ValueError("actual_outcome must be CORRECT, INCORRECT or FLAT")
        outcome = actual_outcome.strip().upper()
        if outcome not in ("CORRECT", "INCORRECT", "FLAT"):
            raise ValueError("actual_outcome must be CORRECT, INCORRECT or FLAT")
        for entry in self.history:
            if entry.get("id") != prediction_id:
                continue
            if entry.get("evidence_class") != "observed_prediction":
                return None
            if entry.get("actual_outcome") is not None:
                raise ValueError(f"prediction '{prediction_id}' has already been resolved")
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

        pending = sum(1 for entry in observed_all if entry.get("actual_outcome") is None)
        flat = sum(1 for entry in observed_all if entry.get("actual_outcome") == "FLAT")
        if not observed_all:
            evidence_class = "no_observations"
        elif resolved:
            evidence_class = "observed_resolved_predictions"
        elif flat and not pending:
            evidence_class = "observed_predictions_without_scoreable_outcomes"
        else:
            evidence_class = "observed_predictions_pending_resolution"

        report: dict[str, Any] = {
            "total_predictions_evaluated": len(resolved),
            "observed_predictions_recorded": len(observed_all),
            "pending_resolution": pending,
            "flat_outcomes_excluded_from_directional_accuracy": flat,
            "fixture_rows_excluded": fixtures,
            "overall_directional_accuracy_pct": None,
            "sub_model_accuracies": {},
            "calibration_score_0_to_1": None,
            "calibration_basis": "withheld: no scoreable resolved observations to calibrate against",
            "accuracy_status": "INSUFFICIENT_DATA",
            "min_observations_required": MIN_OBSERVATIONS_FOR_SCORE,
            "baseline_accuracy_pct": BASELINE_ACCURACY_PCT,
            "evidence_class": evidence_class,
            "basis": (
                f"computed from {len(resolved)} scoreable resolved observed predictions; "
                f"{flat} flat outcomes excluded from directional scoring; "
                f"{pending} predictions await resolution; "
                f"{fixtures} demonstration rows excluded; "
                f"{MIN_OBSERVATIONS_FOR_SCORE} scoreable resolved predictions are required "
                "before an accuracy figure is published"
            ),
        }

        if len(resolved) < MIN_OBSERVATIONS_FOR_SCORE:
            return report

        # The scoreable set contains only CORRECT / INCORRECT. FLAT is explicitly counted
        # and reported above but is not a directional hit or miss.
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
