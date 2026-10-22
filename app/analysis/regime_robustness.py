"""Market Regime Robustness Analysis and Whipsaw Transition Stress Testing."""

from typing import Any


class RegimeRobustnessEngine:
    """Evaluates strategy performance consistency across Bull, Bear, High Volatility, and Chop regimes."""

    def test_regime_robustness(
        self,
        strategy_name: str,
        regime_metrics: dict[str, dict[str, float]]
    ) -> dict[str, Any]:
        """Calculates multi-regime consistency, worst-regime performance, and whipsaw transition score."""
        # No measured regime data means no score. Before this, four hard-coded regime results
        # (bull win rate 0.68 ... crisis PF 1.15) were substituted when the caller sent none, so
        # every request without data received the same "robustness" verdict.
        if not regime_metrics:
            return {
                "strategy_name": strategy_name,
                "status": "not_measured",
                "robustness_score": None,
                "worst_regime_profit_factor": None,
                "worst_regime_max_drawdown_pct": None,
                "regime_dependency_classification": None,
                "regime_breakdown": {},
                "whipsaw_transition_survival": None,
                "advisory_notes": "No per-regime metrics were supplied; no robustness score is reported.",
            }

        # A regime entry without its own profit factor or drawdown is not measured either.
        missing = sorted(
            name for name, m in regime_metrics.items()
            if "profit_factor" not in m or "max_drawdown_pct" not in m
        )
        if missing:
            return {
                "strategy_name": strategy_name,
                "status": "insufficient_inputs",
                "robustness_score": None,
                "missing_fields_for_regimes": missing,
                "regime_breakdown": regime_metrics,
                "advisory_notes": "Each regime needs profit_factor and max_drawdown_pct; no score is reported.",
            }

        pfs = [m["profit_factor"] for m in regime_metrics.values()]
        dds = [m["max_drawdown_pct"] for m in regime_metrics.values()]

        worst_pf = min(pfs)
        worst_dd = max(dds)
        avg_pf = sum(pfs) / len(pfs)

        # Robustness score (0 - 100)
        # Penalizes strategies that catastrophically fail in chop or crisis
        consistency_penalty = (max(pfs) - min(pfs)) * 15.0
        robustness_score = round(max(10.0, min(95.0, (avg_pf * 35.0) - (worst_dd * 2.0) - consistency_penalty)), 1)

        regime_dependency = "BALANCED_MULTI_REGIME" if worst_pf >= 1.0 else "VULNERABLE_TO_SIDEWAYS_CHOP"

        return {
            "strategy_name": strategy_name,
            "status": "measured",
            "robustness_score": robustness_score,
            "worst_regime_profit_factor": worst_pf,
            "worst_regime_max_drawdown_pct": worst_dd,
            "regime_dependency_classification": regime_dependency,
            "regime_breakdown": regime_metrics,
            "whipsaw_transition_survival": "HIGH" if worst_dd < 10.0 else "LOW",
            "advisory_notes": "Strategy handles directional regimes well but suffers drag during sideways range consolidation." if worst_pf < 1.0 else "Solid cross-regime stability."
        }


regime_robustness_engine = RegimeRobustnessEngine()
