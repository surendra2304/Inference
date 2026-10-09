"""Conservative Recommendation Engine prioritizing capital preservation and risk reduction."""

from typing import Any


class ConservativeRecommendationEngine:
    """Enforces strict capital preservation hierarchy: REDUCE_RISK > NO_CHANGE > OPTIMIZE."""

    def generate_conservative_recommendation(
        self,
        strategy_name: str,
        current_drawdown_pct: float,
        profit_factor: float,
        confidence: float,
        proposed_action: str | None = None
    ) -> dict[str, Any]:
        """Generates bounded conservative recommendation with explicit 'what could go wrong' analysis."""
        # 1. Immediate risk reduction if drawdown is elevated
        if current_drawdown_pct >= 5.0 or profit_factor < 1.0:
            rec_action = "REDUCE_RISK"
            rationale = "Portfolio drawdown elevated or profit factor sub-optimal. Position sizing reduction prioritized."
            worst_case = "Prolonged chop causing drag if sizing is not reduced quickly."

        # 2. Prefer NO_CHANGE if confidence is modest
        elif confidence < 0.80:
            rec_action = "NO_CHANGE"
            rationale = "Current telemetry is stable but confidence score does not warrant parameter perturbation on real capital."
            worst_case = "Opportunity cost of not capturing marginally higher alpha."

        # 3. High-confidence optimization
        else:
            rec_action = proposed_action or "OPTIMIZE_PARAMETERS"
            # Built from the caller's numbers. The former fixed sentence claimed "multi-agent consensus",
            # which this path never computes; the confidence is only what the caller supplied.
            rationale = (
                f"Caller-supplied confidence {confidence:.2f} clears the 0.80 threshold with profit factor "
                f"{profit_factor:.2f} and drawdown {current_drawdown_pct:.2f}%. No multi-agent consensus "
                "was computed for this recommendation."
            )
            worst_case = "Regime change post-parameter shift."

        return {
            "strategy_name": strategy_name,
            "recommended_action": rec_action,
            "confidence": confidence,
            "rationale": rationale,
            "what_could_go_wrong": worst_case,
            "capital_preservation_officer_verdict": "APPROVED_DEFENSIVE" if rec_action in ("REDUCE_RISK", "NO_CHANGE") else "CONDITIONAL_APPROVAL"
        }


conservative_engine = ConservativeRecommendationEngine()
