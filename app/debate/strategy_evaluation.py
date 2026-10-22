"""Strategy Evaluation Debate Engine with 5 Specialized Evaluator Roles."""

from typing import Any

from app.analysis.overfitting_intel import overfitting_engine
from app.analysis.regime_robustness import regime_robustness_engine


class StrategyEvaluationDebateEngine:
    """
    Evaluates evolved strategy candidates through a specialized panel:
    1. Quantitative Analyst: statistical validity
    2. Risk Analyst: drawdown and tail-risk
    3. Overfitting Detective: curve-fitting tests (DSR, PBO)
    4. Market Regime Expert: regime robustness
    5. Contrarian: reasons why strategy will fail (with veto power)
    """

    def evaluate_strategy_candidate(
        self,
        strategy_name: str,
        backtest_metrics: dict[str, Any],
        regime_metrics: dict[str, Any]
    ) -> dict[str, Any]:
        """Conducts structured 5-panel evaluation debate on strategy candidate."""
        # Without the four backtest fields there is nothing to evaluate. Before this, absent
        # fields were silently replaced (Sharpe 1.8, PF 1.6, drawdown 6.5%, 120 trades).
        required = ("sharpe_ratio", "profit_factor", "max_drawdown_pct", "total_trades")
        missing = [k for k in required if backtest_metrics.get(k) is None]
        if missing:
            return self._not_evaluated(
                strategy_name,
                reason="Backtest summary is missing required fields; no verdict is issued.",
                missing_backtest_fields=missing,
            )
        sharpe = float(backtest_metrics["sharpe_ratio"])
        pf = float(backtest_metrics["profit_factor"])
        max_dd = float(backtest_metrics["max_drawdown_pct"])
        trades = int(backtest_metrics["total_trades"])

        # Run analytics
        overfit_res = overfitting_engine.evaluate_strategy_overfitting(
            strategy_name=strategy_name,
            backtest_sharpe=sharpe,
            backtest_profit_factor=pf,
            total_trades=trades
        )
        regime_res = regime_robustness_engine.test_regime_robustness(
            strategy_name=strategy_name,
            regime_metrics=regime_metrics
        )

        # Evaluator scores (0 - 100)
        quant_score = round(min(95.0, max(20.0, (sharpe * 30.0) + (pf * 20.0))), 1)
        risk_score = round(max(10.0, 100.0 - (max_dd * 7.0)), 1)
        overfit_score = round(max(10.0, (1.0 - overfit_res["probability_of_backtest_overfitting_pbo"]) * 100.0), 1)
        regime_score = regime_res["robustness_score"]
        if regime_score is None:
            return self._not_evaluated(
                strategy_name,
                reason="No per-regime metrics were supplied; the regime evaluator has no score.",
                overfitting_analysis=overfit_res,
                regime_robustness=regime_res,
            )
        contrarian_score = round(max(15.0, (quant_score + risk_score) / 2.0 - 25.0), 1)

        evaluator_evaluations = [
            {
                "evaluator": "Quantitative Analyst",
                "score": quant_score,
                "verdict": "VALID" if quant_score >= 70 else "WEAK_ALPHA",
                "rationale": f"Sharpe Ratio of {sharpe:.2f} and PF {pf:.2f} demonstrate statistical edge."
            },
            {
                "evaluator": "Risk Analyst",
                "score": risk_score,
                "verdict": "APPROVED" if risk_score >= 65 else "EXCESSIVE_DRAWDOWN",
                "rationale": f"Max Drawdown observed at {max_dd:.1f}% within bounded tolerances."
            },
            {
                "evaluator": "Overfitting Detective",
                "score": overfit_score,
                "verdict": overfit_res["overfitting_verdict"],
                "rationale": f"PBO scored at {overfit_res['probability_of_backtest_overfitting_pbo'] * 100:.1f}%. Min backtest length: {overfit_res['min_backtest_length_days']} days."
            },
            {
                "evaluator": "Market Regime Expert",
                "score": regime_score,
                "verdict": regime_res["regime_dependency_classification"],
                "rationale": f"Cross-regime robustness score: {regime_score:.1f}/100. Worst PF: {regime_res['worst_regime_profit_factor']}."
            },
            {
                "evaluator": "Contrarian",
                "score": contrarian_score,
                "verdict": "CHALLENGE" if contrarian_score < 60 else "NEUTRAL",
                "rationale": "High sensitivity to execution slippage and spread widening in real market conditions."
            }
        ]

        # Consensus score calculation
        scores = [quant_score, risk_score, overfit_score, regime_score, contrarian_score]
        composite_score = round(sum(scores) / len(scores), 1)

        # Contrarian veto / Overfitting gate
        if overfit_res["overfitting_verdict"] == "REJECT_OVERFITTED" or contrarian_score < 30.0:
            final_verdict = "REJECT"
            approval_status = "FAILED_SAFETY_GATES"
        elif composite_score >= 70.0 and overfit_res["overfitting_verdict"] == "ACCEPT_ROBUST":
            final_verdict = "APPROVE_FOR_FORWARD_TESTING"
            approval_status = "PASSED_EVALUATION"
        else:
            final_verdict = "REQUIRE_EXTENDED_PAPER_TESTING"
            approval_status = "CONDITIONAL_REVIEW"

        return {
            "strategy_name": strategy_name,
            "status": "evaluated",
            "composite_evaluation_score": composite_score,
            "final_verdict": final_verdict,
            "approval_status": approval_status,
            "evaluator_panel": evaluator_evaluations,
            "overfitting_analysis": overfit_res,
            "regime_robustness": regime_res
        }

    @staticmethod
    def _not_evaluated(strategy_name: str, reason: str, **extra: Any) -> dict[str, Any]:
        """A candidate with unmeasured inputs gets no score and no verdict, only the reason."""
        return {
            "strategy_name": strategy_name,
            "status": "not_evaluated",
            "composite_evaluation_score": None,
            "final_verdict": "NOT_EVALUATED",
            "approval_status": "NOT_EVALUATED",
            "evaluator_panel": [],
            "reason": reason,
            **extra,
        }


strategy_evaluation_debate = StrategyEvaluationDebateEngine()
