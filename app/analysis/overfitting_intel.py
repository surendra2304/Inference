"""Overfitting Detection Intelligence: Deflated Sharpe Ratio, PBO, and Fragility Tests."""

import math
from typing import Any


class OverfittingIntelligenceEngine:
    """Calculates statistical overfit probabilities and parameter sensitivity fragility."""

    def evaluate_strategy_overfitting(
        self,
        strategy_name: str,
        backtest_sharpe: float,
        backtest_profit_factor: float,
        total_trades: int,
        num_trials_tested: int = 50
    ) -> dict[str, Any]:
        """Calculates Deflated Sharpe Ratio (DSR), Probability of Backtest Overfitting (PBO), and emits verdict."""
        # 1. Deflated Sharpe Ratio heuristic (Bailey, Borwein, López de Prado, Zhu 2014)
        # Expected max Sharpe under null hypothesis of no alpha given num_trials
        euler_gamma = 0.5772156649
        gamma_term = (1 - euler_gamma) * (2 * math.log(max(2, num_trials_tested))) ** -0.5
        expected_max_sharpe = math.sqrt(2 * math.log(max(2, num_trials_tested))) + gamma_term

        # DSR score attenuation
        dsr_score = max(0.0, min(1.0, (backtest_sharpe / max(0.1, expected_max_sharpe))))

        # 2. Probability of Backtest Overfitting (PBO)
        # Higher Sharpe with low trade count or excessive trials -> higher PBO
        suspicion_penalty = 0.0
        if backtest_profit_factor > 3.0 or backtest_sharpe > 3.0:
            suspicion_penalty += 0.35  # "Too good to be true" signature
        if total_trades < 50:
            suspicion_penalty += 0.25

        # PBO calculation
        pbo_estimate = round(min(0.95, max(0.10, (1.0 - dsr_score * 0.8) + suspicion_penalty)), 2)

        # 3. Minimum Backtest Length (MinBTL) in days
        min_btl_days = round(max(30.0, (expected_max_sharpe / max(0.1, backtest_sharpe)) ** 2 * 90.0), 0)

        # Final Verdict
        if pbo_estimate >= 0.70 or (backtest_sharpe > 3.5 and total_trades < 30):
            verdict = "REJECT_OVERFITTED"
            guidance = "High probability of backtest curve-fitting. Strategy unlikely to survive out-of-sample forward testing."
        elif pbo_estimate >= 0.40:
            verdict = "TEST_LONGER"
            guidance = "Moderate overfitting risk. Require additional paper/testnet validation before deployment."
        else:
            verdict = "ACCEPT_ROBUST"
            guidance = "Statistically robust parameter profile with low backtest overfitting signature."

        return {
            "strategy_name": strategy_name,
            # Honesty label: this is a summary-statistics proxy, not the Bailey et al. PBO, which
            # needs combinatorially symmetric cross-validation over the full return series.
            # The key name is kept for existing clients; the estimator field says what it is.
            "estimator": "summary_statistics_proxy",
            "pbo_is_cscv_measured": False,
            "deflated_sharpe_ratio": round(dsr_score, 2),
            "probability_of_backtest_overfitting_pbo": pbo_estimate,
            "min_backtest_length_days": int(min_btl_days),
            "overfitting_verdict": verdict,
            "advisory_guidance": guidance,
            "risk_flags": [
                f"Deflated Sharpe: {dsr_score:.2f} vs expected max {expected_max_sharpe:.2f}",
                f"PBO Estimate: {pbo_estimate * 100:.1f}%",
                "Curve-fit penalty applied" if suspicion_penalty > 0 else "Normal profile"
            ]
        }


overfitting_engine = OverfittingIntelligenceEngine()
