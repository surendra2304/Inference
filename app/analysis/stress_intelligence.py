"""Market Stress Intelligence and Historical Crisis Scenario Testing."""

from typing import Any


class StressIntelligenceEngine:
    """Detects liquidity drains, cross-asset correlation spikes, and runs historical stress scenarios."""

    def evaluate_market_stress(
        self,
        bid_ask_spread_pct: float,
        cross_asset_correlation: float,
        volatility_atr_pct: float
    ) -> dict[str, Any]:
        """Calculates aggregate stress index."""
        # Baseline score: 0 to 100
        stress_score = 0.0
        stress_score += min(40.0, (bid_ask_spread_pct / 0.002) * 20.0)
        stress_score += min(30.0, max(0.0, cross_asset_correlation - 0.5) * 60.0)
        stress_score += min(30.0, (volatility_atr_pct / 0.03) * 15.0)

        stress_score = round(min(100.0, stress_score), 1)
        regime = "NORMAL" if stress_score < 35 else ("ELEVATED_STRESS" if stress_score < 70 else "ACUTE_CRISIS_STRESS")

        return {
            "market_stress_score": stress_score,
            "stress_regime": regime,
            "metrics": {
                "bid_ask_spread_pct": bid_ask_spread_pct,
                "cross_asset_correlation": cross_asset_correlation,
                "volatility_atr_pct": volatility_atr_pct
            },
            "advisory_guidance": "REDUCE_EXPOSURE_AND_PAUSE_OPTIMIZATION" if stress_score >= 70 else ("MONITOR_CLOSELY" if stress_score >= 35 else "NORMAL_TRADING_CONDITIONS")
        }

    def run_historical_stress_test(self, portfolio_equity: float, active_notional: float) -> dict[str, Any]:
        """Simulates portfolio impact under classic historical market shock scenarios.

        Three defects were fixed here, all reachable from a request body the published
        schema accepts (``{"portfolio_equity": 0, "active_notional": 0}``):

        1. ``ZeroDivisionError``. Each scenario computed the loss ratio twice — once for
           ``estimated_drawdown_pct`` guarded by ``if portfolio_equity > 0``, and again on
           the very next line for ``survival_probability`` *without* the guard. The guard
           proves the author knew equity could be zero; it was applied to only half of a
           duplicated expression. The ratio is now computed once and reused, so the two
           fields cannot disagree and there is only one division to guard.
        2. ``survival_probability`` was derived only from the ratio, so a wiped-out
           account was reported as ``HIGH``. With no equity, any notional is unbacked and
           the honest answer is ``CRITICAL``.
        3. ``stress_resilience_rating`` was the constant string ``"PASSING"`` on every
           response regardless of the arithmetic: a portfolio that loses its entire
           equity in the modelled shock was still rated PASSING. It is now derived from
           the worst-case scenarios, so the summary cannot contradict its own detail.
        """
        equity = float(portfolio_equity)
        notional = float(active_notional)

        shocks = [
            ("COVID March 2020 Liquidity Crunch (-35% Price Shock / 5x Spread)", 0.35),
            ("FTX Insolvency Cascading Deleveraging (-22% Shock)", 0.22),
            ("May 2021 Flash Liquidation (-15% Rapid Wick)", 0.15),
        ]

        scenarios: list[dict[str, Any]] = []
        worst_ratio = 0.0
        for scenario_name, shock in shocks:
            drawdown_usd = round(notional * shock, 2)
            # One division, one guard: the ratio is the single source of truth for both
            # the percentage and the survival verdict below.
            if equity > 0:
                loss_ratio = (notional * shock) / equity
            else:
                loss_ratio = float("inf") if notional > 0 else 0.0
            worst_ratio = max(worst_ratio, loss_ratio)

            if loss_ratio == float("inf"):
                survival = "CRITICAL"
            elif loss_ratio < 0.20:
                survival = "HIGH"
            elif loss_ratio < 0.50:
                survival = "MEDIUM"
            else:
                survival = "LOW"

            scenarios.append({
                "scenario_name": scenario_name,
                "estimated_drawdown_usd": drawdown_usd,
                "estimated_drawdown_pct": (
                    round(loss_ratio * 100, 2) if loss_ratio != float("inf") else None
                ),
                "exceeds_available_equity": drawdown_usd > equity,
                "survival_probability": survival,
            })

        if equity <= 0 and notional > 0:
            rating = "FAILING"
            rationale = (
                "No portfolio equity is available to absorb the modelled shocks, so every "
                "scenario is unbacked. Any active notional is fully exposed."
            )
        elif worst_ratio < 0.20:
            rating = "PASSING"
            rationale = "Worst modelled loss stays below 20% of portfolio equity."
        elif worst_ratio < 0.50:
            rating = "MARGINAL"
            rationale = "Worst modelled loss consumes between 20% and 50% of portfolio equity."
        else:
            rating = "FAILING"
            rationale = "Worst modelled loss consumes 50% or more of portfolio equity."

        return {
            "portfolio_equity": equity,
            "active_notional": notional,
            "scenario_results": scenarios,
            "worst_case_loss_pct_of_equity": (
                round(worst_ratio * 100, 2) if worst_ratio != float("inf") else None
            ),
            "stress_resilience_rating": rating,
            "rating_rationale": rationale,
        }


stress_intelligence_engine = StressIntelligenceEngine()
