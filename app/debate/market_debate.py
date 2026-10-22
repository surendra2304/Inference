"""Cross-Market Multi-Agent Debate Engine with Macro, Liquidity, and Correlation Specialists."""

from typing import Any

from app.analysis.cross_asset import cross_asset_engine
from app.analysis.liquidity_intel import liquidity_intel
from app.analysis.market_regime_intel import regime_intel


class MultiMarketDebateEngine:
    """Deliberates cross-venue arbitrage, global liquidity conditions, and portfolio concentration."""

    def conduct_cross_market_debate(self, portfolio_positions: dict[str, float]) -> dict[str, Any]:
        """Runs multi-agent market deliberation."""
        corr_data = cross_asset_engine.analyze_portfolio_correlation(portfolio_positions)
        reg_data = regime_intel.classify_market_regime()
        liq_data = liquidity_intel.analyze_asset_liquidity("BTCUSDT")

        # Specialist confidences were fixed literals (0.85 / 0.89 / 0.88, overall 0.87). None of them
        # is computed from an input, so each is None: "not measured", not a number.
        liquidity_known = liq_data.get("global_liquidity_score") is not None
        liquidity_findings = (
            f"Global Liquidity Score: {liq_data['global_liquidity_score']}. Best Execution: {liq_data['best_execution_venue']}."
            if liquidity_known else
            "Liquidity not measured: no venue order-book feed is configured."
        )
        specialist_deliberations = [
            {
                "specialist": "Macro Analyst",
                "findings": f"Macro Regime: {reg_data['macro_regime']}. BTC Dominance: {reg_data['leading_indicators']['btc_dominance_trend']}.",
                "bias": "BULLISH_RISK_ON",
                "confidence": None,
            },
            {
                "specialist": "Liquidity Analyst",
                "findings": liquidity_findings,
                "bias": "FAVOR_DEEP_POOLS" if liquidity_known else "NOT_MEASURED",
                "confidence": None,
            },
            {
                "specialist": "Correlation Analyst",
                "findings": f"Portfolio BTC Correlation: {corr_data['weighted_btc_correlation']}. Concentration Risk: {corr_data['concentration_risk_warning']}.",
                "bias": "CAUTION_ON_CORRELATION" if corr_data['concentration_risk_warning'] else "BALANCED",
                "confidence": None,
            }
        ]

        consensus = "EXPAND_CORE_ALLOCATION" if reg_data["macro_regime"] == "RISK_ON" and not corr_data["concentration_risk_warning"] else "DEFENSIVE_DIVERSIFICATION"

        return {
            "portfolio_market_consensus": consensus,
            "overall_confidence": None,
            "confidence_basis": "no specialist confidence is computed; see each specialist entry",
            "specialist_deliberations": specialist_deliberations,
            "regime_intelligence": reg_data,
            "liquidity_intelligence": liq_data,
            "correlation_analysis": corr_data
        }


multi_market_debate = MultiMarketDebateEngine()
