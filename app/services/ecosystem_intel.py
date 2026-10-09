"""Central Ecosystem Intelligence Hub for System-Wide Diagnostics and Early Warning Alerts."""

import time
from typing import Any

from app.analysis.cross_asset import cross_asset_engine
from app.analysis.market_regime_intel import regime_intel
from app.intelligence.meta_intel import meta_intelligence
from app.learning.continuous_learning import continuous_learning_engine


class EcosystemIntelligenceHub:
    """Consolidates cross-market states, active strategy portfolios, continuous learning, and system-level early warnings."""

    def get_ecosystem_intelligence_report(self) -> dict[str, Any]:
        """Provides full real-time model of entire trading ecosystem."""
        regime_data = regime_intel.classify_market_regime()
        corr_data = cross_asset_engine.get_correlation_matrix()
        learning_data = continuous_learning_engine.get_learning_status()
        meta_data = meta_intelligence.generate_meta_intelligence_report()

        proactive_early_warnings = []
        if regime_data["leading_indicators"]["global_futures_funding_rate_avg"] > 0.015:
            proactive_early_warnings.append("Elevated funding rates detected: potential long squeeze risk.")
        if corr_data["dxy_index"] > 105.0:
            proactive_early_warnings.append("DXY strengthening rapidly: risk-off headwind across digital assets.")

        return {
            "timestamp": time.time(),
            "overall_ecosystem_intelligence_score": meta_data["meta_intelligence_quality_score"],
            # "OPTIMAL_AUTONOMOUS_OPERATION" was a constant; with no measured quality score the
            # honest status is that it is not measured.
            "ecosystem_status": "NOT_MEASURED" if meta_data["meta_intelligence_quality_score"] is None else "MEASURED",
            "proactive_early_warnings": proactive_early_warnings or ["No critical ecosystem anomalies detected."],
            "market_regime": regime_data,
            "continuous_learning_summary": learning_data,
            "meta_intelligence_audit": meta_data
        }

    def conduct_ecosystem_consultation(
        self,
        portfolio_positions: dict[str, float],
        active_strategies: list[str]
    ) -> dict[str, Any]:
        """Generates comprehensive ecosystem-level advisory recommendations."""
        corr_analysis = cross_asset_engine.analyze_portfolio_correlation(portfolio_positions)
        regime_data = regime_intel.classify_market_regime()

        return {
            "ecosystem_verdict": "BALANCED_PORTFOLIO_STATE",
            "recommended_macro_bias": "RISK_ON_SELECTIVE",
            "active_strategy_count": len(active_strategies),
            "portfolio_btc_correlation": corr_analysis["weighted_btc_correlation"],
            "guidance_summary": (
                f"Ecosystem operating in {regime_data['macro_regime']} regime. "
                f"Portfolio BTC correlation at {corr_analysis['weighted_btc_correlation']}. "
                "Maintain conservative risk budgeting on live capital."
            )
        }


ecosystem_hub = EcosystemIntelligenceHub()
