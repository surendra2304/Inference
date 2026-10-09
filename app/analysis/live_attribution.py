"""Live Capital Performance Attribution and Slippage Disparity Analytics.

Reliability needs a measured testnet win rate. The previous version defaulted it to 0.60, so every
live book was scored against a benchmark nobody measured. Without ``testnet_metrics["win_rate"]``
the reliability score and the allocation recommendation are None.
"""

from typing import Any


class LiveAttributionEngine:
    """Analyzes real execution slippage, testnet vs live divergence, and computes strategy reliability scores."""

    def evaluate_live_attribution(
        self,
        live_trades: list[dict[str, Any]],
        testnet_metrics: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Live vs testnet alpha retention, execution drag, and (when measurable) a reliability score."""
        total_live_pnl = sum(float(t["pnl"]) for t in live_trades)
        total_slippage_usd = sum(
            abs(float(t["expected_price"]) - float(t["fill_price"])) * float(t["qty"]) for t in live_trades
        )
        win_count = sum(1 for t in live_trades if float(t["pnl"]) > 0)
        total_trades = len(live_trades)

        live_win_rate = (win_count / total_trades) if total_trades > 0 else None
        testnet_win_rate = (testnet_metrics or {}).get("win_rate")

        # Heuristic blend (0.6 parity + 0.4 sample size). The weights are not fitted to outcomes; the
        # result is only computed when a measured testnet win rate exists to compare against.
        reliability_score: float | None = None
        if live_win_rate is not None and testnet_win_rate is not None:
            wr_parity = max(0.0, 1.0 - abs(live_win_rate - float(testnet_win_rate)))
            sample_weight = min(1.0, total_trades / 50.0)
            reliability_score = round(0.6 * wr_parity + 0.4 * sample_weight, 2)

        return {
            "total_live_trades": total_trades,
            "total_live_pnl_usd": round(total_live_pnl, 2),
            "live_win_rate": round(live_win_rate, 2) if live_win_rate is not None else None,
            "testnet_win_rate": round(float(testnet_win_rate), 2) if testnet_win_rate is not None else None,
            "total_slippage_usd": round(total_slippage_usd, 2),
            "strategy_reliability_score": reliability_score,
            "reliability_status": (
                "computed (heuristic blend; not calibrated)" if reliability_score is not None
                else "not_measured: no testnet win rate supplied"
            ),
            "is_recommended_for_capital_allocation": (
                reliability_score >= 0.70 if reliability_score is not None else None
            ),
            "execution_quality_assessment": (
                "NO_TRADES" if total_trades == 0
                else ("EXCELLENT" if total_slippage_usd < 50.0 else ("ACCEPTABLE" if total_slippage_usd < 200.0 else "POOR_SLIPPAGE_DRAG"))
            ),
        }


live_attribution_engine = LiveAttributionEngine()
