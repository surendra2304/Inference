"""Cross-Exchange Liquidity Depth Scoring and Price Impact Simulation.

No order-book or venue feed is wired into this process. The previous version returned a fixed
``global_liquidity_score`` of 92.5, a fixed best venue of BINANCE, fixed slippage figures for
every asset and ``liquidity_crisis_flag: False``. Those are not observations, so they are now
reported as not measured.
"""

from typing import Any

NOT_MEASURED_REASON = "No venue order-book feed is configured; liquidity has not been measured."


class CrossExchangeLiquidityIntel:
    """Liquidity depth and slippage are reported only when a venue feed supplies them (none does)."""

    def analyze_asset_liquidity(self, symbol: str = "BTCUSDT") -> dict[str, Any]:
        """Liquidity for ``symbol``; every measured field is None until a feed exists."""
        base_asset = symbol.replace("USDT", "").replace("USD", "").upper()
        return {
            "status": "not_measured",
            "reason": NOT_MEASURED_REASON,
            "symbol": symbol.upper(),
            "base_asset": base_asset,
            "global_liquidity_score": None,
            "liquidity_trend": None,
            "slippage_estimates": None,
            "best_execution_venue": None,
            "liquidity_crisis_flag": None,
        }


liquidity_intel = CrossExchangeLiquidityIntel()
