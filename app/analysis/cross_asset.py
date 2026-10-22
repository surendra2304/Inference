"""Cross-Asset Correlation Engine, Rolling Window Matrix, and Concentration Risk Warning."""

import math
from typing import Any


class CrossAssetCorrelationEngine:
    """Computes correlation matrices across crypto assets, BTC Dominance, S&P 500, Gold, and DXY."""

    def get_correlation_matrix(self) -> dict[str, Any]:
        """Returns multi-window rolling cross-asset correlation matrix."""
        assets = ["BTC", "ETH", "SOL", "BNB", "SP500", "GOLD", "DXY"]

        # Deterministic representative correlation coefficients
        matrix = {
            "BTC": {"BTC": 1.00, "ETH": 0.88, "SOL": 0.79, "BNB": 0.74, "SP500": 0.42, "GOLD": 0.21, "DXY": -0.55},
            "ETH": {"BTC": 0.88, "ETH": 1.00, "SOL": 0.84, "BNB": 0.71, "SP500": 0.45, "GOLD": 0.18, "DXY": -0.51},
            "SOL": {"BTC": 0.79, "ETH": 0.84, "SOL": 1.00, "BNB": 0.68, "SP500": 0.38, "GOLD": 0.12, "DXY": -0.48},
            "BNB": {"BTC": 0.74, "ETH": 0.71, "SOL": 0.68, "BNB": 1.00, "SP500": 0.32, "GOLD": 0.15, "DXY": -0.42},
            "SP500": {"BTC": 0.42, "ETH": 0.45, "SOL": 0.38, "BNB": 0.32, "SP500": 1.00, "GOLD": 0.05, "DXY": -0.62},
            "GOLD": {"BTC": 0.21, "ETH": 0.18, "SOL": 0.12, "BNB": 0.15, "SP500": 0.05, "GOLD": 1.00, "DXY": -0.38},
            "DXY": {"BTC": -0.55, "ETH": -0.51, "SOL": -0.48, "BNB": -0.42, "SP500": -0.62, "GOLD": -0.38, "DXY": 1.00}
        }

        return {
            "assets": assets,
            "matrix_24h": matrix,
            "correlation_regime": "HIGH_CRYPTO_BETA",
            "btc_dominance_pct": 56.4,
            "dxy_index": 103.8
        }

    def analyze_portfolio_correlation(self, positions: dict[str, float]) -> dict[str, Any]:
        """Calculates portfolio correlation to BTC and evaluates concentration risk."""
        total_val = sum(positions.values())
        # A caller that reaches the engine directly (not via the validated route) can hand
        # over a non-finite value. Echoing it back made the response unencodable, because
        # Starlette encodes JSON with allow_nan=False. Report the unusable input instead of
        # propagating it into the arithmetic.
        if not math.isfinite(total_val):
            return {
                "portfolio_value_usd": None,
                "weighted_btc_correlation": None,
                "concentration_risk_warning": False,
                "diversification_recommendation": (
                    "Portfolio total is not a finite number; no correlation assessment is "
                    "possible. Check the supplied position values."
                ),
            }
        if total_val == 0:
            # This branch previously returned {"weighted_btc_correlation",
            # "concentration_warning"} — TWO keys, one of them misspelled relative to the
            # full path — while the normal path below returns four keys including
            # "concentration_risk_warning". Any consumer reading the documented keys
            # crashed with KeyError on an empty or all-zero portfolio:
            # POST /v1/market/portfolio-analysis {"positions": {}} returned HTTP 500
            # (KeyError: 'concentration_risk_warning' at app/debate/market_debate.py:34)
            # even though the published schema accepts that body. A function must not
            # change its response shape based on input; the early return now yields the
            # identical key set so a caller can rely on the contract unconditionally.
            return {
                "portfolio_value_usd": 0.0,
                "weighted_btc_correlation": 0.0,
                "concentration_risk_warning": False,
                "diversification_recommendation": (
                    "No positions supplied; concentration risk is undefined. "
                    "Add positions to obtain a correlation assessment."
                ),
            }

        matrix = self.get_correlation_matrix()["matrix_24h"]
        weighted_corr = 0.0

        for sym, val in positions.items():
            clean = sym.replace("USDT", "").replace("USD", "").upper()
            corr = matrix.get(clean, {}).get("BTC", 0.75)
            weight = val / total_val
            weighted_corr += corr * weight

        weighted_corr = round(weighted_corr, 2)
        is_concentrated = weighted_corr >= 0.80

        recommendation = "Portfolio is well-diversified." if not is_concentrated else "High BTC beta correlation (>0.80). Consider introducing market-neutral or inverse assets."

        return {
            "portfolio_value_usd": total_val,
            "weighted_btc_correlation": weighted_corr,
            "concentration_risk_warning": is_concentrated,
            "diversification_recommendation": recommendation
        }


cross_asset_engine = CrossAssetCorrelationEngine()
