"""Alternative Data Aggregation Engine (News NLP, Social Spikes, On-Chain Whales, Macro)."""

import time
from typing import Any


class AlternativeDataEngine:
    """Ingests and normalizes alternative data feeds for alpha generation."""

    def get_consolidated_alternative_data(self, asset: str = "BTC") -> dict[str, Any]:
        """Provides consolidated sentiment, social spikes, on-chain flows, and macro indicators."""
        clean_asset = asset.upper().replace("USDT", "").replace("USD", "")

        return {
            "asset": clean_asset,
            "timestamp": time.time(),
            # Declared provenance. This engine is a static table: it ingests no feed, and it
            # returns the same numbers for every asset (BTC and DOGE alike), so a reader must
            # not treat these as observations about a market. Before this block the payload
            # carried no provenance at all and fed the directional signal at
            # GET /v1/predict/{asset} as if it were live data.
            "evidence_class": "synthetic_fixture",
            "inputs_simulated": True,
            "asset_specific": False,
            "generator": (
                "static fixture table in app/data/alternative_data.py; no news, social, "
                "on-chain or macro feed is configured in this process"
            ),
            "values_identical_across_assets": True,
            "news_intelligence": {
                "sentiment_score": 0.42,
                "urgency_level": "MODERATE",
                "dominant_event": "Institutional ETF Accumulation Inflows",
                "impact_score_0_100": 78.5
            },
            "social_intelligence": {
                "reddit_crypto_sentiment": 0.58,
                "twitter_attention_score": 84.0,
                "social_volume_spike_detected": True,
                "sentiment_trend": "ACCELERATING_BULLISH"
            },
            "onchain_intelligence": {
                "exchange_netflow_24h_usd": -145_000_000.0,
                "whale_wallet_bias": "STRONG_ACCUMULATION",
                "active_addresses_change_pct": 4.8,
                "defi_tvl_trend": "EXPANDING"
            },
            "macro_intelligence": {
                "dxy_index": 103.4,
                "sp500_futures_trend": "POSITIVE",
                "gold_trend": "STABLE",
                "vix_volatility_index": 14.8,
                "macro_regime": "RISK_ON"
            }
        }


alt_data_engine = AlternativeDataEngine()
