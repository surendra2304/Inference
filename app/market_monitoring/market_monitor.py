"""Real-time Multi-Dimensional Market Monitoring Engine."""

import time
from typing import Any


class MarketMonitor:
    """Detects microstructure anomalies, momentum exhaustion, and regime shifts."""

    def evaluate_market_alerts(
        self,
        symbol: str,
        indicators: dict[str, Any],
        sentiment: dict[str, Any],
        orderbook: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Generates real-time anomaly alerts based on market telemetry."""
        alerts = []
        now = time.time()

        indicators = indicators or {}
        sentiment = sentiment or {}
        orderbook = orderbook or {}

        rsi = indicators.get("rsi_14")
        rsi = float(rsi) if rsi is not None else 50.0

        bb = indicators.get("bollinger_bands")
        if not isinstance(bb, dict):
            bb = {}
        bw_val = bb.get("bandwidth_pct")
        bw = float(bw_val) if bw_val is not None else 4.0

        sent_val = sentiment.get("overall_score")
        sent_score = float(sent_val) if sent_val is not None else 0.0

        # 1. RSI Extremes
        if rsi >= 75.0:
            alerts.append({
                "severity": "WARNING",
                "type": "MOMENTUM_OVERBOUGHT",
                "message": f"{symbol} RSI reached {rsi:.1f} (Overbought threshold). Caution on long breakouts.",
                "timestamp": now,
            })
        elif rsi <= 25.0:
            alerts.append({
                "severity": "WARNING",
                "type": "MOMENTUM_OVERSOLD",
                "message": f"{symbol} RSI dropped to {rsi:.1f} (Oversold threshold). Mean reversion potential.",
                "timestamp": now,
            })

        # 2. Volatility Compression
        if bw < 2.5:
            alerts.append({
                "severity": "INFO",
                "type": "VOLATILITY_SQUEEZE",
                "message": f"{symbol} Bollinger Bandwidth compressed to {bw:.2f}%. High-probability explosive expansion imminent.",
                "timestamp": now,
            })

        # 3. Sentiment Shift
        if abs(sent_score) > 0.6:
            alerts.append({
                "severity": "INFO",
                "type": "SENTIMENT_EXTREME",
                "message": f"Social/News sentiment scored extreme at {sent_score:+.2f}.",
                "timestamp": now,
            })

        # 4. Orderbook Imbalance
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])
        if bids and asks and isinstance(bids, list) and isinstance(asks, list):
            try:
                bid_vol = sum(q for _, q in bids[:5])
                ask_vol = sum(q for _, q in asks[:5])
                ratio = (bid_vol / ask_vol) if ask_vol > 0 else 1.0
                if ratio > 2.0:
                    alerts.append({
                        "severity": "INFO",
                        "type": "ORDERBOOK_BID_WALL",
                        "message": f"Significant bid-side support ({bid_vol:.1f} vs {ask_vol:.1f} asks in top 5 levels).",
                        "timestamp": now,
                    })
            except Exception:
                pass

        return alerts


market_monitor = MarketMonitor()
