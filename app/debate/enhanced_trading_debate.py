"""Enhanced trading deliberation: each specialist reports only what its inputs support.

Before this rewrite the four specialists had fixed confidences (0.85, 0.82, 0.88 and a 0.80
fallback), the on-chain specialist was always BULLISH whatever the flows were, the consensus
was a confidence-weighted average of those literals, and the code crashed when a value was
missing (``float(None)``, a ``:,.2f`` format on a null price). Now:

* a specialist's bias is derived from its inputs, or is ``UNKNOWN`` when the inputs are missing;
* a specialist has a ``confidence`` only where a measured quantity backs it; otherwise ``None``;
* the consensus averages only the specialists that have a bias, with equal weight, and no
  overall confidence is reported because none has been calibrated against outcomes.
"""

from __future__ import annotations

from typing import Any

from app.analysis.onchain_analytics import onchain_engine
from app.analysis.sentiment_analysis import sentiment_engine
from app.analysis.technical_analysis import ta_engine
from app.ml.price_prediction import ml_prediction_model

BULLISH = "BULLISH"
BEARISH = "BEARISH"
NEUTRAL = "NEUTRAL"
UNKNOWN = "UNKNOWN"


def _fmt_money(value: Any) -> str:
    return f"${value:,.2f}" if isinstance(value, (int, float)) else "not measured"


def _fmt_pct(value: Any) -> str:
    return f"{value:+.2f}%" if isinstance(value, (int, float)) else "not measured"


class EnhancedTradingDebateEngine:
    """Deliberation over measured inputs: technical, sentiment, on-chain, and quantitative views."""

    async def conduct_advanced_market_deliberation(
        self,
        symbol: str,
        candles: list[dict[str, Any]],
        news_feed: list[dict[str, Any]],
        orderbook: dict[str, Any],
    ) -> dict[str, Any]:
        """Runs one deliberation over the inputs given. Nothing is assumed when an input is absent."""
        indicators = ta_engine.calculate_indicators(candles)
        sentiment = sentiment_engine.analyze_news(news_feed)
        onchain = onchain_engine.get_onchain_metrics(symbol)
        curr_price: float | None = candles[-1]["close"] if candles else None
        ml_pred = (
            ml_prediction_model.predict_price_trajectory(curr_price, indicators, sentiment, onchain)
            if curr_price is not None
            else {"forecast_direction": None, "overall_confidence": None, "horizons": {}, "missing_inputs": ["candles"]}
        )

        rsi = indicators.get("rsi_14")
        macd_hist = (indicators.get("macd") or {}).get("histogram")
        if rsi is None:
            ta_bias = UNKNOWN
        elif rsi > 52:
            ta_bias = BULLISH
        elif rsi < 48:
            ta_bias = BEARISH
        else:
            ta_bias = NEUTRAL
        ta_perspective = {
            "specialist": "Technical Analyst",
            "findings": (
                f"Regime: {indicators.get('market_regime') or 'not measured'}. RSI: {rsi if rsi is not None else 'not measured'}, "
                f"MACD histogram: {macd_hist if macd_hist is not None else 'not measured'}. "
                f"Patterns: {len(indicators.get('patterns') or [])} detected."
            ),
            "bias": ta_bias,
            "confidence": None,
            "confidence_basis": "rule_based_not_calibrated",
        }

        classification = sentiment.get("classification")
        sent_score = sentiment.get("overall_score")
        if sent_score is None:
            sent_bias = UNKNOWN
        elif sent_score > 0.1:
            sent_bias = BULLISH
        elif sent_score < -0.1:
            sent_bias = BEARISH
        else:
            sent_bias = NEUTRAL
        sent_perspective = {
            "specialist": "Sentiment Analyst",
            "findings": (
                f"Overall sentiment: {classification} (score: {sent_score if sent_score is not None else 'not measured'}). "
                f"Items: {sentiment.get('item_count', 0)}."
            ),
            "bias": sent_bias,
            "confidence": None,
            "confidence_basis": "rule_based_not_calibrated",
        }

        flows = onchain.get("exchange_flows") or {}
        net_flow = flows.get("net_flow_usd")
        if net_flow is None:
            onchain_bias = UNKNOWN
        elif net_flow < 0:
            onchain_bias = BULLISH  # net outflow from exchanges, a conventional accumulation read
        elif net_flow > 0:
            onchain_bias = BEARISH
        else:
            onchain_bias = NEUTRAL
        onchain_perspective = {
            "specialist": "On-Chain Analyst",
            "findings": (
                f"Exchange flow bias: {flows.get('flow_bias') or 'not measured'}. "
                f"Net flow: {_fmt_money(net_flow)}."
                + ("" if onchain.get("status") != "not_measured" else " No on-chain provider is configured.")
            ),
            "bias": onchain_bias,
            "confidence": None,
            "confidence_basis": "not_measured" if onchain_bias == UNKNOWN else "rule_based_not_calibrated",
        }

        direction = ml_pred.get("forecast_direction")
        if direction is None:
            quant_bias = UNKNOWN
        elif "BULLISH" in direction:
            quant_bias = BULLISH
        elif "BEARISH" in direction:
            quant_bias = BEARISH
        else:
            quant_bias = NEUTRAL
        h24 = (ml_pred.get("horizons") or {}).get("24h") or {}
        quant_perspective = {
            "specialist": "Quantitative ML Modeler",
            "findings": (
                f"Forecast direction: {direction or 'not measured'}. "
                f"24H target: {_fmt_money(h24.get('predicted_price'))} ({_fmt_pct(h24.get('change_pct'))})."
            ),
            "bias": quant_bias,
            "confidence": ml_pred.get("overall_confidence"),
            "confidence_basis": ml_pred.get("confidence_basis", "not_measured"),
        }

        specialists: list[dict[str, Any]] = [ta_perspective, sent_perspective, onchain_perspective, quant_perspective]
        voting = [s for s in specialists if s["bias"] != UNKNOWN]
        score = (
            sum({BULLISH: 1.0, BEARISH: -1.0, NEUTRAL: 0.0}[s["bias"]] for s in voting) / len(voting)
            if voting
            else None
        )
        if score is None:
            overall_consensus = "NO_EVIDENCE"
        elif score >= 0.3:
            overall_consensus = "BULLISH_CONVERGENCE"
        elif score <= -0.3:
            overall_consensus = "BEARISH_DIVERGENCE"
        else:
            overall_consensus = "NEUTRAL_CONSOLIDATION"

        synthesis_summary = (
            f"Deliberation over {len(voting)}/{len(specialists)} specialists with measured inputs "
            f"({overall_consensus}; mean bias score: {score:+.2f}). "
            f"Overall confidence is not reported: no component has been calibrated against outcomes."
            if score is not None
            else "No specialist had measured inputs; no consensus is reported."
        )

        return {
            "symbol": symbol.upper(),
            "current_price": curr_price,
            "overall_consensus": overall_consensus,
            "overall_bias_score": None if score is None else round(score, 3),
            "overall_confidence": None,
            "overall_confidence_basis": "not_calibrated",
            "specialists_with_evidence": len(voting),
            "synthesis_summary": synthesis_summary,
            "specialist_deliberations": specialists,
            "technical_indicators": indicators,
            "sentiment_analysis": sentiment,
            "onchain_analytics": onchain,
            "price_predictions": ml_pred,
        }


enhanced_trading_debate = EnhancedTradingDebateEngine()
