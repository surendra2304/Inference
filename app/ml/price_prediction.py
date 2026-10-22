"""Machine Learning Ensemble Price Prediction and Uncertainty Engine."""

from typing import Any


class PricePredictionModel:
    """Ensemble price prediction engine combining trend momentum, on-chain flows, and sentiment."""

    def predict_price_trajectory(
        self,
        current_price: float,
        indicators: dict[str, Any],
        sentiment: dict[str, Any],
        onchain: dict[str, Any]
    ) -> dict[str, Any]:
        """Generates multi-horizon price trajectory forecast with confidence intervals.

        Every number here is a function of the inputs. Before this, missing indicators were
        replaced by neutral defaults (RSI 50, ADX 25, ATR 1.5% of price), the feature attributions
        were a fixed 50/30/20 split, and the overall confidence was an ADX heuristic presented as
        if calibrated. Now missing inputs are listed, the interval is null without a measured ATR,
        attributions are each factor's share of the blended alpha, and the confidence is labelled.
        """
        missing: list[str] = []
        rsi_raw = indicators.get("rsi_14")
        macd_raw = (indicators.get("macd") or {}).get("histogram")
        adx_raw = indicators.get("adx")
        atr_raw = indicators.get("atr_14")
        if rsi_raw is None:
            missing.append("rsi_14")
        if macd_raw is None:
            missing.append("macd_histogram")
        if adx_raw is None:
            missing.append("adx")
        if atr_raw is None:
            missing.append("atr_14")
        sent_raw = sentiment.get("overall_score")
        if sent_raw is None:
            missing.append("sentiment_score")
        flow_raw = (onchain.get("exchange_flows") or {}).get("net_flow_usd")
        if flow_raw is None:
            missing.append("onchain_net_flow_usd")

        # Missing momentum and sentiment contribute nothing; they are not assumed neutral-positive.
        mom_factor = ((rsi_raw - 50.0) / 100.0 if rsi_raw is not None else 0.0) + (
            (macd_raw / 100.0) if macd_raw is not None else 0.0
        )
        sent_factor = (sent_raw * 0.4) if sent_raw is not None else 0.0
        flow_factor = 0.0 if flow_raw is None else (0.02 if flow_raw < 0 else -0.02)

        contributions = {
            "technical_momentum": 0.5 * mom_factor,
            "sentiment_nlp": 0.3 * sent_factor,
            "onchain_flow": 0.2 * flow_factor,
        }
        combined_alpha = sum(contributions.values())
        total_abs = sum(abs(v) for v in contributions.values())
        attributions = {
            f"{name}_pct": (round(abs(v) / total_abs * 100.0, 1) if total_abs > 0 else None)
            for name, v in contributions.items()
        }
        # Bound trajectory shift to realistic range (-3% to +3% per horizon)
        bounded_alpha = max(-0.03, min(0.03, combined_alpha))

        def horizon(mult: float, ci_mult: float | None) -> tuple[float, dict[str, float] | None]:
            pred = round(current_price * (1.0 + bounded_alpha * mult), 2)
            if atr_raw is None or current_price <= 0:
                return pred, None
            band = (atr_raw / current_price) * current_price * ci_mult
            return pred, {"lower": round(pred - band, 2), "upper": round(pred + band, 2)}

        pred_1h, ci_1h = horizon(0.3, 0.8)
        pred_4h, ci_4h = horizon(0.8, 1.5)
        pred_24h, ci_24h = horizon(1.5, 2.5)

        direction = "BULLISH_CONTINUATION" if bounded_alpha > 0.005 else (
            "BEARISH_PULLBACK" if bounded_alpha < -0.005 else "RANGE_BOUND")
        confidence = (
            round(min(0.92, max(0.65, 0.70 + (adx_raw / 200.0))), 2) if adx_raw is not None else None
        )

        def change(pred: float) -> float:
            return round(((pred - current_price) / current_price) * 100, 2)

        return {
            "current_price": current_price,
            "forecast_direction": direction,
            "overall_confidence": confidence,
            "confidence_basis": "adx_heuristic_not_calibrated" if confidence is not None else "not_measured",
            "missing_inputs": missing,
            "horizons": {
                "1h": {"predicted_price": pred_1h, "confidence_interval": ci_1h, "change_pct": change(pred_1h)},
                "4h": {"predicted_price": pred_4h, "confidence_interval": ci_4h, "change_pct": change(pred_4h)},
                "24h": {"predicted_price": pred_24h, "confidence_interval": ci_24h, "change_pct": change(pred_24h)},
            },
            "feature_attributions": attributions,
        }


ml_prediction_model = PricePredictionModel()
