"""Prediction Aggregation Engine combining Deep Learning, Technicals, Sentiment, and On-Chain."""

from typing import Any

from app.data.alternative_data import alt_data_engine
from app.ml.deep_models import deep_models_engine


class PredictionAggregationEngine:
    """Ensemble aggregator combining ML sequence models, TA signals, NLP sentiment, and on-chain metrics."""

    def aggregate_prediction(
        self,
        symbol: str,
        current_price: float,
        recent_returns: list[float],
        *,
        inputs_simulated: bool = False,
        input_source: str = "caller_supplied",
    ) -> dict[str, Any]:
        """Generates unified ensemble directional signal and key drivers.

        ``inputs_simulated`` must be set by any caller that invented its inputs (see
        ``GET /v1/predict/{asset}``, which used to pass a hardcoded price and a fixed
        return series and still published a ``unified_confidence`` of at least 0.60).
        When the inputs are simulated, the directional read is still computed — the math is
        real given its inputs — but the confidence is **withheld** (``None``) and the
        evidence class says why, because a confidence attached to invented inputs is a
        fabricated measurement.
        """
        dl_pred = deep_models_engine.predict_horizons(symbol, current_price, recent_returns)
        alt_data = alt_data_engine.get_consolidated_alternative_data(symbol)

        # Signal components
        # Three states, three values. The previous ternary mapped NEUTRAL to -1 (bearish): a 24h read
        # of NEUTRAL (the mean-reversion damping case) was published as a BEARISH call.
        lstm_direction_24h = dl_pred["horizons"]["24h"]["predicted_direction"]
        lstm_signal = {"BULLISH": 1.0, "BEARISH": -1.0}.get(lstm_direction_24h, 0.0)

        # The news and on-chain legs come from the alternative-data table, which is a static
        # fixture (evidence_class "synthetic_fixture"). A fixture cannot move a directional call,
        # so those legs are excluded (weight 0) while they are synthetic. Before this they carried
        # 60% of the composite, so the "direction" was set by invented sentiment and netflow.
        alt_is_synthetic = alt_data.get("evidence_class") == "synthetic_fixture"
        if alt_is_synthetic:
            news_signal = 0.0
            onchain_signal = 0.0
        else:
            news_signal = 1.0 if alt_data["news_intelligence"]["sentiment_score"] > 0.2 else (-1.0 if alt_data["news_intelligence"]["sentiment_score"] < -0.2 else 0.0)
            onchain_signal = 1.0 if alt_data["onchain_intelligence"]["exchange_netflow_24h_usd"] < 0 else -1.0

        # Weighted composite score: 40% DL, 30% On-Chain, 30% News/Social (measured legs only)
        composite_score = (0.40 * lstm_signal) + (0.30 * onchain_signal) + (0.30 * news_signal)

        direction = "BULLISH" if composite_score >= 0.25 else ("BEARISH" if composite_score <= -0.25 else "NEUTRAL")
        # No confidence is reported from the composite. The former mapping
        # ``min(0.95, 0.5 + |composite| * 0.45)`` was an invented formula; nothing calibrates it
        # against realised outcomes, so it is withheld in every case.
        raw_confidence = None

        # Which legs of the composite came from a fixture rather than an observation. The
        # alternative-data engine is a static table (app/data/alternative_data.py), so its
        # four legs are *always* synthetic here; only the price/return series can arrive as
        # an observation from the caller. Before this, a caller-supplied price plus invented
        # on-chain flows produced an unqualified "confidence".
        synthetic_legs = [
            "news_intelligence (sentiment_score, urgency, dominant_event)",
            "social_intelligence (reddit/twitter scores)",
            "onchain_intelligence (netflow, whale bias, active addresses)",
            "macro_intelligence (dxy, sp500 futures, gold, vix)",
        ]
        if inputs_simulated:
            synthetic_legs.insert(0, "price_series (no current_price supplied)")
            synthetic_legs.insert(1, "return_series (no recent_returns supplied)")

        confidence: float | None = raw_confidence
        if inputs_simulated:
            confidence_basis = (
                "withheld: the price and return inputs were simulated by the caller "
                "(inputs_simulated=true), so no measurement of this signal's reliability exists"
            )
        else:
            confidence_basis = (
                "withheld: no calibration against realised outcomes exists for this composite "
                f"(direction score {composite_score:+.2f}; measured legs only)"
            )

        key_drivers = []
        if onchain_signal > 0:
            key_drivers.append("Exchange net outflows (institutional accumulation)")
        if news_signal > 0:
            key_drivers.append("Bullish news NLP sentiment")
        if lstm_signal > 0:
            key_drivers.append("LSTM/Transformer sequence momentum")

        return {
            "symbol": symbol.upper(),
            "current_price": current_price,
            "unified_direction": direction,
            "unified_confidence": confidence,
            "confidence_basis": confidence_basis,
            "horizon": "24H",
            "key_drivers": key_drivers,
            "conflicting_signals": [] if abs(lstm_signal - news_signal) <= 1.0 else ["Deep Learning vs News Divergence"],
            "evidence_class": (
                "synthetic_fixture_inputs"
                if inputs_simulated
                else "caller_supplied_price_and_returns_with_synthetic_alt_data"
            ),
            "inputs_simulated": inputs_simulated,
            "input_source": input_source,
            "synthetic_legs": synthetic_legs,
            "synthetic_leg_count": len(synthetic_legs),
            "advisory_only": True,
            "deep_learning_forecast": dl_pred,
            "alternative_data_snapshot": alt_data
        }


prediction_aggregator = PredictionAggregationEngine()
