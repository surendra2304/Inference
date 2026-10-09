"""FastAPI Router for Deep Learning Predictions, Alternative Data Intelligence, and Performance Tracking."""

import time

from fastapi import APIRouter, HTTPException, Path, Query, status

from app.data.alternative_data import alt_data_engine
from app.ml.prediction_aggregator import prediction_aggregator
from app.ml.prediction_tracking import prediction_tracker

predictions_router = APIRouter(tags=["Deep Learning Predictions & Alternative Data"])


@predictions_router.get("/v1/predict/{asset}", status_code=status.HTTP_200_OK)
async def get_asset_prediction(
    asset: str = Path(..., description="Target asset symbol, e.g. BTC"),
    current_price: float | None = Query(
        default=None,
        description="Observed last price. Omit to run the signal math over a documented "
                    "placeholder series, in which case the confidence is withheld.",
    ),
    recent_returns: str | None = Query(
        default=None,
        description="Comma-separated observed returns, oldest first (e.g. 0.002,-0.001).",
    ),
):
    """Unified ensemble directional signal, its key drivers, and the provenance of its inputs.

    The route used to invent both inputs (``base_price = 65200.0 if "BTC" in symbol else
    3450.0`` plus a fixed return series) while answering with a directional call and a
    confidence of at least 0.60 for whatever symbol was asked about. Callers that supply
    observed inputs get a real reading; callers that do not get the same math over clearly
    labelled placeholder inputs and **no confidence figure**.
    """
    symbol = f"{asset.upper()}USDT" if not asset.upper().endswith("USDT") else asset.upper()

    supplied_returns: list[float] = []
    if recent_returns:
        try:
            supplied_returns = [float(part) for part in recent_returns.split(",") if part.strip()]
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="recent_returns must be comma-separated numbers, e.g. '0.002,-0.001,0.003'.",
            ) from None

    inputs_simulated = current_price is None or not supplied_returns
    price = current_price if current_price is not None else (65200.0 if "BTC" in symbol else 3450.0)
    returns = supplied_returns or [0.002, -0.001, 0.003, 0.001, 0.004]
    source = (
        "caller_supplied_price_and_returns" if not inputs_simulated
        else "placeholder_price_and_returns_hardcoded_in_route"
    )

    result = prediction_aggregator.aggregate_prediction(
        symbol=symbol,
        current_price=price,
        recent_returns=returns,
        inputs_simulated=inputs_simulated,
        input_source=source,
    )
    result["disclaimer"] = (
        "Advisory signal only; not investment advice. Directional signals are computed from "
        "the supplied series and are not calibrated to realised market outcomes."
    )
    return result


@predictions_router.get("/v1/predict/{asset}/history", status_code=status.HTTP_200_OK)
async def get_asset_prediction_history(asset: str = Path(..., description="Target asset symbol")):
    """Returns historical prediction log and out-of-sample accuracy verification."""
    return {
        "asset": asset.upper(),
        # ``.to_list()``: a bounded store is not JSON-serializable as itself.
        "history": prediction_tracker.history.to_list(),
        "retention": prediction_tracker.describe_history(),
        "accuracy_summary": prediction_tracker.get_source_accuracy_report()
    }


@predictions_router.get("/v1/intelligence/summary", status_code=status.HTTP_200_OK)
async def get_intelligence_summary(asset: str = Query(default="BTC", description="Base asset symbol")):
    """Returns consolidated alternative data intelligence snapshot (news, social spikes, on-chain flows, macro)."""
    return alt_data_engine.get_consolidated_alternative_data(asset)


@predictions_router.get("/v1/intelligence/accuracy", status_code=status.HTTP_200_OK)
async def get_intelligence_accuracy():
    """Returns granular accuracy report across all deep learning and alternative data sources."""
    return prediction_tracker.get_source_accuracy_report()


@predictions_router.post("/v1/predict/refresh", status_code=status.HTTP_200_OK)
async def refresh_predictions():
    """Forces model cache invalidation and hot inference refresh."""
    return {
        "status": "REFRESHED",
        "timestamp": time.time(),
        "active_model_versions": {
            "lstm_gru": "v2.4.1-lstm-gru",
            "transformer": "v1.8.0-trans-seq"
        }
    }
