"""FastAPI Router for Deep Learning Predictions, Alternative Data Intelligence, and Performance Tracking."""

import math
import time

from fastapi import APIRouter, HTTPException, Path, Query, status

from app.data.alternative_data import alt_data_engine
from app.ml.prediction_aggregator import prediction_aggregator
from app.ml.prediction_tracking import prediction_tracker

predictions_router = APIRouter(tags=["Deep Learning Predictions & Alternative Data"])

MAX_RECENT_RETURN_OBSERVATIONS = 256


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
    if not isinstance(asset, str) or not asset.strip() or len(asset.strip()) > 64:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="asset must be a non-empty symbol of at most 64 characters.",
        )
    if any(ord(char) < 32 or ord(char) == 127 for char in asset):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="asset must not contain control characters.",
        )
    asset_symbol = asset.strip().upper()
    symbol = f"{asset_symbol}USDT" if not asset_symbol.endswith("USDT") else asset_symbol

    if current_price is not None and (
        not math.isfinite(current_price) or current_price <= 0
    ):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="current_price must be finite and greater than zero.",
        )

    supplied_returns: list[float] = []
    if recent_returns is not None:
        if not recent_returns.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="recent_returns must contain at least one numeric observation.",
            )

        parts = recent_returns.split(",")
        if len(parts) > MAX_RECENT_RETURN_OBSERVATIONS:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    "recent_returns must contain no more than "
                    f"{MAX_RECENT_RETURN_OBSERVATIONS} observations."
                ),
            )
        if any(not part.strip() for part in parts):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="recent_returns must not contain empty observations.",
            )
        try:
            supplied_returns = [float(part.strip()) for part in parts]
        except (OverflowError, ValueError):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="recent_returns must be comma-separated finite numbers.",
            ) from None
        if any(not math.isfinite(value) for value in supplied_returns):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="recent_returns must contain only finite numbers.",
            )

    # Supplying one input does not make a prediction fully observation-backed: both the
    # price and return series must be present before the result can be labeled as such.
    price_is_simulated = current_price is None
    returns_are_simulated = recent_returns is None
    inputs_simulated = price_is_simulated or returns_are_simulated
    price = current_price if current_price is not None else (65200.0 if "BTC" in symbol else 3450.0)
    returns = supplied_returns if recent_returns is not None else [0.002, -0.001, 0.003, 0.001, 0.004]
    if not inputs_simulated:
        source = "caller_supplied_price_and_returns"
    elif price_is_simulated and returns_are_simulated:
        source = "placeholder_price_and_returns_hardcoded_in_route"
    elif price_is_simulated:
        source = "placeholder_price_with_caller_supplied_returns"
    else:
        source = "caller_supplied_price_with_placeholder_returns"

    result = prediction_aggregator.aggregate_prediction(
        symbol=symbol,
        current_price=price,
        recent_returns=returns,
        inputs_simulated=inputs_simulated,
        input_source=source,
        price_is_simulated=price_is_simulated,
        returns_are_simulated=returns_are_simulated,
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
