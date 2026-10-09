"""FastAPI Router for Enhanced Market Intelligence, Sentiment, On-Chain, and ML Predictions."""

import time

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.analysis.onchain_analytics import onchain_engine
from app.analysis.sentiment_analysis import sentiment_engine
from app.analysis.technical_analysis import ta_engine
from app.debate.enhanced_trading_debate import enhanced_trading_debate
from app.integrations.market_data import (
    BINANCE_SOURCE,
    InvalidMarketParameter,
    market_data_fetcher,
    normalize_symbol,
)
from app.market_monitoring.market_monitor import market_monitor
from app.ml.price_prediction import ml_prediction_model

enhanced_router = APIRouter(prefix="/v1/trading", tags=["Enhanced Market Intelligence"])


class PredictionRequest(BaseModel):
    """Payload for ML price trajectory prediction."""
    symbol: str = Field(default="BTCUSDT", pattern=r"^[A-Za-z0-9]{3,20}$", description="Target trading symbol")
    current_price: float | None = Field(default=None, gt=0, description="Current price (taken from the last candle if None)")


async def _candles_or_422(symbol: str, limit: int) -> tuple[list[dict], str]:
    """Market candles with their provenance. A malformed symbol is a 422, not an upstream call."""
    try:
        normalize_symbol(symbol)
        return await market_data_fetcher.get_ohlcv_sourced(symbol=symbol, limit=limit)
    except InvalidMarketParameter as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@enhanced_router.get("/market/analysis", status_code=status.HTTP_200_OK)
async def get_market_analysis(symbol: str = Query(default="BTCUSDT", description="Crypto trading pair")):
    """Returns comprehensive multi-agent market deliberation including technical indicators and consensus."""
    candles, source = await _candles_or_422(symbol, 100)
    news = await market_data_fetcher.get_news_and_social_feed(symbol=symbol[:3])
    orderbook = await market_data_fetcher.get_orderbook(symbol=symbol)

    analysis = await enhanced_trading_debate.conduct_advanced_market_deliberation(
        symbol=symbol,
        candles=candles,
        news_feed=news,
        orderbook=orderbook
    )
    # Provenance travels with the analysis. A synthetic series is not market data.
    if isinstance(analysis, dict):
        analysis["market_data_source"] = source
        analysis["news_source"] = "not_implemented"
    return analysis


@enhanced_router.get("/market/sentiment", status_code=status.HTTP_200_OK)
async def get_market_sentiment(symbol: str = Query(default="BTC", description="Base asset symbol")):
    """Returns NLP-extracted news sentiment, social indicators, and event classifications."""
    news = await market_data_fetcher.get_news_and_social_feed(symbol=symbol)
    sentiment = sentiment_engine.analyze_news(news)
    return {
        "symbol": symbol.upper(),
        "sentiment": sentiment,
        "raw_feed_count": len(news)
    }


@enhanced_router.get("/market/onchain", status_code=status.HTTP_200_OK)
async def get_onchain_metrics(symbol: str = Query(default="BTC", description="Base asset symbol")):
    """Returns on-chain network health, whale transfers, and exchange netflow statistics."""
    return onchain_engine.get_onchain_metrics(symbol=symbol)


@enhanced_router.post("/predict", status_code=status.HTTP_200_OK)
async def predict_price(req: PredictionRequest):
    """Generates multi-horizon ML price forecasts with confidence intervals and feature attributions."""
    candles, source = await _candles_or_422(req.symbol, 100)
    # Refuse rather than forecast on the synthetic baseline. Before this, a failed market-data
    # fetch produced a price trajectory from a zig-zag series with a fixed 65000 starting price.
    if source != BINANCE_SOURCE or not candles:
        return {
            "symbol": req.symbol.upper(),
            "status": "no_market_data",
            "market_data_source": source,
            "prediction": None,
            "reason": "public market data is unavailable, so no forecast is produced",
        }
    curr_price = req.current_price if req.current_price is not None else candles[-1]["close"]

    indicators = ta_engine.calculate_indicators(candles)
    news = await market_data_fetcher.get_news_and_social_feed(symbol=req.symbol[:3])
    sentiment = sentiment_engine.analyze_news(news)
    onchain = onchain_engine.get_onchain_metrics(symbol=req.symbol[:3])

    prediction = ml_prediction_model.predict_price_trajectory(
        current_price=curr_price,
        indicators=indicators,
        sentiment=sentiment,
        onchain=onchain
    )
    return {
        "symbol": req.symbol.upper(),
        "status": "forecast",
        "market_data_source": source,
        "prediction": prediction,
    }


@enhanced_router.get("/monitor/alerts", status_code=status.HTTP_200_OK)
async def get_market_alerts(symbol: str = Query(default="BTCUSDT", description="Target symbol")):
    """Returns real-time anomaly alerts including RSI extremes, volatility squeezes, and orderbook walls."""
    candles, source = await _candles_or_422(symbol, 50)
    if source != BINANCE_SOURCE or not candles:
        return {
            "symbol": symbol.upper(),
            "status": "no_market_data",
            "market_data_source": source,
            "active_alerts_count": 0,
            "alerts": [],
            "timestamp": time.time(),
        }
    curr_price = candles[-1]["close"]
    indicators = ta_engine.calculate_indicators(candles)
    news = await market_data_fetcher.get_news_and_social_feed(symbol=symbol[:3])
    sentiment = sentiment_engine.analyze_news(news)
    orderbook = await market_data_fetcher.get_orderbook(symbol=symbol)

    alerts = market_monitor.evaluate_market_alerts(
        symbol=symbol,
        current_price=curr_price,
        indicators=indicators,
        sentiment=sentiment,
        orderbook=orderbook
    )
    return {
        "symbol": symbol.upper(),
        "status": "evaluated",
        "market_data_source": source,
        "active_alerts_count": len(alerts),
        "alerts": alerts,
        "timestamp": time.time()
    }


@enhanced_router.get("/history/analysis", status_code=status.HTTP_200_OK)
async def get_historical_analysis(symbol: str = Query(default="BTCUSDT", description="Crypto pair")):
    """Returns historical technical indicator series and pattern markers."""
    candles, source = await _candles_or_422(symbol, 100)
    indicators = ta_engine.calculate_indicators(candles)
    return {
        "symbol": symbol.upper(),
        "market_data_source": source,
        "candle_count": len(candles),
        "indicators": indicators,
        "last_updated": time.time()
    }
