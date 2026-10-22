# Enhanced Trading Intelligence & Market Analysis

Inference provides advanced quantitative market intelligence, multi-agent deliberation, sentiment extraction, on-chain flows, and machine learning price trajectory forecasting for algorithmic trading systems.

> [!IMPORTANT]
> **Advisory Invariant**: Inference produces intelligence, confidence scores, and structured parameter bounds. It NEVER executes trades directly or calls exchange endpoints with private keys.

---

## 1. Advanced Architecture

```
                                +---------------------------+
                                | Market Data & News Ingest |
                                +-------------+-------------+
                                              |
        +-----------------------+-------------+-------------+-----------------------+
        |                       |                           |                       |
        v                       v                           v                       v
+---------------+       +---------------+           +---------------+       +---------------+
|  50+ TA Engine|       |  NLP Sentiment|           |  On-Chain Data|       |  ML Forecaster|
| (MACD, RSI,   |       | (News, Social,|           | (Whales, NVT, |       | (1h/4h/24h    |
|  Patterns)    |       |  Event NLP)   |           |  Netflow)     |       |  Confidence)  |
+-------+-------+       +-------+-------+           +-------+-------+       +-------+-------+
        |                       |                           |                       |
        +-----------------------+-------------+-------------+-----------------------+
                                              |
                                              v
                              +-------------------------------+
                              | Enhanced Multi-Agent Deliberation |
                              | (Confidence-Weighted Consensus)|
                              +---------------+---------------+
                                              |
                                              v
                              +-------------------------------+
                              | Structured Advisory Decision  |
                              +-------------------------------+
```

---

## 2. API Endpoints

### 1. Market Analysis & Multi-Agent Deliberation
`GET /v1/trading/market/analysis?symbol=BTCUSDT`

#### Response
```json
{
  "symbol": "BTCUSDT",
  "current_price": 65240.0,
  "overall_consensus": "BULLISH_CONVERGENCE",
  "overall_confidence": 0.84,
  "synthesis_summary": "Enhanced Multi-Agent Debate Consensus (BULLISH_CONVERGENCE | Confidence: 0.84)...",
  "specialist_deliberations": [
    {
      "specialist": "Technical Analyst",
      "findings": "Regime: TRENDING_BULL. RSI: 62.4, MACD Histogram: 14.2. Patterns: 1 detected.",
      "bias": "BULLISH",
      "confidence": 0.85
    },
    {
      "specialist": "Sentiment Analyst",
      "findings": "Overall Sentiment: BULLISH (Score: 0.42). Extracted events: ['ETF Inflow Acceleration'].",
      "bias": "BULLISH",
      "confidence": 0.82
    },
    {
      "specialist": "On-Chain Analyst",
      "findings": "Exchange Flows: NET_OUTFLOW_ACCUMULATION. Whale Outflow: $165,000,000.",
      "bias": "BULLISH",
      "confidence": 0.88
    },
    {
      "specialist": "Quantitative ML Modeler",
      "findings": "Ensemble Direction: BULLISH_CONTINUATION. 24H Target: $66,700.00 (+2.24%).",
      "bias": "BULLISH",
      "confidence": 0.81
    }
  ],
  "technical_indicators": { ... },
  "sentiment_analysis": { ... },
  "onchain_analytics": { ... },
  "price_predictions": { ... }
}
```

---

### 2. NLP Sentiment Analysis
`GET /v1/trading/market/sentiment?symbol=BTC`

---

### 3. On-Chain Metrics & Whale Activity
`GET /v1/trading/market/onchain?symbol=BTC`

---

### 4. ML Price Trajectory Prediction
`POST /v1/trading/predict`

#### Request Payload
```json
{
  "symbol": "BTCUSDT",
  "current_price": 65240.0
}
```

#### Response
```json
{
  "symbol": "BTCUSDT",
  "prediction": {
    "current_price": 65240.0,
    "forecast_direction": "BULLISH_CONTINUATION",
    "overall_confidence": 0.82,
    "horizons": {
      "1h": {"predicted_price": 65420.0, "change_pct": 0.28, "confidence_interval": {"lower": 64800.0, "upper": 66040.0}},
      "4h": {"predicted_price": 65860.0, "change_pct": 0.95, "confidence_interval": {"lower": 64400.0, "upper": 67320.0}},
      "24h": {"predicted_price": 66700.0, "change_pct": 2.24, "confidence_interval": {"lower": 63800.0, "upper": 69600.0}}
    },
    "feature_attributions": {
      "technical_momentum_pct": 50.0,
      "sentiment_nlp_pct": 30.0,
      "onchain_flow_pct": 20.0
    }
  }
}
```

---

### 5. Real-Time Market Alerts & Anomaly Monitor
`GET /v1/trading/monitor/alerts?symbol=BTCUSDT`
