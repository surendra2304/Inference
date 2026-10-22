# Deep Learning Prediction & Alternative Data Architecture

Inference integrates high-performance sequence modeling (LSTM, Transformers), GARCH-LSTM volatility forecasts, and multi-source alternative data (news NLP, Reddit/X social attention spikes, on-chain whale tracking, and macro variables).

> [!IMPORTANT]
> **Advisory Invariant**: Deep learning forecasts and alternative data serve strictly as advisory inputs to trading consultations and risk gates. Inference never executes trades directly.

---

## 1. Multi-Source Prediction Pipeline

```
┌─────────────────────────────────┐   ┌─────────────────────────────────┐
│     Sequential Price History    │   │        Alternative Data         │
└────────────────┬────────────────┘   └────────────────┬────────────────┘
                 │                                     │
         ┌───────┴───────┐                     ┌───────┴───────┐
         ▼               ▼                     ▼               ▼
┌────────────────┐┌──────────────┐   ┌────────────────┐┌──────────────┐
│ LSTM / GRU     ││ Transformer  │   │ NLP News Feed  ││ On-Chain Net │
│ 1h/4h Sequence ││ 24h Pattern  │   │ Sentiment NLP  ││ Whale Flows  │
└────────┬───────┘└──────┬───────┘   └────────┬───────┘└──────┬───────┘
         │               │                    │               │
         └───────┬───────┴────────────────────┴───────┬───────┘
                 │                                    │
                 ▼                                    ▼
       ┌────────────────────────────────────────────────────────┐
       │             Prediction Aggregation Engine              │
       │       (Weighted Ensemble & Conflict Detection)         │
       └──────────────────────────┬─────────────────────────────┘
                                  │
                                  ▼
       ┌────────────────────────────────────────────────────────┐
       │   Unified Directional Forecast & Accuracy Calibration  │
       └────────────────────────────────────────────────────────┘
```

---

## 2. Model Latency & Architecture Specifications

- **LSTM / GRU Models (`v2.4.1`)**: Optimized for short-horizon direction ($1\text{h}$ and $4\text{h}$) with inference latency $< 20\text{ms}$.
- **Transformer Sequence Models (`v1.8.0`)**: Multi-day pattern recognition ($24\text{h}$ horizon).
- **GARCH-LSTM Volatility Predictor**: Generates $24\text{h}$ realized volatility forecasts and detects volatility regime expansion.

---

## 3. Endpoints Reference

### 3.1 Unified Asset Prediction
`GET /v1/predict/{asset}`
- Returns unified ensemble direction (`BULLISH` / `BEARISH` / `NEUTRAL`), confidence level, horizon breakdown, and key alpha drivers.

### 3.2 Prediction History & Out-of-Sample Accuracy
`GET /v1/predict/{asset}/history`
- Returns historical prediction log and empirical directional accuracy metrics.

### 3.3 Consolidated Alternative Data Intelligence
`GET /v1/intelligence/summary?asset=BTC`
- Ingests news sentiment scores, social volume spike flags, on-chain netflows, and macro regime indicators.

### 3.4 Multi-Source Accuracy Calibration Report
`GET /v1/intelligence/accuracy`
- Evaluates individual accuracy contributions across LSTM models, news NLP, whale tracking, and technical momentum.

### 3.5 Model Inference Refresh
`POST /v1/predict/refresh`
- Clears cached inference buffers and runs hot model synchronization.
