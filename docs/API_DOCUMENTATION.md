# Inference Production API Documentation & Integration Reference

Version: `v1.0.0-PROD`  
Protocol: `HTTPS / JSON`  
Authentication: `Bearer <API_KEY>`

---

## 1. Authentication & Security

All private production endpoints accept an `Authorization` header:
```http
Authorization: Bearer aiu_live_sec_9948271049281726
```

### Security Headers Enforced:
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`
- `Strict-Transport-Security: max-age=31536000; includeSubDomains`

---

## 2. API Endpoints Reference

### 2.1 Multi-Agent Trading Consultation
- **Endpoint**: `POST /v1/trading/consult`
- **Purpose**: Evaluates bot performance telemetry, triggers multi-agent debate, and returns structured parameter recommendations.

### 2.2 Advanced Market Intelligence
- `GET /v1/trading/market/analysis?symbol=BTCUSDT`: 50+ Technical indicators, multi-agent consensus, and regime.
- `GET /v1/trading/market/sentiment?symbol=BTC`: NLP news/social sentiment and entity extraction.
- `GET /v1/trading/market/onchain?symbol=BTC`: Blockchain network health, whale transfers, and exchange flows.
- `POST /v1/trading/predict`: Multi-horizon (1h, 4h, 24h) ML price predictions with confidence intervals.
- `GET /v1/trading/monitor/alerts?symbol=BTCUSDT`: Real-time market volatility and orderbook imbalance alerts.

### 2.3 Production Health & Prometheus Metrics
- `GET /health`: Liveness probe (`{"status": "healthy"}`).
- `GET /health/detailed`: Latency percentiles (p50/p95/p99), active connections, and cache hit rates.
- `GET /metrics`: Standard Prometheus metrics exporter.

---

## 3. High Availability & Failover Policy

Inference operates an automated failover chain:
`Groq` $\rightarrow$ `Gemini` $\rightarrow$ `OpenAI` $\rightarrow$ `Anthropic` $\rightarrow$ `Ollama`.

If a provider incurs 3 consecutive failures, the node transitions to `DEGRADED` status and requests automatically route to the next available provider in the chain.
