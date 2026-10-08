# Inference — Production API & Performance Deployment Reference

This guide details the production architecture, performance optimizations, monitoring setup, and deployment standards for the **Inference Advisory Consultation System**.

---

## 1. Production Architecture & Performance Optimizations

### Caching Layer (`app/optimization.py`)
- **Telemetry Pattern Hashing**: Normalizes continuous performance parameters (win rate, profit factor, drawdown bucket, trade count) into discrete risk buckets.
- **LRU In-Memory Cache**: 24-hour TTL with instant retrieval ($< 1$ms) on recurring identical telemetry envelopes.
- **Cache Invalidation**: Automatic eviction when reaching capacity (5,000 entries) or expiring past TTL.

### Provider Gateway & Circuit Breaker
- **Tripping Threshold**: 3 consecutive failed provider calls.
- **Recovery Time**: 60 seconds half-open probe before full restoration.
- **Failover Chain**: `Groq` $\rightarrow$ `Gemini` $\rightarrow$ `OpenAI` $\rightarrow$ `Anthropic` $\rightarrow$ `Ollama` $\rightarrow$ `Deterministic Mathematical Engine`.

### Concurrency & Compression
- **Concurrency Throttling**: Semaphore bounding active consultations to 100 simultaneous requests.
- **GZip Response Compression**: Enabled globally for responses $\ge 500$ bytes.

---

## 2. Production Health & Prometheus Monitoring

### Health Endpoints (`app/health.py`)
- `GET /health`: Basic operational liveness check.
- `GET /health/detailed`: Live system telemetry, p50/p95/p99 latency percentiles, cache hit rate, and active concurrency.
- `GET /health/providers`: Per-provider success rates and average response times.
- `GET /status`: Active capabilities and specialist agent rosters.
- `GET /metrics`: Standard Prometheus metrics exporter.

### Key Prometheus Metrics
`GET /metrics` (public, dependency-free) and `GET /v1/governance/prometheus-metrics`
(authenticated) both render through `app/observability/prometheus.py`, so these names, their
values and the handling of absent data are identical on both. A series with no measurement is
omitted rather than printed as a placeholder such as `None`, which no scraper accepts.

| Metric Name | Type | Description |
| :--- | :--- | :--- |
| `inference_requests_total` | Counter | API requests recorded by this process |
| `inference_requests_failed_total` | Counter | API requests that failed or returned 5xx |
| `inference_latency_samples` | Gauge | Latency samples behind the percentiles below |
| `inference_request_duration_seconds{quantile="0.50"}` | Gauge | Median response latency |
| `inference_request_duration_seconds{quantile="0.95"}` | Gauge | 95th percentile response latency ($< 30$s SLA) |
| `inference_request_duration_seconds{quantile="0.99"}` | Gauge | 99th percentile response latency |
| `inference_error_rate_percent` | Gauge | Percentage of recorded requests that failed |
| `inference_cache_hit_rate_percent` | Gauge | Cache hit percentage |
| `inference_active_requests` | Gauge | In-flight requests |
| `inference_uptime_seconds` | Gauge | Process uptime |

The governance endpoint additionally publishes `inference_tracked_tenants`,
`inference_provider_circuit_available`, `inference_provider_circuit_state`,
`inference_provider_success_ratio`, `inference_provider_health_score` and
`inference_provider_rate_limited_keys`. Those stay on the authenticated endpoint on purpose:
which providers exist and which of them are failing is deployment detail, and `/metrics` is
reachable without a key.

---

## 3. SLA & Performance Benchmarks

Validated via `tests/load_test.py` across 100 concurrent multi-scenario requests:
- **P50 Latency**: $< 0.05$s (Cached) / $\sim 1.2$s (Debate)
- **P95 Latency**: Strictly **$< 30.0$s**
- **Success Rate**: **100.0%**
- **Error Rate**: **0.0%**
- **Quality Score**: **100/100** on recommendation coherence, bounds, and gating.

---

## 4. Production Deployment Pre-Flight Checklist

Run the automated pre-flight audit:
```bash
python deploy_production.py
```
This script verifies:
1. All health endpoints (`/health`, `/health/detailed`, `/health/providers`, `/status`, `/metrics`).
2. Trading advisory invariant adherence (`advisory_only: true`, `exchange_execution: false`).
3. A/B testing and testnet comparative analytics.
4. Concurrency controller and memory initialization.
