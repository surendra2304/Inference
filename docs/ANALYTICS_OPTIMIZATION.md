# Provider Analytics & Self-Optimizing Intelligence Architecture

Inference implements comprehensive usage tracking, cost management, downstream outcome attribution, and self-optimizing provider rebalancing.

---

## 1. Multi-Consumer Usage & Cost Analytics

- **Per-Request Telemetry**: Captures consumer identity, service name, providers utilized, token consumption, latency, and proxy cost attribution.
- **Budget Ceilings**: Enforces daily expenditure caps ($10.00 default) and automatically triggers proactive alerts when usage crosses the 80% ceiling.

---

## 2. Downstream Outcome Feedback & Dynamic Weight Adaptation

```
┌─────────────────────────────────┐
│     Inference Generation      │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│ Downstream Consumer Execution   │
│ (FORGE Builds / Trading Bot)    │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  POST /v1/analytics/outcome     │
│  • verification_passed (FORGE)  │
│  • drawdown_mitigated (Bot)     │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│     Self-Optimizing Router      │
│  • Rebalances provider weights  │
│  • Explores 10% underused pool  │
│  • Min 2 active providers bound │
└─────────────────────────────────┘
```

---

## 3. Endpoints Reference

### 3.1 Analytics & Telemetry
- `GET /v1/analytics/overview`: High-level call count, token usage, latency, and cost totals.
- `GET /v1/analytics/consumer/{id}`: Granular stats filtered by consumer (`forge`, `trading_bot`, `friday`, `human`).
- `GET /v1/analytics/service/{name}`: Performance breakdown per service.
- `GET /v1/analytics/providers`: Comparison matrix across all 7 cloud providers.
- `GET /v1/analytics/provider-intel`: Failure analysis and strategic routing recommendations.
- `GET /v1/analytics/quality`: Syntactic AST validation and confidence calibration reports.
- `POST /v1/analytics/outcome`: Downstream outcome reporting from FORGE and Trading Bot.

### 3.2 Admin Operations
- `GET /v1/admin/dashboard`: Unified executive dashboard spanning usage, provider comparison, predictive forecasts, outcomes, and active alerts.
- `GET /v1/admin/optimization/status`: Current dynamic routing weights and audit logs.
- `GET /v1/admin/alerts`: Active operational alerts.
