# API Governance, Multi-Tenancy & Production Operations

Inference provides enterprise-grade API governance, strict multi-tenant isolation, automated key rotation, circuit breaking, request deduplication, Prometheus observability, and graceful degradation.

---

## 1. Multi-Tenant Isolation & Key Management

- **Tenant Scoping**: All requests, persistent memories, outcomes, and audit logs are partitioned by `tenant_id`.
- **Policy Enforcement**:
  - `tenant_forge`: $1000\text{ reqs/hr}$, $\$50.00$ daily budget limit.
  - `tenant_nexus`: $500\text{ reqs/hr}$, $\$25.00$ daily budget limit.
  - `tenant_trading`: $100\text{ reqs/hr}$, $\$15.00$ daily budget limit.
  - `tenant_default`: $200\text{ reqs/hr}$, $\$10.00$ daily budget limit.
- **Hard Budget Cutoffs**: Rejects outbound provider requests gracefully when a tenant crosses its daily budget ceiling.
- **Key Rotation**: `POST /v1/governance/tenants/{tenant_id}/rotate-key` securely rotates API keys with zero downtime.

---

## 2. Request Deduplication & Idempotency

- In-memory 5-minute idempotency window keyed by `request_id`.
- Re-submitting the same `request_id` within 5 minutes returns the cached response instantly without consuming upstream provider tokens.

---

## 3. Circuit Breaker & Graceful Degradation

- **Trip Threshold**: 5 consecutive errors on any provider transitions its state from `CLOSED` to `OPEN` for 60 seconds.
- **Half-Open Cooldown**: Probes one request after 60s cooldown; automatically closes circuit on success.
- **Degradation Cascade**:
  - If external AI providers fail $\rightarrow$ Returns deterministic template fallbacks.
  - If memory fails $\rightarrow$ Bypasses StrategyBank context without interrupting reasoning.
  - If analytics fails $\rightarrow$ Enqueues telemetry for asynchronous retry.

---

## 4. Prometheus Observability & Governance APIs

- `GET /v1/governance/tenants/{tenant_id}`: Tenant policy and spend status.
- `POST /v1/governance/tenants/{tenant_id}/rotate-key`: Key rotation.
- `GET /v1/governance/circuits`: Live circuit breaker state across all 7 cloud providers.
- `GET /v1/governance/prometheus-metrics`: Formatted Prometheus metrics, rendered by the same `app/observability/prometheus.py` as the public `/metrics` endpoint (`inference_requests_total`, `inference_requests_failed_total`, `inference_request_duration_seconds{quantile=...}`, `inference_provider_circuit_state`, `inference_provider_success_ratio`, `inference_provider_health_score`). The provider-level families exist only here, because this route is authenticated and `/metrics` is not.
