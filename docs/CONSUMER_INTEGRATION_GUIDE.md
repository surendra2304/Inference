# Inference Multi-Consumer Integration Guide

Inference serves seven specialized autonomous consumers with dedicated rate limits, priority queues, and monthly budget allocations.

---

## 1. Consumer Registry & Governance Matrix

| Consumer | Role & Workload Focus | Rate Limit | Priority Queue Policy | Monthly Budget | Soft Limit (80%) | Hard Limit (100%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Trading Bot** | Algorithmic trading parameters advisory | $20\text{ reqs/hr}$ | **Ultra-High Priority** *(Isolated, non-blocking queue)* | $\$50.00$ | $\$40.00$ *(Warning)* | $\$50.00$ *(Rejected)* |
| **FORGE** | Autonomous software code generation & planning | $200\text{ reqs/hr}$ | **Burst-Tolerant Queue** *(Heavy parallel batches)* | $\$200.00$ | $\$160.00$ *(Warning)* | $\$200.00$ *(Rejected)* |
| **Nexus** | High-throughput multi-mode decision engine | $200\text{ reqs/hr}$ | **Standard Interactive Queue** | $\$100.00$ | $\$80.00$ *(Warning)* | $\$100.00$ *(Rejected)* |
| **Sentinel** | Cybersecurity posture & threat reasoning | $100\text{ reqs/hr}$ | **Dedicated Security Queue** | $\$100.00$ | $\$80.00$ *(Warning)* | $\$100.00$ *(Rejected)* |
| **IntelX** | Deep research & claim verification | $200\text{ reqs/hr}$ | **Batched Research Queue** | $\$150.00$ | $\$120.00$ *(Warning)* | $\$150.00$ *(Rejected)* |
| **Futuris** | Predictive forecasting & statistical grounding | $150\text{ reqs/hr}$ | **Analytical Priority Queue** | $\$100.00$ | $\$80.00$ *(Warning)* | $\$100.00$ *(Rejected)* |
| **FRIDAY** | General assistant & workspace automation | $100\text{ reqs/hr}$ | **Interactive User Queue** | $\$50.00$ | $\$40.00$ *(Warning)* | $\$50.00$ *(Rejected)* |

---

## 2. Authentication & Header Standards

All intelligence requests require an API key passed either via the `Authorization: Bearer <KEY>` header or `X-API-Key: <KEY>`.

```http
POST /v1/nexus/intelligence HTTP/1.1
Host: localhost:8000
Authorization: Bearer key_nexus_prod_01
Content-Type: application/json
```

---

## 3. Dedicated Consumer Integration Examples

### A. Trading Bot Integration
```python
import os
from sdk.python.inference_client import AIUniverseClient
client = AIUniverseClient(base_url="http://localhost:8000", api_key=os.environ["INFERENCE_API_KEY"])

# Query bounded advisory
advisory = client.client.post("/v1/trading/consult", json={
    "symbol": "BTC/USDT",
    "current_regime": "high_volatility",
    "drawdown_pct": 2.1
}).json()
```

### B. FORGE Code Generation & Review
```python
# Generate syntax-checked code
code_res = client.generate_code(
    file_type="python",
    filename="app/middleware/auth.py",
    context={"goal": "Bearer token authentication with timing attack protection"}
)
```

### C. Nexus Mode-Based Intelligence
```python
# Run 4-round adversarial debate
debate_res = client.client.post("/v1/nexus/intelligence", json={
    "request_id": "nex-str-01",
    "task_type": "strategic_decision",
    "goal": "Evaluate Q4 cloud migration timeline",
    "mode": "debate"
}).json()
```

### D. Sentinel Security Posture & Remediation
```python
# Run attack path reasoning and remediation prioritization
sentinel_res = client.query_sentinel_analysis(
    request_id="sent-01",
    analysis_type="attack_path_reasoning",
    target_context={"asset_type": "api_gateway", "exposure_level": "public_internet"},
    findings=[{"finding_id": "F-01", "severity": "CRITICAL", "title": "Unauthenticated Parameter Exposure"}]
)
```

### E. IntelX Research & Verification
```python
# Fact Checker + Critic claim verification
research_res = client.query_intelx_research(
    request_id="intelx-01",
    role="verifier",
    context={"question": "Verify TPS benchmarks across rollup frameworks"},
    evidence_with_spans=[{
        "claim": "Rollup A achieves 10k TPS",
        "verbatim_span": "demonstrated sustained 10,000 TPS under load",
        "document_source": "https://wire.news/report",
        "credibility_score": 0.90
    }]
)
```

### F. Universal Outcome Learning
```python
# Ingest downstream outcomes to improve multi-model routing weights
client.report_outcome(
    consumer="intelx",
    request_id="intelx-01",
    outcome="success",
    metrics={"verification_accuracy_boost": 0.23}
)
```

---

## 4. Cost Optimization & Semantic Caching

- **Context Compression**: Token reduction averaging $40-65\%$ before sending prompts to LLM providers.
- **Domain-Specific Cache TTLs**:
  - Trading Advisory: $5\text{ minutes}$ (Fast-shifting financial state)
  - Security Telemetry: $15\text{ minutes}$ (Scan state)
  - Decision Intelligence: $30\text{ minutes}$
  - Architecture Manifests: $24\text{ hours}$
