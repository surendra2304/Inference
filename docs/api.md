# Inference REST API Reference

Inference exposes clean, typed FastAPI REST endpoints for orchestration, structured multi-agent debates, evaluations, benchmarks, and dedicated FRIDAY peer integration.

---

## Endpoints

### 1. Ask a Question
`POST /ask`

Auto-routes the question to Fast, Review, or Debate mode based on semantic triggers, budget, and latency guardrails.

**Request Body:**
```json
{
  "question": "How do I implement an asynchronous token bucket in Python?",
  "mode": "auto",
  "max_agents": 5,
  "require_evidence": true,
  "max_budget": 0.05,
  "max_latency": 15.0
}
```

---

### 2. Trigger Structured Debate
`POST /debate`

Explicitly triggers the 6-Round Structured Multi-Agent Debate Engine.

**Request Body:**
```json
{
  "question": "Compare microservices vs modular monolith for low-latency fintech platforms.",
  "max_agents": 5,
  "require_evidence": true
}
```

---

### 3. Trigger Benchmark / Comparison Experiment
`POST /experiments`

**Request Body:**
```json
{
  "experiment_type": "baseline_vs_debate",
  "question": "What is the optimal caching architecture for read-heavy workloads?"
}
```

---

### 4. FRIDAY Integration Endpoints

All FRIDAY endpoints require authentication via the `X-FRIDAY-API-Key` HTTP header.

#### `POST /v1/friday/ask`
Fast consultation endpoint for FRIDAY sub-modules with provenance lineage.

#### `POST /v1/friday/debate`
6-Round Multi-Agent Adversarial Debate consultation for high-stakes decisions.

**Headers:**
```http
X-FRIDAY-API-Key: your_friday_api_key_here
Content-Type: application/json
```

**Request Body:**
```json
{
  "question": "Assess memory isolation risks for FRIDAY subagents running untrusted plugins.",
  "caller_id": "friday_security_core",
  "max_latency": 30.0,
  "require_evidence": true
}
```

**Response (200 OK):**
```json
{
  "task_id": "task_9876543210ab",
  "run_id": "deb_1234567890cd",
  "answer": "FRIDAY debate consensus: Enforce cryptographic capabilities across process boundaries...",
  "mode_used": "debate",
  "confidence": 0.88,
  "unresolved_disagreements": [
    "Overhead of full VM sandboxing vs WASM lightweight process isolation."
  ],
  "key_evidence": [
    "Capabilities prevent privilege escalation even during memory corruption."
  ],
  "agents_used": ["security_analyst", "architect", "critic"],
  "models_used": ["gemini-2.5-pro", "gemini-2.5-pro", "gemini-2.5-pro"],
  "latency_seconds": 3.12,
  "total_tokens": 940,
  "provenance": {
    "caller_id": "friday_security_core",
    "debate_id": "deb_1234567890cd",
    "platform": "Inference",
    "version": "1.0.0",
    "rounds_completed": 6
  }
}
```

---

### 5. Utility Endpoints
- `GET /tasks/{task_id}`: Returns audit details, execution status, and final answer.
- `GET /experiments/{experiment_id}`: Returns experiment metrics and configuration.
- `GET /health`: Returns service health status (`{"status": "healthy"}`).
