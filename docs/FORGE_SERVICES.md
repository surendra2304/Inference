# FORGE Supporting Intelligence Services

Inference acts as the high-throughput reasoning and code generation backbone for **FORGE** (autonomous software engineering engine), while maintaining strict resource isolation for the Algorithmic Trading Bot and FRIDAY.

---

## 1. FORGE Intelligence Services Reference

### 1.1 Per-File Code Generation
`POST /v1/forge/generate-code`
- **Request**:
```json
{
  "file_type": "python",
  "filename": "app/services/calculator.py",
  "context": {
    "project_goal": "Financial telemetry parser",
    "architecture_spec": "Modular service pattern"
  },
  "requirements": ["Compute Sharpe and Sortino ratios", "Strict typing"],
  "language_features": ["PEP 8", "dataclasses"]
}
```
- **Response**:
```json
{
  "code": "import math\nfrom typing import List...",
  "confidence": 0.92,
  "generation_path": "agent",
  "token_usage": 450,
  "latency_ms": 34.2,
  "filename": "app/services/calculator.py"
}
```

---

### 1.2 Architecture Planning & File Manifest Generation
`POST /v1/forge/plan-architecture`
- Deconstructs goals into file trees, inter-file dependencies, and tech stack configurations.

---

### 1.3 Multi-Agent Code Review Debate
`POST /v1/forge/review-code`
- Evaluates code across **Coder**, **Security Analyst**, and **Critic** specialists.
- Automatically flags `fix_required` for OWASP vulnerabilities or hardcoded secrets.

---

### 1.4 Automated Debugging & Patch Formulation
`POST /v1/forge/debug`
- Cross-references stack traces against code context to synthesize targeted surgical patches.

---

### 1.5 Automated Test Generation
`POST /v1/forge/generate-tests`
- Generates exhaustive unit test suites in `pytest`, `jest`, or `playwright`.

---

### 1.6 Batch Parallel Code Generation
`POST /v1/forge/batch-generate`
- Executes up to 10 file generation requests in parallel with partial failure resilience.

---

### 1.7 Health & Provider Telemetry
- `GET /v1/forge/health`: Health status and provider availability poll.
- `GET /v1/forge/capabilities`: Active services and throughput capacity.
- `GET /v1/admin/usage`: Usage metrics per consumer (`forge`, `trading_bot`, `friday`, `human`).
- `GET /v1/admin/providers/performance`: Provider latency and demotion telemetry.
