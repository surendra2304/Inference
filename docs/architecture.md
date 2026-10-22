# Architecture of Inference

Inference is a local-first, multi-agent intelligence platform designed for deep reasoning, structured adversarial debate, persistent empirical memory, and provider-agnostic LLM orchestration.

---

## 1. Core Architectural Tenets

1. **Independent Intelligence**: Inference is an autonomous intelligence engine. FRIDAY is an autonomous agent operating at the OS and application layer. They communicate over strict, typed API boundaries.
2. **FRIDAY is a Peer Client, Not a Wrapper**: Inference is completely standalone. FRIDAY interfaces with Inference via secure, authenticated HTTP endpoints (`/v1/friday/*`), receiving full deliberation provenance, surviving claims, and dissenting views.
3. **Confidence is Not Correctness**: Authoritative prose is separated from empirical validity. LLM confidence scores are calibrated, and meaningful dissent is preserved rather than smoothed over.
4. **Graceful Cost & Latency Degradation**: Trivial questions are routed to fast single-agent execution, while complex trade-offs trigger 6-Round adversarial debates.
5. **Zero Secrets in Code or Memory**: All keys are strictly loaded via environment variables and sanitized in logs and database stores.

---

## 2. System Layering

```
 ┌──────────────────────────────────────────────────────────┐
 │                     FastAPI & CLI Layer                  │
 │   /ask   │   /debate   │   /experiments   │   /health    │
 │            /v1/friday/ask   │   /v1/friday/debate        │
 └────────────────────────────┬─────────────────────────────┘
                              │
 ┌────────────────────────────▼─────────────────────────────┐
 │                     Task Router Layer                    │
 │  - Domain Specialist Allocation (10 Specialist Roles)    │
 │  - Budget & Latency Guardrails (Degradation Engine)      │
 │  - Strategy Store Learned Recommendations                │
 └────────────────────────────┬─────────────────────────────┘
                              │
 ┌────────────────────────────▼─────────────────────────────┐
 │                    Orchestrator Core                     │
 │  - Fast Mode (Single Specialist)                         │
 │  - Review Mode (Specialist Pair)                         │
 │  - Debate Engine (6-Round Structured Adversarial Panel)   │
 └──────────────────────┬──────────────────────┬────────────┘
                        │                      │
 ┌──────────────────────▼───────┐  ┌───────────▼────────────┐
 │       Provider Gateway       │  │   Memory Subsystem     │
 │  - Google Gemini Adapter     │  │  - SQLite (aiosqlite)  │
 │  - Groq Adapter              │  │  - Agents & Tasks      │
 │  - NVIDIA NIM Adapter        │  │  - Runs & Messages     │
 │  - Mistral Adapter           │  │  - Memories (Scoped)   │
 │  - OpenRouter Adapter        │  │  - Strategies & Exp.   │
 └──────────────────────────────┘  └────────────────────────┘
```

---

## 3. The 6-Round Structured Debate Protocol

- **Round 0: Problem Framing**: Primary strategist produces a canonical problem statement with declared assumptions.
- **Round 1: Parallel Independent Analysis**: Specialists execute concurrently via `asyncio.gather` without seeing peer responses.
- **Round 2: Cross-Review & Adversarial Critique**: The `Critic` attacks weak assumptions, unstated trade-offs, and failure modes.
- **Round 3: Specialist Rebuttal**: Proposal authors respond, defend validated decisions, and concede valid critiques.
- **Round 4: Evidence & Fact Checking**: The `Fact Checker` separates claims from verifiable evidence.
- **Round 5: Consensus Synthesis**: The `Synthesizer` builds a final recommendation integrating surviving claims.
- **Round 6: Confidence Calibration**: Unresolved disagreements are permanently recorded and confidence is calibrated.

---

## 4. FRIDAY Integration Boundary

Inference provides dedicated, authenticated integration endpoints for FRIDAY:
- **Authentication**: Validated via `X-FRIDAY-API-Key` header with constant-time comparison.
- **Provenance & Dissent**: Returns explicit `provenance` lineage, `key_evidence`, and `unresolved_disagreements`, allowing FRIDAY to autonomously decide whether to act upon or escalate recommendations.
