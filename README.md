---
title: Inference
emoji: 🌌
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
---

# Inference

> **Independent Multi-Agent Intelligence System**  
> Master Architecture, Multi-Model Reasoning, Debate Engine, and FRIDAY Integration.

---

## 🌟 Overview

**Inference** is a local-first, provider-agnostic multi-agent intelligence platform. Its purpose is to make multiple AI systems work together so that you can ask one complex question and receive a stronger, critically evaluated, and empirically verified answer than any single model would provide alone.

### Core Principles
- **Inference is not FRIDAY, and FRIDAY is not a wrapper around Inference.** They are two independent intelligent systems. Later, Inference can exchange specialized reasoning, analysis, and debate outcomes with FRIDAY through a secure API boundary.
- **Multi-Model / Provider Agnostic**: Connects to **Gemini, Groq, Mistral, OpenRouter, NVIDIA, Cohere, and HuggingFace** through a unified provider gateway.
- **Structured Debate Protocol**: Uses multi-agent adversarial critique, rebuttal, evidence checking, and uncertainty-aware synthesis.
- **Persistent Memory & Learning**: Persistent SQLite memory with strict agent scoping, evaluation rubrics, and routing strategy optimization.
- **Local-First, Cloud Inference**: Orchestration, storage, and UI run locally on your laptop, while model inference is dispatched to fast cloud provider APIs without GPU strain.

---

## 🏛️ System Architecture

```
                      USER / CLIENT
                            |
                            v
                 +----------------------+
                 |     FastAPI / UI     |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 |  ORCHESTRATOR CORE   |
                 +----------+-----------+
                            |
         +------------------+------------------+
         |                  |                  |
         v                  v                  v
    Task Router       Agent Registry        Memory
         |                  |                  |
         +------------------+------------------+
                            |
                            v
                 +----------------------+
                 |    Debate Engine     |
                 | (10 Specialist Roles)|
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 | Evaluator / FactCheck|
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 |     Synthesizer      |
                 | (Audit Trail + Conf) |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 |  Learning & Strategy |
                 +----------------------+
```

---

## 🤖 Initial 10-Specialist Agent Panel

| Agent Role | Primary Responsibility | Typical Use Case |
|---|---|---|
| **Researcher** | Find & organize relevant information | Knowledge synthesis, comparisons |
| **Architect** | Design systems and structure | Software design, technical planning |
| **Coder** | Propose implementation approaches | Code generation, refactoring |
| **Debugger** | Trace failures & inconsistencies | Root-cause analysis, error diagnosis |
| **Security Analyst** | Identify vulnerabilities & risks | Threat modeling, safe design |
| **Data Analyst** | Reason from structured data | Metrics, quantitative analysis |
| **Critic** | Attack weak assumptions & fallacies | Red-team reasoning, counterexamples |
| **Fact Checker** | Separate claims from evidence | Claim verification, contradiction detection |
| **Strategist** | Compare alternatives & tradeoffs | Prioritization, decision support |
| **Synthesizer** | Produce final coherent answer | Integrates claims, reports uncertainty |
| **Trading Analyst** | Analyze quantitative trading metrics & drawdown | Strategy parameter advice (SL/TP, leverage) |

### 📈 Algorithmic Trading Bot Integration (Advisory Authority)
- **Specialist Agent (`TradingAnalyst`)**: Evaluates performance telemetry (win rate, profit factor, drawdown, consecutive loss streaks) sent by FRIDAY.
- **Debate & Consultation**: Engages in multi-agent deliberation with the **Strategist** and **Critic** to formulate calibrated recommendations (e.g. tightening stop loss, adjusting position sizes).
- **Strict Safety Boundary**: Inference **NEVER** calls exchange APIs or executes trades directly; it returns structured `AIUniverseDecision` recommendations for FRIDAY to record and present to the user.

---

## 📖 Project Diary & Memory Standard

Development in this repository strictly adheres to the project memory and diary policy:
- **Master Index**: [inference_DIARY.md](inference_DIARY.md)
- **Specification Standard**: [inference_DIARY_SPEC.md](inference_DIARY_SPEC.md)
- **Daily Logs**: Located in [`diary/`](diary/) (e.g. [`diary/2026-08-24.md`](diary/2026-08-24.md))

---

## 🚀 Getting Started

### 1. Clone & Setup
```bash
git clone https://github.com/surendra2304/Inference.git
cd Inference
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Configure Environment Variables
```bash
cp .env.example .env
# Edit .env and supply your API keys (GEMINI_API_KEY, GROQ_API_KEY, etc.)
```

To inspect configured provider-key counts locally without printing credential
values, run `python -m app.core.key_inventory` from this repository. The report
counts deduplicated values recognized by the current settings model. It does not
verify that credentials work, belong to separate provider accounts, or have
available quota; those require separate provider checks. Do not hardcode the
count in ecosystem manifests because the local and deployed environment pools
can change independently.

### 3. Run the Service
```bash
uvicorn app.main:app --reload --port 8000
```

### 4. Or Run It With Docker Compose
```bash
# Optional but recommended: supply provider keys
cp .env.example .env

docker compose -f docker-compose.production.yml up --build
# API on http://localhost:8000  (health: /health, dashboard: /ui)
```

The compose file provisions a single API container with a persistent volume at
`/app/data` for the SQLite store. The `.env` file is optional at start-up: the
container boots and serves `/health` without it, but model calls fail closed
with `503` until provider keys are configured.

---

## 📚 Documentation

- [Architecture & Design](docs/architecture.md)
- [Specialist Agents](docs/agents.md)
- [REST API Specifications](docs/api.md)
- [Trading Consultation API](docs/TRADING_CONSULT_API.md)
- [Memory & Persistence](docs/memory.md)
- [Experiments & Benchmarks](docs/experiments.md)
