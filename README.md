# 🌌 Inference — Centralized LLM Gateway & Deliberation Engine

[![CI](https://github.com/surendra2304/Inference/actions/workflows/ci.yml/badge.svg)](https://github.com/surendra2304/Inference/actions/workflows/ci.yml)
[![Docker](https://img.shields.io/badge/docker-python%3A3.11--slim-blue)](Dockerfile)
[![Mesh](https://img.shields.io/badge/network-friday__mesh-purple)](docker-compose.production.yml)

Inference is the centralized LLM gateway, multi-provider fallback router, and high-speed token streaming engine of the **FRIDAY Universe**. It aggregates and load-balances API keys across leading model providers—**Groq, Cerebras, OpenRouter, and Google Gemini**—providing automated cascading fallback, calibrated multi-agent adversarial debate, and OpenAI-compatible endpoints with sub-second response times.

---

## 🏛️ Topology & Network Architecture

Inference is deployed within the unified private container mesh (`friday_mesh`) on Oracle Cloud Always Free and fronted by an automated HTTPS Caddy reverse proxy.

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Identity** | `inference` | Specialist Gateway Agent |
| **Internal Container Port** | `8000` | Application listen port |
| **Host Port** | `8004` | Published host port on host |
| **Internal Docker Mesh URL**| `http://inference:8000` | Mesh discovery address for universe agents (`friday`, `memora`, etc.) |
| **Public Gateway Route** | `https://friday-universe.duckdns.org/inference/*` | External public ingress routed via Caddy |
| **Gateway Host** | Oracle Cloud Always Free | 2 OCPUs, 12 GB RAM Ampere A1 (ARM64 / x86_64) |
| **Runtime Base** | `python:3.11-slim` | Non-root `appuser` (UID 1000) with curl healthchecks |

```
                       FRIDAY UNIVERSE CONTAINER MESH (friday_mesh)
                   
   [Public Client] ── HTTPS ──> Caddy (Reverse Proxy :443)
                                      |
                                      +── /inference/* ──> [Inference :8000] (Host: 8004)
                                      |                          |
                                      |                          +── Primary: Groq (LPU)
                                      |                          +── Primary: Cerebras (Ultra-Fast)
                                      |                          +── Primary: Google Gemini (3.8 Flash)
                                      |                          +── Fallback: OpenRouter (Multi-model)
                                      |                          +── Local: Self-Hosted Tier (llama.cpp/vLLM)
                                      |
                                      +── /friday/*    ──> [FRIDAY Core :10000]
                                      +── /memora/*    ──> [Memora :8000]
```

---

## ⚡ Cascading Fallback Engine

Inference implements an automated cascading fallback ladder across providers. If a primary provider experiences rate limits (`429`), timeouts (`504`), socket failures, or service outages (`503`), Inference seamlessly cascades to the next healthy provider within the same request lifecycle without leaking provider exceptions to the client:

```
[Incoming Request]
        │
        ▼
1. Fast-Path Speculative Race ──────► Groq & Cerebras (Ultra-Low Latency LPU)
        │ (Timeout / Rate Limit)
        ▼
2. Primary Reasoning Engine ────────► Google Gemini (Gemini 3.8 Flash)
        │ (429 / Outage)
        ▼
3. Broad Provider Mesh ─────────────► OpenRouter (Aggregated Free/Low-Cost Frontier Models)
        │ (Network Failure)
        ▼
4. Fallback Cloud Keys ─────────────► Secondary & Third-Party Key Pools
        │ (Air-Gapped / Outage)
        ▼
5. Terminal Self-Hosted Tier ───────► Local Server (127.0.0.1:11434 / Ollama / llama.cpp)
```

- **Health Recovery Probing**: Quarantined keys or providers automatically enter a half-open probe state after `PROVIDER_HEALTH_RECOVERY_SECONDS` (30s default) to recover capacity without human intervention.
- **Deadline Reservation**: 40% of the request timeout is reserved for fallback execution (`PRIMARY_ATTEMPT_FRACTION = 0.60`), guaranteeing that a slow primary cannot starve the fallback ladder.

---

## 📡 API Specification & Curl Examples

### 1. OpenAI-Compatible Chat Completions (`POST /v1/chat/completions`)

Standard endpoint compatible with the OpenAI SDK, LangChain, LiteLLM, and any ecosystem agent.

#### Request:
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-INFERENCE-API-KEY: your_api_key_here" \
  -d '{
    "model": "auto",
    "messages": [
      {"role": "system", "content": "You are a concise engineering specialist."},
      {"role": "user", "content": "Explain circuit breaker pattern in two sentences."}
    ],
    "temperature": 0.5,
    "max_tokens": 256,
    "stream": false
  }'
```

#### Response:
```json
{
  "id": "chatcmpl-a1b2c3d4e5f6",
  "object": "chat.completion",
  "created": 1729000000,
  "model": "openai/gpt-oss-120b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The circuit breaker pattern detects service failures and prevents applications from repeatedly trying to execute an operation doomed to fail. When failures exceed a threshold, the breaker trips to open state, immediately failing calls and allowing degraded dependencies to recover."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 24,
    "completion_tokens": 42,
    "total_tokens": 66
  }
}
```

---

### 2. Available Models Catalog (`GET /v1/models`)

Lists the active model catalog available across configured providers.

#### Request:
```bash
curl -X GET http://localhost:8000/v1/models \
  -H "X-INFERENCE-API-KEY: your_api_key_here"
```

#### Response:
```json
{
  "object": "list",
  "data": [
    {"id": "gemini-3.8-flash", "object": "model", "owned_by": "google", "permission": []},
    {"id": "openai/gpt-oss-120b", "object": "model", "owned_by": "groq", "permission": []},
    {"id": "llama-3.3-70b-versatile", "object": "model", "owned_by": "groq", "permission": []},
    {"id": "cerebras/llama-3.3-70b", "object": "model", "owned_by": "cerebras", "permission": []},
    {"id": "nvidia/nemotron-3-super-120b-a12b", "object": "model", "owned_by": "nvidia", "permission": []},
    {"id": "liquid/lfm-2.5-2.6b:free", "object": "model", "owned_by": "openrouter", "permission": []},
    {"id": "auto", "object": "model", "owned_by": "inference", "permission": []}
  ]
}
```

---

### 3. High-Speed SSE Token Streaming (`POST /stream`)

Sub-300ms Time-To-First-Token (TTFT) Server-Sent Events (SSE) streaming endpoint for real-time conversational UI.

#### Request:
```bash
curl -N -X POST http://localhost:8000/stream \
  -H "Content-Type: application/json" \
  -H "X-INFERENCE-API-KEY: your_api_key_here" \
  -d '{
    "prompt": "List 3 high-throughput caching strategies.",
    "model": "auto"
  }'
```

#### Stream Output:
```
data: {"token": "1.", "done": false}

data: {"token": " Cache", "done": false}

data: {"token": "-Aside", "done": false}

data: {"token": " pattern\n", "done": false}

...

data: {"done": true}
```

---

### 4. Process Liveness & Health Check (`GET /health`)

Lightweight unauthenticated liveness probe compatible with Kubernetes and Docker Healthcheck.

#### Request:
```bash
curl -X GET http://localhost:8000/health
```

#### Response:
```json
{
  "status": "ok",
  "service": "inference",
  "evidence_class": "process_liveness",
  "observed_at": "2026-10-14T16:00:00.000000+00:00",
  "providers": ["groq", "cerebras", "openrouter", "gemini"],
  "active_specialist_agents": 10,
  "version": "2.0.0"
}
```

---

## 🔐 Environment Variables

All configuration is loaded from environment variables and `.env`. Keys support both singular and comma-separated lists for automatic round-robin pooling.

| Variable | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `INFERENCE_API_KEY` | `string` | *(Required)* | Primary secret token for authenticating gateway calls |
| `GEMINI_API_KEYS` | `string` | `""` | Comma-separated Google Gemini API keys (Gemini 3.8 Flash) |
| `GROQ_API_KEYS` | `string` | `""` | Comma-separated Groq API keys (LPU ultra-fast inference) |
| `CEREBRAS_API_KEYS` | `string` | `""` | Comma-separated Cerebras API keys |
| `OPENROUTER_API_KEYS`| `string` | `""` | Comma-separated OpenRouter API keys |
| `NVIDIA_API_KEYS` | `string` | `""` | Comma-separated NVIDIA NIM API keys |
| `THIRD_PARTY_*_KEYS` | `string` | `""` | Dedicated fallback key pools for tertiary redundancy |
| `PORT` | `int` | `8000` | Application server port |
| `HOST` | `string` | `0.0.0.0` | Bind interface |
| `DATABASE_URL` | `string` | `sqlite:///data/universe.db` | Persistent consultation & task memory database |
| `WEB_CONCURRENCY` | `int` | `1` | Uvicorn worker process count |
| `MEMORY_TRIM_ENABLED`| `bool` | `true` | Proactive glibc memory trim to prevent container OOM |
| `MEMORY_TRIM_THRESHOLD_MB`| `float` | `400.0` | RSS high-water trigger for idle memory reclamation |
| `LOCAL_ENABLED` | `bool` | `false` | Enable self-hosted local model server fallback |
| `LOCAL_BASE_URL` | `string` | `http://127.0.0.1:11434/v1` | Local server base URL |

---

## 🚀 Quickstart & Container Deployment

### Local Development
```bash
# 1. Clone repository & setup virtual environment
git clone https://github.com/surendra2304/Inference.git
cd Inference
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install production dependencies
pip install -r requirements.txt
pip install -e ".[dev]"

# 3. Configure credentials
cp .env.example .env
# Edit .env with your provider API keys and INFERENCE_API_KEY

# 4. Start local development server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Docker Production Deployment
```bash
# Build production container image
docker build -t inference .

# Run container with persistent data volume and port mapping
docker run -d \
  --name inference \
  -p 8004:8000 \
  --env-file .env \
  -v inference-data:/app/data \
  --restart unless-stopped \
  inference
```

### Docker Compose Mesh Deployment
```bash
docker compose -f docker-compose.production.yml up -d --build
```
