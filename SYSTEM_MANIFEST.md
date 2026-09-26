# 🏛️ SYSTEM MANIFEST — Inference AI Intelligence Gateway

> **Official Subsystem Name:** Inference
> **Role in Ecosystem:** Central Multi-Model Intelligence & Deliberation Gateway (configured provider pool)
> **Repository:** [surendra2304/Inference](https://github.com/surendra2304/Inference) (Branch: main)
> **Workspace Path:** d:\FRIDAY Universe\Inference

---

## ☁️ 1. Live Cloud Infrastructure & Deployment

| Attribute | Production Configuration |
| :--- | :--- |
| **Live Production URL** | [https://inference-r1sn.onrender.com](https://inference-r1sn.onrender.com) |
| **Health Check Endpoint** | https://inference-r1sn.onrender.com/health |
| **Master API Key Variable** | `INFERENCE_API_KEY` (set a unique secret outside source control) |
| **Authentication Header** | `X-INFERENCE-API-KEY: <configured key>` or `Authorization: Bearer <configured key>` |
| **Database Topology** | In-Memory Consultation Cache / Connected to Memora Cloud |
| **Database Connection** | memora://inference/private |
| **Hosting Platform** | Render Docker Web Service (Singapore / AWS Mumbai) |

---

## 🎯 2. Purpose & Responsibilities

### What Inference IS:
* Inference is the model gateway. Its available provider pool is determined by the runtime environment and may change; consult runtime configuration rather than relying on a fixed key count.

### What Inference DOES:
* Operates as the **Central Multi-Model Intelligence & Deliberation Gateway** within the 9-agent FRIDAY Universe.
* Communicates directly with peer agents via authenticated REST and WebSocket protocols.
* Persists private long-term memory records to **Memora** under memora://inference/private.

---

## 🌐 3. Full Ecosystem Network Connectivity

Every agent in the universe communicates using standard environment variables:

`env
# ============================================================================== #
#               FRIDAY UNIVERSE MASTER ECOSYSTEM CONFIGURATION                  #
# ============================================================================== #

# 1. ⚡ Inference AI Multi-Model Gateway (runtime-configured providers)
INFERENCE_URL=https://inference-r1sn.onrender.com
INFERENCE_API_KEY=<configure locally; do not commit>

# 2. 🧠 Memora Cloud Persistent Memory (9 GB Turso AWS Mumbai)
MEMORA_URL=https://memora-cavc.onrender.com
MEMORA_API_KEY=<configure locally; do not commit>

# 3. 📈 Stratex 24/7 Algorithmic Trading Platform (Binance Futures)
STRATEX_URL=https://stratex-8wj1.onrender.com
STRATEX_API_KEY=<configure locally; do not commit>

# 4. 🧠 IntelX Evidence & Intelligence Research Engine (Turso AWS Mumbai)
INTELX_URL=https://intelx-mygl.onrender.com
INTELX_API_KEY=<configure locally; do not commit>

# 5. 🔮 Futuris Calibrated Predictive Forecasting Engine
FUTURIS_URL=https://futuris-th6f.onrender.com
FUTURIS_API_KEY=<configure locally; do not commit>

# 6. 🌐 Cortex Autonomous Web Operations & Intelligence
CORTEX_URL=https://cortex-0m7c.onrender.com
CORTEX_API_KEY=<configure locally; do not commit>

# 7. 🛠️ Forge Local Software Engineering Engine
FORGE_URL=https://forge-e9kl.onrender.com
FORGE_API_KEY=<configure locally; do not commit>

# 8. 🛡️ Sentinel Local Cybersecurity & Threat Defense Shield
SENTINEL_URL=https://sentinel-a861.onrender.com
SENTINEL_API_KEY=<configure locally; do not commit>

# 9. 🤖 FRIDAY Central Desktop Operating System
FRIDAY_URL=https://friday-zw59.onrender.com
FRIDAY_API_KEY=<configure locally; do not commit>
`

---

## 🤖 4. Antigravity AI Session Guide

When opening this directory in **Antigravity AI**:
* **Identity:** You are working inside **Inference** (d:\FRIDAY Universe\Inference).
* **Live Service:** This service is deployed live at https://inference-r1sn.onrender.com.
* **Authentication:** Incoming requests require the configured `INFERENCE_API_KEY`; use a unique strong secret per service.
* **Never Fake Tests:** All tests and verifications must be executed against real code and real endpoints.
* **No Unapproved Git Pushes:** Keep modifications local unless explicitly instructed to push.
