# 🏛️ SYSTEM MANIFEST — Inference AI Intelligence Gateway

> **Official Subsystem Name:** Inference
> **Role in Ecosystem:** Central Multi-Model Intelligence & Deliberation Gateway (configured provider pool)
> **Repository:** [surendra2304/Inference](https://github.com/surendra2304/Inference) (Branch: main)
> **Workspace Path:** d:\FRIDAY Universe\Inference

---

## ☁️ 1. Configured Cloud Infrastructure

| Attribute | Repository configuration |
| :--- | :--- |
| **Configured Service URL** | [https://inference-r1sn.onrender.com](https://inference-r1sn.onrender.com) (current deployment state not verified here) |
| **Configured Health Route** | `/health` (response is not proof of provider availability) |
| **API key variable (keep value in secret environment)** | `INFERENCE_API_KEY` (set a unique secret outside source control) |
| **Authentication Header** | `X-INFERENCE-API-KEY: <configured key>` or `Authorization: Bearer <configured key>` |
| **Configured database topology (runtime unverified)** | In-memory consultation cache plus Memora integration; current remote durability/connectivity is not verified here |
| **Database Namespace** | `memora://inference/private` (configured intent; access not verified here) |
| **Hosting** | Render is configured in the repository; current deployment revision is not verified here |

---

## 🎯 2. Purpose & Responsibilities

### Intended role
* Inference is intended to act as the model gateway. Its available provider pool is determined by the runtime environment and may change; consult runtime configuration rather than relying on a fixed key count.

### What Inference is intended to do
* Route model requests across providers configured for the current runtime.
* Support authenticated peer requests where configured; live peer connectivity is not proven by this manifest.
* Use Memora's private namespace when remote memory is configured and reachable; persistence requires runtime verification.

---

## 🌐 3. Ecosystem endpoint configuration

These variable names and URLs are configuration references, not proof of live connectivity. Set real credentials in secret environments; do not commit them:

`````env
# ============================================================================== #
#               FRIDAY UNIVERSE MASTER ECOSYSTEM CONFIGURATION                  #
# ============================================================================== #

# 1. ⚡ Inference AI Multi-Model Gateway (runtime-configured providers)
INFERENCE_URL=https://inference-r1sn.onrender.com
INFERENCE_API_KEY=<configure locally; do not commit>

# 2. 🧠 Memora cloud memory service (active backend/capacity not verified)
MEMORA_URL=https://memora-cavc.onrender.com
MEMORA_API_KEY=<configure locally; do not commit>

# 3. 📈 Stratex paper/testnet strategy service (live-money orders blocked in source)
STRATEX_URL=https://stratex-8wj1.onrender.com
STRATEX_API_KEY=<configure locally; do not commit>

# 4. 🧠 IntelX research service (active corpus backend unverified)
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
```

---

## 🤖 4. Repository guide

When opening this repository:
* **Identity:** You are working inside **Inference** (d:\FRIDAY Universe\Inference).
* **Configured Service URL:** https://inference-r1sn.onrender.com; confirm current deployment status in Render.
* **Authentication:** Incoming requests require the configured `INFERENCE_API_KEY`; use a unique strong secret per service.
* **Verification:** Distinguish source tests, local integrations, mocked results, and live service evidence.
* **Secrets:** Never commit live credentials or copy placeholder examples into service environments.
