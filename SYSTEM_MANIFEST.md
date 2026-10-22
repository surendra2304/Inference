# 🏛️ SYSTEM MANIFEST — Inference AI Intelligence Gateway

> **Official Subsystem Name:** Inference
> **Role in Ecosystem:** Central Multi-Model Intelligence & Deliberation Gateway (configured provider pool)
> **Repository:** [surendra2304/Inference](https://github.com/surendra2304/Inference) (Branch: main)
> **Workspace Path:** d:\FRIDAY Universe\Inference

---

## ☁️ 1. Configured Cloud Infrastructure

| Attribute | Repository configuration |
| :--- | :--- |
| **Configured Service URL** | [https://friday-universe.duckdns.org/inference](https://friday-universe.duckdns.org/inference) (current deployment state not verified here) |
| **Configured Health Route** | `/health` (response is not proof of provider availability) |
| **API key variable (keep value in secret environment)** | `INFERENCE_API_KEY` (set a unique secret outside source control) |
| **Authentication Header** | `X-INFERENCE-API-KEY: <configured key>` or `Authorization: Bearer <configured key>` |
| **Configured database topology (runtime unverified)** | In-memory consultation cache plus Memora integration; current remote durability/connectivity is not verified here |
| **Database Namespace** | `memora://inference/private` (configured intent; access not verified here) |
| **Hosting** | Oracle Cloud Always Free (friday_mesh) |

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
INFERENCE_URL=https://friday-universe.duckdns.org/inference
INFERENCE_API_KEY=<configure locally; do not commit>

# 2. 🧠 Memora cloud memory service (active backend/capacity not verified)
MEMORA_URL=https://friday-universe.duckdns.org/memora
MEMORA_API_KEY=<configure locally; do not commit>

# 3. 📈 Stratex paper/testnet strategy service (live-money orders blocked in source)
STRATEX_URL=https://friday-universe.duckdns.org/stratex
STRATEX_API_KEY=<configure locally; do not commit>

# 4. 🧠 IntelX research service (active corpus backend unverified)
INTELX_URL=https://friday-universe.duckdns.org/intelx
INTELX_API_KEY=<configure locally; do not commit>

# 5. 🔮 Futuris Calibrated Predictive Forecasting Engine
FUTURIS_URL=https://friday-universe.duckdns.org/futuris
FUTURIS_API_KEY=<configure locally; do not commit>

# 6. 🌐 Cortex Autonomous Web Operations & Intelligence
CORTEX_URL=https://friday-universe.duckdns.org/cortex
CORTEX_API_KEY=<configure locally; do not commit>

# 7. 🛠️ Forge Local Software Engineering Engine
FORGE_URL=https://friday-universe.duckdns.org/forge
FORGE_API_KEY=<configure locally; do not commit>

# 8. 🛡️ Sentinel Local Cybersecurity & Threat Defense Shield
SENTINEL_URL=https://friday-universe.duckdns.org/sentinel
SENTINEL_API_KEY=<configure locally; do not commit>

# 9. 🤖 FRIDAY Central Desktop Operating System
FRIDAY_URL=https://friday-universe.duckdns.org/friday
FRIDAY_API_KEY=<configure locally; do not commit>
```

---

## 🤖 4. Repository guide

When opening this repository:
* **Identity:** You are working inside **Inference** (d:\FRIDAY Universe\Inference).
* **Configured Service URL:** https://friday-universe.duckdns.org/inference; deployed on Oracle Cloud Always Free mesh.
* **Authentication:** Incoming requests require the configured `INFERENCE_API_KEY`; use a unique strong secret per service.
* **Verification:** Distinguish source tests, local integrations, mocked results, and live service evidence.
* **Secrets:** Never commit live credentials or copy placeholder examples into service environments.
