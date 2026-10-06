# Phase 0–2 Report — Inference

Branch: `arena/01a10cbc-inference` · Base commit: `59846da88337de79c112427b8732d659b16b0cc2` · Date: 2026-10-05

Every claim below is followed by the command that produced it.

---

## PHASE 0 — TOTAL COMPREHENSION

### (a) What this project IS, and its dream state

**Inference** is a local-first, provider-agnostic **multi-agent intelligence gateway**: a FastAPI
service that accepts a question, routes it to a panel of specialist agents, lets them reason
through a "collaborate-first, debate-on-conflict" engine, and returns a synthesized answer with a
calibrated confidence score plus a persistent SQLite audit trail. It is one service inside a larger
"FRIDAY Universe" mesh (Memora, Stratex, IntelX, Futuris, Cortex, Forge, Sentinel, FRIDAY).

**Dream state (inferred — the vision line in the brief was left as the placeholder):**
> A *trustworthy* answer engine. One question in, a genuinely cross-checked answer out — and if the
> system did not actually consult models and evidence, it must say so instead of inventing confidence.
> Provider pool behind a unified gateway; trading consultation strictly advisory; FRIDAY as a
> peer client over an authenticated boundary.

Why this is inferable from the code: the repo already encodes that intent in its own words —
`v1_core_routes.py` guarantee #6 *"If evidence is insufficient, low confidence and missing data are
returned"*, `STRICT_EMPIRICAL` evidence policy, and test assertions like *"avoids_healthy_claim"*.
The gap between that intent and the observed behaviour is the core story of this report.

### (b) Request/execution flow (real files, classes, functions)

**HTTP → orchestrator → debate → gateway → provider:**

```
uvicorn app.main:app
└─ app/main.py: app = FastAPI(lifespan=lifespan)
   lifespan(): register_software_specialists() → orchestrator.memory.initialize()
               → http_client_pool.get_client() + prewarm()
   middleware stack (added in app/main.py):
     EnhancedRateLimiterMiddleware (app/middleware/rate_limiter.py)
     → ProductionSecurityMiddleware (app/security/api_security.py)
     → GZipMiddleware → CORSMiddleware
   routers: 29 include_router() calls (app/main.py:118-146)
```

1. **`POST /ask`** → `app/api/routes.py::ask_question()` → builds `OrchestrationRequest`
   (`app/core/orchestrator.py`) → **`Orchestrator.process_task()`**
2. `Orchestrator.process_task()`:
   - `classify_task_complexity()` (`app/core/dag.py`)
   - `StrategyStore.recommend_strategy()` (`app/learning/strategy_store.py`)
   - `router.route_task()` (`app/agents/router.py`) → `decision.selected_agent_ids` / `decision.telemetry`
   - `self.registry.get_agent(aid)` (`app/agents/registry.py`)
   - `build_execution_dag()` → `ExecutionDAG` + `dag.build_layers()`  ← **built, logged, then discarded**
   - `asyncio.create_task(self.memory.save_task(...))` (`app/memory/sqlite.py::SQLiteMemory.save_task`)
   - **`CollaborationEngine.run_collaboration()`** (`app/agents/debate.py`) ← actual execution
3. `CollaborationEngine`:
   - `_execute_agent_call()` → `_invoke_single_model()` → **`model_gateway.execute()`**
     (`app/providers/gateway.py::ModelGateway.execute`)
   - `KeyPool.choose()` (round-robin + 60s quarantine), `ProviderRateLimiter.acquire()`
   - `provider_health_tracker` health gate (`app/providers/health.py`)
   - fallback chain: OpenRouter capability fallback → `ProviderSwitchingPolicy` → LiteLLM
   - `Adjudicator.reconcile_panel_adjudication()` / `calculate_system_confidence()` (`app/agents/adjudication.py`)
4. Result → `OrchestrationResult` → persisted via `SQLiteMemory.save_task()` → `data/universe.db`

**Parallel flagship path:** `POST /v1/trading/consult`
→ `app/routers/trading.py::consult_trading_bot()` (size check → `_scan_for_forbidden_keys()` →
schema → `_check_rate_limit()`) → `TradingConsultService.consult()` → `telemetry_cache` →
`concurrency_controller.run()` → `_consult_internal()` → 4 × `_invoke_agent()` →
`_rule_based_synthesis()` → `AIUniverseDecision`.

**Security boundary:** only `app/api/friday_routes.py:25` carries
`dependencies=[Depends(verify_friday_api_key)]` (`app/core/security.py::verify_friday_api_key`).

### (c) The 5 most important files

| # | File | Why |
|---|---|---|
| 1 | `app/agents/debate.py` (693 LOC) | The product. `CollaborationEngine` is the multi-agent engine everything else calls; it also contains the failure-masking behaviour (below). |
| 2 | `app/providers/gateway.py` (618 LOC) | `ModelGateway` — key rotation, per-provider rate limiting, quarantine, 3-tier fallback. Every model call in the system flows through it. |
| 3 | `app/core/orchestrator.py` (333 LOC) | The glue: routing → agents → DAG → debate → persistence. Decides what the user actually gets. |
| 4 | `app/providers/unified_manager.py` (238 LOC) | `UnifiedProviderManager` — the alternate dispatch path used by `/v1/ask`, `/v1/debate`, `/v1/task/execute`, `/v1/agent/*`. Site of the worst bug. |
| 5 | `app/memory/sqlite.py` (547 LOC) | `SQLiteMemory` — the audit trail. If records here are wrong, every downstream claim is wrong. |

### (d) What surprised me

1. **Two `pytest` config files, and pytest warns it is ignoring one.** Proven below.
2. **A 58-module, 2,105-LOC "hardened runtime" (`app/inference_runtime/`) that nothing calls.**
3. **Two Python files that cannot be imported at all** due to module shadowing — and `mypy` and
   `ruff` both pass because neither resolves imports the way the interpreter does.
4. **The DAG that the log line advertises is never executed.** `app/core/dag_executor.py`
   (173 LOC, complete: retries, cancellation, timeouts) is imported by **zero** files.
5. **The model names I suspected were hallucinated are real.** I checked instead of assuming —
   `gemini-3.8-flash` released 2026-09-02, `qwen/qwen3.8-27b` and `openai/gpt-oss-120b` live on Groq,
   `nvidia/nemotron-3-super-120b-a12b` on NVIDIA NIM. **Not bugs.**
6. **Green CI over a broken system**: 20 tests, ruff clean, mypy clean, build clean — and the core
   request path returns fabricated results with fabricated confidence.

---

## PHASE 1 — TRUTH AUDIT

### 1. Fresh install + gauntlet

```
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
→ Successfully installed Inference-2.0.0 ... fastapi-0.142.2 pydantic-2.13.5 pytest-9.1.1
  pytest-asyncio-1.4.0 ruff/mypy installed separately (CI installs them ad hoc)
```

| Gate | Command | Result |
|---|---|---|
| Tests | `.venv/bin/python -m pytest -ra -q tests/` | `20 passed in 0.64s` |
| Lint | `.venv/bin/ruff check .` | `All checks passed!` |
| Types | `.venv/bin/python -m mypy app` | `Success: no issues found in 231 source files` |
| Build | `.venv/bin/python -m build --wheel` | `Successfully built inference-2.0.0-py3-none-any.whl` |
| Import all | `pkgutil.walk_packages(app.__path__)` | `modules walked: 185  FAILED IMPORTS: 0` |

**All four CI gates are green.** Note the import walk only imports modules — it does not import the
two shadowed files as submodules of their parent (see Bug #9).

### 2. Boot + user flows exercised

```
uvicorn app.main:app --host 0.0.0.0 --port 8000
→ "Application startup complete."  117 paths / 120 operations / 19 specialist agents
```

| Flow | Result |
|---|---|
| `GET /health` | ✅ 200 `"status":"responding"` |
| `GET /ui`, `GET /` (Accept: text/html) | ✅ 200, 19,944 bytes |
| `GET /status`, `GET /agents` | ✅ 19 agents with real 2026 model IDs |
| `POST /ask` (no provider keys) | ⚠️ **200, `confidence:0.92`, answer = error string** |
| `GET /tasks/{id}` | ⚠️ 200, persists `status:"completed"` + `confidence:0.92` for a failure |
| `POST /v1/ask` (no provider keys) | 🔴 **200, `status:"SUCCESS"`, `confidence:0.92`, `evidence:"Empirical provider verification via groq:openai/gpt-oss-20b"`, `total_tokens:120`** |
| `POST /v1/friday/ask` (no key) | ✅ 401 — correctly enforced |
| `POST /v1/friday/ask` (correct key) | ✅ 200 |
| `POST /v1/ask` (bogus key / no key) | 🔴 **200** — declared header never validated |
| `POST /v1/trading/consult` (valid payload) | ⚠️ 200 in **0.052 s** for a "5-agent deliberation" (canned text) |
| `POST /v1/governance/.../rotate-key` (no auth) | 🔴 **200, minted a key** |
| `GET /v1/governance/prometheus-metrics` | 🔴 **hardcoded fake metrics** |
| `POST /v1/agent/stream` (SSE) | ✅ correctly reports `{"error":"GEMINI_API_KEY is not configured."}` |
| `POST /v1/instant/stream` with `{"question":..}` | ✅ 422 (schema wants `prompt`) |
| `python -m app.cli --help`, `python -m app.core.key_inventory` | ✅ both work |
| SQLite persistence | ✅ `data/universe.db` 110,592 bytes |

### 3. Docs vs reality

| Claim | Source | Reality |
|---|---|---|
| "Incoming requests require the configured `INFERENCE_API_KEY`" | `SYSTEM_MANIFEST.md:10` | 🔴 Only `/v1/friday/*` enforces it. 120 ops, **0 with a security scheme**; `/v1/ask` returns 200 with no key. |
| "In-memory consultation cache plus **Memora integration**" | `SYSTEM_MANIFEST.md:18` | 🔴 `app/memora_client.py` + `app/memora_cloud_fallback.py` imported by **nothing**. Memory is local SQLite only. |
| "Confidential credentials are scrubbed before any model invocation" | `v1_core_routes.py` guarantee #3 | ⚠️ Works for `k=v` form; **leaks** on other natural forms (Bug #12). |
| "If evidence is insufficient, low confidence … are returned" | `v1_core_routes.py` guarantee #6 | 🔴 Total provider failure returns `confidence 0.92` + `status SUCCESS`. |
| "ENGAGES in multi-agent deliberation" (trading) | `README.md` | ⚠️ Deterministic rules + 4 canned sentences when providers fail; runs recorded as `status="completed"`. |
| `docs/openapi.yaml` `/v1/trading/consult` needs `symbol, current_regime` | `docs/openapi.yaml:14-30` | 🔴 **400** — real schema needs `bot_id, trading_mode, telemetry, consultation_reason`. |
| `docs/openapi.yaml` `version: 2.5.0`, 11 paths | `docs/openapi.yaml:3` | 🔴 Actual: `2.0.0`, **117 paths**. |
| `docs/api.md:116` `/health` → `{"status":"healthy"}` | `docs/api.md` | ⚠️ `/health` → `"responding"`; `/v1/health` → `"healthy"`. |
| README → `inference_DIARY.md`, `inference_DIARY_SPEC.md` | `README.md` | 🔴 **Both files do not exist** (`git ls-files` → only `INFERENCE_DIARY.md`). |
| README architecture: "Debate Engine (10 Specialist Roles)" | `README.md` | ⚠️ **19** agents registered (`/status`). |
| README "Connects to Gemini, Groq, Mistral, OpenRouter, NVIDIA, Cohere, HuggingFace" | `README.md` | ✅ Real — all 7 adapters exist; model IDs verified live for 2026. |
| `python -m app.core.key_inventory` | `README.md` | ✅ Works, prints counts only. |
| docker-compose production stack | `docker-compose.production.yml` | 🔴 `nginx/` absent, api containers publish no ports, Postgres/Redis unused. |

**Working features nobody documented:** `/ui` control room; SSE streaming (`/v1/agent/stream`,
`/v1/instant/stream`); `/metrics` (Prometheus text, real values); 29 routers / 117 paths (docs cover
~5); `app/cli.py`; `scripts/` verification harness.

### 4. Test coverage

**20 tests / 411 test LOC against 23,724 app LOC (1.7% by line).** No tests exist for:
`app.api.routes`, `app.core.orchestrator`, `app.agents.debate`,
`app.services.trading_consult_service`, `app.routers.trading`, `app.security.prompt_isolation`,
`app.governance.tenant_manager`, `app.memory.sqlite`, `app.middleware.rate_limiter`,
`app.routers.governance` (0 test files reference any of them).

→ **"Write the safety net" is now part of Phase 3.**

---

## PHASE 2 — BUG REPORT (prioritized)

### 🔴 CRITICAL-1 — Total provider failure returns HTTP 200, `status: SUCCESS`, `confidence: 0.92`, and invented "empirical" evidence

* **Where:** `app/providers/unified_manager.py:221-235`; `app/api/v1_core_routes.py:205-240`
* **Root cause:** `execute()` catches *every* provider exception and returns a canned template with a
  hardcoded `token_usage={"total_tokens": 120}` and `status="fallback_success"` — it never raises.
  The caller never inspects `resp.status`, so it emits `status="SUCCESS"`, `confidence=0.92`,
  `uncertainty=0.08`, and `evidence=["Empirical provider verification via {provider}:{model}"]`
  regardless. The `answer` also echoes the first 150 chars of the *internal* ASTRA system instruction
  back to the caller (prompt-scaffold leak).
* **Evidence:**
  ```
  POST /v1/ask {"prompt":"What is the capital of France?","mode":"fast"}
  → status: "SUCCESS"      confidence: 0.92   uncertainty: 0.08   failure_state: null
    evidence: ["Empirical provider verification via groq:openai/gpt-oss-20b"]
    provider_metadata: prompt_tokens: 0  completion_tokens: 0  total_tokens: 120  latency_ms: 0
    answer: "[system_architect] Analysis completed for prompt: [ASTRA DELIBERATIVE REASONING COUNCIL ...
  ```
  `prompt_tokens:0 / latency_ms:0` self-refutes the "empirical verification" claim in the same object.
  Zero provider keys were configured during this call.
* **Fix:** distinguish degradation from success: propagate an error, return `status="DEGRADED"` /
  `failure_state`, set `confidence` to a low floor, drop `evidence`, and stop synthesizing fake token
  counts. Do not echo the internal instruction.

### 🔴 CRITICAL-2 — Unauthenticated, unvalidated tenant key rotation that mints unlimited keys

* **Where:** `app/routers/governance.py:26-33`; `app/governance/tenant_manager.py:107-116`
* **Root cause:** endpoint has **no** `Depends`; `rotate_tenant_key()` never verifies `old_key`
  (it only `remove()`s it *if present*), then unconditionally appends a freshly minted key.
* **Evidence:**
  ```
  curl -X POST /v1/governance/tenants/tenant_nexus/rotate-key -d '{"old_key":""}'
  → 200 {"status":"ROTATED","new_key":"key_tenant_nexus_8518e95aba81"}      # no headers sent
  curl -X POST /v1/governance/tenants/tenant_forge/rotate-key -d '{"old_key":"totally-wrong-key-12345"}'
  → 200 {"status":"ROTATED","new_key":"key_tenant_forge_b26e3ccf63f1"}
  curl /v1/governance/tenants/tenant_trading   → 200 leaking active_keys: ["key_trading_live_01", ...]
  ```
* **Fix:** add `Depends(verify_friday_api_key)`; require `old_key` to match `policy.active_keys`
  (403 otherwise); cap `active_keys`; log the rotation.

### 🔴 CRITICAL-3 — 114/120 operations are anonymous; 3 declare an auth header that is never read

* **Where:** `app/api/v1_core_routes.py:133,272`; `app/api/universal_task_routes.py:67`;
  `app/security/api_security.py:42`
* **Evidence:**
  ```
  OpenAPI: paths=117 operations=120 with_security_scheme=0
  POST /v1/friday/ask  (no key) → 401 ✅          ← only protected family
  POST /v1/ask  (bogus key)     → 200 🔴
  POST /v1/ask  (no key)        → 200 🔴
  POST /v1/task/execute (bogus) → 200 🔴
  grep -n "x_friday_api_key" app/api/*.py | grep -v 'Header(None'  →  no matches (never read)
  grep -rn "validate_api_key" --include=*.py .  →  tests/ only
  ```
  `APISecurityManager.validate_api_key()` is production-dead; `sanitize_input()` has zero callers.
  Contradicts `SYSTEM_MANIFEST.md:10`.
* **Fix:** central auth dependency on sensitive routers; remove the decorative `x_friday_api_key`
  parameters (or actually validate them); declare an `APIKey` scheme so OpenAPI reflects reality.

### 🔴 CRITICAL-4 — Fabricated Prometheus metrics served to monitoring

* **Where:** `app/routers/governance.py:44-69`
* **Root cause:** the response body is a hardcoded string literal — including
  `inference_provider_health{provider="gemini"} 1` for all 7 providers, claiming healthy while the
  process holds zero keys. It is additionally wrapped in JSON (`{"content_type":...,"metrics":...}`),
  so Prometheus could not scrape it anyway. The real `/metrics` (`app/health.py::prometheus_metrics`)
  exists alongside it.
* **Evidence:**
  ```
  GET /v1/governance/prometheus-metrics → inference_requests_total{service="nexus"} 412
                                           inference_requests_total{service="forge"} 1240
                                           inference_provider_health{provider="gemini"} 1   ← constant
  GET /metrics (real)                   → inference_requests_total 0                         ← truth
  ```
* **Fix:** delete the fake endpoint, or make it a 301 to `/metrics`. Never ship constant metrics.

### 🟠 HIGH-5 — `/ask` returns 200 + `confidence 0.92` when every model call failed

* **Where:** `app/agents/debate.py:283` (fallback text), `:455` (hardcoded `model_confidence=0.90`),
  `:458-470` (single-agent fast path), `app/api/routes.py::ask_question`
* **Root cause:** `_execute_agent_call()` returns an error *string* on failure — indistinguishable
  from a real answer. `SpecialistAssessment.model_confidence` is hardcoded to `0.90` for every
  specialist regardless of outcome; `calculate_system_confidence()` then yields
  `0.90 + evidence_bonus(0) + complexity(0.02) = 0.92`.
* **Evidence:**
  ```
  POST /ask {"question":"What is 2+2?","mode":"fast"}
  → 200  answer:"*[Specialist Researcher temporarily offline ... GROQ_API_KEY is not configured.]*"
         confidence:0.92  models_used:["gemini-3.8-flash"]  total_tokens:0
  GET /tasks/task_55b473f43b11 → status:"completed"  confidence:0.92  (error string as result)
  ```
* **Fix:** raise/propagate a typed failure from `_execute_agent_call` instead of returning prose;
  set `model_confidence` from actual call success; the fast path must not certify a failure as 0.92.

### 🟠 HIGH-6 — Trading consultation presents canned text as agent deliberation, recorded as successful runs

* **Where:** `app/services/trading_consult_service.py:236-257`, `:176-234`, `:638-644`
* **Root cause:** `_deterministic_fallback_for_agent()` returns 4 fixed sentences; the fallback path
  saves `RunRecord(status="completed", error=...)` — the audit trail marks a failed invocation as
  completed. `_rule_based_synthesis()` then reports `confidence: 0.88`, `status:"RECOMMENDATION"`,
  and a `debate_summary` containing the canned text, with **no degraded indicator**.
* **Evidence:** full 5-agent consult returned in `real 0m0.052s` with
  `"debate_summary":"Multi-Agent Deliberation:\n- TradingAnalyst: Quantitative Analysis: Evaluated win
  rate, profit factor, consecutive losses and drawdown against empirical baselines..."` — the exact
  literals at `trading_consult_service.py:241-256`.
* **Fix:** mark the decision `source: "deterministic_rule_fallback"`, record runs as
  `status="failed"` + surface `provider_degraded: true`, and stop calling it deliberation.
  *(The underlying numeric rules are legitimate — the mislabelling is the bug.)*

### 🟠 HIGH-7 — Rate limiting keyed on `request.client.host` with no proxy awareness

* **Where:** `app/middleware/rate_limiter.py:41`; `app/security/api_security.py:92`
* **Evidence:** `grep -rni "x-forwarded" --include=*.py .` → **0 hits** (whole repo).
  Additionally `api_security.py` unconditionally bypasses `127.0.0.1/localhost/testclient` with **no
  env guard** (`return True`), unlike `rate_limiter.py` which requires dev flags.
* **Impact:** behind Render/nginx every user shares one bucket (over-blocking), or — if the proxy
  source is loopback — limits vanish entirely. The security middleware's bypass has no `APP_ENV` check.
* **Fix:** honour `X-Forwarded-For` only from a trusted-proxy list; gate the loopback bypass on
  `settings.INSECURE_DEV_AUTH` / `APP_ENV != production`.

### 🟠 HIGH-8 — Naive/aware `datetime` mix corrupts persisted task records

* **Where:** `app/memory/base.py:18,30,48,60,73,84` (`default_factory=datetime.utcnow` → naive);
  `app/core/orchestrator.py` (`datetime.now(timezone.utc)` → aware);
  `app/services/trading_consult_service.py:649` (`datetime.utcnow()` → naive). 10 `utcnow()` vs 13
  tz-aware call sites.
* **Evidence (read from the live `data/universe.db`, through the app's own model):**
  ```
  task_55b473f43b11
    created_at  = datetime(2026,10,5,18,59,2,114572)                 tzinfo=None
    completed_at= datetime(2026,10,5,18,59,2,115890, tzinfo=utc)     tzinfo=UTC
    >>> CRASH: TypeError: can't subtract offset-naive and offset-aware datetimes
  ```
* **Fix:** make every timestamp `datetime.now(timezone.utc)` (incl. `TaskRecord` defaults) and
  normalize on read.

### 🟡 MEDIUM-9 — Two Python files are physically unimportable (module shadowing)

* **Where:** `app/monitoring/alerts.py` (61 LOC), `app/optimization/token_optimizer.py` (92 LOC)
* **Root cause:** `app/monitoring.py` and `app/optimization.py` are *modules*; the same-named
  *directories* have no `__init__.py`, so the interpreter resolves the module and the submodules
  become unreachable.
* **Evidence:**
  ```
  import app.monitoring.alerts     → ModuleNotFoundError: No module named 'app.monitoring.alerts';
                                     'app.monitoring' is not a package
  import app.optimization.token_optimizer → ModuleNotFoundError: same shape
  ```
  `app/monitoring/alerts.py` is a byte-for-byte duplicate of the live `app/alerts.py`.
  **ruff and mypy both report clean** — neither resolves this the way Python does.
* **Fix:** delete the duplicates (the live counterparts are `app/alerts.py`, `app/token_optimizer.py`),
  or add `__init__.py` if the package form is intended.

### 🟡 MEDIUM-10 — 69 modules / 2,910 LOC unreachable from any entry point; flagship subsystems wired to nothing

Static reachability from `app.main` + `app.cli`:

| Dead module | LOC | Why it matters |
|---|---|---|
| `app/inference_runtime/*` (55 files) | ~2,105 | Exports `HardenedGateway` (budget, circuit, dedupe, key pool, retry, telemetry). **Never instantiated.** Only 3 `contracts` imports from outside. |
| `app/core/dag_executor.py` | 173 | Complete DAG executor. Orchestrator builds the DAG, logs `len(dag.layers)`, then runs flat `gather` — **the DAG is decorative**. |
| `app/agents/trading_analyst.py` | 123 | Never imported; the role comes from `roles/__init__.py`. |
| `app/memora_client.py` + `memora_cloud_fallback.py` | 132 | Basis of SYSTEM_MANIFEST's "Memora integration" claim. |
| `app/token_optimizer.py` **and** `app/optimization/token_optimizer.py` | 184 | Two token optimizers, **both** dead. |
| `app/ha/high_availability.py` | 53 | No HA despite the name. |
| `app/observability_system.py` | 43 | Unused; `/metrics` is hand-rolled. |
| `config/model_registry.json`, `config/key_pool.json` | — | **0 readers** in the whole repo; contains stale placeholder entries (`anthropic_key_1`). |

### 🟡 MEDIUM-11 — `docs/openapi.yaml` is a different API than the one running

* 11 paths vs 117 live; `version: 2.5.0` vs actual `2.0.0`.
* **Evidence — sending exactly the documented payload:**
  ```
  POST /v1/trading/consult {"symbol":"BTC/USDT","current_regime":"high_volatility","drawdown_pct":3.5}
  → 400 "4 validation errors … bot_id Field required … trading_mode … telemetry … consultation_reason"
  ```
* `docs/api.md:116` `/health` shape wrong; README links to two files that don't exist.

### 🟡 MEDIUM-12 — Credential scrubber misses real key formats (and the auditor reports nothing)

* **Where:** `app/security/prompt_isolation.py:47-50` (`_GENERIC_SECRET_FIELD_PATTERN` requires
  `[:=]` **and** ≥16-char value); `detect_credentials()` at `:172-211` never checks the generic pattern.
* **Evidence:**
  ```
  CASE                                    SCRUBBED  detect
  GEMINI_API_KEY=<AIza 39 chars>          True      []
  GEMINI_API_KEY is AIzaSyAbbbb…           FALSE 🔴  []     ← prose form leaks
  AIzaSyAbbbb…  (bare)                    FALSE 🔴  []     ← no Google-key pattern at all
  password=hunter2                        FALSE 🔴  []
  my api_key is ZZZ…                      FALSE 🔴  []
  hmac secret: sss…                       FALSE 🔴  []
  api_key=<40 chars>                      True      []     ← redacted, yet auditor says "clean"
  ```
  `GEMINI_API_KEY` is *this project's* primary credential.
* **Fix:** add standalone patterns for `AIza…`, `gsk_…`, `nvapi-…`, `sk-…`, `hf_…`; relax the
  separator requirement; make `detect_credentials` use the same pattern set as `scrub_credentials`.

### 🟡 MEDIUM-13 — Production docker-compose stack cannot run

* **Evidence:**
  ```
  git ls-files | grep -i nginx  → (no output; exit 1)     # nginx/nginx.conf does not exist
  ls nginx/ → No such file or directory
  docker-compose api-1/2/3 → publish NO ports; only nginx fronts them
  SQLiteMemory(db_path="postgresql://…").db_path
      → 'postgresql://postgres:postgres@postgres:5432/inference'   # treated as a FILE PATH
  grep -rn 'sqlalchemy' --include=*.py app/ scripts/ tests/ → 0 hits
  grep -rn 'asyncpg|psycopg|create_engine'                  → 0 hits
  grep -rni redis --include=*.py app/ → only demo prompt strings (3), no client
  ```
* Also: `POSTGRES_PASSWORD=postgres` hardcoded; `sqlalchemy` declared in both dep files but imported
  nowhere; `prometheus-client` is in `requirements.txt` but **not** in `pyproject.toml` (and missing
  from a fresh `.[dev]` venv) → Docker/CI/local resolve **different dependency sets**.

### 🟡 MEDIUM-14 — `pytest.ini` shadows `pyproject` pytest config (pytest itself warns)

* **Evidence (pytest header):**
  ```
  configfile: pytest.ini (WARNING: ignoring pytest config in pyproject.toml!)
  asyncio: mode=Mode.STRICT, debug=False …
  ```
* So `asyncio_mode = "auto"` and `addopts = "-ra -q -p no:fugue_test -p no:fugue_test_fixtures"` are
  **not applied**. It passes today only because the single async test carries an explicit
  `@pytest.mark.asyncio`; the next marker-less async test will fail.
* **Fix:** merge into one config source (delete `pytest.ini` and fold `python_*` keys into pyproject).

### 🟡 MEDIUM-15 — Unauthenticated unbounded memory growth (DoS)

* `app/routers/trading.py:31` — `_bot_request_timestamps` is a `defaultdict(list)` keyed by an
  unauthenticated, attacker-chosen `bot_id`; per-bot timestamps are pruned but **keys never are**.
* `app/security/api_security.py:72-83` — `_rate_limits` prunes timestamps per IP but **never removes IPs**.
* `app/governance/tenant_manager.py:115` — `active_keys` grows on every unauthenticated rotation (proven in #2).
* `app/governance/tenant_manager.py:130` — `dedup_cache` entries expire but are only deleted when
  the *same* `request_id` is seen again.

### 🔵 LOW-16 — Dead / stale code

* `api_security.py::sanitize_input()` — **zero callers anywhere** (and it strips `--`, which would
  mangle legitimate Markdown if ever wired up).
* `settings.get_friday_api_key` field `inference_api_KEY: str | None` — typo-cased field name
  (`app/core/config.py:96`), works but is confusing/error-prone.
* `.env.example` sets `APP_ENV="development"` while both config classes default `"production"` —
  copying the example (as the README instructs) into prod changes rate-limit behaviour.
* `Dockerfile` installs `requirements.txt`; README and CI install `.[dev]` — divergent dep sets.

### 🔵 LOW-17 — Deprecated / hygiene

* 10 × `datetime.utcnow()` (removed-in-3.12 trajectory).
* `docker-compose.production.yml` `version: '3.8'` (obsolete key).
* `docs/` has 29 files; only 5 are reachable from README's documentation list.

---

### NOT bugs (checked, explicitly cleared)

* **Model IDs** `gemini-3.8-flash`, `qwen/qwen3.8-27b`, `openai/gpt-oss-120b`,
  `nvidia/nemotron-3-super-120b-a12b` — verified live against Google/Groq/NVIDIA catalogs (2026).
* **No secrets in tree or history** (`git rev-list --all` = 1 commit; scans for `AIza…`, `sk-…`,
  `gsk_…`, `nvapi-…`, `ghp_…`, `AKIA…`, PEM blocks → 0 hits). `config/key_pool.json` values are
  placeholders, not credentials — but the file is dead config (Medium-10).
* **ruff / mypy / build** genuinely clean; **all 185 modules import**; UI, CLI, SSE, `/metrics`,
  task persistence and the trading *rules engine* all function.
* The trading consult's **numeric rules** (drawdown/loss-streak → bounded parameter change) are sound;
  only the labelling of their source is wrong.

---

## Deferral request

Phase 3 (fix) and Phase 4 (upgrades) are blocked on your go-ahead per the protocol.
My proposed ordering for Phase 3:

1. **CRITICAL-1 + HIGH-5** (fabricated success) — one coherent fix: a single `degraded/failure`
   signal that every response path must honour, plus regression tests.
2. **CRITICAL-2 + CRITICAL-3** (auth) — auth dependency + `old_key` validation + remove decoy params.
3. **CRITICAL-4** (fake metrics) — delete or redirect.
4. **HIGH-6, HIGH-7, HIGH-8** — audit-trail honesty, proxy-aware rate limiting, tz normalization.
5. **Safety net** — tests for `/ask`, `/v1/ask`, `/v1/trading/consult`, governance, scrubber, memory.
6. **MEDIUM** batch — shadowed files, dead-code inventory, docs/openapi, pytest config merge.

**Awaiting your go-ahead before touching any code.**
