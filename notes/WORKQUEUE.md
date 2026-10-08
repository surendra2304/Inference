# WORKQUEUE — source of truth

Rules: an item is checked off ONLY with evidence attached (command run + observed output).
Every new defect gets appended immediately with a measurement. If empty, re-audit.

## Infrastructure state
- venv `/tmp/venv` (rebuilt after the sandbox restart; `pip install -e ".[dev]" ruff mypy httpx`)
- rig: `python scripts/local_model_server.py --port 11434 --slots 4` (needs restart)
- agent: uvicorn `app.main:app` :8000 with `INFERENCE_API_KEY=drive_key APP_ENV=production
  ALLOW_DEV_RATE_LIMIT_BYPASS=true LOCAL_ENABLED=true LOCAL_PREFERRED=true
  LOCAL_BASE_URL=http://127.0.0.1:11434/v1 LOCAL_MODEL=rig-8b REQUEST_TIMEOUT=30`

## Open

- [x] Q7. 6 residual `PytestUnhandledThreadExceptionWarning` (loop ownership). Repro + fix.
      CLOSED in increment 28 — see "Q7 (aiosqlite loop ownership) — FIXED" below for the
      owner-loop design, the 3× suite runs with 0 warnings, and the regression file.
- [x] Q9. "distinct answers 1/3" — CLOSED in increment 30: `mode` is a validated `Literal`, each declared mode reports `mode_used`, and `deliberative`/`consensus` map to `review` with a `mode_mapping_note`; `tests/test_declared_modes_are_honored.py` (5 tests) pins it.

## Closed (with evidence)

- [x] Q6. Direct `get_provider(...).generate(...)` call sites audited. Two bypassed the gateway:
      `app/evaluation/evaluator.py:204` (LLM-as-judge) and `app/experiments/harness.py:175`
      (provider matrix). Both now call `model_gateway.execute(...)`, inheriting per-provider rate
      limiting, key rotation/quarantine, the deadline budget with its fallback reservation, health
      recording and provenance. Live re-check: `POST /experiments` -> 200 in 0.29s, rig +20 calls.
- [x] #54 the evaluator fabricated its own grades. When the judge was unavailable (provider error
      **or** a non-JSON reply), `evaluator.py` substituted six hardcoded scores (0.85/0.90/0.80/0.85/
      0.80/1.0), the strength "Structured output generated successfully", `flaws=[]`, `confidence=0.80`,
      averaged them with the deterministic scores, and returned them as
      `evaluator_model="gemini:gemini-2.5-pro"` — a model that never ran. Now: `judge_ran: False`,
      `judge_error` names the cause, `confidence: None`, `evaluator_model: None`, only the
      deterministic dimensions (`latency`, `usage_efficiency`) are scored, and
      `metadata.partial_evaluation: True` + `overall_score_basis` say so. Live: judge_ran False with
      `JSONDecodeError: Expecting value: line 1 column 2` (the local rig answers the judge prompt
      with prose; the prompt does carry a correct JSON schema contract).
      #54b tolerant parsing added: a JSON object wrapped in prose is recovered (`raw_decode` from each
      `{`) and the recovery is flagged in `judge_error` rather than being silently accepted.
      #54c judge payloads are validated: non-list `scores`, missing `criterion`, non-numeric or
      out-of-range scores (e.g. 1.4) are rejected as unusable instead of raising a deep
      `KeyError`/`ValidationError`; a missing `calibrated_confidence` no longer becomes 0.85.
- [x] #55 the benchmark published a quality verdict it had not measured: `winner: "fast"`,
      `score_difference: -0.15` from latency/efficiency scores alone. Live now: `winner` retained but
      `quality_verdict_withheld: True`, `hypothesis_tested: False`, `verdict_basis: "deterministic
      dimensions only (latency/efficiency); the semantic judge did not run, so no comparison of
      reasoning quality was made"`, and per-side `judge_ran` / `scored_dimensions` / `judge_error`.
- [x] Falsification (#54/#55): `git stash push` of evaluator.py + harness.py ->
      `tests/test_evaluator_and_experiment_honesty.py` **6 failed**; `git stash pop` -> **6 passed**.
- [x] Test-suite integrity defect found and fixed while landing the above: patching
      `"app.evaluation.evaluator.model_gateway.execute"` sets an **instance** attribute, and
      monkeypatch's teardown reinstates a bound method that shadows the class for the rest of the
      session — so `monkeypatch.setattr(ModelGateway, "execute", ...)` in
      `tests/test_multi_agent_self_healing.py` silently stopped working and **6 tests failed only in
      the full run** (they passed in isolation). Now patched on the class everywhere, plus a guard
      test asserting the singleton carries no instance-level `execute`.
- [x] Suite: 359 passed; ruff clean; mypy clean (234 files).


- [x] Q3. Mode validator: `mode` ∈ {`bogus`, ``(empty)``, `FAST`, `"debate "`, `123`, `null`} all -> **422**
      `literal_error "Input should be 'fast', 'review' or 'debate'"` (no silent fallback). Live, 6/6.
- [x] Q4/Q5. 404 triage: none of the guessed paths (`/runs/recent`, `/runs`, `/tasks`, `/stats`,
      `/providers/health`, `/analytics/summary`, ...) appears in the OpenAPI schema (117 paths) or in
      the repo's docs — the only repo hits were my own notes plus `app/providers/health.py` (a module,
      not a route). The real routes exist under prefixes (`/v1/agent/stats` -> 200, `/v1/experiments`
      -> 200, `/tasks/{id}` -> documented). Contract check over the live schema: **118 documented
      operations, 78 x 200, 29 x 422, 11 x 404 (all GET-with-path-param lookups for a dummy id, which
      is correct), 0 documented-but-unrouted**.
- [x] Two durable invariant tests added (falsified): every `APIRouter(prefix=...)` declared under
      `app/` must appear among the resolved schema paths, and every documented operation must not
      404. Falsification: commenting out `include_router(sentinel_router)` -> `F` (failed); restored ->
      pass. `tests/test_api_contract_fuzz.py` is now 80 tests.


- [x] Q2. Pressure wave: 64 req @ c=40 -> 0 x 5xx, 0 transport failures, p95 <= 165ms; 24 req @ c=20 ->
      0 failures. 180s sustained soak @ c=20: **30,072 req, 0 failures, p50 22ms / p95 64ms /
      p99 239ms**. (Pre-fix numbers for the same mix: FD 924->70, throughput 39.7->81.6 req/s.)
- [x] Q8. Recovery, 4 x (30s load + 5s idle) cycles: FD 47 -> 65 under load -> **exactly 47** in
      every idle window; threads constant 7; RSS 151.3 -> 152.0 MB across 10,757 requests, with the
      per-cycle delta shrinking (+0.3/+0.1/+0.2/+0.1 MB). No FD, thread or RSS leak remains.
- [x] #40 unbounded in-process accumulators (the leak). Repro: 60s soak @ c=20 (17,765 req, 0 errors)
      grew RSS 77.9 -> 143.2 MB, still 144.3 MB 3s after the load stopped. Per-route attribution with
      gc.collect() before sampling: **nexus +9.03 MB/1k req (retained +19.7), sentinel +5.19 (+13.7),
      analytics +2.42 (+10.4)**, market/stress/assist ~+0.02 (no accumulator). Root cause: a plain
      {} / [] on a module-level singleton that every request appended to (nexus/sentinel
      provenance_store, debate reasoning_traces, outcome_records, strategy_bank, outcome_history,
      optimization_logs, cost/usage records). Fix: `app/utils/bounded_store.py`
      (BoundedStore/BoundedList/BoundedSeries + a process-wide registry). After: **+7.2 MB over 180s
      and 30,072 requests** (~15x less per request) and the curve decelerates.
- [x] Eviction is disclosed, not silent. `/metrics/runtime` now reports `retention` (store count,
      within_bound, total evictions, stores at capacity) + `process` (rss/fds/threads). Live:
      `within_bound True | stores 10 | process {'rss_mb': 71.0, 'open_fds': 22, 'threads': 6}`.
      Live eviction probe (2,601 unique nexus ids): 2048/2048 slots, 1,106 evictions, within_bound.
- [x] #52 a miss message must not claim "never recorded" for an evicted key. Live: id MSG-FIRST was
      HTTP 200, then evicted; lookup answered "No ... was ever recorded". Cause: the 64-key eviction
      memory. Now: recent eviction -> "expired ... Re-run"; evictions but key unknown -> "either never
      recorded, or evicted (keeps the most recent 2048, has evicted 553)"; nothing evicted -> "never
      recorded".
- [x] #41 seeded demonstration outcomes were counted as observations by the computed statistics
      (`conversion_delta_pct: 14.5` etc.). Fix: `source: Literal["observed","fixture"]` on
      DetailedOutcomeReport, four seeds labelled `fixture`, every statistic reports
      `records_observed` / `records_fixture_excluded` and computes from observed rows only.
- [x] #42 fields nobody reported were recorded as measurements: `provider_used="gemini"`,
      `task_type="code_generation"`, `agent_composition=["strategist","critic"]`,
      `stated_confidence=0.85`, `detail="verification_passed"`. All now default to None and the
      statistics skip them. Same for the legacy `OutcomeReportRequest`.
- [x] #43 nexus/sentinel outcomes were relabelled "forge" on the way into the legacy tracker, and a
      missing provider/service was filled with "gemini"/"code_generation" (admin_analytics.py:64-69).
      Fix: pass through unchanged; only the four native consumers are forwarded.
- [x] #44 fabricated 90% provider prior, served live. `outcomes.py:47 prov_quality[p] = 90.0` and
      `outcome_learning.py:157 {"success_rate_pct": 90.0, "total_samples": 0}` -> `success_rate_pct:
      null` + `samples: 0` + `insufficient_data`. Live `/v1/admin/dashboard`: all five providers null.
      #44b `self_optimizer.py:37-38 pass_rates.get("groq", 90.0)` used those priors to **set live
      routing weights**; now a missing rate skips the rebalance and logs REBALANCE_SKIPPED.
      #44c the router's log was pre-seeded with "98% verification pass rate observed from FORGE
      builds"; removed (test asserts "98%" absent).
- [x] #45 four hardcoded "cross-consumer patterns" ("verification improves accuracy 23%...",
      "research synthesis succeeds 91% but attack path reasoning only 71%...") -> computed from
      observed reports, each carrying its sample, withheld entirely below 20 reports per task type
      with `patterns_insufficient_data`.
- [x] #46 calibration curve was four literal rows labelled `HONEST_CALIBRATED` -> computed per bin
      from reports carrying `stated_confidence`; <10 samples -> INSUFFICIENT_DATA with nulls;
      otherwise CALIBRATED / OVERCONFIDENT / UNDERCONFIDENT from the measured delta.
- [x] #47 `95.0 if c == "futuris" else 94.0 ... 92.0` consumer rates for consumers with **zero**
      evaluations -> `success_rate_pct: null`, `alert_status: NO_DATA`.
- [x] #48 `self.daily_average_cost_usd = 0.50` literal driving cost-anomaly alerts -> computed
      `daily_average_cost_usd()` over observed rows (None when there are none).
- [x] #49 cost leaderboard invented `p_cost = 0.001` + 1 success for providers with no records,
      producing a concrete $/success and "PROMOTE_PRIMARY" -> NO_DATA, null cost, excluded from
      ranking; seeds labelled `fixture` and excluded from spend.
- [x] #50 zero-as-measurement: `success_rate_pct: 0.0` for a consumer with zero calls and
      `avg_latency_ms: 0.0` with no samples (`max(1, total_calls)`); also `confidence: 0.90` default
      on every analytics record. Now None + a note.
- [x] #51 `record_outcome` invented a strategy-bank record: `success_rate=0.90` for every success,
      `recommendation=f"Recommended actions for {task_type}"` (advice nobody gave), and
      `pattern_id=f"PAT-{int(time.time())}"` (same id for every success in the same second). Now:
      rate taken from measured_metrics or None, recommendation None, `PAT-<ms>-<seq>` unique.
- [x] #51b `query_strategy_bank` reassigned `self.strategy_bank = [e for e in ...]`, replacing the
      bounded store with a plain list and disabling the bound. Expiry is now applied at read time.
- [x] #53 (self-inflicted, caught by the suite) `get_routing_status()` returned the BoundedList
      object -> not JSON-serializable, the status endpoint would fail at render. Now `.to_list()`
      plus a retention block.
- [x] Falsification of this cluster: `git stash push` of outcome_learning.py / outcomes.py /
      self_optimizer.py / usage_analytics.py / cost_tracking.py / admin_analytics.py ->
      `tests/test_bounded_stores_and_analytics_honesty.py` **14 failed, 12 passed** (the 12 that pass
      are the new primitives, which did not exist before); `git stash pop` -> **26 passed**.
- [x] Suite: 350 passed; ruff clean; mypy clean (234 files).

- [x] Q1. Drove 16 real operator tasks (`/v1/debate`, `/debate`, `/v1/trading/consult`, `/v1/friday/*`,
      `/v1/instant/ask`, `/v1/task/execute`, `/v1/forge/batch-generate`, `/experiments`,
      `/v1/agent/assist`, `/v1/providers/execute`, `/v1/nexus/*`, `/v1/sentinel/analyze`,
      `/v1/intelx/research`, `/v1/market/*`, `/v1/trading/live/stress-test`).
      Evidence: `16 real tasks driven | non-200: 2` (both strict enums with my payloads, 422 correct);
      `200-with-no-model-and-no-prose: 0`; rig deltas 20/11/8/2/1 per model-backed route.
- [x] #35 fabricated debate measurements. Repro: clean `total_debates = 24`; stats published
      `objection_rate_pct: 24.5`, `93.8 / 81.2 / diversity_lift 12.6`, compositions `94.2/92.8/91.5`.
      After: counters start at 0, rates derived with sample sizes, `insufficient_data` until n>=20,
      `provider_diversity_impact: null` + reason. Live `/v1/debate/statistics` -> `total_structured_debates: 0`.
- [x] #35b `confidence_evolution=[0.82,0.76,0.88,0.86]` hardcoded -> derived from evidence (0.088 no
      evidence / 0.256 untrusted / 0.928 verified / 0.930 system_fact).
- [x] #35c "(4 rounds executed)" literal contradicting `rounds_conducted` -> prose == field (4 == 4).
- [x] #35d `[GEMINI]`/`[GROQ]` vendor tags on text no provider produced -> no vendor tags;
      `provider_allocation_exercised: False`, `model_calls: 0` on the trace.
- [x] #35e canned rounds ("constant market liquidity", "dynamic stop boundaries") -> derived from the
      weakest evidence item; hardcoded objection -> 0 objections when nothing is unresolved.
- [x] evidence ids were renumbered `EVD-001` -> caller ids preserved (`ev-index-drop`, `ev-user-note`).
- [x] #36 fictional precedent: `query_strategy_bank` fallback served PAT-001 (SDR sales) for a
      Postgres question -> unmatched queries return `[]`; fixtures labelled `illustrative_fixture`
      and cited as "Illustrative fixture (not an observed outcome)".
- [x] `compute_agent_composition_performance` hardcoded 78.4/91.2/92.5/90.8/94.0 -> computed from
      `outcome_records` with `sample_size` + `insufficient_data`.
- [x] Falsification: `tests/test_no_fabricated_measurements.py` = 13 tests; all 11 originally-written
      fail against pre-fix code (verified via `git stash push` on the three files) and pass after.
- [x] Suite: 324 tests pass; ruff + mypy clean (233 files).


- [x] #22 unauthenticated admin endpoints → `nokey 401 / key 200`
- [x] #23 provider semaphore leak on cancellation → falsified: pre-fix `DEAD`, shipped `usable`
- [x] #14 orphaned `running` tasks → startup log `Reconciled N task(s)`
- [x] panel budget bounds the specialist phase → `503 at 20.01s` (was 30.01s)
- [x] quota contradiction `human` 50/h vs 120/min → settings-driven, `HUMAN_RATE_LIMIT_PER_HOUR`
- [x] #24 unknown experiment_type 500→400
- [x] #25 `/v1/trading/consult` published no request body → 12 properties, 0 dangling refs
- [x] #26 speculative race reported a vendor that never ran → `provider=local, served_by=local`
- [x] #27 `KeyError` on empty portfolio → single contract, empty portfolio 422
- [x] #28 `ZeroDivisionError` + hardcoded `PASSING` → derived rating, 0 divisions by zero
- [x] #29 cross-endpoint dedup collision → namespaced keys, typed lookup
- [x] #30 friday debate outage 500→503; #31 non-finite floats 500→422 (all routes)
- [x] #32 security refusal 500→400; #33 degenerate hex false positive
- [x] 40-route degenerate sweep: 0 5xx; 341-request malformed sweep after fix: 0 unhandled 5xx
- [x] suite 311 passed; ruff + mypy clean (233 files)

## Increment 28 — Q7 closed; #56/#57 found by live driving and fixed

- [x] **Q7 (aiosqlite loop ownership) — FIXED.** One process-wide owner loop
      (`_DbExecutor`, daemon thread `sqlite-db-loop`) owns every connection; every statement,
      cursor method and `close()` is submitted there (`app/memory/sqlite.py`). Abandonment
      closed deterministically by a `weakref.finalize` hook (`_schedule_owned_close`).
      Evidence: full suite **365 passed** ×3 consecutive runs with **0**
      `PytestUnhandledThreadExceptionWarning` (was 2–3 per run); no-close probe
      (`/tmp/drive/loop_repro4.py`, 5× `asyncio.run` without `close()`): **0 thread
      exceptions, 0 warnings, 0 stranded workers**; `loop_repro2.py` abandoned file-pool
      scenario: 0 exceptions, only `sqlite-db-loop` alive. Regression file
      `tests/test_sqlite_loop_ownership.py` (6 tests) with baseline-relative worker
      assertions.
- [x] **#56 unbounded `Orchestrator._recent_tasks` — FIXED.** Plain dict, written on every
      task, never pruned. Measured in-process (`/tmp/drive/leak_attribution.py`): 400
      `POST /ask` → 400 entries, RSS +10.9 MB (**27.9 KB/request**, unbounded). Now a
      `BoundedStore("orchestrator.recent_tasks", max_entries=1024)`; durable history stays in
      SQLite and `get_task_status` still resolves evicted ids. After: 1,600 requests → store
      pinned at exactly **1024**, first 800 cost +14.4 MB and the second 800 only **+1.7 MB**
      (residual slope 2.1 KB/req). Miss wording distinguishes never-recorded from evicted
      (`missing_entry_detail`).
- [x] **#57 `POST /v1/debate` handed out an unresolvable task id — FIXED.** The route drives
      the collaboration engine directly, so nothing was persisted or registered: measured live
      `POST /v1/debate` → 200 `task_53017bc07162`, `GET /tasks/task_53017bc07162` → **404**
      "Task ... not found." All three return paths now call `orchestrator.record_task(...)`.
      After: `POST /v1/debate` → 200, read-back → **200** (`status=completed mode=debate
      confidence=0.8 result 1178 chars`).
- [x] Falsification: restoring the old shapes makes both assertions fail —
      plain dict holds **1074** entries (cap 1024) and a no-op `record_task` reproduces the
      **404** read-back. Regression file `tests/test_task_record_resolvability.py` (3 tests).
- [x] Suite 365 passed; ruff + mypy clean (234 files).

## Increment 29 — prediction/monitoring honesty, dedup cache ceiling, BoundedStore protocol

- [x] **#58 fabricated prediction analytics — FIXED.** `app/ml/prediction_tracking.py`
      published `lstm_transformer_model: 78.4`, `news_sentiment_nlp: 71.2`,
      `onchain_whale_signals: 81.5`, `technical_momentum: 69.8`,
      `calibration_score_0_to_1: 0.88`, `accuracy_status: HIGH_CONFIDENCE_PRODUCTION_READY`
      over four hardcoded predictions (fallback `75.0`), and the class had **no method to
      record a prediction**. Now: `record_prediction`/`resolve_prediction`,
      `MIN_OBSERVATIONS_FOR_SCORE=20`, per-source scores with `sample_size`, Brier-based
      calibration, `INSUFFICIENT_DATA` + `overall_directional_accuracy_pct: None` until the
      floor is met, fixtures labelled `source: "fixture"` and counted in
      `fixture_rows_excluded`. `GET /v1/predict/{asset}` accepts `current_price` /
      `recent_returns`; without them `inputs_simulated=true`, `evidence_class:
      synthetic_fixture_inputs`, **confidence withheld**; the old
      `max(0.60, 0.70 + |composite|*0.25)` floor is gone.
      Measured before → after (same script, `/tmp/drive/pre_post_honesty.py`):
      `{75.0, HIGH_CONFIDENCE_PRODUCTION_READY, 0.88}` → `{None, INSUFFICIENT_DATA, None}`;
      sub-models `{78.4, 71.2, 81.5, 69.8}` → `{}`; `/v1/predict/btc` confidence `0.95` →
      `None` with `inputs_simulated: true`; history labels `[None]` → `['fixture']`;
      `/v1/intelligence/accuracy` contains "78.4"/"0.88" → **False**.
- [x] **#59 fabricated monitoring metrics + malformed Prometheus line — FIXED.**
      `get_debate_metrics()` dropped the constant `recommendation_quality_score: 100.0`
      ("Certified perfect 100/100 by audit suite") for `None` + `recommendation_quality_basis`,
      and `total_deliberations` no longer means "every HTTP request" (measured:
      `100.0` → `None`, `total_deliberations` 0 until a real deliberation is recorded).
      `get_api_metrics()` with no samples reported `error_rate_pct: 0.0` and `p50..p99: 0.0`
      without a sample count → now `None` + `samples: 0` + `basis`. The exporter then
      emitted the literal text `None` (`inference_error_rate_percent None`, measured) —
      a malformed exposition line; `None` values are now omitted (Prometheus convention) and
      `inference_api_latency_samples` publishes the window size. `record_request` had exactly
      one caller (trading consult), so the numbers did not describe the process they were
      reported for; a new `app/middleware/request_metrics.py` records every route once and
      the double-counting calls were removed.
- [x] **#60 dedup cache invisible to the audit and oversized — FIXED.** It held a full
      response payload per caller-supplied `request_id`, was the largest retained store, and
      appeared in no retention report (measured: 12 registered stores, none of them this one).
      Now a registered `BoundedStore` with a `DEDUP_CACHE_MAX_ENTRIES` ceiling (1000),
      `describe_dedup_cache()`, and an eviction-aware miss detail. Measured on 12,000 unique-id
      `POST /v1/nexus/intelligence` requests, same driver: RSS ceiling **123.4 MB → 113.8 MB**,
      `retention.store_count` 12 → **13**, `governance.dedup_cache` now listed at
      `1000/1000 used, 11,000 evicted`.
- [x] **#61 `BoundedStore` protocol gaps — FIXED** (found by these edits): no `__delitem__`
      (the dedup cache does `del cache[key]`), no `__iter__` (`any("stale-" in k for k in
      store)` fell back to the *sequence* protocol and raised `KeyError: '0'`), and no
      settable ceiling (`mgr.max_dedup_entries = 25` was silently ignored — 300 entries kept
      under a 25 cap). Added all three; the ceiling setter trims immediately, and
      `max_dedup_entries` is a property delegating to the store so the attribute cannot lie.
- [x] **#62–#65 still open** (cost-tracking invented prior + `0.001` defaults, usage-analytics
      `confidence: 0.90` default, self-optimizer seeded strategy bank/unbounded logs,
      fallback-executor unbounded `history`).
- [x] Suite: **386 passed**; ruff clean; mypy clean (235 files). New tests:
      `tests/test_prediction_and_monitoring_honesty.py` (13), `tests/test_declared_paths_are_served.py`
      (3, incl. a falsified detector), `tests/test_task_record_resolvability.py` (3),
      `tests/test_sqlite_loop_ownership.py` (6).

## Increment 30 — Q9 closed; #67 declared-mode honesty; #68 no default confidence; #66 alt-data provenance

- [x] **Q9 "distinct answers 1/3" — FIXED.** Root cause: `POST /v1/ask`'s `mode` influenced
      exactly one line (`fast_lane=(request.mode == "fast")`, `app/api/v1_core_routes.py`),
      so `deliberative` / `consensus` / `review` all returned the same single-call answer
      with no field saying which mode ran, and `{"mode": "banana"}` was answered **HTTP 200**
      (`app/schemas/v1_models.py:113` declared the field as plain `str`). Measured live
      before → after, same question: modes `fast/deliberative/consensus/review` all
      `agents=2 conf=0.65 mode_used=absent` → `fast` = single call `mode_used="fast"`,
      `review` = `mode_used="review"` (2 agents, 0.83), `debate` = `mode_used="debate"`
      (**5 agents**, 0.80), `deliberative`/`consensus` = mapped to `review` with
      `mode_mapping_note`, `banana` = **422 literal_error**. Regression file
      `tests/test_declared_modes_are_honored.py` (5 tests, all 5 fail against the stashed
      pre-fix endpoint). Consistent within a mode (3/3 identical answers per mode: the
      determinism half of Q9 was already sound).
- [x] **#67 mode validation** — `InferenceAskRequest.mode` is now a `Literal` of the five
      documented values (was `str`, validated by nothing).
- [x] **#68 no default confidence** — `InferenceTaskResponse.confidence` defaulted to
      **0.90** and `uncertainty` to **0.10**, so a response that omitted them published a
      claim nobody made; both are now required and all 8 construction sites pass explicit
      values (asserted by test).
- [x] **#66 alternative-data provenance — FIXED.** `app/data/alternative_data.py` is a static
      table (same numbers for every asset) served by `GET /v1/intelligence/summary` as a live
      snapshot; it now declares `evidence_class: synthetic_fixture`, `inputs_simulated: true`,
      `asset_specific: false` and its generator. The aggregator reports `synthetic_legs` /
      `synthetic_leg_count`, and a caller-supplied price/return pair is disclosed as
      `caller_supplied_price_and_returns_with_synthetic_alt_data` (measured live: confidence
      **0.59**, evidence class set, 4 synthetic legs) instead of an unqualified number.
      `app/ml/deep_models.py` now publishes a `calibration` note (parametric, not fitted).
- [x] **#65 fallback-executor attempt history bounded** (`BoundedList`, 2000) — one record per
      provider attempt with no reader anywhere.
- [x] Suite **394 passed**; ruff clean; mypy clean (235 files).

## Increment 32 — metrics unification (#69), analytics honesty (#70), live routing weights (#71); leak open

- [x] **#69 Prometheus export: one renderer, and the public endpoint stops emitting `None`.**
      [FACT] The two exporters had drifted apart: `/metrics` (`app/health.py`) published
      `inference_requests_total` / `inference_latency_p50_seconds` while
      `/v1/governance/prometheus-metrics` published `inference_api_requests_total` for the same
      counter and `inference_request_duration_seconds{quantile="0.50"}` for the same percentile;
      the `#69` None-skip fix existed only in the public one. Measured pre-fix public output:
      `{'inference_requests_total': '4', 'inference_latency_p50_seconds': '0.03', ...}` (a
      scraper-rejected `... None` line whenever a series had no samples).
      FIX: `app/observability/prometheus.py::render_prometheus(include_provider_detail=...)`,
      used by both endpoints; absent series omitted, every line validated against the exposition
      grammar, provider/circuit/health families only on the authenticated endpoint.
      EVIDENCE: `tests/test_prometheus_metrics.py` 13 passed, including
      `test_both_metrics_endpoints_publish_the_same_names_and_values` (values compared across
      endpoints, tolerance only for the scrape itself) and
      `test_public_metrics_hides_provider_identity`. FALSIFIED: with
      `git show HEAD:app/health.py` restored, the new test fails with
      `inference_requests_failed_total missing from the public /metrics endpoint`. Commit 022f4f4.
- [x] **#70 usage analytics recorded invented provider attribution and spend.** [FACT] All four
      callers (`nexus_intelligence.py:321`, `intelx_intelligence.py:381`,
      `sentinel_intelligence.py:353`, `futuris_enhancement.py:214`) passed
      `provider="gemini"` with fixed `tokens_in/out` (350/300, 350/250, 300/250, 300/200) while
      none of those modules contains a single gateway call site (`grep -c "model_gateway\."` = 0
      for each); `log_request` derived `cost_usd` from those invented counts, and the only
      service that really calls a provider (`trading_consult_service.py:202`) logged nothing at
      all. FIX: provider/tokens/confidence/cost are optional and stored as `None` when not
      measured; `cost_usd` is computed only from measured tokens; `unmeasured_fields` names what
      is missing; `get_overview` publishes `measurement_coverage` + `unmeasured_calls` and
      `budget_basis`; `get_providers_comparison` no longer returns seven
      `calls: 0, tokens: 0, cost_usd: 0.0` rows; `forecast.probability or 0.85` no longer
      substitutes a measured-looking confidence; trading-consult now logs the real
      `ProviderResponse` (provider, prompt/completion tokens, latency); `RunRecord` token counts
      are nullable instead of `0` for unknown. EVIDENCE: live probe — empty store reports
      `total_tokens/total_cost_usd/budget_used_pct = null` with
      `budget_basis: "no observed request reported token usage; spend cannot be stated"`; one
      latency-only log records `unmeasured_calls {tokens: 1, cost: 1}` and the provider table
      shows `unattributed` only. Commit 022f4f4.
- [x] **#71 `SelfOptimizingRouter` published weights and constraints that nothing read or
      enforced.** [FACT] `provider_weights` had no consumer except the endpoints that display
      it (`app/routers/admin_analytics.py:115`); the advertised
      `diversity_constraint` ("min 2 active providers, cap 0.70") was not applied by the router's
      own adaptation (the clamp ran *before* normalisation, so two providers could sit at 0.70
      while a third kept its share and the service's weights summed above 1); and an open
      circuit in `trading_consult_service` went straight to deterministic fallback text with
      every other provider idle. FIX: `choose_provider()` (weighted + bounded exploration +
      explicit `method` + decision counters), `normalize_weights()` enforcing cap-after-normalise,
      `adapt_weights_from_observed_health()` requiring ≥5 measured calls and reporting
      `NO_SUFFICIENT_EVIDENCE` instead of crediting a no-op adaptation, and the trading-consult
      circuit-open path now re-routes by weight and records the substitution.
      EVIDENCE: `tests/test_self_optimizing_router.py` 8 passed (cap/renormalisation, weighted
      distribution within 0.05, exploration ≈10% ±3, unlisted provider reachable, decision
      accounting, evidence threshold); live status payload now carries `effective_weights`,
      `decisions` and `applied_to_request_routing: true` with the exact wiring named.
      Commit 4218d0c.
- [x] **LEAK — CLOSED: it was not a leak.** The 40,000-request soak settles it: RSS
      81.4 -> 98.7 MB while the bounded stores filled, then 26 consecutive rounds (26,000
      requests) with **0.0 MB growth**, every growing store at its ceiling and
      `within_bound: true`. The apparent 1.4 KB/request was the usage-analytics store
      (10,000 records) still filling; in-server tracemalloc showed growth collapsing from
      ~1,500 KiB per 1000 requests to 40-75 KiB per 1000 the moment it saturated. The probe
      rule that produced the false alarm is fixed (a verdict now refuses to judge while a
      grown store has capacity) and pinned by 9 tests. Original measurements retained below.
      --- [FACT] 6,000 unique-id `POST /v1/nexus/intelligence` at
      concurrency 8: RSS 102.8 → 108.7 → 119.8 → 122.2 → 123.7 → 125.2 → 126.5 MB; per-1000
      deltas `[5.9, 11.1, 2.4, 1.5, 1.5, 1.3]` MB; evictions 348 → 13,254 with 4 stores at
      capacity and `within_bound: true`; fds 49, threads 11. Post-saturation growth ≈1.4 KB
      per request ≈36 GB/day at 300 rps. [FACT] The series *decays*, so the first verdict rule
      (trend-only) called it a plateau — the rule was wrong, not the measurement;
      `scripts/leak_probe.py` now also enforces a 0.5 KB/request budget and classifies the
      measured series as `leak` (1.43 KB/request, 38.1 GB/day at 300 rps), pinned by
      `tests/test_leak_probe_verdict.py`. Attribution via the opt-in
      `GET /memory/diagnostics` (in-process tracemalloc + snapshot diff) points at
      `usage_analytics.py:62` (+1187.5 KiB / +5000 objs) and
      `nexus_intelligence.py:309/310/316` (+554.7/+1164.0/+824.2 KiB), i.e. the bounded stores
      filling — which cannot by itself explain sustained growth after 13k evictions, so the
      leak is either retained frame/response state or a store that is not actually bounded.
      NEXT: 12,300-request probe verdict, then fix the named site and re-measure flat RSS.

## Increment 31 — real-life drive hardening: hidden-state bug, error-text leaks, allocator truth

Every item here was found by *driving the agent*, not by reading it. Commit chain:
`405f593` (schema/no-leak/guard) -> `e18844b` (memory verdict) -> `733f537` (error text)
-> `bd60df7`/`75f10fe` (rate limit + fault harness).

- [x] **Hidden state: `/v1/trading/consult` returned HTTP 500 ``no such table: tasks`` on a
      clean database.** `TradingConsultService` builds its own `SQLiteMemory` and nothing
      ever called `initialize()`; the endpoint worked only where some other component had
      already created the schema. Exposed by deleting `data/universe.db` (the harness had
      been passing on a database created by an earlier server run). Fixed in
      `SQLiteMemory.connect()`: `ensure_schema()` creates missing tables once per instance
      before handing out a connection; `initialize()` forces the check. `_create_schema` is
      shared DDL. Pinned by `tests/test_memory_self_heal.py` (8 tests: clean-file writes,
      reads, once-per-instance DDL, forced re-check, the failing route, and that a store
      failure does not return 500 with internals).
- [x] **Internal exception text published to clients — 5 sites.** ``detail=f"... {exc!s}"`` on
      `/ask`, `/v1/ask` (twice: panel-unavailable and deliberation-failed, where the raw text
      was interpolated into ``answer``), `/v1/debate` (``answer`` *and* ``failure_state``), and
      the FRIDAY gateway. Measured in tests: ``password=s3cr3t-marker-2b1c`` and
      ``/srv/private/db.sqlite`` came back in response bodies. `app/utils/errors.py` now owns
      the rule: `internal_error()` logs with a correlation id and returns only the id;
      `unavailable_detail()` publishes specialist ids and failure *kinds*. The engine records
      the exception type explicitly (`app/agents/debate.py`) so the 503 stays informative.
      A legacy assertion that *required* the gateway's raw text ("simulated outage") in the
      503 was rewritten to the safe form of the same intent. 21 tests:
      `tests/test_internal_errors_never_leak.py`.
- [x] **The same text leaked one request later through `GET /tasks/{id}`** — `ask_v1` stored
      `{"error": str(exc)}`, `get_task_status` returns `model_dump()`, and the route publishes
      it. Metadata is now scrubbed at write time *and* on read (`scrub_credentials_dict`).
- [x] **The scrubber missed bare provider key literals** — a rejected key arrives as
      "Invalid API key: sk-...", not as `api_key=sk-...`. `scrub_credentials` now redacts
      `sk-`/`sk-or-v1-`/`gsk_`/`hf_`/`nvapi-`/`xai-`/`pplx-`/`cohere-`/`AIza`/`r8_`/
      `mistral-` literals, and the tests assert both directions: keys are redacted, and
      benign text ("task-1234567890abcdef", "risk-assessment-of-the-write-path", "sk-learn")
      is not mangled. The named-field minimum dropped 16 -> 8 characters, because a short
      password is still a password.
- [x] **Allocator high-water, measured to the end.** RSS-only verdicts were wrong in both
      directions: (a) 1,600 `/ask` requests grew RSS 1.3 MB while `sys.getallocatedblocks()`
      returned exactly to baseline (403,456 rest -> 698,136 load -> 404,371 idle); (b) a sample
      1 s after a burst read +602,105 blocks and 40 s later read *below* its pre-burst value.
      `app/utils/memory_guard.py` returns freed arenas (`malloc_trim`) only when the process is
      idle and above the threshold, and it works: 102 trims reclaimed **222.5 MB** during one
      full drive, 293 MB during the mixed soak — while the same guard reclaimed 0.0 MB on a
      light burst (glibc had no whole arena free). Both numbers are in the module docstring.
      Two bugs in the guard itself were found by driving it: it ignored `MEMORY_TRIM_*`
      entirely (reported threshold 400 when configured 80) and its
      "consider a trim every 25 completions *while idle*" gate made it dead code under
      concurrency (1,500 requests, `trims: 0`).
- [x] **The drive's memory verdict is now object retention**, not RSS: a final
      `check_memory_retention` runs after every other check, polls until the block count holds
      still across four readings, and reports `inconclusive` if it never does. Full drive:
      **12/12 checks**, blocks 559,541 -> 559,545 (**+4**) over ~3,600 mixed requests.
      `check_resource_ceiling` keeps the store audit and the RSS arithmetic and defers the
      retention verdict when the counter is available (with the reason written in the code).
- [x] **Per-bot rate limit was a frozen constant with no `Retry-After`.** `/v1/trading/consult`
      answered 92 x 429 per 60 s soak round (every attempt) under a hardcoded 20/hour, and the
      only headers present described the *global* limiter. Now reads
      `TRADING_CONSULT_RATE_LIMIT_PER_HOUR` (default unchanged), answers `Retry-After` from the
      oldest timestamp in the window, names the setting in the message, and keeps
      `RATE_LIMIT_MAX_REQUESTS` working for existing callers (the first version of the change
      broke `tests/test_bounded_state.py`, which imports it — caught by the suite, fixed in
      `75f10fe`).
- [x] **Failure injection, 14/14 honest** (`scripts/fault_injection.py`, second agent on :8001
      against a rig that can be told to fail): healthy -> outage (every call 500) -> slow (3s
      each) -> past-deadline (client timeout) -> recovery, across `/v1/friday/ask`, `/v1/ask`,
      `/v1/debate`, `/v1/trading/consult`. Outage: 503 or in-band DEGRADED/ERROR at confidence
      0.0. Slow: 12.2s, same honesty. Deadline: answered at 20.0s rather than hanging.
      Recovery: immediate. The rig gained `{"*": ...}` in `fail_for` so "everything is down"
      is expressible without killing the process.
- [x] **The fault harness had three bugs of its own, all fixed and documented:** it judged a
      healthy SUCCESS as dishonest; it mistook a dedup cache hit for a working provider
      (identical question text returns the first answer, so "outage" scenarios echoed the
      baseline); and it set `hang_seconds` without selecting the hang fault, injecting nothing.
- [x] Suite: **472 passed**, ruff clean, mypy clean (239 files).
