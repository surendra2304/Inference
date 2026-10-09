# AGENT_PROGRESS — Inference repo (session `arena/f706c17c-inference`)

Source of truth for this session. Read this first on resume. Restated from `notes/WORKQUEUE.md`
(increment-30 block, 69 items) + the user's standing task.

**Task:** make the agent actually work under real operator usage — drive it with real tasks,
stress it to dead ends, find and FIX every bug surfaced, write substantial new code that
upgrades capability, keep real-life regression tests, keep going until the queue is empty with
evidence attached to every item.

**Current step (operator-tour phase):** items 11-15 done; item 16 blocked on a real model; next is the queue in `notes/WORKQUEUE.md` (startup LOCAL_ENABLED warning, per-request WARNING collapse, 429 banner, harness.py leak). **Earlier step:** items 1-7 done; item 8 (#29 cross-endpoint dedup collision) is the next
queue item; then the Q7 citation check and the final sweep. The tree is green at commit
`e18844b` (**450 passed**, ruff clean, mypy clean, real-time drive **12/12**).

## Checklist

- [x] 1. The "RSS leak" was investigated to a conclusion: **not a leak**. Evidence: a
      40,000-request soak (40 x 1000 unique-id `POST /v1/nexus/intelligence`, concurrency 6)
      grew RSS 81.4 -> 98.7 MB while the bounded stores filled and then **26 consecutive rounds
      (26,000 requests) at exactly 0.0 MB**, with every growing store at its ceiling
      (`/memory/stores`: unfilled 0 for governance.dedup_cache, nexus.provenance_store,
      usage_analytics.records) and verdict `plateau`. The earlier ~1.4 KB/request reading came
      from measuring before the largest store (usage analytics, 10,000 records) had saturated;
      an in-server tracemalloc diff showed the collapse directly (1,500 KiB/1000 requests while
      filling -> 40-75 KiB/1000 after). The measurement bug that produced the false alarm is
      fixed in `scripts/leak_probe.py` (saturation precondition + per-request budget) and pinned
      by `tests/test_leak_probe_verdict.py` (9 tests). Cost: ~17 MB of bounded footprint at full
      occupancy, documented in `app/utils/bounded_store.py`.
- [x] 2. `/metrics` (#69) regression test + falsify. DONE: `app/observability/prometheus.py`
      renders both exporters; `tests/test_prometheus_metrics.py` now has 13 tests including
      `test_both_metrics_endpoints_publish_the_same_names_and_values` and
      `test_public_metrics_hides_provider_identity`. FALSIFIED: with the pre-#69
      `git show HEAD:app/health.py` restored, the new test fails with
      "inference_requests_failed_total missing from the public /metrics endpoint". 403 passed,
      ruff+mypy clean. Commit 022f4f4.
- [x] 3. `resource_ceiling` hardened: final-round <=2MB, tail mean <= half head mean, and a
      0.5 KB/request budget; plus `scripts/leak_probe.py` with
      `tests/test_leak_probe_verdict.py` (7 tests, includes the measured
      `[5.9,11.1,2.4,1.5,1.5,1.3]` series which trend-only logic called a plateau).
- [x] 4. Append #69 + leak findings to `notes/WORKQUEUE.md`. DONE in `65c1c50`/`7b9b59f`; the
      leak entry is marked CLOSED with the 40k-request plateau evidence.
- [x] 5. usage analytics honesty. DONE (bigger than the original item): `log_request` no longer
      invents provider/tokens/confidence/cost; all four services that passed
      `provider="gemini"` + fixed token counts (none of them calls a provider) now pass
      unmeasured=None; `trading_consult_service` logs the real `ProviderResponse` provider /
      token counts / latency; `RunRecord` tokens nullable; aggregates publish
      `measurement_coverage` + `unmeasured_calls`; `get_providers_comparison` no longer emits
      seven zero-rows that look like providers that spent nothing.
- [x] 6. self_optimizer. DONE (bigger than the backlog note): `choose_provider()` performs real
      weighted selection with a bounded exploration budget and reports how each decision was
      made; `normalize_weights()` enforces the published 0.70 cap *after* normalising (the old
      inline clamp let shares sum above 1); `adapt_weights_from_observed_health()` needs >=5
      measured calls and answers NO_SUFFICIENT_EVIDENCE otherwise; `trading_consult_service`
      re-routes by weight when the preferred provider's circuit is open instead of falling
      straight to deterministic text. 8 tests in `tests/test_self_optimizing_router.py`;
      411 passed, ruff+mypy clean. Commit 4218d0c.
- [x] 7. #62–#65 backlog items. #62 fixed (`65c1c50`), #63/#64 verified fixed with new
      regression tests (the #63 test fails against the old `confidence: float = 0.90`
      default - falsified by restoring it), #65 fixed earlier. #29 is item 8.
- [x] 8. #29 cross-endpoint dedup collision. Already fixed and pinned end-to-end by
      `tests/test_api_contract_fuzz.py::test_cross_endpoint_cache_collision_is_impossible_over_http`
      (same `request_id` to `/v1/nexus/intelligence` then `/v1/sentinel/analyze`; the second
      endpoint answers with its own schema), backed by `TenantManager._dedup_key(request_id,
      namespace)` at `app/governance/tenant_manager.py:187`. Test passes on the current tree.
- [x] 9. Q7 citation check. `app/memory/sqlite.py` owns every aiosqlite connection on one
      process-wide loop (`_DB_EXECUTOR`, `:128`, `loop`/`submit`/`run` at `:99-121`) and
      `tests/test_sqlite_loop_ownership.py` passes (6 tests) with no thread-exception warnings.
- [ ] 10. Final sweep: full pytest + ruff + mypy + real-life harness green.

## Operator-tour phase (real tasks, not green tests)

Driven by `scripts/operator_tour.py` (10 real operator tasks through HTTP). Each item below was
REPRODUCED by a tour or direct request, fixed at root cause, falsified against the old code, and
verified live.

- [x] 11. FORGE `generate-code` returned prose as `code` at confidence 0.92. Root cause: a
  constant (`app/services/code_generation.py`, `0.92 if gen_path == "agent" else 0.55`) and the
  syntax validator never being called. Also the fence extractor returned the prose when a fence
  followed a sentence (measured: `'Here it is:'`). FIXED: `extract_code()`, validator-driven
  confidence, `generation_path="invalid_output"` with empty code, not cached.
  Test: `tests/test_forge_output_is_verified.py` (13 tests). Falsified: 13/13 fail on `HEAD`.
- [x] 12. `GET /v1/admin/analytics/quality` served hard-coded figures (96.4, 99.2, WELL_CALIBRATED,
  agent scores) as measurements. FIXED: `app/services/quality_assurance.py` reports only measured
  counts; everything else is `null` with `status: not_measured`. Same test file.
- [x] 13. FORGE `review-code` approved SQL injection at confidence 0.91 (verdict defaulted to
  approve; model output never parsed; summary claimed a "panel consensus" with one provider call).
  FIXED: `app/services/code_review.py` rewritten: AST analysis on the full source, JSON model
  review that must validate, `approve` only with a parsed model review and no medium+ finding,
  taint tracking through local variables. Test: `tests/test_forge_code_review_is_evidence_based.py`
  (30 tests). Falsified against `HEAD`: 30 of 30 fail; with the fix, 30 of 30 pass. Shared parser: `app/utils/model_json.py`.
- [x] 14. FORGE `debug` returned a fixed template as `root_cause`, a fixed `fix_strategy`,
  confidence 0.89 with no model, and prose as `patch_code`. FIXED: `app/services/debugging.py`
  requires validated JSON; otherwise `diagnosis_status` is `unparsed`/`unavailable`, confidence 0,
  no patch. Patches are syntax-checked when the context is Python, else `not_checked`.
  Test: `tests/test_forge_debug_is_evidence_based.py` (10 tests). Falsified: 10/10 fail on `HEAD`.
- [x] 15. Operator tour body/check bugs (mine, not the agent's): wrong enums and required fields
  (schema-derived now), and a leak check that counted the caller's own echoed text. FIXED in
  `scripts/operator_tour.py`. Honest-limited passes are labelled, not hidden: this sandbox's rig
  answers every request with canned prose, so code quality cannot be verified here (see the
  evidence log).
- [ ] 16. Real-model verification of code generation, review and debugging quality. BLOCKED in
  this sandbox: the LLM provider hosts are unreachable from here (probed with curl: only GitHub is
  reachable). The tour reports these checks as honest-limited, not passed.

## Memory-leak investigation, closed (second pass)

The "RSS leak" is **not a leak, and it is not object retention either**. Measured three ways
on the live agent:

1. **40,000-request /nexus/intelligence soak** (`/tmp/soak40k.json`): RSS 81.4 -> 98.7 MB while
   the bounded stores filled, then 26 consecutive rounds (26,000 requests) at exactly 0.0 MB,
   all 13 stores at their ceiling. Verdict `plateau`.
2. **Block-count forensics** `scripts/memory_forensics.py` (tracemalloc off, 1,600 `/ask` at
   concurrency 6): live Python blocks 403,456 at rest -> 698,136 under load -> 404,371 after
   45 s idle, i.e. exactly back to baseline, while RSS ended 1.3 MB higher. Anonymous heap grows
   (RssAnon), not file-backed.
3. **Whole-drive verdict** (`scripts/real_life_drive.py`, new final check
   `memory_retention`): blocks 559,541 -> 559,545 (**+4**) across ~3,600 requests spanning every
   mode, the adversarial corpus and the cancellation probe.

So the residual is allocator slack, and `app/utils/memory_guard.py` now returns it: 102 trims
reclaimed **222.5 MB** cumulatively during that drive (and 0.0 MB on a light burst - both
numbers are in the module docstring). Two bugs in the guard itself were found and fixed by
driving it: it ignored `MEMORY_TRIM_*` (reported `threshold_mb 400` when configured to 80) and
its modulo-based consideration gate made it dead code under concurrency (1,500 requests,
`trims: 0`). The RSS-based rules in the drive were then *demoted* to informational, because the
same run that "failed" them measured +4 blocks.

## Evidence log
(appended per item: command + observed numbers)
