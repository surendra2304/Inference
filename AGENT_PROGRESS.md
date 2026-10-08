# AGENT_PROGRESS — Inference repo (session `arena/f706c17c-inference`)

Source of truth for this session. Read this first on resume. Restated from `notes/WORKQUEUE.md`
(increment-30 block, 69 items) + the user's standing task.

**Task:** make the agent actually work under real operator usage — drive it with real tasks,
stress it to dead ends, find and FIX every bug surfaced, write substantial new code that
upgrades capability, keep real-life regression tests, keep going until the queue is empty with
evidence attached to every item.

**Current step:** items 1-7 done; item 8 (#29 cross-endpoint dedup collision) is the next
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
- [ ] 8. #29 backlog item.
- [ ] 9. Q7 citation check (sqlite loop ownership).
- [ ] 10. Final sweep: full pytest + ruff + mypy + real-life harness green.

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
