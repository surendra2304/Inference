# AGENT_PROGRESS — Inference repo (session `arena/f706c17c-inference`)

Source of truth for this session. Read this first on resume. Restated from `notes/WORKQUEUE.md`
(increment-30 block, 69 items) + the user's standing task.

**Task:** make the agent actually work under real operator usage — drive it with real tasks,
stress it to dead ends, find and FIX every bug surfaced, write substantial new code that
upgrades capability, keep real-life regression tests, keep going until the queue is empty with
evidence attached to every item.

**Current step:** item 1 (RSS leak) — attribution done via new opt-in
`/memory/diagnostics` endpoint; growth diff per burst names
`app/services/nexus_intelligence.py:185,309,310,316,321` and
`app/analytics/usage_analytics.py:62`. Next: separate "store still filling (bounded)" from
"true linear leak".

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
- [ ] 4. Append #69 + leak findings to `notes/WORKQUEUE.md`.
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
- [ ] 7. #62–#65 backlog items.
- [ ] 8. #29 backlog item.
- [ ] 9. Q7 citation check (sqlite loop ownership).
- [ ] 10. Final sweep: full pytest + ruff + mypy + real-life harness green.

## Evidence log
(appended per item: command + observed numbers)
