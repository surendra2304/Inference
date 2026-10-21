# Inference Master Engineering Plan

**Repository:** `surendra2304/Inference`  
**Default branch:** `main`  
**Plan owner:** Inference engineering  
**Last baseline review:** 2026-10-22

## Executive summary

Inference is the FRIDAY Universe's provider-agnostic inference gateway and deliberation runtime. Its repository includes API routes, provider gateways and failover, agent orchestration, asynchronous pipelines, memory/persistence, security middleware, analytics, observability, deployment assets, and a broad pytest suite.

The first objective is a reproducible green baseline. New capabilities must follow evidence of a need, focused implementation, regression tests, CI verification, and reviewable pull requests. A passing lint command alone is not a green release.

## Baseline findings

- The recursive Git tree currently contains **405 files** on the active hardening branch.
- The repository has **66 Python test modules** on the active hardening branch, covering APIs, provider routing, cancellation, security, monitoring, persistence, batch work, and Docker configuration.
- Initial baseline run [39666816838](https://github.com/surendra2304/Inference/actions/runs/39666816838) failed at `ruff check .` with one `I001` import-order error in `app/monitoring.py`, preventing pytest and Docker verification from running. PR [#30](https://github.com/surendra2304/Inference/pull/30) fixed the gate and also corrected PostgreSQL persistence/credential defects.
- PR [#30](https://github.com/surendra2304/Inference/pull/30) merged as `5176b080ee4843a2ed392b76bc1530f99f22ba99`; its current-head CI [run 39672285579](https://github.com/surendra2304/Inference/actions/runs/39672285579) passed lint, the full pytest suite, and Docker build.
- PR [#22](https://github.com/surendra2304/Inference/pull/22) fixed cancellation-token task cleanup and added a pre-cancelled-token regression test. It merged as `6becc799667e02e1830a90cd5710888e64278d32`; its current-head CI [run 39672605944](https://github.com/surendra2304/Inference/actions/runs/39672605944) passed lint, the full pytest suite, and Docker build.
- PR [#32](https://github.com/surendra2304/Inference/pull/32) merged as `6540537bb68aa7b7ebec7ce13d909eca87596072`; current-head CI [39674163591](https://github.com/surendra2304/Inference/actions/runs/39674163591) passed Ruff, the full pytest suite, and Docker build. The post-merge `main` workflow [39674354552](https://github.com/surendra2304/Inference/actions/runs/39674354552) also passed all steps on that commit.
- Configuration review found and removed a hard-coded PostgreSQL password default. Because the prior value remains in public Git history, rotate it in the database and secret store if it was ever valid in a live environment; a source change cannot revoke a committed credential.
- The PostgreSQL adapter's driver (`asyncpg`) is not installed by the base dependency set, and the production Compose file runs SQLite. Treat PostgreSQL persistence as opt-in until the driver and secret-managed connection settings are explicitly provisioned.
- This workspace could not run a local clone/test cycle. GitHub Actions is the verification source for remote changes; only the latest matching commit's workflow is used as evidence.
- PR [#33](https://github.com/surendra2304/Inference/pull/33) merged as `a9059633aa0bc515e585b009353d7c2bb6d1f88f`; its latest-head run [39674879487](https://github.com/surendra2304/Inference/actions/runs/39674879487) and post-merge main run [39675092616](https://github.com/surendra2304/Inference/actions/runs/39675092616) passed Ruff, the full pytest suite, and Docker build. The endpoint now validates finite positive prices, finite return observations, rejects malformed/oversized values, and tracks per-input synthetic provenance.
- `SensitiveDataFilter` was a no-op despite its security contract. The current branch `fix/logging-credential-redaction-20261022` adds standard-library credential redaction for structured assignments, provider key formats, logging arguments, and exception tracebacks; its PR-head CI is the current merge gate.

## Workstreams and acceptance gates

### P0 — Restore trustworthy CI

**Status:** Complete for the verified `main` commit listed above; continue to enforce the gate on every PR.

1. Fix the observed Ruff import-order regression with the smallest possible change.
2. Re-run the complete workflow on the PR head.
3. Confirm that lint, the full pytest suite, and the Docker build all run and pass; do not infer test health from a successful lint step.
4. If a new failure appears once the lint gate is clear, capture the specific log, reproduce the failure with a regression test, and fix the cause rather than skipping or weakening the test.

**Acceptance:** latest PR-head CI is green for all three checks and the merged `main` head is green.

### P1 — Async lifecycle and cancellation safety

**Status:** Cancellation-token race cleanup is merged and verified; broader lifecycle coverage remains in progress.

**Primary surfaces:** `app/inference_runtime/cancellation.py`, `app/core/dag_executor.py`, task/batch routes, microbatching, streaming.

1. Ensure every task created by an operation has a clear owner and cleanup path.
2. Cover normal completion, operation exceptions, token-triggered cancellation, external caller cancellation, and existing-Future inputs.
3. Ensure client disconnects and request deadlines cannot leave provider work, queue futures, or waiter tasks running indefinitely.
4. Verify queue shutdown is graceful, submissions after closure fail deterministically, and all accepted futures resolve.
5. Preserve cancellation semantics: cleanup must not swallow the caller's cancellation or replace the original operation exception.

**Acceptance:** focused race/lifecycle regression tests pass repeatedly, task leaks are asserted absent, and full CI remains green.

### P2 — Provider gateway resilience

**Primary surfaces:** `app/providers/`, `app/inference_runtime/`, high-availability routing, streaming and deadlines.

1. Verify provider attempts share a bounded request deadline; retries and fallbacks must never extend it indefinitely.
2. Test key rotation, cooldown/quarantine recovery, retry classification, and failure propagation with deterministic fakes.
3. Ensure cancelled or timed-out attempts release semaphores, queue slots, HTTP resources, and key leases.
4. Keep provider-specific exceptions and credential material out of client-visible errors and logs.
5. Keep the local tier usable when cloud credentials are absent, while failing closed for authenticated cloud routes when server credentials are not configured.

**Acceptance:** deterministic tests cover primary success, retryable failure, terminal failure, timeout, cancellation, credential absence, and all configured fallback transitions.

### P3 — API security and contract correctness

**Primary surfaces:** `app/core/security.py`, authentication and client-IP middleware, `app/api/`, request/response schemas.

1. Audit every route that can execute models, spend quota, read/write memory, or expose audit data for explicit authentication and authorization.
2. Verify trusted-proxy rules do not accept spoofed forwarding headers from untrusted clients.
3. Validate request bounds before starting expensive work, including batch cardinality, agent assignments, message sizes, and output-token caps.
4. Keep internal stack traces, provider diagnostics, API keys, and sensitive environment values out of public responses.
5. Maintain stable API contracts for success, partial/degraded results, cancellation, rate limiting, and service-unavailable conditions.

**Current change:** PR #33's prediction API request bounds and data provenance fix is merged and verified. The active PR adds logging credential redaction so exception formatting cannot leak raw tokens through the standard logger.

**Acceptance:** authentication, contract fuzz, size/bounds, client-IP trust, and no-error-leak suites pass with no weakened assertions.

### P4 — Honest telemetry and bounded state

**Primary surfaces:** `app/monitoring.py`, `app/observability/`, `app/analytics/`, caches and in-memory stores.

1. Distinguish unobserved, unavailable, estimated, and measured values in output schemas.
2. Do not publish fabricated model quality, latency, provider health, costs, or outcome metrics.
3. Review rolling windows and caches for explicit size/TTL limits and stable behavior under high volume.
4. Ensure concurrent reads/writes and test fixtures cannot leak process-global state across tests.
5. Keep Prometheus exposition syntactically valid when an observation has no samples or is unavailable.
6. Validate prediction-ingestion data before storing it: reject non-finite/out-of-range confidence, malformed labels, and invalid horizons; distinguish unresolved outcomes from terminal FLAT outcomes, and make demo fixtures immutable.

**Acceptance:** tests cover empty samples, bounds, malformed telemetry, concurrency-sensitive state, honest unknown values, and prediction-ingestion edge cases.

### P5 — Orchestration, persistence, and mesh contracts

**Primary surfaces:** `app/core/`, `app/agents/`, `app/memory/`, PostgreSQL/SQLite adapters, `app/api/mesh_routes.py`.

1. Validate DAG topology before execution and propagate cancellation/failure according to an explicit policy.
2. Verify caller-supplied agent selections remain registered, deduplicated, and capped by `max_agents`.
3. Ensure SQLite async resources are used and closed on the owning event loop.
4. Make schema migration idempotent and persistence behavior explicit under concurrent startup or database failures.
5. Keep peer health, identity, configured capacity, and observed availability distinct; never infer live reachability from static configuration.
6. Keep PostgreSQL persistence opt-in unless the driver and secret-managed credentials are installed; preserve unknown measurements as SQL NULL and migrate older schemas without inventing values.

**Acceptance:** DAG, memory, database, selection-bound, and mesh contract suites pass; tests do not require live external services.

### P6 — Deployment and operational readiness

**Primary surfaces:** `Dockerfile`, `docker-compose.production.yml`, `k8s/`, health endpoints and CI configuration.

1. Ensure the image builds from a clean context, runs as a non-root user, and has deterministic health checks.
2. Verify environment configuration is documented without committing secrets or unsafe development defaults.
3. Verify readiness distinguishes process liveness from provider/database dependency health.
4. Keep CI failures visible: lint, test, and image-build steps must not silently skip or downgrade failures.
5. Document deployment rollback and minimum operational signals.

**Acceptance:** Docker builds in CI, health checks match actual service behavior, and documented configuration matches the implementation.


### P7 — Architecture reachability and subsystem ownership

**Status:** Confirmed debt; plan the migration before wiring competing implementations into production.

The repository contains several substantial modules whose production activation is not established by source presence or unit tests alone:

1. `OrchestrationEngine.build_execution_dag()` constructs dependency layers, but `process_task()` delegates the real collaboration to `CollaborationEngine.run_collaboration()` instead of executing nodes through `DAGExecutor`. The graph is currently a topology preview, not the engine driving the live collaboration. Do not describe it as DAG-scheduled execution until the adapter is implemented and validated.
2. `HardenedGateway` is exported and tested as a standalone gateway, but repository call-site review did not find it instantiated by the `app.main` production path; production requests currently route through `app/providers/gateway.py::ModelGateway`. Avoid maintaining two competing routing, retry, budget, and observability stacks indefinitely.
3. A prior reachability report lists adjacent runtime, token-optimizer, Memora, and HA modules as potentially orphaned. Re-run static/import/runtime reachability on the current tree before deleting or promoting any of them; declaration-only search results are not enough to prove a module is safe to remove.
4. The checked-in `docs/openapi.yaml` currently declares only 11 paths and omits live routes such as `/v1/task/batch`, `/v1/intelligence/accuracy`, `/v1/governance/prometheus-metrics`, and `/ready`. Its declared version is `2.5.0`; treat the static file as incomplete until it is generated/compared against the live FastAPI schema in CI. Avoid hand-authoring the full schema as a substitute for the application-generated contract.
5. Repository call-site search found `PredictionTrackingEngine.record_prediction()` and `resolve_prediction()` are exercised in tests but not wired into a production ingestion path. Keep their observed-only accuracy report honest; any integration should happen at a real prediction producer and outcome resolver, not by mutating state from a GET request.

**Next decision gate:** choose a single supported production gateway and define how it will be activated/migrated with endpoint parity, auth, timeout/cancellation, budget accounting, and telemetry tests. Decompose the collaboration algorithm into explicit DAG-node operations before adopting `DAGExecutor`; preserve peer recovery, consensus/rebuttal, evidence requirements, and degradation semantics during any migration.

**Acceptance:** production call-site inventory is current and repeatable; every retained subsystem has a declared supported entry point or is documented as internal/experimental; routing and orchestration migration tests prove behavior parity before any default-path change.

## Pull request and merge policy

- Use a named feature/fix branch per coherent workstream; do not develop directly on `main`.
- Every bug fix includes a regression test where practical.
- Review the full diff and current base branch before opening/updating a PR.
- Merge only after required CI checks pass on the current PR head and conflicts are resolved.
- After merge, inspect the resulting `main` workflow. If it fails, treat that as a release blocker and investigate immediately.
- Do not claim 100% pass rate, complete repository coverage, or production readiness without the corresponding current evidence.
- Prefer focused, reviewable increments to mass-generated code. Additional code is justified by a verified capability requirement or defect.

## Completion checklist

- [x] Latest verified `main` lint passes at `6becc799667e02e1830a90cd5710888e64278d32`.
- [x] Full pytest suite passes on the verified current `main` commit.
- [x] Docker build passes on the verified current `main` commit.
- [x] Cancellation-token race cleanup and pre-cancelled-token regressions are covered and merged; other queue/shutdown acceptance cases remain under P1.
- [ ] Security and request/resource bounds are verified.
- [ ] Provider deadlines, fallback behavior, and credential isolation are verified.
- [x] Confirmed fabricated Prometheus/accuracy/debate metrics on current tree have regression tests; continue the audit for other unmeasured claims.
- [ ] Persistence and mesh contracts are validated.
- [ ] No known failing CI checks or unreviewed release-blocking regressions remain.
