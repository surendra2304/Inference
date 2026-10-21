# Inference Master Engineering Plan

**Repository:** `surendra2304/Inference`  
**Default branch:** `main`  
**Plan owner:** Inference engineering  
**Last baseline review:** 2026-10-22

## Executive summary

Inference is the FRIDAY Universe's provider-agnostic inference gateway and deliberation runtime. Its repository includes API routes, provider gateways and failover, agent orchestration, asynchronous pipelines, memory/persistence, security middleware, analytics, observability, deployment assets, and a broad pytest suite.

The first objective is a reproducible green baseline. New capabilities must follow evidence of a need, focused implementation, regression tests, CI verification, and reviewable pull requests. A passing lint command alone is not a green release.

## Baseline findings

- The recursive Git tree currently contains **401 files**.
- The repository has **60+ test modules** covering APIs, provider routing, cancellation, security, monitoring, persistence, batch work, and Docker configuration.
- The latest recorded CI run on `main` is [run 39666816838](https://github.com/surendra2304/Inference/actions/runs/39666816838). Dependency installation succeeded, but `ruff check .` failed with one `I001` import-order error in `app/monitoring.py`; pytest and Docker verification were skipped.
- The immediate lint failure is caused by a missing blank line before a guarded datetime import in `app/monitoring.py`. A focused fix is being submitted separately and must be validated by GitHub Actions.
- Configuration review also found a hard-coded PostgreSQL password default. The current hardening branch removes the default and fails closed without an explicitly supplied password. Because the old value is in public Git history, rotate it if it was ever valid in a live environment; a source change cannot revoke a committed credential.
- The PostgreSQL adapter's driver (`asyncpg`) is not installed by the base dependency set, and the production Compose file deliberately runs SQLite. Treat PostgreSQL persistence as opt-in until the driver and secret-managed connection settings are explicitly provisioned; do not infer that a PostgreSQL connection is active just because its adapter exists.
- PR [#22](https://github.com/surendra2304/Inference/pull/22) contains an async cancellation cleanup fix and regression tests. Its last reported CI success was against an older base commit, so the latest `main` must be revalidated before it is merged.
- Direct clone/test execution was not available in the current engineering workspace. CI is therefore the authoritative test runner for these remote changes; no tests are considered passed until the matching commit's run reports success.

## Workstreams and acceptance gates

### P0 — Restore trustworthy CI

**Status:** In progress.

1. Fix the observed Ruff import-order regression with the smallest possible change.
2. Re-run the complete workflow on the PR head.
3. Confirm that lint, the full pytest suite, and the Docker build all run and pass; do not infer test health from a successful lint step.
4. If a new failure appears once the lint gate is clear, capture the specific log, reproduce the failure with a regression test, and fix the cause rather than skipping or weakening the test.

**Acceptance:** latest PR-head CI is green for all three checks and the merged `main` head is green.

### P1 — Async lifecycle and cancellation safety

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

**Acceptance:** authentication, contract fuzz, size/bounds, client-IP trust, and no-error-leak suites pass with no weakened assertions.

### P4 — Honest telemetry and bounded state

**Primary surfaces:** `app/monitoring.py`, `app/observability/`, `app/analytics/`, caches and in-memory stores.

1. Distinguish unobserved, unavailable, estimated, and measured values in output schemas.
2. Do not publish fabricated model quality, latency, provider health, costs, or outcome metrics.
3. Review rolling windows and caches for explicit size/TTL limits and stable behavior under high volume.
4. Ensure concurrent reads/writes and test fixtures cannot leak process-global state across tests.
5. Keep Prometheus exposition syntactically valid when an observation has no samples or is unavailable.

**Acceptance:** tests cover empty samples, bounds, malformed telemetry, concurrency-sensitive state, and honest unknown values.

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

## Pull request and merge policy

- Use a named feature/fix branch per coherent workstream; do not develop directly on `main`.
- Every bug fix includes a regression test where practical.
- Review the full diff and current base branch before opening/updating a PR.
- Merge only after required CI checks pass on the current PR head and conflicts are resolved.
- After merge, inspect the resulting `main` workflow. If it fails, treat that as a release blocker and investigate immediately.
- Do not claim 100% pass rate, complete repository coverage, or production readiness without the corresponding current evidence.
- Prefer focused, reviewable increments to mass-generated code. Additional code is justified by a verified capability requirement or defect.

## Completion checklist

- [ ] Latest `main` lint passes.
- [ ] Full pytest suite passes on a current `main` commit.
- [ ] Docker build passes on a current `main` commit.
- [ ] Async cancellation and queue shutdown regressions are covered and merged.
- [ ] Security and request/resource bounds are verified.
- [ ] Provider deadlines, fallback behavior, and credential isolation are verified.
- [ ] Telemetry reports only measured or clearly qualified values.
- [ ] Persistence and mesh contracts are validated.
- [ ] No known failing CI checks or unreviewed release-blocking regressions remain.
