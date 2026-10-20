# Task 1 next-run instructions — Inference

Last updated: 2026-10-20T23:06:08Z

Read .github/friday-continuation/task-1-checkpoint.md before coding. Resume branch automation/friday-universe-task-1; continue existing implementation rather than restarting repository review.

## Previous run evidence

- Result: completed
- Active coding seconds: 1917
- Coding passes: 27
- Source paths changed against main: 6

### Changed source paths

- app/api/friday_routes.py
- app/api/v1_core_routes.py
- app/providers/gateway.py
- pp/api/agent_routes.py
- tests/test_gateway_streaming.py
- tests/test_openai_gateway.py

### Runner failures or blockers

- None recorded by the runner.

## Detailed continuation notes from the previous coding pass

# Task 1 next-run instructions — Inference

Read `.github/friday-continuation/task-1-checkpoint.md` and continue from its concrete next action. Do not repeat gateway timeout/capacity/cleanup work, SSE error sanitization, cancellation tests, empty-primary/empty-fallback completion regressions, degraded agent response/cache handling, `no_cache=True`, agent cache-hit behavior, caller/context/output-settings/task-type cache-scope work, cache-isolation coverage, malformed task/token/temperature cases, unsupported `caller_agent` validation, blank/null prompt validation, or primary-provider success-path forwarding and SSE coverage.

## First coding action

Add an authenticated ASGI fallback-success test for `POST /v1/agent/stream` in `tests/test_openai_gateway.py`. Monkeypatch `ModelGateway.stream` to raise `GatewayError` before yielding on provider `groq`, then emit two chunks on `gemini`; capture provider, `ProviderRequest`, and `stage_name` for both calls. Post a valid Forge code-assistance request and assert HTTP 200, both token SSE events and the terminal `done` event, provider order `groq` then `gemini`, fallback model `gemini-3.6-flash`, forwarded prompt/temperature/max_tokens, and stage name `agent_stream_forge_fb`.

Then run:

- `.venv/bin/pytest -o addopts= tests/test_openai_gateway.py tests/test_gateway_streaming.py -q`
- `.venv/bin/python -m compileall -q app/api/agent_routes.py tests/test_openai_gateway.py tests/test_gateway_streaming.py`
- `git diff --check`

Rewrite both handoffs with the exact implementation outcomes, commands and pass/fail counts, unresolved failure causes, and the subsequent concrete first coding action. Preserve existing changes; do not commit or push.

## Current pass results

- Added `test_agent_stream_forwards_request_and_streams_tokens` in `tests/test_openai_gateway.py:1011`. Authenticated ASGI coverage verifies primary Groq routing, model/prompt/temperature/max_tokens/stage forwarding, two token events, and the terminal `done` event.
- Shared `AgentAssistRequest.prompt` validation in `app/api/agent_routes.py` rejects empty and whitespace-only prompts without altering nonblank prompt text; existing authenticated tests cover `/v1/agent/assist` and `/v1/agent/stream`.
- `.venv/bin/pytest -o addopts= tests/test_openai_gateway.py::test_agent_stream_forwards_request_and_streams_tokens -q` — **1 passed, 0 failed** in 0.63s.
- `.venv/bin/pytest -o addopts= tests/test_openai_gateway.py tests/test_gateway_streaming.py -q` — **47 passed, 0 failed** in 0.85s, 10 Starlette deprecation warnings from expected 422 validation responses.
- `.venv/bin/python -m compileall -q app/api/agent_routes.py tests/test_openai_gateway.py tests/test_gateway_streaming.py` — passed (exit code 0).
- `git diff --check` — passed (exit code 0).
- No unresolved failures or blockers; Ruff remains unavailable/not run.
- Branch/HEAD: `automation/friday-universe-task-1` / `885144a`, no commit. Do not commit or push.

Prior uncommitted implementation is also present in `app/api/agent_routes.py`, `app/api/friday_routes.py`, `app/api/v1_core_routes.py`, `app/providers/gateway.py`, `tests/test_gateway_streaming.py`, and earlier portions of `tests/test_openai_gateway.py`. Preserve those changes while continuing.


## First action next run

Read the checkpoint and these previous notes, inspect the branch diff, run the focused tests for the last implementation, fix failures, and implement the next highest-impact feature or regression fix. Record exact commands/results and revise both handoff files before ending.
