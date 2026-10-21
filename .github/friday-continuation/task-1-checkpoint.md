# Task 1 checkpoint — Inference — 2026-10-22

- Open PR #31: https://github.com/surendra2304/Inference/pull/31, branch `fix/task1-forced-agent-max-bound-20261022`, head `32957964472c1026a784cc97a11e1546e6d7f62c`. `app/core/orchestrator.py` returns [] for `max_agents <= 0` before registry lookup; `tests/test_forced_agent_selection.py` covers 0/-1 and confirms no registry access. CI `39673218580` passed lint, pytest, Docker on exact head.
- PR #34: https://github.com/surendra2304/Inference/pull/34, branch `fix/prediction-api-input-bounds-20261022`, head `491eb9c31e645a0db3cba80a85bc133c15cbbc7e`. Regression `tests/test_prediction_endpoint_input_bounds.py::test_prediction_endpoint_rejects_oversized_raw_return_string` confirms an oversized raw `recent_returns` query is rejected with 422. CI `39674879487` passed lint, pytest and Docker on exact head. PR open; current fetched API says mergeable=false, so recheck current head/main and resolve safely without force-push.
- Main SHA `a9059633aa0bc515e585b009353d7c2bb6d1f88f`, run `39675092616` passed.
- PR #32 prediction tracking is closed, not open. New separate branch `fix/prediction-api-input-bounds-20261022` is now represented by PR #34.
- Local tests unavailable (laptop MCP HTTP 404). No merge/force-push, duration unmeasured.

Next: inspect PR #34 latest mergeability/current checks and PR #31 current status; don't use historical CI as evidence for a changed head.