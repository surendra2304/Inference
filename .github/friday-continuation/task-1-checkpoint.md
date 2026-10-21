# Task 1 checkpoint — Inference — 2026-10-22

- Open PR #31: https://github.com/surendra2304/Inference/pull/31, branch `fix/task1-forced-agent-max-bound-20261022`, head `32957964472c1026a784cc97a11e1546e6d7f62c`. `app/core/orchestrator.py` returns [] for `max_agents <= 0` before registry lookup; `tests/test_forced_agent_selection.py` covers 0/-1 and asserts no registry calls. CI `39673218580` passed lint, pytest and Docker on exact head.
- New PR #34: https://github.com/surendra2304/Inference/pull/34, branch `fix/prediction-api-input-bounds-20261022`, exact head `491eb9c31e645a0db3cba80a85bc133c15cbbc7e`. Its latest commit adds `tests/test_prediction_endpoint_input_bounds.py::test_prediction_endpoint_rejects_oversized_raw_return_string`, proving an oversized `recent_returns` raw query string returns 422 with the 8192-character contract. CI `39674879487` passed lint, pytest and Docker on that exact head. PR #34 is open, currently API says mergeable=false; no merge attempted.
- Latest main CI `39675092616` passed at SHA `a9059633aa0bc515e585b009353d7c2bb6d1f88f`.
- Earlier prediction-tracking PR #32 is closed (not current work); run `39674147330` at `bb5dfb46764319384fd0767752f6714c6b1fc069` passed.
- Local tests unavailable because laptop MCP returns HTTP 404; exact GitHub CI is validation. No merge/force push; active duration not reliably measured.

Next concrete action: inspect why PR #34 is not mergeable (base/current main divergence and checks), fix via additive non-force-push if needed, then verify #31. No duplicate max_agents PR.