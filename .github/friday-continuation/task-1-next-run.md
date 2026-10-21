# Task 1 next-run instructions — Inference (2026-10-22)

1. Recheck PR #30 current head and combined checks. The prior green runs `39671781899` at `0199cf81c69437f0314b94ed2d10ae62aa249e8a` and `39671769752` at `64c758a11499e0c378340aadca8668cea86c838b` passed Ruff, pytest, Docker, but may be behind current API head `daf519d8f401446838017fb9bb7dcff5db0ab2fa`; don't infer current-head status.
2. Review PR #30's scope, especially nullable confidence and `ALTER TABLE deliberations ALTER COLUMN confidence DROP NOT NULL`; inspect how schema DDL runs on fresh vs existing deployments before any merge.
3. Continue existing PR #22 cancellation cleanup, do not create a duplicate PR.
4. Next regression candidate: `app/core/orchestrator.py::_resolve_forced_agents` and `tests/test_forced_agent_selection.py`; inspect zero/negative `max_agents` contract and add a focused boundary test. Preserve candidate scan bounds.
5. Laptop execution endpoint returned HTTP 404; no local tests ran this invocation. Use connected GitHub and actual CI evidence. Update all three handoff files and never auto-merge/force push.