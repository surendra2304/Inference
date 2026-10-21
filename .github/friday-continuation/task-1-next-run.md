# Task 1 next-run — Inference

Continue from PR #18, branch fix/bound-forced-agent-selection-20261021, head 4cd82fdb575edc7451076f7f11a85dbdcfe2035e. CI run 39557358134 passed Ruff, pytest, and Docker build.

1. Re-read the PR diff and tests/test_forced_agent_selection.py before editing.
2. Check latest CI status and review any new findings. Keep max_agents enforced for caller-provided agent IDs, bound candidate scanning, and ignore malformed/duplicate values.
3. Select one concrete follow-up issue from the existing source/tests; make a focused change and regression test rather than broad refactoring.
4. Run the focused tests and lint through an available runner, or report that they were not run. Do not infer a local pass from CI on a different commit.
5. Update this checkpoint, next-run file, and last-run JSON with exact commit SHA, files, test command/results, CI URL/status, and next concrete change.
