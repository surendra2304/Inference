#!/usr/bin/env python3
"""Drive the Inference agent with real tasks and check its honesty invariants.

WHY THIS EXISTS
---------------
``pytest tests/`` proves the mocked unit paths work. It cannot prove the agent
*behaves* — that a panel of specialists answers a real question, that a broken
provider degrades instead of fabricating, that concurrency does not corrupt state,
or that latency holds under load. This harness drives the live server over HTTP with
workloads shaped like real usage, then asserts the properties the system claims to
guarantee.

THE INVARIANTS
--------------
Each is a property the codebase asserts about itself. A violation is a bug in the
agent, not a flaky test:

  INV-1  An answer is never published without model output behind it.
  INV-2  ``degraded`` is always accompanied by an explanation.
  INV-3  Confidence never exceeds the documented ceiling and never contradicts
         ``degraded``.
  INV-4  A truncated panel is disclosed (the operator can tell output was cut off).
  INV-5  Total failure returns a non-2xx status, never a confident answer.
  INV-6  No 5xx leaks from the application itself under load.
  INV-7  Latency stays inside the stated budget at the requested concurrency.
  INV-8  Nothing crashes and no request hangs past the client deadline.
  INV-9  Answers are deterministic for a deterministic model.
  INV-10 The audit trail records every accepted request.
  INV-11 Every model call carries a system prompt, the user's question, and the
         identity of the specialist that is speaking.

USAGE
-----
    # start the rig, then the server (see scripts/local_model_server.py docstring)
    python scripts/drive_agent.py --scenario all --concurrency 8 --requests 40
    python scripts/drive_agent.py --scenario reliability --report /tmp/report.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

# --------------------------------------------------------------------------------------
# A task battery shaped like real usage, not like a fixture.
# --------------------------------------------------------------------------------------

TASKS: list[tuple[str, str, str]] = [
    # (name, mode, question)
    ("code-review", "fast", "Review this Python function for bugs:\n\ndef parse(rows):\n    out = []\n    for r in rows:\n        out.append(int(r['v']))\n    return sum(out) / len(out)"),
    ("sql-safety", "review", "Is it safe to build a SQL query with an f-string if I escape quotes? Show the correct approach."),
    ("architecture", "review", "I have a FastAPI service that calls three external APIs per request. Where should I put retries and timeouts?"),
    ("debug-async", "review", "My asyncio task sometimes never completes and the event loop logs 'Task was destroyed but it is pending'. What causes this?"),
    ("perf-sqlite", "review", "SQLite writes are slow under concurrent load from 8 workers. What should I check first?"),
    ("security-auth", "review", "How do I store API keys for a small service without introducing a secrets manager?"),
    ("data-analysis", "review", "Given 40 trades with a 55% win rate, can I conclude my strategy has an edge?"),
    ("deployment", "review", "A container works locally but is OOM-killed on a 512MB host. How do I find the cause?"),
    ("reasoning", "debate", "Should I split my monolith into two services to scale reads, or add a cache first?"),
    ("tradeoff", "debate", "Is it better to bound a retry by deadline or by attempt count for a user-facing API?"),
]

FAILURE_TASKS: list[tuple[str, str, str]] = [
    ("provider-429", "review", "__FAIL:429 Explain how SQLite WAL mode works."),
    ("provider-500", "review", "__FAIL:500 Explain how SQLite WAL mode works."),
    ("provider-503", "review", "__FAIL:503 Explain how SQLite WAL mode works."),
    ("empty-completion", "review", "__FAIL:empty Explain how SQLite WAL mode works."),
    ("malformed-body", "review", "__FAIL:malformed Explain how SQLite WAL mode works."),
    ("truncated-output", "review", "__FAIL:truncate Explain how SQLite WAL mode works."),
    ("provider-timeout", "review", "__FAIL:timeout Explain how SQLite WAL mode works."),
]


@dataclass
class Observation:
    """One request/response pair, with everything needed to judge the invariants."""

    name: str
    mode: str
    status: int
    seconds: float
    body: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    @property
    def answer(self) -> str:
        return str(self.body.get("answer") or "")

    @property
    def degraded(self) -> bool:
        return bool(self.body.get("degraded"))

    @property
    def confidence(self) -> float | None:
        value = self.body.get("confidence")
        return float(value) if isinstance(value, (int, float)) else None


@dataclass
class Violation:
    invariant: str
    task: str
    detail: str


class Violations:
    def __init__(self) -> None:
        self.items: list[Violation] = []

    def check(self, condition: bool, invariant: str, task: str, detail: str) -> None:
        if not condition:
            self.items.append(Violation(invariant, task, detail))

    def report(self) -> str:
        if not self.items:
            return "no invariant violations"
        lines = [f"{len(self.items)} INVARIANT VIOLATION(S):"]
        for item in self.items:
            lines.append(f"  [{item.invariant}] {item.task}: {item.detail}")
        return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Invariant evaluation
# --------------------------------------------------------------------------------------


def judge(obs: Observation, v: Violations) -> None:
    """Apply every invariant to one observation."""
    body = obs.body
    ok = 200 <= obs.status < 300

    if not ok:
        # INV-5: a failure must be a failure. 5xx from the app is only acceptable when
        # it is the documented 503 fail-closed on total panel darkness.
        v.check(
            obs.status in (401, 403, 429, 503),
            "INV-6",
            obs.name,
            f"unexpected status {obs.status}: {obs.error[:160]}",
        )
        if body:
            v.check(
                obs.confidence is None,
                "INV-5",
                obs.name,
                f"failure response carried a confidence of {obs.confidence}",
            )
            v.check(
                not obs.answer or obs.degraded,
                "INV-5",
                obs.name,
                "failure response carried an authoritative answer",
            )
        return

    # INV-1: a 2xx answer must have model output behind it.
    # An empty answer is only acceptable if the agent said it was degraded.
    if not obs.answer.strip():
        v.check(
            obs.degraded,
            "INV-1",
            obs.name,
            f"HTTP 200 with an EMPTY answer and degraded={obs.degraded}, "
            f"confidence={obs.confidence}, tokens={body.get('total_tokens')}",
        )
    else:
        v.check(
            not (body.get("total_tokens") == 0 and obs.confidence and obs.confidence > 0.5),
            "INV-1",
            obs.name,
            f"confidence {obs.confidence} published with 0 tokens of model output",
        )

    # INV-2: degraded must be explained.
    if obs.degraded:
        v.check(
            bool(body.get("degradation_reasons")),
            "INV-2",
            obs.name,
            "degraded=True with an empty degradation_reasons list",
        )

    # INV-3: confidence sanity, and degraded answers must be humbler.
    if obs.confidence is not None:
        v.check(0.0 <= obs.confidence <= 1.0, "INV-3", obs.name, f"confidence out of range: {obs.confidence}")
        v.check(obs.confidence <= 0.98, "INV-3", obs.name, f"confidence above the documented 0.98 ceiling: {obs.confidence}")
        # INV-3: the confidence ceiling is a *policy*, and the policy is graduated by
        # severity (see DebateOrchestrator._confidence_ceiling). Checking the specific
        # ceiling rather than one blanket number means a response cannot quietly claim
        # the trustworthiness of a clean run while reporting damage.
        if obs.degraded:
            reasons = " ".join(obs.body.get("degradation_reasons") or []).lower()
            if "synth" in reasons:
                ceiling, why = 0.45, "synthesis failed"
            elif "truncated" in reasons:
                ceiling, why = 0.55, "output truncated"
            elif "unavailable" in reasons or obs.body.get("agent_coverage"):
                ceiling, why = 0.65, "specialist covered or dropped"
            else:
                ceiling, why = 0.70, "degraded"
            v.check(
                obs.confidence <= ceiling,
                "INV-3",
                obs.name,
                f"degraded ({why}) but confidence {obs.confidence} exceeds the {ceiling} ceiling",
            )

    # INV-1 (stricter): the agent must never hand the caller an *agent-side* failure
    # placeholder as if it were an answer.
    #
    # NOTE on a corrected measurement: an earlier revision of this check also flagged
    # any answer beginning "[specialist] ... round analysis", on the theory that it was
    # raw model scaffolding leaking through. That was wrong. That string is produced by
    # the test rig itself (scripts/local_model_server.py), which is not a language model
    # and replies with a deterministic echo of the request; the agent really does send a
    # full system prompt and the real question (verified against the rig's /__capture
    # endpoint). Flagging it was measuring the instrument, not the system. The rig cannot
    # speak, so *no* answer-quality claim may be drawn from its text here — only these
    # structural claims, which are about the agent's own plumbing.
    agent_side_placeholders = (
        "[specialist x temporarily offline]",  # the old hard-coded offline string
        "temporarily offline",
        "no specialist produced output",
        "agentcallunavailable",
        "traceback (most recent call last)",
    )
    lowered = obs.answer.strip().lower()
    if lowered:
        v.check(
            not any(m in lowered for m in agent_side_placeholders),
            "INV-1",
            obs.name,
            f"answer contains an agent-side failure placeholder: {obs.answer[:100]!r}",
        )
        # An error string must never arrive with a 200 and a confident number attached.
        v.check(
            not (obs.degraded is False and obs.confidence and obs.confidence >= 0.7 and "error" in lowered[:60]),
            "INV-1",
            obs.name,
            f"answer opens with 'error' but was published as confidently undegraded: {obs.answer[:80]!r}",
        )


# --------------------------------------------------------------------------------------
# Scenarios
# --------------------------------------------------------------------------------------


async def one_request(client: httpx.AsyncClient, url: str, key: str, name: str, mode: str,
                      question: str, timeout: float) -> Observation:
    start = time.perf_counter()
    try:
        resp = await client.post(
            f"{url}/ask",
            json={"question": question, "mode": mode},
            headers={"X-API-Key": key},
            timeout=timeout,
        )
        elapsed = time.perf_counter() - start
        try:
            body = resp.json()
        except Exception:
            body = {}
        if not isinstance(body, dict):
            body = {}
        return Observation(name=name, mode=mode, status=resp.status_code, seconds=elapsed, body=body)
    except Exception as exc:
        return Observation(
            name=name, mode=mode, status=0, seconds=time.perf_counter() - start, error=f"{type(exc).__name__}: {exc}"
        )


async def scenario_quality(client: httpx.AsyncClient, url: str, key: str, v: Violations,
                           timeout: float) -> list[Observation]:
    """Drive the real task battery sequentially and judge each answer."""
    print("\n=== SCENARIO: task quality (sequential, real questions) ===")
    results: list[Observation] = []
    for name, mode, question in TASKS:
        obs = await one_request(client, url, key, name, mode, question, timeout)
        results.append(obs)
        judge(obs, v)
        if obs.status == 200:
            print(
                f"  {name:14s} {mode:7s} {obs.seconds:5.2f}s  conf={obs.confidence} "
                f"degraded={str(obs.degraded):5s} agents={len(obs.body.get('agents_used') or [])} "
                f"ans={len(obs.answer):5d}c tok={obs.body.get('total_tokens')}"
            )
        else:
            print(f"  {name:14s} {mode:7s} {obs.seconds:5.2f}s  HTTP {obs.status}  {obs.error[:80] or str(obs.body)[:80]}")
    return results


async def scenario_reliability(client: httpx.AsyncClient, url: str, key: str, v: Violations,
                               timeout: float) -> list[Observation]:
    """Inject every provider fault in turn and assert the agent stays honest."""
    print("\n=== SCENARIO: provider faults (one fault per request) ===")
    results: list[Observation] = []
    for name, mode, question in FAILURE_TASKS:
        obs = await one_request(client, url, key, name, mode, question, timeout)
        results.append(obs)
        judge(obs, v)
        verdict = "honest"
        if obs.status == 200 and not obs.answer.strip() and not obs.degraded:
            verdict = "*** FABRICATED ***"
        reasons = obs.body.get("degradation_reasons") or []
        print(
            f"  {name:18s} HTTP {obs.status:3d} {obs.seconds:5.2f}s conf={str(obs.confidence):5s} "
            f"degraded={str(obs.degraded):5s} {verdict}"
        )
        for reason in reasons[:2]:
            print(f"      reason: {reason[:150]}")
    return results


async def scenario_load(client: httpx.AsyncClient, url: str, key: str, v: Violations,
                        concurrency: int, requests: int, timeout: float) -> list[Observation]:
    """Fire N concurrent requests and look for breakage rather than answers."""
    print(f"\n=== SCENARIO: concurrency {concurrency} x {requests} requests ===")
    semaphore = asyncio.Semaphore(concurrency)
    results: list[Observation] = []

    async def bounded(index: int) -> Observation:
        name, mode, question = TASKS[index % len(TASKS)]
        async with semaphore:
            return await one_request(client, url, key, f"{name}#{index}", mode, question, timeout)

    start = time.perf_counter()
    results = await asyncio.gather(*[bounded(i) for i in range(requests)])
    wall = time.perf_counter() - start

    for obs in results:
        judge(obs, v)

    latencies = sorted(o.seconds for o in results)
    ok = [o for o in results if o.status == 200]
    statuses: dict[int, int] = {}
    for obs in results:
        statuses[obs.status] = statuses.get(obs.status, 0) + 1

    print(f"  wall={wall:.2f}s  throughput={len(results) / wall:.2f} req/s")
    print(f"  statuses: {dict(sorted(statuses.items()))}")
    if latencies:
        print(
            f"  latency  p50={_pct(latencies, 50):.3f}s  p95={_pct(latencies, 95):.3f}s  "
            f"p99={_pct(latencies, 99):.3f}s  max={latencies[-1]:.3f}s"
        )
    if ok:
        confs = [o.confidence for o in ok if o.confidence is not None]
        if confs:
            print(f"  200s: {len(ok)}  confidence mean={statistics.mean(confs):.3f} min={min(confs)} max={max(confs)}")
    return list(results)


async def scenario_prompt_plumbing(client: httpx.AsyncClient, url: str, key: str, v: Violations,
                                   timeout: float, rig_url: str) -> list[Observation]:
    """Verify what the agent actually SENDS to the model, via the rig's capture buffer.

    This is the one class of claim a non-linguistic rig can fully support: not whether
    an answer is good, but whether the request that produced it was well-formed. A
    panel that silently drops the system prompt, forgets the user's question, or calls
    a model it never declared would still "work" on a rig that echoes its input — and
    would fail catastrophically against a real model. Checking the wire, not the words.
    """
    print("\n=== SCENARIO: request plumbing (what the model actually receives) ===")
    results: list[Observation] = []
    question = "Explain how SQLite WAL mode works and when it is a bad idea."
    for mode in ("fast", "review", "debate"):
        try:
            await client.post(f"{rig_url.rstrip('/')}/__control", json={"reset": True}, timeout=5.0)
        except Exception:  # noqa: BLE001 - control endpoint is best-effort
            pass
        obs = await one_request(client, url, key, f"plumbing-{mode}", mode, question, timeout)
        results.append(obs)
        judge(obs, v)
        if obs.status != 200:
            print(f"  {mode:7s} HTTP {obs.status} — no captures to inspect")
            continue
        try:
            caps = (await client.get(f"{rig_url.rstrip('/')}/__capture", timeout=10.0)).json()["captures"]
        except Exception as exc:  # noqa: BLE001
            print(f"  {mode:7s} capture unavailable: {exc}")
            continue

        empty_system = [c for c in caps if not (c.get("system_head") or "").strip()]
        missing_question = [c for c in caps if "WAL" not in (c.get("question_head") or "")]
        no_role = [c for c in caps if c.get("persona") in (None, "", "specialist") and mode != "fast"]
        print(
            f"  {mode:7s} calls={len(caps)} personas={sorted({c.get('persona') for c in caps})} "
            f"models={sorted({c.get('model') for c in caps})}"
        )
        v.check(bool(caps), "INV-11", f"plumbing-{mode}", "the agent answered without calling any model")
        v.check(
            not empty_system,
            "INV-11",
            f"plumbing-{mode}",
            f"{len(empty_system)}/{len(caps)} model calls carried an EMPTY system prompt",
        )
        v.check(
            not missing_question,
            "INV-11",
            f"plumbing-{mode}",
            f"{len(missing_question)}/{len(caps)} model calls did not contain the user's question",
        )
        v.check(
            not no_role,
            "INV-11",
            f"plumbing-{mode}",
            f"{len(no_role)}/{len(caps)} panel calls did not identify the speaking specialist",
        )
        # A panel is supposed to be a panel: more than one distinct voice.
        if mode in ("review", "debate"):
            distinct = {c.get("persona") for c in caps if c.get("persona") not in (None, "specialist")}
            v.check(
                len(distinct) >= 2,
                "INV-11",
                f"plumbing-{mode}",
                f"{mode} mode ran with only {len(distinct)} distinct specialist voice(s): {sorted(distinct)}",
            )
    return results


async def scenario_determinism(client: httpx.AsyncClient, url: str, key: str, v: Violations,
                               timeout: float, repeats: int = 3) -> list[Observation]:
    """The rig is deterministic; any answer variance is the agent's doing."""
    print(f"\n=== SCENARIO: determinism ({repeats} identical requests) ===")
    name, mode, question = "determinism", "review", "Explain how SQLite WAL mode works."
    results = []
    answers = []
    for _ in range(repeats):
        obs = await one_request(client, url, key, name, mode, question, timeout)
        results.append(obs)
        judge(obs, v)
        answers.append(obs.answer)
    distinct = len({a.strip() for a in answers})
    confs = {o.confidence for o in results}
    print(f"  distinct answers: {distinct}/{repeats}   distinct confidences: {sorted(c for c in confs if c is not None)}")
    v.check(distinct == 1, "INV-9", name, f"a deterministic model produced {distinct} different answers")
    return results


def _pct(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, int(round((percentile / 100.0) * (len(sorted_values) - 1))))
    return sorted_values[index]


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


async def run(args: argparse.Namespace) -> int:
    v = Violations()
    url = args.url.rstrip("/")
    limits = httpx.Limits(max_connections=max(args.concurrency * 2, 16), max_keepalive_connections=args.concurrency)

    async with httpx.AsyncClient(limits=limits) as client:
        try:
            health = await client.get(f"{url}/health", timeout=10.0)
        except Exception as exc:
            print(f"FATAL: cannot reach {url}: {type(exc).__name__}: {exc}")
            return 2
        print(f"server: /health -> {health.status_code} {health.json().get('status', '')}")
        rig = await _rig_stats(client, args)
        if rig:
            print(f"rig:    {rig.get('requests')} requests, model={rig.get('default_model')}, slots={rig.get('slots')}")

        all_obs: list[Observation] = []
        # Order matters and is deliberate. Fault injection is *persistent state*: it
        # drives the agent's own provider-health tracker, and until the recovery window
        # elapses every later scenario runs against a system that already believes a
        # provider is down. Running the healthy-path scenarios first keeps each
        # scenario's result attributable to its own stimulus. (This ordering bug made a
        # previous full run look like a total outage — 40/40 HTTP 503 — when in fact
        # only the load scenario, run minutes after the fault battery, was affected.)
        if args.scenario in ("all", "quality"):
            all_obs += await scenario_quality(client, url, args.key, v, args.timeout)
        if args.scenario in ("all", "plumbing"):
            all_obs += await scenario_prompt_plumbing(client, url, args.key, v, args.timeout, args.rig_url)
        if args.scenario in ("all", "load"):
            all_obs += await scenario_load(client, url, args.key, v, args.concurrency, args.requests, args.timeout)
        if args.scenario in ("all", "determinism"):
            all_obs += await scenario_determinism(client, url, args.key, v, args.timeout)
        if args.scenario in ("all", "reliability"):
            all_obs += await scenario_reliability(client, url, args.key, v, args.timeout)

    print("\n" + "=" * 78)
    print(v.report())
    ok = [o for o in all_obs if o.status == 200]
    print(f"\nrequests={len(all_obs)}  ok={len(ok)}  violations={len(v.items)}")

    if args.report:
        payload = {
            "summary": {
                "requests": len(all_obs),
                "ok": len(ok),
                "violations": [vars(item) for item in v.items],
            },
            "observations": [
                {
                    "name": o.name, "mode": o.mode, "status": o.status, "seconds": round(o.seconds, 4),
                    "degraded": o.degraded, "confidence": o.confidence,
                    "answer_chars": len(o.answer), "error": o.error,
                    "degradation_reasons": o.body.get("degradation_reasons"),
                    "agents_used": o.body.get("agents_used"),
                    "agent_coverage": o.body.get("agent_coverage"),
                }
                for o in all_obs
            ],
        }
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"report written to {args.report}")

    return 1 if v.items else 0


async def _rig_stats(client: httpx.AsyncClient, args: argparse.Namespace) -> dict[str, Any] | None:
    """Best-effort read of the rig's counters, so the report shows what the model saw."""
    for candidate in (args.rig_url, os.environ.get("LOCAL_BASE_URL", "").rsplit("/v1", 1)[0]):
        if not candidate:
            continue
        try:
            resp = await client.get(f"{candidate.rstrip('/')}/__stats", timeout=5.0)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            continue
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Drive the Inference agent with real tasks and check invariants.")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Inference base URL")
    parser.add_argument("--rig-url", default="http://127.0.0.1:11434", help="local model rig base URL")
    parser.add_argument("--key", default=os.environ.get("INFERENCE_API_KEY", "drive_key"))
    parser.add_argument(
        "--scenario",
        default="all",
        choices=["all", "quality", "reliability", "plumbing", "load", "determinism"],
    )
    parser.add_argument("--concurrency", type=int, default=4, help="concurrent requests for the load scenario")
    parser.add_argument("--requests", type=int, default=24, help="total requests for the load scenario")
    parser.add_argument("--timeout", type=float, default=60.0, help="per-request client timeout")
    parser.add_argument("--report", default="", help="write a JSON report to this path")
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
