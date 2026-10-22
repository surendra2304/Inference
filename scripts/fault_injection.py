"""Failure injection against a live agent: does it stay honest, and does it stay up?

The rig (``scripts/local_model_server.py``) can be told to fail, hang or misbehave through
``POST /__control``. This script uses that to answer the questions an operator actually cares
about when a provider goes down:

* does any endpoint return 5xx (a crash) rather than an honest in-band failure?
* does anything claim success (``status`` SUCCESS / non-null confidence with no model output)?
* does a substitution or fallback get *reported* rather than hidden?
* does the agent recover once the provider comes back?

Each scenario is a real HTTP request driven through the public API; the report keeps the
status code, the honesty verdict and the response fields that carry the claim.

    python scripts/fault_injection.py --base http://127.0.0.1:8001 --rig http://127.0.0.1:11435
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

ASK = {"question": "Should the write path be sharded by tenant?", "mode": "review"}
DEBATE = {"question": "Is a single-writer SQLite service viable at 300 rps?", "mode": "debate"}
FRIDAY = {"question": "What breaks first when the retry budget is exhausted?"}
CONSULT = {
    "bot_id": "fault-bot", "trading_mode": "PAPER", "consultation_reason": "SCHEDULED",
    "telemetry": {"equity": 1000.0, "unrealized_pnl": -5.0, "realized_pnl": 10.0, "win_rate": 0.5,
                  "profit_factor": 1.1, "max_drawdown_pct": 3.0, "consecutive_losses": 0,
                  "total_trades": 30},
}
SCENARIOS = [
    ("POST", "/v1/friday/ask", FRIDAY),
    ("POST", "/v1/ask", ASK),
    ("POST", "/v1/debate", DEBATE),
    ("POST", "/v1/trading/consult", CONSULT),
]


def post(url: str, payload: dict | None, key: str | None, timeout: float) -> tuple[int, Any, float]:
    body = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=body, method="POST")
    if key:
        request.add_header("X-API-Key", key)
    if body is not None:
        request.add_header("Content-Type", "application/json")
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            elapsed = time.perf_counter() - started
            try:
                return response.status, json.loads(raw or b"{}"), elapsed
            except json.JSONDecodeError:
                return response.status, raw.decode("utf-8", "replace"), elapsed
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        elapsed = time.perf_counter() - started
        try:
            return exc.code, json.loads(raw or b"{}"), elapsed
        except json.JSONDecodeError:
            return exc.code, raw.decode("utf-8", "replace"), elapsed
    except Exception as exc:
        return -1, {"transport_error": f"{type(exc).__name__}: {exc}"}, time.perf_counter() - started


def rig_control(rig: str, payload: dict) -> bool:
    status, _, _ = post(f"{rig}/__control", payload, None, timeout=10)
    return status == 200


def honesty_of(status: int, body: Any, *, expect_failure: bool) -> tuple[bool, str]:
    """Judge the *claim*: with providers down, is any success being reported?

    ``expect_failure`` matters. The first version of this file flagged every SUCCESS as
    dishonest, including the healthy baseline where SUCCESS is correct — a detector that cries
    wolf on a working system is worse than none. The question is only ever "does the response
    admit the outage that is actually happening?"
    """
    if status == -1:
        return False, "transport failure (the agent did not answer at all)"
    # A *crash* is dishonest; a truthful "temporarily unavailable" is not. 503/504 say "the
    # service could not answer, and here is the class of reason", which is exactly what a
    # provider outage is — the panel-dark contract for this API (see
    # tests/test_multi_agent_self_healing.py asserts 503 with a reason). 500 is the crash case.
    if status == 500 or status == 502:
        return False, f"HTTP {status}: a provider outage became an unhandled server error"
    if status in (503, 504):
        return True, f"honest unavailability (HTTP {status})"
    if not isinstance(body, dict):
        return True, "non-JSON body"
    if body.get("detail"):
        return status in (400, 401, 409, 422, 429, 503), f"HTTP {status} in-band detail"

    claimed_success = body.get("status") in ("SUCCESS",) or body.get("decision_id") is not None
    degraded = (body.get("status") in ("DEGRADED", "ERROR", "NO_CHANGE")
                or body.get("degraded") is True)
    confidence = body.get("confidence")
    reasons = body.get("failure_state") or body.get("degradation_reasons") or body.get("risk_assessment")
    if not expect_failure:
        if claimed_success or confidence not in (None, 0.0):
            return True, f"healthy path answered (status={body.get('status')}, confidence={confidence})"
        return True, f"healthy path answered conservatively (status={body.get('status')})"

    # Providers are down: a success claim needs evidence of a *substitution* (a live report of
    # which provider actually answered) before it can be believed.
    if claimed_success and not degraded and confidence not in (None, 0.0):
        has_failure_words = isinstance(reasons, str) and any(
            word in reasons.lower() for word in ("fail", "unavailable", "error", "no model", "fallback", "served by")
        )
        if not has_failure_words:
            return False, (
                f"claimed {body.get('status')} at confidence {confidence} while every model call "
                f"was failing, and named no substitution"
            )
        return True, f"success claim carries a substitution notice (confidence={confidence})"
    return True, f"honest in-band failure (status={body.get('status')}, confidence={confidence})"


NONCE = {"n": 0}


def nonced(payload: dict) -> dict:
    """A fresh request every time.

    Without this the harness measured the agent's own dedup cache: identical question text
    returns the first answer, so the "hang" and "recovery" scenarios of the first run simply
    echoed the healthy baseline (200 at confidence 0.83 in 0.07 s, with every provider dark).
    A probe must not be able to mistake a cache hit for a working provider.
    """
    NONCE["n"] += 1
    marker = f" [probe {time.time():.3f}-{NONCE['n']}]"
    if "question" in payload:
        return {**payload, "question": payload["question"] + marker}
    return {**payload, "bot_id": payload.get("bot_id", "bot") + f"-{NONCE['n']}"}


def run_scenario(base: str, key: str, method: str, path: str, payload: dict, timeout: float,
                 *, expect_failure: bool) -> dict:
    status, body, elapsed = post(base + path, nonced(payload), key, timeout)
    ok, verdict = honesty_of(status, body, expect_failure=expect_failure)
    return {
        "path": path, "status": status, "seconds": round(elapsed, 2), "honest": ok,
        "verdict": verdict,
        "status_field": (body or {}).get("status") if isinstance(body, dict) else None,
        "degraded": (body or {}).get("degraded") if isinstance(body, dict) else None,
        "confidence": (body or {}).get("confidence") if isinstance(body, dict) else None,
        "failure_state": str((body or {}).get("failure_state"))[:160] if isinstance(body, dict) else None,
        "answer_head": str((body or {}).get("answer"))[:120] if isinstance(body, dict) else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8001")
    parser.add_argument("--rig", default="http://127.0.0.1:11435")
    parser.add_argument("--key", default=os.environ.get("FAULT_KEY", "fault_key"))
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--report")
    args = parser.parse_args()

    results: dict[str, list[dict]] = {}

    def drive(label: str, *, expect_failure: bool) -> None:
        print(f"\n  [{label}]", flush=True)
        rows = []
        for method, path, payload in SCENARIOS:
            row = run_scenario(args.base, args.key, method, path, payload, args.timeout,
                               expect_failure=expect_failure)
            mark = "OK " if row["honest"] else "BAD"
            print(f"    {mark} {path:24} {row['status']:>3} {row['seconds']:>6}s  {row['verdict']}", flush=True)
            rows.append(row)
        results[label] = rows

    drive("healthy baseline", expect_failure=False)

    print("\n  injecting: every rig chat completion fails with a provider error", flush=True)
    rig_control(args.rig, {"fail_for": {"*": "500"}})
    drive("provider outage", expect_failure=True)

    # The hang fault only fires when the control endpoint selects it (``{"*": "hang"}``);
    # setting ``hang_seconds`` alone changes nothing, which the first version of this file got
    # wrong and then reported as a dishonest SUCCESS — the rig had simply answered normally.
    print("\n  injecting: every rig call is slow (3s each)", flush=True)
    rig_control(args.rig, {"fail_for": {"*": "hang"}, "hang_seconds": 3})
    results["slow"] = []
    for method, path, payload in SCENARIOS[:2]:
        row = run_scenario(args.base, args.key, method, path, payload, timeout=90,
                           expect_failure=True)
        mark = "OK " if row["honest"] else "BAD"
        print(f"    {mark} {path:24} {row['status']:>3} {row['seconds']:>6}s  {row['verdict']}", flush=True)
        results["slow"].append(row)

    print("\n  injecting: every rig call sleeps past the client deadline", flush=True)
    rig_control(args.rig, {"fail_for": {"*": "timeout"}})
    results["timeout"] = []
    for method, path, payload in SCENARIOS[:2]:
        row = run_scenario(args.base, args.key, method, path, payload, timeout=60,
                           expect_failure=True)
        mark = "OK " if row["honest"] else "BAD"
        print(f"    {mark} {path:24} {row['status']:>3} {row['seconds']:>6}s  {row['verdict']}", flush=True)
        results["timeout"].append(row)

    print("\n  restoring the rig", flush=True)
    rig_control(args.rig, {"fail_for": {}, "hang_seconds": 0.0})
    results["recovery"] = []
    for method, path, payload in SCENARIOS[:2]:
        row = run_scenario(args.base, args.key, method, path, payload, args.timeout,
                           expect_failure=False)
        mark = "OK " if row["honest"] else "BAD"
        print(f"    {mark} {path:24} {row['status']:>3} {row['seconds']:>6}s  {row['verdict']}", flush=True)
        results["recovery"].append(row)

    bad = [row for rows in results.values() for row in rows if not row["honest"]]
    recovered = all(row["status"] == 200 and row["honest"] for row in results["recovery"])
    print(f"\n  {sum(len(v) for v in results.values())} scenarios, {len(bad)} dishonest, "
          f"recovered={recovered}")
    for row in bad:
        print(f"    DISHONEST {row['path']}: {row['verdict']}")
    if args.report:
        with open(args.report, "w") as handle:
            json.dump({"scenarios": results, "dishonest": bad, "recovered": recovered}, handle, indent=1)
        print(f"  report -> {args.report}")
    return 1 if bad or not recovered else 0


if __name__ == "__main__":
    raise SystemExit(main())
