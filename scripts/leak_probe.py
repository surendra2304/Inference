#!/usr/bin/env python3
"""Sustained-pressure memory probe: does the agent leak, plateau, or just fill its caches?

Why this exists
---------------
"Memory grows during load" has three very different causes and only one of them is a bug:

* **caches filling** — a bounded store growing towards its ceiling, then flat;
* **allocator high-water** — RSS stays high because arenas are not returned to the OS even
  though Python freed the objects (annoying, not a leak);
* **a real leak** — retained growth that keeps returning, request after request, forever.

The honest way to tell them apart is to hold pressure long past every store's ceiling and
watch whether the growth *decays to zero*. This script does that, per round, and reports a
verdict. It also uses the agent's own opt-in allocation diagnostics
(``GET /memory/diagnostics``, enabled with ``ENABLE_MEMORY_DIAGNOSTICS=true``) to name the
allocation site when the growth turns out to be Python-level.

Usage::

    python scripts/leak_probe.py --rounds 12 --per-round 1000 --warm 300
    python scripts/leak_probe.py --path /v1/nexus/intelligence --report /tmp/leak.json

Exit code is 0 for plateau/filling, 1 for a detected leak, 2 when the run could not be
completed (so it can be used as a real-life regression gate).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any


def request(
    base: str,
    api_key: str,
    path: str,
    payload: dict[str, Any] | None = None,
    method: str = "GET",
    timeout: float = 60.0,
) -> tuple[int, bytes]:
    data = None
    if payload is not None:
        data = json.dumps(payload).encode()
    req = urllib.request.Request(base + path, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("X-API-Key", api_key)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except Exception as exc:  # noqa: BLE001 - reported, not raised: a probe survives a blip
        return -1, str(exc).encode()


def emit(text: str = "") -> None:
    """Print immediately: a probe that runs for twenty minutes must be inspectable while it
    runs, not only once it has finished (a buffered run reports nothing, which is
    indistinguishable from a hung one)."""
    print(text, flush=True)


def write_report(path: str, report: dict[str, Any]) -> None:
    """Write the report after every round, so an interrupted run still yields its evidence."""
    if not path:
        return
    with open(path, "w") as handle:
        json.dump(report, handle, indent=2)


def store_rows(base: str, api_key: str) -> dict[str, dict[str, Any]]:
    """Every in-process accumulator, keyed by name (empty when the endpoint is absent)."""
    status, body = request(base, api_key, "/memory/stores")
    if status != 200:
        return {}
    try:
        audit = json.loads(body)
    except json.JSONDecodeError:
        return {}
    return {row["name"]: row for row in audit.get("stores", [])}


def unfilled_capacity(
    before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Remaining capacity of the stores this workload actually grew.

    Not every store can be filled by one route: the outcome and provenance ledgers only fill
    from their own endpoints, so summing the remaining capacity of *all* stores would report
    "still filling" forever and the probe could never reach a verdict. What matters is whether
    a store that the measured traffic *did* grow still had room — that is the case where an
    RSS rise is explained by design rather than by retention.
    """
    grew: dict[str, int] = {}
    for name, row in after.items():
        previous = (before.get(name) or {}).get("used") or 0
        used = row.get("used") or 0
        ceiling = row.get("max_entries") or 0
        if used > previous and ceiling:
            grew[name] = max(0, ceiling - used)
    return {
        "remaining_entries": sum(grew.values()),
        "growing_stores": {name: {"unfilled": room} for name, room in sorted(grew.items())},
        "stores": len(after),
    }


def runtime(base: str, api_key: str) -> dict[str, Any]:
    status, body = request(base, api_key, "/metrics/runtime")
    if status != 200:
        return {}
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return {}


def diagnostics(base: str, api_key: str, action: str, limit: int = 10) -> dict[str, Any]:
    """Opt-in allocation diagnostics; empty dict when the endpoint is disabled (404)."""
    status, body = request(
        base, api_key, f"/memory/diagnostics?action={action}&limit={limit}"
    )
    if status != 200:
        return {}
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return {}


def build_payload(path: str, tag: str, index: int) -> dict[str, Any]:
    """Request shape derived from the route's own schema, not from guesswork.

    ``/v1/nexus/intelligence`` takes a required ``request_id``, ``task_type`` and ``goal``;
    omitting any of them returns 422 before the handler runs and the probe measures nothing
    (that mistake produced a "no leak" reading once already).
    """
    if path.endswith("/nexus/intelligence"):
        return {
            "request_id": f"{tag}-{index}",
            "task_type": "incident_analysis",
            "goal": f"pressure probe {index}",
            "mode": "fast",
            "context": {"probe": tag},
        }
    if path.endswith("/ask"):
        return {"query": f"pressure probe {index}"}
    return {"prompt": f"pressure probe {index}", "request_id": f"{tag}-{index}"}


def drive_round(
    base: str,
    api_key: str,
    path: str,
    tag: str,
    count: int,
    concurrency: int,
    offset: int,
) -> tuple[dict[int, int], float]:
    codes: dict[int, int] = {}

    def one(i: int) -> int:
        status, _ = request(
            base, api_key, path, build_payload(path, tag, offset + i), "POST"
        )
        return status

    started = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for status in pool.map(one, range(count)):
            codes[status] = codes.get(status, 0) + 1
    return codes, time.time() - started


#: Retained bytes per request that are tolerated once the caches have saturated. Below this,
#: allocator behaviour and cache churn dominate; above it, the process is keeping memory for
#: every request it serves, which is a leak by definition.
PER_REQUEST_KB_BUDGET = 0.5


def verdict(
    per_round_mb: list[float],
    per_round_requests: int = 1000,
    remaining_capacity_entries: int | None = None,
) -> tuple[str, str]:
    """Classify the growth trend from the per-round RSS deltas.

    A bounded store still filling shows deltas that *shrink* towards zero; a leak shows
    deltas that stay put. Trend alone is not enough to decide, though: the measured leak in
    this repo produced deltas ``[5.9, 11.1, 2.4, 1.5, 1.5, 1.3]`` MB per 1000 requests — a
    decaying series that a trend-only rule calls a plateau, while the process was in fact
    retaining ~1.4 KB per request (~36 GB/day at 300 rps) forever. So the decay test is
    combined with an absolute per-request budget, which cannot be satisfied by a run that
    only slowed down.
    """
    if len(per_round_mb) < 4:
        return "inconclusive", "need at least 4 rounds to judge a trend"

    # A store that is not yet full grows *by design*, and no rule based on RSS or on traced
    # bytes can tell that apart from a leak. Measured (12,000 requests, 12 rounds): traced
    # growth sat at ~1.5 MB per 1000 requests while the 10,000-record usage-analytics store
    # filled, then dropped to 40 KiB and 75 KiB per 1000 in the two rounds after it saturated
    # — a 20x collapse that a rate-based rule alone reads as a leak. So saturation is a
    # precondition, not a detail: report "filling" and say how much capacity remains.
    if remaining_capacity_entries:
        return (
            "filling",
            f"{remaining_capacity_entries} entries of store capacity remain unfilled; growth "
            f"cannot be called a leak until every ceiling is reached — drive more traffic",
        )
    window = max(2, len(per_round_mb) // 3)
    tail = per_round_mb[-window:]
    head = per_round_mb[:window]
    tail_mean = statistics.fmean(tail)
    head_mean = statistics.fmean(head)

    requests = max(1, len(tail) * per_round_requests)
    retained_kb = (sum(tail) * 1024.0) / requests
    if retained_kb > PER_REQUEST_KB_BUDGET:
        return (
            "leak",
            f"retains {retained_kb:.2f} KB/request in the final {len(tail)} rounds "
            f"(budget {PER_REQUEST_KB_BUDGET}), i.e. {retained_kb * 1024 * 300 * 86400 / 1e9:.1f} "
            f"GB/day at 300 rps; tail mean {tail_mean:.2f} MB/round",
        )
    if tail_mean <= 0.5:
        return "plateau", f"tail growth {tail_mean:.2f} MB/round is within noise"
    if tail_mean >= 0.5 * head_mean:
        return (
            "leak",
            f"growth did not decay: head {head_mean:.2f} MB/round -> tail "
            f"{tail_mean:.2f} MB/round",
        )
    return (
        "plateau",
        f"growth decaying and within budget: head {head_mean:.2f} -> tail {tail_mean:.2f} "
        f"MB/round, {retained_kb:.2f} KB/request",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--api-key", default="drive_key")
    ap.add_argument("--path", default="/v1/nexus/intelligence")
    ap.add_argument("--tag", default="leakprobe")
    ap.add_argument("--warm", type=int, default=300, help="requests before measuring")
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--per-round", type=int, default=1000)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--trend-threshold-mb", type=float, default=0.5)
    ap.add_argument("--report", default="")
    args = ap.parse_args()

    base = args.base.rstrip("/")
    tag = f"{args.tag}-{int(time.time()) % 100000}"
    report: dict[str, Any] = {
        "base": base,
        "path": args.path,
        "tag": tag,
        "rounds": [],
        "config": {
            "warm": args.warm,
            "rounds": args.rounds,
            "per_round": args.per_round,
            "concurrency": args.concurrency,
        },
    }

    before = runtime(base, args.api_key)
    if not before:
        emit("cannot read /metrics/runtime - is the agent up and the key right?")
        return 2
    diagnostics_available = bool(diagnostics(base, args.api_key, "reset"))
    emit(
        f"agent rss={before.get('process', {}).get('rss_mb')}MB "
        f"store_count={before.get('retention', {}).get('store_count')} "
        f"allocation_diagnostics={'on' if diagnostics_available else 'off'}"
    )
    if diagnostics_available:
        diagnostics(base, args.api_key, "reset")

    if args.warm:
        codes, secs = drive_round(
            base, args.api_key, args.path, tag, args.warm, args.concurrency, 0
        )
        emit(f"  warm {args.warm} in {secs:.1f}s codes={codes}")
    if diagnostics_available:
        diagnostics(base, args.api_key, "snapshot")
    # Snapshot of every store *after* warm-up, so the verdict can tell which store the
    # measured traffic actually grew and whether that store still had room.
    stores_before = store_rows(base, args.api_key)

    offset = args.warm
    rss_series: list[float] = []
    per_round_mb: list[float] = []
    all_codes: dict[int, int] = {}
    previous_rss = before.get("process", {}).get("rss_mb") or 0.0
    for rnd in range(args.rounds):
        codes, secs = drive_round(
            base, args.api_key, args.path, tag, args.per_round, args.concurrency, offset
        )
        offset += args.per_round
        for code, n in codes.items():
            all_codes[code] = all_codes.get(code, 0) + n
        now = runtime(base, args.api_key)
        rss = now.get("process", {}).get("rss_mb")
        retention = now.get("retention", {})
        delta = round((rss or 0.0) - previous_rss, 1)
        previous_rss = rss or 0.0
        rss_series.append(rss or 0.0)
        if rnd:
            per_round_mb.append(delta)
        entry: dict[str, Any] = {
            "round": rnd + 1,
            "requests": args.per_round,
            "codes": {str(k): v for k, v in codes.items()},
            "seconds": round(secs, 1),
            "rss_mb": rss,
            "rss_delta_mb": delta,
            "evictions": retention.get("total_evictions"),
            "stores_at_capacity": len(retention.get("stores_at_capacity") or []),
            "within_bound": retention.get("within_bound"),
        }
        if diagnostics_available:
            snap = diagnostics(base, args.api_key, "snapshot")
            entry["traced_mb"] = snap.get("traced_current_mb")
            growth = snap.get("growth_since_previous_snapshot") or []
            entry["traced_growth_kib"] = round(
                sum(row.get("size_diff_kib", 0.0) for row in growth), 1
            )
            entry["top_growth_sites"] = [row.get("allocation_site") for row in growth[:3]]
            if growth:
                emit(
                    f"  round {rnd + 1}: rss={rss}MB ({delta:+.1f}) "
                    f"traced_growth={entry['traced_growth_kib']}KiB "
                    f"top={growth[0]['allocation_site'].replace('/home/user/Inference/', '')}"
                )
        report["rounds"].append(entry)
        emit(
            f"  round {rnd + 1}: rss={rss}MB ({delta:+.1f} MB/round) "
            f"codes={codes} evictions={entry['evictions']}"
        )
        write_report(args.report, report)

    capacity = unfilled_capacity(stores_before, store_rows(base, args.api_key))
    remaining = capacity.get("remaining_entries") if capacity else None
    kind, reason = verdict(per_round_mb, args.per_round, remaining)
    final = runtime(base, args.api_key)
    report["verdict"] = kind
    report["verdict_reason"] = reason
    report["per_round_mb"] = per_round_mb
    report["codes"] = {str(k): v for k, v in all_codes.items()}
    report["final_rss_mb"] = final.get("process", {}).get("rss_mb")
    report["store_capacity"] = capacity
    report["final_retention"] = final.get("retention", {})
    emit(
        f"\nVERDICT: {kind} ({reason})\n"
        f"  rss {rss_series[0] if rss_series else '?'} -> {report['final_rss_mb']} MB over "
        f"{args.rounds * args.per_round} requests; per-round {per_round_mb}"
    )
    if args.report:
        write_report(args.report, report)
        emit(f"  report -> {args.report}")
    return 1 if kind == "leak" else 0  # "filling"/"inconclusive"/"plateau" are not failures


if __name__ == "__main__":
    sys.exit(main())
