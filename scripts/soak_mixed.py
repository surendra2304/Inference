"""Mixed-endpoint soak: sustained real traffic across the whole surface, with memory verdicts.

Unlike ``leak_probe.py`` (one endpoint, saturation verdict), this drives the *mix* an operator
actually generates — deliberative asks, debate, nexus intelligence, sentinel analysis, trading
consult, analytics reads, memory reads — and answers three questions on one timeline:

* does anything 5xx or time out under sustained mixed load?
* does the resident set plateau, or does it grow with every round?
* does the *object* count plateau (the only memory signal that cannot be faked by the
  allocator; see ``scripts/memory_forensics.py`` for why RSS alone is not enough)?

The report keeps the per-round series for every metric, so a verdict can be re-derived from the
stored numbers instead of trusted.

    python scripts/soak_mixed.py --minutes 30 --concurrency 6 --report /tmp/soak_mixed.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any

WORKLOADS: list[tuple[str, str, dict[str, Any]]] = [
    ("POST", "/ask", {"question": "Should the write path be sharded by tenant?", "mode": "fast"}),
    ("POST", "/ask", {"question": "Review the retry policy for payment capture.", "mode": "review"}),
    ("POST", "/v1/debate", {"question": "Is a single-writer SQLite service viable at 300 rps?",
                            "mode": "debate"}),
    ("POST", "/v1/nexus/intelligence", {
        "request_id": "soak-{i}", "task_type": "incident_analysis",
        "goal": "Assess the incident response plan for the checkout service",
        "mode": "fast", "context": {"region": "ap-south-1", "severity": "sev2"},
    }),
    ("POST", "/v1/sentinel/analyze", {
        "request_id": "soak-sent-{i}", "analysis_type": "risk_scoring",
        "target_context": {"asset_type": "web-service", "technologies_detected": ["nginx"],
                           "exposure_level": "public_internet"},
        "findings": [{"finding_id": "f1", "title": "TLS 1.0 enabled", "severity": "MEDIUM",
                      "description": "Old TLS."}],
    }),
    ("POST", "/v1/trading/consult", {
        "bot_id": "soak-bot", "trading_mode": "PAPER", "consultation_reason": "SCHEDULED",
        "telemetry": {"equity": 1000.0, "unrealized_pnl": -12.0, "realized_pnl": 40.0,
                      "win_rate": 0.52, "profit_factor": 1.15, "max_drawdown_pct": 4.0,
                      "consecutive_losses": 1, "total_trades": 60},
    }),
    ("GET", "/v1/admin/costs", {}),
    ("GET", "/v1/admin/usage", {}),
    ("GET", "/metrics/runtime", {}),
    ("GET", "/memory/stores", {}),
    ("GET", "/v1/analytics/strategy-bank?task_type=code_generation&query=retry", {}),
    ("GET", "/health", {}),
]


def call(base: str, key: str, method: str, path: str, payload: dict | None, timeout: float,
         index: int) -> tuple[str, int, float]:
    body = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(base + path, data=body, method=method)
    request.add_header("X-API-Key", key)
    if body is not None:
        request.add_header("Content-Type", "application/json")
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read()
            return path, response.status, time.perf_counter() - started
    except urllib.error.HTTPError as exc:
        try:
            exc.read()
        except Exception:
            pass
        return path, exc.code, time.perf_counter() - started
    except Exception:
        return path, -1, time.perf_counter() - started


def overview(base: str, key: str, path: str) -> Any:
    request = urllib.request.Request(base + path)
    request.add_header("X-API-Key", key)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read() or b"{}")
    except Exception:
        return {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--key", default=os.environ.get("AGENT_KEY", "drive_key"))
    parser.add_argument("--minutes", type=float, default=30.0)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--round-seconds", type=float, default=60.0)
    parser.add_argument("--report")
    args = parser.parse_args()

    base = args.base.rstrip("/")
    deadline = time.time() + args.minutes * 60.0
    rounds: list[dict[str, Any]] = []
    counter = 0

    def one() -> tuple[str, int, float]:
        nonlocal counter
        counter += 1
        method, path, payload = WORKLOADS[counter % len(WORKLOADS)]
        if payload:
            payload = json.loads(json.dumps(payload).replace("{i}", str(counter)))
        return call(base, args.key, method, path, payload, timeout=180.0, index=counter)

    print(f"  soak: {args.minutes:.0f} min, concurrency {args.concurrency}, "
          f"{len(WORKLOADS)} workload shapes", flush=True)
    while time.time() < deadline:
        round_started = time.time()
        codes: Counter[int] = Counter()
        latencies: list[float] = []
        per_path_errors: Counter[str] = Counter()
        requests = 0
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            while time.time() - round_started < args.round_seconds and time.time() < deadline:
                batch = [pool.submit(one) for _ in range(args.concurrency * 4)]
                for future in batch:
                    path, status, elapsed = future.result()
                    codes[status] += 1
                    latencies.append(elapsed)
                    requests += 1
                    if status >= 500 or status == -1:
                        per_path_errors[path] += 1

        runtime = overview(base, args.key, "/metrics/runtime")
        blocks = overview(base, args.key, "/memory/diagnostics?action=blocks")
        guard = runtime.get("memory_guard") or {}
        row = {
            "round": len(rounds) + 1,
            "requests": requests,
            "codes": {str(k): v for k, v in sorted(codes.items())},
            "p50_ms": round(statistics.median(latencies) * 1000, 1) if latencies else None,
            "p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)] * 1000, 1) if latencies else None,
            "rss_mb": (runtime.get("process") or {}).get("rss_mb"),
            "blocks": blocks.get("allocated_blocks"),
            "live_objects": blocks.get("live_object_count"),
            "trims": guard.get("trims"),
            "reclaimed_mb": guard.get("reclaimed_total_mb"),
            "server_errors_by_path": dict(per_path_errors),
            "within_bound": (runtime.get("retention") or {}).get("within_bound"),
        }
        rounds.append(row)
        print(
            f"  round {row['round']}: {requests} req, codes {row['codes']}, "
            f"p50 {row['p50_ms']}ms p95 {row['p95_ms']}ms, rss {row['rss_mb']}MB, "
            f"blocks {row['blocks']}, trims {row['trims']} ({row['reclaimed_mb']} MB)",
            flush=True,
        )

    total = sum(row["requests"] for row in rounds)
    errors = sum(sum(v for k, v in row["codes"].items() if int(k) >= 500 or int(k) == -1)
                 for row in rounds)
    rss_series = [row["rss_mb"] for row in rounds if row["rss_mb"] is not None]
    block_series = [row["blocks"] for row in rounds if row["blocks"] is not None]
    growth = [round(rss_series[i + 1] - rss_series[i], 1) for i in range(len(rss_series) - 1)]

    # Two signals, judged on their own terms, because they fail differently:
    #
    # * RSS is allocator-sensitive. It plateaus when the process is healthy and the allocator
    #   has stopped growing arenas; it can also move with nothing retained at all (measured:
    #   +1.3 MB over 1,600 requests with live blocks returning exactly to baseline).
    # * Live Python blocks are the leak signal, but a *busy* process holds request-scoped work
    #   (measured: a sample 1 s after a burst read +602,105 blocks and 40 s later read below
    #   its pre-burst value). Under continuous load this soak cannot ever be quiet, so the
    #   comparable quantity is growth per request, and the shape test is whether it keeps
    #   climbing round after round (retention) or levels off once the bounded stores fill.
    #
    # The authoritative quiet-process verdict lives in the real-time drive
    # (``check_memory_retention``); this soak's job is to catch growth under sustained mixed
    # traffic and to keep the series for inspection.
    total_requests = max(1, sum(row["requests"] for row in rounds))
    blocks_per_request: float | None = None
    verdict = "inconclusive"
    detail = "not enough rounds to judge"
    if len(growth) >= 4:
        tail = statistics.fmean(growth[-max(2, len(growth) // 3):])
        if block_series and len(block_series) >= 4:
            blocks_per_request = (block_series[-1] - block_series[0]) / total_requests
            deltas = [block_series[i + 1] - block_series[i] for i in range(len(block_series) - 1)]
            tail_deltas = deltas[-max(2, len(deltas) // 3):]
            head_deltas = deltas[: max(1, len(deltas) // 3)]
            # 20 blocks/request is ~0.6 KB/request at 32 B/block: an order of magnitude below
            # the leak this repo was measured with (1.4 KB/request) and well above the
            # request-scoped noise seen under load.
            climbing = statistics.fmean(tail_deltas) > statistics.fmean(head_deltas)
            if blocks_per_request > 20.0 and climbing:
                verdict, detail = "object-retention", (
                    f"live blocks {block_series[0]} -> {block_series[-1]} over {total_requests} "
                    f"requests ({blocks_per_request:.1f} blocks/request) and still climbing"
                )
            elif tail <= 0.5:
                verdict, detail = "plateau", (
                    f"tail RSS growth {tail:.2f} MB/round; blocks {blocks_per_request:+.1f}/request"
                )
            else:
                verdict, detail = "growing", (
                    f"tail RSS growth {tail:.2f} MB/round (blocks {blocks_per_request:+.1f}/request)"
                )
        elif tail <= 0.5:
            verdict, detail = "plateau", f"tail RSS growth {tail:.2f} MB/round"
        else:
            verdict, detail = "growing", f"tail RSS growth {tail:.2f} MB/round (no block counter)"

    report = {
        "minutes": args.minutes, "concurrency": args.concurrency, "rounds": rounds,
        "total_requests": total, "server_or_transport_errors": errors,
        "rss_series": rss_series, "block_series": block_series, "growth_mb_per_round": growth,
        "verdict": verdict, "detail": detail,
        "blocks_per_request": blocks_per_request,
    }
    print(f"\nVERDICT: {verdict} — {detail}")
    print(f"  {total} requests, {errors} server/transport errors, "
          f"rss {rss_series[0] if rss_series else '?'} -> {rss_series[-1] if rss_series else '?'} MB, "
          f"blocks {block_series[0] if block_series else '?'} -> {block_series[-1] if block_series else '?'}")
    if args.report:
        with open(args.report, "w") as handle:
            json.dump(report, handle, indent=1)
        print(f"  report -> {args.report}")
    return 1 if errors or verdict == "object-retention" else 0


if __name__ == "__main__":
    raise SystemExit(main())
