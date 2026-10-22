"""Memory forensics on a live agent: heap vs file, Python blocks vs C memory.

Answers, in order, the questions a leak hunt actually has:

1. Is the growth anonymous heap (candidate leak) or file-backed (mmap/cache, not a leak)?
2. Is it Python objects (``sys.getallocatedblocks`` grows; the type census names them) or
   C-level memory such as sqlite3/malloc arenas (blocks flat, RSS grows)?
3. Does it come back after the load stops (allocator high-water) or stay (retention)?

Tracing is switched **off** for the whole run: tracemalloc inflates the very numbers this
script measures. Use ``/memory/diagnostics?action=snapshot`` separately for site attribution.

    python scripts/memory_forensics.py --rounds 4 --per-round 400 --report /tmp/forensics.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any


def api(base: str, key: str, path: str, payload: dict | None = None, method: str = "GET",
        timeout: float = 180.0) -> tuple[int, Any]:
    request = urllib.request.Request(base + path, method=method)
    request.add_header("X-API-Key", key)
    data = None
    if payload is not None:
        data = json.dumps(payload).encode()
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, data=data, timeout=timeout) as response:
            raw = response.read()
            try:
                return response.status, json.loads(raw or b"{}")
            except json.JSONDecodeError:
                return response.status, raw.decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read() or b"{}")
        except json.JSONDecodeError:
            return exc.code, {}


def listening_pid(port: int = 8000) -> int | None:
    """The pid actually listening on ``port``.

    Not ``pgrep -f uvicorn``: that pattern also matches the shell wrapper around the server,
    and it will happily return a pid from a previous incarnation that is still draining. A
    probe that measures the wrong process produces *plausible* numbers (measured: a frozen
    process read back byte-identical RSS for fifty seconds while the load kept running).
    """
    result = subprocess.run(["ss", "-ltnp"], capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if f":{port} " not in line:
            continue
        marker = line.find("pid=")
        if marker != -1:
            digits = ""
            for char in line[marker + 4:]:
                if char.isdigit():
                    digits += char
                else:
                    break
            if digits and os.path.exists(f"/proc/{digits}"):
                return int(digits)
    return None


def memory_mb(pid: int) -> dict[str, float]:
    wanted = ("VmRSS", "RssAnon", "RssFile", "RssShmem", "VmData")
    values: dict[str, float] = {}
    try:
        with open(f"/proc/{pid}/status") as handle:
            for line in handle:
                key = line.split(":", 1)[0]
                if key in wanted:
                    values[key] = int(line.split()[1]) / 1024.0
    except (FileNotFoundError, ProcessLookupError):
        return {"gone": 1.0}
    return values


def snapshot(base: str, key: str, pid: int) -> dict[str, Any]:
    status, blocks = api(base, key, "/memory/diagnostics?action=blocks")
    mem = memory_mb(pid)
    return {
        "rss_mb": round(mem.get("VmRSS", 0.0), 1),
        "anon_mb": round(mem.get("RssAnon", 0.0), 1),
        "file_mb": round(mem.get("RssFile", 0.0), 1),
        "vmdata_mb": round(mem.get("VmData", 0.0), 1),
        "allocated_blocks": (blocks or {}).get("allocated_blocks") if status == 200 else None,
        "gc_collected": (blocks or {}).get("gc_collected") if status == 200 else None,
        "live_object_count": (blocks or {}).get("live_object_count") if status == 200 else None,
        "top_types": (blocks or {}).get("live_object_types") if status == 200 else None,
        "blocks_status": status,
    }


def burst(base: str, key: str, count: int, concurrency: int, offset: int) -> dict[int, int]:
    def one(index: int) -> int:
        status, _ = api(
            base, key, "/ask",
            {"question": f"forensics probe {offset + index}", "mode": "fast"}, "POST",
        )
        return status

    codes: dict[int, int] = {}
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for status in pool.map(one, range(count)):
            codes[status] = codes.get(status, 0) + 1
    return codes


def show(tag: str, row: dict[str, Any]) -> None:
    print(
        f"  {tag:>14}  rss={row['rss_mb']:7.1f}  anon={row['anon_mb']:7.1f}  "
        f"vmdata={row['vmdata_mb']:7.1f}  blocks={row['allocated_blocks']}  "
        f"live={row['live_object_count']}  gc={row['gc_collected']}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--key", default=os.environ.get("AGENT_KEY", "drive_key"))
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--per-round", type=int, default=400)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--idle-seconds", type=float, default=30.0)
    parser.add_argument("--report")
    args = parser.parse_args()

    port = 8000
    if ":" in args.base.rsplit("/", 1)[-1]:
        port = int(args.base.rsplit(":", 1)[-1].rstrip("/"))
    pid = listening_pid(port)
    if pid is None:
        print(f"no process listening on :{port}", file=sys.stderr)
        return 2

    # The endpoint is GET; a POST returns 405 and the probe would keep tracing on, inflating
    # exactly the numbers it is trying to measure.
    status, body = api(args.base, args.key, "/memory/diagnostics?action=stop")
    print(f"agent pid {pid}; diagnostics stop -> {status} {body}", flush=True)
    if status != 200:
        print("this agent has no diagnostics endpoint; blocks attribution unavailable",
              file=sys.stderr)

    rows: list[dict[str, Any]] = []
    baseline = snapshot(args.base, args.key, pid)
    show("baseline", baseline)
    rows.append({"phase": "baseline", **baseline})

    for round_number in range(1, args.rounds + 1):
        codes = burst(args.base, args.key, args.per_round, args.concurrency,
                      offset=round_number * args.per_round)
        row = snapshot(args.base, args.key, pid)
        row["codes"] = {str(k): v for k, v in codes.items()}
        show(f"round {round_number}", row)
        rows.append({"phase": f"round {round_number}", **row})

    if args.idle_seconds > 0:
        time.sleep(args.idle_seconds)
        idle = snapshot(args.base, args.key, pid)
        show("idle", idle)
        rows.append({"phase": "idle", **idle})

    print(json.dumps(rows, indent=1), flush=True)
    if args.report:
        with open(args.report, "w") as handle:
            json.dump({"pid": pid, "rows": rows}, handle, indent=1)
        print(f"report -> {args.report}")

    # Reading of the run, printed so the numbers are not left to interpretation.
    if len(rows) >= 3 and rows[0].get("allocated_blocks") is not None:
        first, last = rows[0], rows[-1]
        rss_delta = last["rss_mb"] - first["rss_mb"]
        block_delta = (last["allocated_blocks"] or 0) - (first["allocated_blocks"] or 0)
        print(
            f"  reading: RSS {rss_delta:+.1f} MB, python blocks {block_delta:+d} "
            f"({block_delta * 32 / 1024.0:+.1f} MB if ~32 B/block) — "
            + ("retention is in Python objects" if block_delta > 1000 * len(rows)
               else "growth is NOT Python object retention (C-level/arena memory)")
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
