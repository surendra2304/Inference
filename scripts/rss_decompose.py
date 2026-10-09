"""Decompose the agent's RSS: is the growth heap (RssAnon) or file-backed (RssFile)?

Heap growth that survives every round is a leak. File-backed growth is mmap'd file pages —
reclaimable, not a leak. Also correlates growth with the SQLite database size, because a
process that caches a growing database looks like a leak to RSS-only probes.
"""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = os.environ.get("AGENT_BASE", "http://127.0.0.1:8000")
KEY = os.environ.get("AGENT_KEY", "drive_key")


def api(path, payload=None, method="GET", timeout=180):
    req = urllib.request.Request(BASE + path, method=method)
    req.add_header("X-API-Key", KEY)
    data = None
    if payload is not None:
        data = json.dumps(payload).encode()
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, data=data, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as exc:
        return exc.code, {}


def agent_pid():
    out = subprocess.run(["pgrep", "-f", "uvicorn app.main:app"], capture_output=True, text=True)
    pids = [int(p) for p in out.stdout.split()]
    return pids[0] if pids else None


def memory_mb(pid):
    """RssAnon / RssFile / total from /proc/<pid>/status, in MB."""
    values = {}
    with open(f"/proc/{pid}/status") as handle:
        for line in handle:
            for key in ("VmRSS", "RssAnon", "RssFile", "RssShmem"):
                if line.startswith(key):
                    values[key] = int(line.split()[1]) / 1024.0
    return values


def db_mb():
    total = 0.0
    root = "data"
    for dirpath, _, names in os.walk(root):
        for name in names:
            if name.endswith((".db", ".db-wal", ".db-shm", ".sqlite", ".sqlite3")):
                try:
                    total += os.path.getsize(os.path.join(dirpath, name))
                except OSError:
                    pass
    return total / (1024.0 * 1024.0)


def burst(n, c=8):
    def one(i):
        status, _ = api("/ask", {"question": f"memory decomposition probe {i}", "mode": "fast"}, "POST")
        return status

    codes: dict[int, int] = {}
    with ThreadPoolExecutor(max_workers=c) as pool:
        for status in pool.map(one, range(n)):
            codes[status] = codes.get(status, 0) + 1
    return codes


def line(tag, pid):
    mem = memory_mb(pid)
    print(f"  {tag:>22}  rss={mem.get('VmRSS', 0):7.1f}  anon={mem.get('RssAnon', 0):7.1f}  "
          f"file={mem.get('RssFile', 0):6.1f}  shmem={mem.get('RssShmem', 0):5.1f}  "
          f"db={db_mb():7.1f} MB", flush=True)
    return mem


if __name__ == "__main__":
    pid = agent_pid()
    if pid is None:
        sys.exit("agent not running")
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    per_round = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    print(f"agent pid {pid}; tracing off; {rounds} rounds x {per_round} /ask", flush=True)
    print("  diagnostics stop:", api("/memory/diagnostics?action=stop", method="POST")[0], flush=True)
    line("baseline", pid)
    rows = []
    for i in range(rounds):
        codes = burst(per_round)
        mem = line(f"round {i + 1}", pid)
        rows.append({"round": i + 1, "codes": codes, "rss": mem.get("VmRSS"),
                     "anon": mem.get("RssAnon"), "file": mem.get("RssFile"), "db_mb": round(db_mb(), 1)})
        time.sleep(0.5)
    print(json.dumps(rows, indent=1))
