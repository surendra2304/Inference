#!/usr/bin/env python3
"""Local OpenAI-compatible model server: a conformance, stress and failure rig.

WHAT THIS IS
------------
A dependency-free HTTP server that speaks the OpenAI ``/v1/chat/completions``
contract (JSON and SSE) so the Inference agent can be driven end-to-end on a
machine with no cloud credentials. It exists because every failure mode that
actually matters in this codebase — provider timeouts, 429 key quarantine,
fallback escalation, empty completions, malformed bodies, truncated answers,
panel self-healing — requires *a provider that behaves badly on demand*, and no
cloud vendor will do that to order.

WHAT THIS IS NOT
----------------
It is **not** a language model and it does not reason. ``synthesize()`` is a
deterministic template engine keyed on the persona and the question, seeded by a
hash so the same request always produces the same answer. That determinism is the
point: it makes the *agent's* behaviour reproducible while the fake model is held
constant. Any claim about answer quality produced with this rig is invalid; claims
about the agent's routing, resilience, honesty, persistence and concurrency are
valid, because those are exercised over real HTTP through the real
``OpenAICompatibleProvider`` code path.

USAGE
-----
    # terminal 1 — the rig, on Ollama's default port so no config is needed
    python scripts/local_model_server.py --port 11434

    # terminal 2 — point the agent at it
    LOCAL_ENABLED=true LOCAL_MODEL=rig-8b python -m uvicorn app.main:app --port 8000

PER-REQUEST FAULT DIRECTIVES
----------------------------
Embed a directive anywhere in the prompt (system or user) to fault one call:

    __LATENCY_MS:1500      delay the response by 1.5s
    __FAIL:429              HTTP 429 with Retry-After
    __FAIL:503              HTTP 503
    __FAIL:500              HTTP 500 with an error body
    __FAIL:timeout          sleep past the client deadline (no response)
    __FAIL:empty            200 with an empty choices array (no model output)
    __FAIL:malformed        200 with a body that is not JSON
    __FAIL:truncate         200 with finish_reason="length" and a cut-off answer
    __FAIL:sse              fail mid-stream on the streaming endpoint
    __NO_METRICS            omit metric-bearing bullets (test evidence extraction)

GLOBAL CONTROL (affects every subsequent request)
-------------------------------------------------
    curl -X POST localhost:11434/__control -d '{"slots":2,"base_latency_ms":50}'
    curl -X POST localhost:11434/__control -d '{"flaky_pct":30}'
    curl -X POST localhost:11434/__control -d '{"fail_for":{"security_analyst":"503"}}'
    curl -X POST localhost:11434/__control -d '{"conflict_mode":true}'
    curl localhost:11434/__stats
    curl -X POST localhost:11434/__control -d '{"reset":true}'

``fail_for`` is the important one: it breaks a *single named specialist* while the
rest of the panel stays healthy, which is the only way to test whether peer
coverage and self-healing actually work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

# --------------------------------------------------------------------------------------
# Persona and prompt handling
# --------------------------------------------------------------------------------------

#: Bracketed tag, when a caller supplies one, e.g. "[security_analyst]".
_TAG_RE = re.compile(r"\[([a-z0-9_]{3,40})\]", re.IGNORECASE)
#: The Inference panel addresses each specialist as "As the <Role>, ..." in the user
#: turn — the system instruction is shared boilerplate and carries no identity. Fault
#: injection has to key off this form or it cannot target a single specialist.
_ROLE_RE = re.compile(r"\bAs the ([A-Z][A-Za-z0-9 _-]{2,40}?)[,.:]", re.MULTILINE)
_DIRECTIVE_RE = re.compile(
    r"__(LATENCY_MS|FAIL|NO_METRICS)(?::([A-Za-z0-9_]+))?", re.IGNORECASE
)


def slugify_role(role: str) -> str:
    """'Security Analyst' -> 'security_analyst'."""
    return re.sub(r"[^a-z0-9]+", "_", role.strip().lower()).strip("_")

ARCHETYPES: dict[str, tuple[str, ...]] = {
    "security": ("security", "critic", "reviewer", "fact_check", "audit", "sentinel", "risk"),
    "architecture": ("architect", "strateg", "proposer", "requirements", "planner", "analyst"),
    "implementation": ("coder", "generator", "debugger", "test", "documentation", "writer", "devops"),
    "data": ("data", "trading", "market", "forecast", "metric", "performance"),
}


def detect_persona(system_instruction: str, messages: list[dict[str, Any]]) -> str:
    """Identify the calling specialist.

    Three sources, in decreasing reliability:

    1. a bracketed ``[agent_id]`` tag anywhere in the system turn, if a caller adds one;
    2. the ``As the <Role>,`` phrasing the Inference panel uses in the user turn — this
       is the form actually emitted today, and the role is slugified to an agent id;
    3. ``specialist``, when neither is present.

    Getting this wrong silently defeats fault injection: a schedule naming
    ``critic`` matches nothing if the rig resolved the caller to ``specialist``, and
    the resulting test passes while injecting no fault at all.
    """
    match = _TAG_RE.search(system_instruction or "")
    if match:
        return match.group(1).lower()
    haystack = [system_instruction or ""] + [str(m.get("content", "")) for m in messages]
    for text in haystack:
        role_match = _ROLE_RE.search(text)
        if role_match:
            return slugify_role(role_match.group(1))
    for message in messages:
        if message.get("role") == "system":
            inner = _TAG_RE.search(str(message.get("content", "")))
            if inner:
                return inner.group(1).lower()
    return "specialist"


def resolve_fault_schedule(schedule: dict[str, str], persona: str) -> str:
    """Match a fault schedule entry against a resolved persona.

    Accepts an exact persona match or a substring match in either direction, so both
    ``{"critic": "503"}`` and ``{"researcher": "429"}`` target the right specialist
    without the operator having to know how the rig slugified the role.
    """
    if not schedule:
        return ""
    if persona in schedule:
        return schedule[persona]
    for key, value in schedule.items():
        clean = key.strip().lower()
        if clean and (clean in persona or persona in clean):
            return value
    return ""


def detect_archetype(persona: str, system_instruction: str) -> str:
    haystack = f"{persona} {system_instruction}".lower()
    for archetype, needles in ARCHETYPES.items():
        if any(needle in haystack for needle in needles):
            return archetype
    return "architecture"


def extract_question(messages: list[dict[str, Any]]) -> str:
    """Pull the human's question out of the message list, trailing markers stripped."""
    for message in reversed(messages):
        if message.get("role") == "user":
            text = str(message.get("content", "")).strip()
            if text:
                return " ".join(text.split())[:400]
    return "(no user message)"


def parse_directives(text: str) -> dict[str, str]:
    """Extract ``__FAIL:x`` / ``__LATENCY_MS:n`` / ``__NO_METRICS`` from prompt text."""
    directives: dict[str, str] = {}
    for name, value in _DIRECTIVE_RE.findall(text):
        directives[name.upper()] = (value or "").lower()
    return directives


# --------------------------------------------------------------------------------------
# Deterministic answer synthesis
# --------------------------------------------------------------------------------------

_BULLETS: dict[str, tuple[str, ...]] = {
    "security": (
        "Bind every parameter instead of building SQL by concatenation; a blacklist of tokens such as "
        "``DROP TABLE`` or ``--`` is bypassed by encoding and corrupts legitimate input.",
        "Measured overhead of bound parameters is <2 ms p95 versus a full-table ``LIKE`` scan at about 40 ms "
        "p95 on 100k rows, so the safe path is also the fast path.",
        "Collect the DBAPI call graph plus a regression test that fails when ``%`` string formatting appears in "
        "any query, and treat an absent test as an unverified claim.",
        "The residual risk is privilege, not injection: the application role should hold no DDL grant, which caps "
        "the blast radius at 0 schema mutations.",
    ),
    "architecture": (
        "Put the decision behind one choke point; today the same concern is implemented twice, so any fix applied "
        "to one path silently leaves the other open.",
        "Keep the synchronous request path free of blocking I/O: a single blocking call on the event loop adds its "
        "full latency to every concurrent request sharing that loop.",
        "Reserve a bounded slice of the request deadline for degradation, otherwise a slow dependency consumes the "
        "whole budget and the fallback becomes unreachable exactly when it is needed.",
        "Make the boundary explicit in types rather than comments: a mode that is unrepresentable cannot be reached "
        "by accident, which is worth more than a warning that only a reader sees.",
    ),
    "implementation": (
        "Land the change in three steps so each is independently revertible: the abstraction, then the call-site "
        "migration, then the deletion of the old path.",
        "Add a regression test that fails on the old behaviour, not merely a test that passes on the new one — the "
        "value is in the failure it would have caught.",
        "Expect a 15-30 ms p95 increase per added network hop; measure at the 95th percentile rather than the mean, "
        "because the mean hides the tail that users actually feel.",
        "Delete the superseded code in the same change; leaving both paths is how the second implementation becomes "
        "the one that drifts.",
    ),
    "data": (
        "Sample size dominates: below roughly 20 observations the interval on any rate estimate is wider than the "
        "effect being claimed, so report insufficient data rather than a direction.",
        "Track p50, p95 and p99 separately; a distribution with a stable mean and an exploding p99 is a queueing "
        "problem, not a capacity problem, and adding replicas will not fix it.",
        "Attribute every write to a correlation id so a missing row is provable rather than suspected; a "
        "fire-and-forget write has no failure signal.",
        "Cap retained state explicitly, since an unbounded dict keyed on caller-supplied identifiers is an "
        "unauthenticated memory-growth path.",
    ),
}

_OPENERS: dict[str, tuple[str, ...]] = {
    "security": (
        "The decisive control for this is input handling at the data-access layer, not validation at the edge.",
        "I would treat this as an untrusted-boundary problem first and a code-style problem second.",
    ),
    "architecture": (
        "The structural choice that matters here is where the single source of truth lives.",
        "This is a boundary problem before it is an implementation problem.",
    ),
    "implementation": (
        "Concretely, I would sequence this as a migration with a delete at the end.",
        "The work splits cleanly into an interface change and a call-site sweep.",
    ),
    "data": (
        "The measurement design has to come before the recommendation.",
        "This is answerable only with an explicit sample size and a stated interval.",
    ),
}

_COMPLICATIONS: dict[str, tuple[str, ...]] = {
    "security": (
        "This conflicts with the assumption that edge validation is sufficient, which is why a bypass exists.",
        "I am flagging a flaw in the current approach: the two implementations of this check disagree.",
    ),
    "architecture": (
        "This contradicts the premise that one code path owns this concern.",
        "There is a bottleneck risk here: the request holds a lock across an await.",
    ),
    "implementation": (
        "The risk is that this leaves the old path live and the old path is the one under load.",
        "I would flag a flaw in the plan: the migration has no observability until the final step.",
    ),
    "data": (
        "This conflicts with the assumption that the sample is large enough to support the claim.",
        "The bottleneck is the retained buffer, not the query planner.",
    ),
}


def synthesize(
    persona: str,
    archetype: str,
    question: str,
    directives: dict[str, str],
    conflict_mode: bool,
    max_tokens: int,
) -> str:
    """Build a deterministic, specialist-shaped answer.

    Determinism is deliberate: the rig is a controlled constant so that any change
    in observed behaviour is attributable to the agent, not to the fake model.
    """
    seed = int(hashlib.sha256(f"{persona}|{question}".encode()).hexdigest()[:12], 16)
    rng = random.Random(seed)

    opener = rng.choice(_OPENERS[archetype])
    bullet_pool = list(_BULLETS[archetype])
    rng.shuffle(bullet_pool)
    bullet_count = 3 if max_tokens and max_tokens < 400 else 4
    bullets = bullet_pool[:bullet_count]

    if "NO_METRICS" in directives:
        # Strip any bullet carrying a metric token, so the adjudicator's evidence
        # extractor has nothing to latch onto. Tests the "no fabricated evidence" path.
        bullets = [b for b in bullets if not re.search(r"\b\d+\s*(ms|s|%|x|mb|kb|rows)\b", b, re.IGNORECASE)]
        bullets = bullets or ["No measurable claim can be supported from the information provided."]

    lines = [
        f"[{persona}] round analysis",
        "",
        f"On: {question[:200]}",
        "",
        opener,
        "",
    ]
    lines.extend(f"- {bullet}" for bullet in bullets)

    if conflict_mode and seed % 3 == 0:
        lines.append("")
        lines.append(rng.choice(_COMPLICATIONS[archetype]))

    lines.extend(
        [
            "",
            f"Scope: {persona} perspective only; contradictions with other panel members are stated rather than "
            "silently resolved.",
            "Confidence: moderate-high, bounded by the parts of the system not shown to me.",
        ]
    )
    return "\n".join(lines)


def count_tokens(text: str) -> int:
    """Approximate token count.

    A word/punctuation heuristic, not a real BPE tokenizer: it is documented as an
    approximation so no downstream metric is mistaken for an exact figure.
    """
    if not text:
        return 0
    return max(1, len(re.findall(r"\w+|[^\w\s]", text)))


# --------------------------------------------------------------------------------------
# Server state
# --------------------------------------------------------------------------------------


class ServerState:
    """Thread-safe global configuration and telemetry for the rig."""

    def __init__(self, slots: int, base_latency_ms: int) -> None:
        self._lock = threading.Lock()
        self.slots = slots
        self._semaphore = threading.BoundedSemaphore(slots)
        self.base_latency_ms = base_latency_ms
        self.flaky_pct = 0
        self.conflict_mode = False
        self.fail_for: dict[str, str] = {}
        self.hang_seconds = 600.0
        self.default_model = "rig-8b"
        self.requests = 0
        self.failures = 0
        self.authorized_requests = 0
        self.unauthorized_requests = 0
        self.rejected_preemptively = 0
        self.in_flight = 0
        self.queue_depth = 0
        self.max_queue_depth = 0
        self.latencies: list[float] = []
        self.per_persona: dict[str, int] = {}
        self.captures: list[dict[str, Any]] = []
        self._t0 = time.monotonic()
        self.shutdown = threading.Event()
        self.started = time.time()

    def acquire_slot(self) -> None:
        with self._lock:
            self.queue_depth += 1
            self.max_queue_depth = max(self.max_queue_depth, self.queue_depth)
        self._semaphore.acquire()
        with self._lock:
            self.queue_depth -= 1
            self.in_flight += 1

    def release_slot(self) -> None:
        with self._lock:
            self.in_flight = max(0, self.in_flight - 1)
        self._semaphore.release()

    def update(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            # NOTE: a reset must NOT short-circuit the rest of the payload. An earlier
            # revision returned here, so `{"reset": true, "fail_for": {...}}` silently
            # discarded the fault schedule and produced a test that passed while
            # injecting nothing. Apply the reset, then continue applying overrides.
            if payload.get("reset"):
                self.requests = self.failures = self.authorized_requests = 0
                self.unauthorized_requests = self.rejected_preemptively = 0
                self.latencies.clear()
                self.per_persona.clear()
                self.max_queue_depth = 0
                self.flaky_pct = 0
                self.conflict_mode = False
                self.fail_for = {}
                self.captures.clear()
            if "slots" in payload:
                new_slots = max(1, int(payload["slots"]))
                if new_slots != self.slots:
                    self.slots = new_slots
                    self._semaphore = threading.BoundedSemaphore(new_slots)
            if "base_latency_ms" in payload:
                self.base_latency_ms = max(0, int(payload["base_latency_ms"]))
            if "flaky_pct" in payload:
                self.flaky_pct = min(100, max(0, int(payload["flaky_pct"])))
            if "conflict_mode" in payload:
                self.conflict_mode = bool(payload["conflict_mode"])
            if "fail_for" in payload:
                self.fail_for = {str(k): str(v) for k, v in dict(payload["fail_for"]).items()}
            if "hang_seconds" in payload:
                self.hang_seconds = float(payload["hang_seconds"])
            if "default_model" in payload:
                self.default_model = str(payload["default_model"])
            return self.snapshot_locked()

    def capture(
        self,
        *,
        persona: str,
        archetype: str,
        model: str,
        system_head: str,
        question: str,
        max_tokens: Any,
        fault: str,
    ) -> None:
        """Keep a small ring buffer of what the agent actually sent.

        Indispensable when a fault schedule keys on persona: without this you cannot
        tell whether the persona the rig *saw* matches the one the schedule *named*.
        """
        with self._lock:
            self.captures.append(
                {
                    "t": round(time.monotonic() - self._t0, 4),
                    "persona": persona,
                    "archetype": archetype,
                    "model": model,
                    "max_tokens": max_tokens,
                    "system_head": system_head[:160],
                    "question_head": question[:80],
                    "fault": fault or "",
                }
            )
            if len(self.captures) > 50:
                del self.captures[:-50]

    def record(
        self,
        *,
        persona: str,
        latency: float,
        failed: bool,
        authorized: bool | None = None,
        preemptive: bool = False,
    ) -> None:
        with self._lock:
            self.requests += 1
            self.latencies.append(latency)
            if len(self.latencies) > 20000:
                del self.latencies[:10000]
            if failed:
                self.failures += 1
            self.per_persona[persona] = self.per_persona.get(persona, 0) + 1
            if authorized is True:
                self.authorized_requests += 1
            elif authorized is False:
                self.unauthorized_requests += 1
            if preemptive:
                self.rejected_preemptively += 1

    def snapshot_locked(self) -> dict[str, Any]:
        return {
            "slots": self.slots,
            "base_latency_ms": self.base_latency_ms,
            "flaky_pct": self.flaky_pct,
            "conflict_mode": self.conflict_mode,
            "fail_for": dict(self.fail_for),
            "hang_seconds": self.hang_seconds,
            "default_model": self.default_model,
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            latencies = sorted(self.latencies)
            uptime = max(0.001, time.time() - self.started)
            return {
                **self.snapshot_locked(),
                "uptime_seconds": round(uptime, 2),
                "requests": self.requests,
                "failures": self.failures,
                "error_rate_pct": round(100.0 * self.failures / self.requests, 2) if self.requests else 0.0,
                "requests_per_second": round(self.requests / uptime, 2),
                "in_flight": self.in_flight,
                "queue_depth": self.queue_depth,
                "max_queue_depth": self.max_queue_depth,
                "latency_p50_ms": round(_pct(latencies, 50) * 1000, 2),
                "latency_p95_ms": round(_pct(latencies, 95) * 1000, 2),
                "latency_p99_ms": round(_pct(latencies, 99) * 1000, 2),
                "latency_max_ms": round((max(latencies) if latencies else 0.0) * 1000, 2),
                "per_persona": dict(sorted(self.per_persona.items(), key=lambda kv: -kv[1])),
                "authorized_requests": self.authorized_requests,
                "unauthorized_requests": self.unauthorized_requests,
                "rejected_preemptively": self.rejected_preemptively,
                "captures": list(self.captures[-10:]),
            }


def _pct(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, int(round((percentile / 100.0) * (len(sorted_values) - 1))))
    return sorted_values[index]


STATE: ServerState
MODEL_NAME = "rig-8b"


def sleep_interruptible(seconds: float) -> None:
    """Sleep, but return promptly on shutdown so Ctrl+C is never blocked by a hang."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if STATE.shutdown.is_set():
            return
        time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))


# --------------------------------------------------------------------------------------
# HTTP handler
# --------------------------------------------------------------------------------------


class RigHandler(BaseHTTPRequestHandler):
    """OpenAI-compatible endpoints plus fault-injection control surfaces."""

    protocol_version = "HTTP/1.1"
    server_version = "LocalModelRig/1.0"
    sys_version = ""

    # -- plumbing -----------------------------------------------------------------

    def log_message(self, fmt: str, *args: Any) -> None:
        if _VERBOSE:
            sys.stderr.write(f"[rig] {self.address_string()} {fmt % args}\n")

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length else b""

    def _write_safely(self, data: bytes) -> None:
        """Write to the socket, tolerating a client that has already disconnected.

        A client that times out closes its end, and the rig's write then raises
        BrokenPipeError inside the request thread. That is *expected* during
        timeout-injection tests, so swallowing it keeps the server log readable
        instead of burying real findings under tracebacks.
        """
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def _send_json(self, status: int, payload: dict[str, Any], extra: dict[str, str] | None = None) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if status != 204:
            self._write_safely(body)

    def _send_raw(self, status: int, body: bytes, content_type: str = "text/plain") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self._write_safely(body)

    def _openai_error(self, status: int, message: str, extra: dict[str, str] | None = None) -> None:
        self._send_json(
            status,
            {"error": {"message": message, "type": "rig_injected_fault", "code": status}},
            extra=extra,
        )

    # -- routing ------------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = self.path.split("?")[0]
        if path in ("/health", "/healthz"):
            self._send_json(200, {"status": "ok", "model": MODEL_NAME, "slots": STATE.slots})
        elif path.endswith("/models"):
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {"id": MODEL_NAME, "object": "model", "owned_by": "local-rig"},
                        {"id": "rig-tiny", "object": "model", "owned_by": "local-rig"},
                    ],
                },
            )
        elif path == "/__stats":
            self._send_json(200, STATE.snapshot())
        elif path == "/__capture":
            self._send_json(200, {"captures": list(STATE.captures)})
        elif path == "/":
            self._send_raw(200, b"Local model rig. See /__stats, POST /__control, POST /v1/chat/completions.")
        else:
            self._openai_error(404, f"Unknown path {path}")

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = self.path.split("?")[0]
        if path == "/__control":
            try:
                payload = json.loads(self._read_body() or b"{}")
            except json.JSONDecodeError:
                self._openai_error(400, "control body must be JSON")
                return
            self._send_json(200, STATE.update(payload))
        elif path.endswith("/chat/completions"):
            self._handle_completion()
        elif path == "/__shutdown":
            self._send_json(200, {"shutting_down": True})
            threading.Thread(target=self.server.shutdown, daemon=True).start()
        else:
            self._openai_error(404, f"Unknown path {path}")

    # -- completion ----------------------------------------------------------------

    def _handle_completion(self) -> None:
        raw = self._read_body()
        try:
            payload = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            STATE.record(persona="?", latency=0.0, failed=True)
            self._openai_error(400, "request body is not valid JSON")
            return

        messages = payload.get("messages") or []
        system_instruction = ""
        for message in messages:
            if message.get("role") == "system":
                system_instruction += str(message.get("content", "")) + "\n"
        prompt_text = system_instruction + "\n".join(str(m.get("content", "")) for m in messages)

        persona = detect_persona(system_instruction, messages)
        archetype = detect_archetype(persona, system_instruction)
        question = extract_question(messages)
        directives = parse_directives(prompt_text)
        model = payload.get("model") or STATE.default_model
        max_tokens = payload.get("max_tokens") or 1024
        wants_stream = bool(payload.get("stream"))

        # ---- authorization surface -------------------------------------------------
        # The rig records whether a bearer token was presented, which is how we prove
        # the agent's keyless local path really is keyless (and that a configured
        # LOCAL_API_KEY is forwarded).
        authorized = bool(self.headers.get("Authorization"))

        # ---- resolve the fault to inject -------------------------------------------
        fault = directives.get("FAIL") or resolve_fault_schedule(STATE.fail_for, persona) or ""
        if not fault and STATE.flaky_pct:
            # Deterministic flakiness: seeded by the request counter so a rerun of the
            # same load pattern fails at the same points.
            with STATE._lock:  # noqa: SLF001 - internal counter read, intentional
                counter = STATE.requests
            if random.Random(counter).randint(1, 100) <= STATE.flaky_pct:
                fault = random.Random(counter).choice(["429", "500", "503"])

        STATE.capture(
            persona=persona,
            archetype=archetype,
            model=model,
            system_head=system_instruction.strip().replace("\n", " "),
            question=question,
            max_tokens=max_tokens,
            fault=fault,
        )

        if not fault and STATE.base_latency_ms:
            sleep_interruptible(STATE.base_latency_ms / 1000.0)

        latency_directive = directives.get("LATENCY_MS")
        if latency_directive and latency_directive.isdigit():
            sleep_interruptible(int(latency_directive) / 1000.0)

        if fault:
            self._inject_fault(fault, persona, model, wants_stream)
            return

        # ---- healthy path ----------------------------------------------------------
        STATE.acquire_slot()
        start = time.perf_counter()
        try:
            answer = synthesize(persona, archetype, question, directives, STATE.conflict_mode, max_tokens)
            # Emulate a model whose answer is cut off by the token ceiling.
            if max_tokens and len(answer) > max_tokens * 4:
                answer = answer[: max_tokens * 4].rsplit(" ", 1)[0]
            sleep_interruptible(min(0.05, STATE.base_latency_ms / 2000.0 + 0.005))
            if wants_stream:
                self._stream_answer(answer, model, persona, start)
            else:
                self._json_answer(answer, model, persona, start, authorized=authorized)
        finally:
            STATE.release_slot()

    def _inject_fault(self, fault: str, persona: str, model: str, wants_stream: bool) -> None:
        fault = fault.lower()
        if fault in ("429", "rate_limit"):
            STATE.record(persona=persona, latency=0.0, failed=True)
            self._openai_error(429, "rig-injected rate limit", extra={"Retry-After": "2"})
        elif fault in ("503", "unavailable"):
            STATE.record(persona=persona, latency=0.0, failed=True)
            self._openai_error(503, "rig-injected service unavailable")
        elif fault in ("500", "error"):
            STATE.record(persona=persona, latency=0.0, failed=True)
            self._openai_error(500, "rig-injected internal error")
        elif fault in ("timeout", "hang"):
            if fault == "hang":
                duration = STATE.hang_seconds
            else:
                # Sleep slightly past any plausible client deadline, then answer.
                duration = float(DIRECTIVE_TIMEOUT_S)
            STATE.record(persona=persona, latency=duration, failed=True)
            sleep_interruptible(duration)
            try:
                self._openai_error(504, "rig-injected timeout")
            except (BrokenPipeError, ConnectionResetError):
                pass  # the client already gave up, which is the point
        elif fault == "empty":
            STATE.record(persona=persona, latency=0.0, failed=True)
            # A 200 with no choices is the most dangerous provider response: the adapter
            # returns an empty content string, and any caller that treats "the HTTP call
            # succeeded" as "the model answered" will fabricate a result.
            self._send_json(
                200,
                {
                    "id": "rig-empty",
                    "object": "chat.completion",
                    "model": model,
                    "choices": [],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                },
            )
        elif fault == "malformed":
            STATE.record(persona=persona, latency=0.0, failed=True)
            self._send_raw(200, b"<html>not json at all</html>", content_type="application/json")
        elif fault == "truncate":
            STATE.record(persona=persona, latency=0.0, failed=False)
            self._send_json(
                200,
                {
                    "id": "rig-trunc",
                    "object": "chat.completion",
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": f"[{persona}] analysis truncated at the"},
                            "finish_reason": "length",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                },
            )
        elif fault == "sse":
            STATE.record(persona=persona, latency=0.0, failed=True)
            self._stream_failure()
        else:
            STATE.record(persona=persona, latency=0.0, failed=True)
            self._openai_error(400, f"unknown rig fault '{fault}'")

    def _json_answer(
        self, answer: str, model: str, persona: str, start: float, *, authorized: bool
    ) -> None:
        latency = time.perf_counter() - start
        prompt_tokens = 128
        completion_tokens = count_tokens(answer)
        STATE.record(persona=persona, latency=latency, failed=False, authorized=authorized)
        self._send_json(
            200,
            {
                "id": f"rig-{int(start * 1000)}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": answer},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            },
        )

    def _stream_answer(self, answer: str, model: str, persona: str, start: float) -> None:
        """Emit Server-Sent Events chunk-by-chunk, as a real OpenAI server does."""
        words = answer.split(" ")
        chunks = [" ".join(words[i : i + 4]) for i in range(0, len(words), 4)] or [answer]

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def write_chunk(data: str) -> None:
            encoded = data.encode()
            self.wfile.write(f"{len(encoded):X}\r\n".encode() + encoded + b"\r\n")

        try:
            for index, chunk in enumerate(chunks):
                frame = {
                    "id": "rig-stream",
                    "object": "chat.completion.chunk",
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": chunk + " "}, "finish_reason": None}],
                }
                write_chunk(f"data: {json.dumps(frame)}\n\n")
                if index % 8 == 0:
                    sleep_interruptible(0.01)
            write_chunk("data: [DONE]\n\n")
            self._write_safely(b"0\r\n\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            STATE.record(persona=persona, latency=time.perf_counter() - start, failed=False)

    def _stream_failure(self) -> None:
        """Open a valid SSE stream, emit one chunk, then sever the connection."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            frame = json.dumps(
                {
                    "id": "rig-stream-fail",
                    "object": "chat.completion.chunk",
                    "model": MODEL_NAME,
                    "choices": [{"index": 0, "delta": {"content": "partial "}, "finish_reason": None}],
                }
            )
            encoded = f"data: {frame}\n\n".encode()
            self.wfile.write(f"{len(encoded):X}\r\n".encode() + encoded + b"\r\n")
            self.wfile.flush()
            self.connection.close()  # abrupt mid-stream failure
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass


_VERBOSE = False
DIRECTIVE_TIMEOUT_S = 90.0


def run(port: int, host: str, slots: int, base_latency_ms: int, model: str, verbose: bool) -> int:
    global STATE, MODEL_NAME, _VERBOSE
    _VERBOSE = verbose
    MODEL_NAME = model
    STATE = ServerState(slots=slots, base_latency_ms=base_latency_ms)
    STATE.default_model = model

    server = ThreadingHTTPServer((host, port), RigHandler)
    server.daemon_threads = True
    sys.stderr.write(
        f"[rig] listening on http://{host}:{port}  model={model}  slots={slots}  "
        f"base_latency_ms={base_latency_ms}\n"
        f"[rig] endpoints: POST /v1/chat/completions  GET /v1/models  GET /health  "
        f"GET /__stats  POST /__control\n"
        f"[rig] NOTE: this is a deterministic conformance rig, not a language model.\n"
    )
    sys.stderr.flush()
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        sys.stderr.write("\n[rig] shutting down\n")
    finally:
        STATE.shutdown.set()
        server.server_close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Local OpenAI-compatible model rig for driving Inference.")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=11434, help="port (default 11434, matches Ollama)")
    parser.add_argument("--slots", type=int, default=4, help="concurrent model slots (default 4)")
    parser.add_argument("--base-latency-ms", type=int, default=0, help="baseline latency per request")
    parser.add_argument("--model", default="rig-8b", help="model id to advertise (default rig-8b)")
    parser.add_argument("-v", "--verbose", action="store_true", help="log every request")
    args = parser.parse_args()
    return run(args.port, args.host, args.slots, args.base_latency_ms, args.model, args.verbose)


if __name__ == "__main__":
    raise SystemExit(main())
