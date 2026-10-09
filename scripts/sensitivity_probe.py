#!/usr/bin/env python3
"""Input-sensitivity probe: does each POST endpoint's judgement depend on its input?

Why this exists. A number that never changes when the input changes is not a measurement of
the input. The operator tour found hard-coded confidence values (0.91, 0.89, 0.92, 96.4) on
endpoints that claimed to analyse the input. Reading 66 literal confidence assignments by hand
does not scale, so this probe measures the behaviour instead:

1. Build two realistic, *different* request bodies for each POST operation, from the live
   OpenAPI schema (``$ref``s resolved, required fields filled, enums and literals chosen,
   numbers and strings varied between the two bodies).
2. Send both. Record status, latency, and every numeric leaf in the JSON response.
3. Classify each operation:

   * ``insensitive``: judgement fields (confidence, score, probability, ...) are present and
     identical across the two different inputs, while the response text differs or the
     response is otherwise identical. This is the fabrication signature.
   * ``sensitive``: at least one judgement field changed with the input.
   * ``no_judgement``: no judgement fields in the response; nothing to test for constancy.
   * ``server_error`` / ``rejected`` / ``unreachable``: the probe could not get an answer.
   * ``skipped``: side-effecting operation (orders, deletes, resets, deploys); not sent.

Writes a JSON report. The exit code is 0 even when findings exist: this is a measurement, and
the findings are reviewed per endpoint, not gated blindly. Use ``--fail-on insensitive`` to gate.

Usage:
    python scripts/sensitivity_probe.py --base http://127.0.0.1:8000 --api-key KEY \
        --report /tmp/sensitivity.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any

#: Judgement fields: a number here is a claim about the input, so it must respond to it.
JUDGEMENT_KEYS = re.compile(
    r"(confidence|score|probability|certainty|rating|grade|quality|accuracy|calibrat|"
    r"conviction|sentiment|risk|severity_score|win_rate|likelihood|uncertainty|edge)", re.IGNORECASE)
#: Timing and bookkeeping numbers change on every call and say nothing about the input.
NOISE_KEYS = re.compile(r"(latency|_ms$|seconds|duration|elapsed|timestamp|created|updated|token|"
                        r"cost|count|_at$|_id$|^id$|tick|epoch|pid|version|port)", re.IGNORECASE)
#: Operations with side effects are never sent by this probe.
SIDE_EFFECT = re.compile(r"(order|trade|execute|delete|reset|shutdown|deploy|submit|purge|"
                         r"restart|kill|cancel|admin|upload|webhook|notify|send|publish|write)", re.IGNORECASE)

TEXT_A = ("Our payments service retries failed charges three times with no backoff, and the retry "
          "queue has no dead-letter handling. Assess the operational risk for a Black Friday load.")
TEXT_B = ("Migrating the marketing site to a static generator: the build takes four seconds and "
          "the content team edits Markdown files in git. Is this a sound architecture for a team of two?")


@dataclass
class OperationResult:
    path: str
    method: str
    classification: str
    status_a: int | None = None
    status_b: int | None = None
    judgement_a: dict[str, Any] = field(default_factory=dict)
    judgement_b: dict[str, Any] = field(default_factory=dict)
    texts_differ: bool | None = None
    note: str = ""


class SchemaBodies:
    """Builds two different bodies from an OpenAPI schema. Deterministic."""

    def __init__(self, spec: dict[str, Any]) -> None:
        self.spec = spec

    def resolve(self, schema: dict[str, Any]) -> dict[str, Any]:
        seen = 0
        while "$ref" in schema and seen < 20:
            name = schema["$ref"].split("/")[-1]
            schema = self.spec["components"]["schemas"].get(name, {})
            seen += 1
        return schema

    def build(self, schema: dict[str, Any], variant: str, depth: int = 0) -> Any:
        schema = self.resolve(schema)
        if depth > 6:
            return None
        if "default" in schema and schema["default"] is not None and variant == "A":
            return schema["default"]
        for key in ("anyOf", "oneOf"):
            if key in schema:
                options = [o for o in schema[key] if o.get("type") != "null"]
                if options:
                    return self.build(options[0] if variant == "A" else options[-1], variant, depth + 1)
                return None
        if "enum" in schema:
            values = schema["enum"]
            return values[0] if variant == "A" else values[-1]
        kind = schema.get("type")
        if "const" in schema:
            return schema["const"]
        if kind == "object" or "properties" in schema:
            body: dict[str, Any] = {}
            required = set(schema.get("required", []))
            for name, sub in schema.get("properties", {}).items():
                if name in required or depth == 0:
                    value = self.build(sub, variant, depth + 1)
                    if value is not None:
                        body[name] = value
            return body
        if kind == "array":
            items = schema.get("items", {})
            count = max(1, int(schema.get("minItems", 1)))
            return [self.build(items, variant, depth + 1) for _ in range(count)]
        if kind == "string":
            if "file" in (schema.get("format") or ""):
                return "a.py"
            text = TEXT_A if variant == "A" else TEXT_B
            return text[: int(schema.get("maxLength", 200))]
        if kind in ("integer", "number"):
            low = schema.get("minimum", 0)
            high = schema.get("maximum", 1000)
            base = 5 if variant == "A" else 17
            value = min(max(base, low), high)
            return int(value) if kind == "integer" else float(value)
        if kind == "boolean":
            return variant == "A"
        return None


def _numeric_leaves(value: Any, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(value, bool):
        return out
    if isinstance(value, (int, float)):
        out[prefix] = float(value)
    elif isinstance(value, dict):
        for k, v in value.items():
            out.update(_numeric_leaves(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            out.update(_numeric_leaves(v, f"{prefix}[{i}]"))
    return out


def _judgement(leaves: dict[str, float]) -> dict[str, float]:
    return {path: v for path, v in leaves.items()
            if JUDGEMENT_KEYS.search(path.split(".")[-1].split("[")[0]) and not NOISE_KEYS.search(path)}


def _strip_noise(value: Any) -> Any:
    """Remove fields that legitimately differ between calls, so 'identical' means identical."""
    if isinstance(value, dict):
        return {k: _strip_noise(v) for k, v in value.items()
                if not NOISE_KEYS.search(k) and k not in ("correlation_id", "request_id")}
    if isinstance(value, list):
        return [_strip_noise(v) for v in value]
    return value


def send(base: str, api_key: str, path: str, body: Any, timeout: float) -> tuple[int | None, Any, float]:
    data = json.dumps(body).encode()
    req = urllib.request.Request(base.rstrip("/") + path, data=data, method="POST",
                                 headers={"Content-Type": "application/json", "X-API-Key": api_key})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", "replace")
            status = resp.status
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        status = exc.code
    except Exception:  # noqa: BLE001 - an unreachable operation is a result
        return None, None, time.perf_counter() - started
    try:
        parsed = json.loads(raw)
    except ValueError:
        parsed = {"_raw": raw}
    return status, parsed, time.perf_counter() - started


def probe(base: str, api_key: str, spec: dict[str, Any], timeout: float) -> list[OperationResult]:
    builder = SchemaBodies(spec)
    results: list[OperationResult] = []
    for path, methods in sorted(spec["paths"].items()):
        op = methods.get("post")
        if not op:
            continue
        if SIDE_EFFECT.search(path):
            results.append(OperationResult(path, "POST", "skipped", note="side-effect word in path"))
            continue
        schema = (op.get("requestBody", {}).get("content", {}).get("application/json", {}).get("schema"))
        if schema is None:
            results.append(OperationResult(path, "POST", "skipped", note="no JSON body"))
            continue
        body_a = builder.build(schema, "A")
        body_b = builder.build(schema, "B")
        body_a = body_a if body_a is not None else {}
        body_b = body_b if body_b is not None else {}
        status_a, resp_a, _ = send(base, api_key, path, body_a, timeout)
        status_b, resp_b, _ = send(base, api_key, path, body_b, timeout)
        result = OperationResult(path, "POST", "", status_a, status_b)
        if status_a is None or status_b is None:
            result.classification = "unreachable"
            results.append(result)
            continue
        if status_a >= 500 or status_b >= 500:
            result.classification = "server_error"
            results.append(result)
            continue
        if status_a != 200 or status_b != 200:
            result.classification = "rejected"
            result.note = f"body A {status_a}, body B {status_b}"
            results.append(result)
            continue
        leaves_a = _numeric_leaves(_strip_noise(resp_a))
        leaves_b = _numeric_leaves(_strip_noise(resp_b))
        judge_a, judge_b = _judgement(leaves_a), _judgement(leaves_b)
        result.judgement_a, result.judgement_b = judge_a, judge_b
        result.texts_differ = json.dumps(_strip_noise(resp_a), sort_keys=True) != json.dumps(
            _strip_noise(resp_b), sort_keys=True)
        if not judge_a and not judge_b:
            result.classification = "no_judgement"
        elif judge_a == judge_b:
            result.classification = "insensitive"
            result.note = "judgement fields identical for two different inputs"
        else:
            result.classification = "sensitive"
        results.append(result)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--report", help="write the JSON report here")
    parser.add_argument("--fail-on", choices=["insensitive", "never"], default="never")
    args = parser.parse_args()

    with urllib.request.urlopen(urllib.request.Request(
            args.base.rstrip("/") + "/openapi.json", headers={"X-API-Key": args.api_key}), timeout=30) as resp:
        spec = json.loads(resp.read().decode())

    results = probe(args.base, args.api_key, spec, args.timeout)
    counts: dict[str, int] = {}
    for r in results:
        counts[r.classification] = counts.get(r.classification, 0) + 1
        mark = {"insensitive": "!!", "sensitive": "ok", "no_judgement": "--"}.get(r.classification, "??")
        extra = f" {r.note}" if r.note else ""
        print(f"[{mark}] {r.classification:13} {r.path:42} A={r.status_a} B={r.status_b}{extra}")
    print("\nsummary:", json.dumps(dict(sorted(counts.items())), indent=None))
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump([asdict(r) for r in results], fh, indent=2)
    if args.fail_on == "insensitive" and counts.get("insensitive"):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
