"""Find float literals bound to judgement names anywhere in ``app/``.

A judgement value (confidence, score, probability, likelihood, credibility, bias weight) that is
a fixed float literal is a fabrication candidate: the response then reports the same "measured"
number whatever the input. This AST scan lists every such binding so each one can be triaged:

* a dict key ``"confidence": 0.85`` or a keyword ``confidence=0.85``;
* an assignment ``confidence = 0.85`` or annotated ``confidence: float = 0.85``;
* a ``Field(default=0.85)`` on a judgement-named field.

Only literal constants are reported. Computed values are not, because they are not constant by
construction (the sensitivity probe covers those). Known-safe bindings can be listed in
``ALLOWED`` with a reason, so the scan stays a gate rather than a suppression list.

Usage: ``python scripts/find_literal_judgements.py [--json]``. Exit code 1 when unallowed hits exist.
"""

from __future__ import annotations

import ast
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
APP = ROOT / "app"

JUDGEMENT = re.compile(
    r"(confidence|score|probability|likelihood|credibility|verdict_weight|robustness|pbo|sharpe_prob)",
    re.IGNORECASE,
)
# Thresholds and config bounds are legitimate constants; they are not outputs.
THRESHOLD_HINT = re.compile(r"(threshold|minimum|min_|max_|_min|_max|bound|floor|ceiling|cutoff|limit|weight_of|expected)", re.IGNORECASE)

# (relative path, line) -> reason, for a single reviewed binding.
ALLOWED: dict[tuple[str, int], str] = {}

# relative path -> (expected number of literal bindings, reason). A file is allowed only up to the
# expected count, so a NEW literal added to an allowed file still fails the gate. Every entry was
# reviewed by reading its binding in context; the reason says why the value is not a fabrication.
ALLOWED_FILES: dict[str, tuple[int, str]] = {
    "app/utils/confidence.py": (5, "the named confidence states themselves (their definitions)"),
    "app/evaluation/evaluator.py": (11, "scoring rubric: a fixed table from MEASURED latency and token counts to a 0-1 score; the inputs are measured"),
    "app/data/alternative_data.py": (3, "labelled evidence_class=synthetic_fixture with inputs_simulated=true; no longer sets a directional call"),
    "app/analytics/outcome_learning.py": (4, "labelled source=fixture rows, excluded from every aggregate (tests/test_bounded_stores_and_analytics_honesty.py)"),
    "app/analytics/usage_analytics.py": (4, "labelled source=fixture rows, excluded from every aggregate (same test file)"),
    "app/ml/prediction_tracking.py": (4, "labelled source=fixture rows, excluded from statistics (tests/test_prediction_and_monitoring_honesty.py)"),
    "app/analysis/sentiment_analysis.py": (2, "neutral score when no keyword is present (an absence of evidence, not a bias) and the accumulator initialiser"),
    "app/analysis/stress_intelligence.py": (1, "additive baseline of the stress index; measured terms are added to it"),
}


def _float_const(node: ast.AST) -> float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, float) and not isinstance(node.value, bool):
        return node.value

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _float_const(node.operand)
        return -inner if inner is not None else None
    return None


def _name_of(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def scan_file(path: pathlib.Path) -> list[dict[str, object]]:
    rel = path.relative_to(ROOT).as_posix()
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=rel)
    hits: list[dict[str, object]] = []

    def record(node: ast.AST, name: str, value: float, kind: str) -> None:
        if not JUDGEMENT.search(name) or THRESHOLD_HINT.search(name):
            return
        line = getattr(node, "lineno", 0)
        hits.append({"file": rel, "line": line, "name": name, "value": value, "kind": kind,
                     "allowed": ALLOWED.get((rel, line))})

    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                name = _name_of(key) if key is not None else None
                num = _float_const(value)
                if name and num is not None:
                    record(value, name, num, "dict")
        elif isinstance(node, ast.keyword) and node.arg:
            num = _float_const(node.value)
            if num is not None:
                record(node.value, node.arg, num, "kwarg")
        elif isinstance(node, ast.Assign):
            num = _float_const(node.value)
            if num is not None:
                for target in node.targets:
                    name = _name_of(target)
                    if name:
                        record(node.value, name, num, "assign")
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            name = _name_of(node.target)
            num = _float_const(node.value)
            if name and num is None and isinstance(node.value, ast.Call):
                # pydantic ``confidence: float = Field(default=0.85)`` is the same fabricated default.
                for kw in node.value.keywords:
                    if kw.arg == "default" and _float_const(kw.value) is not None:
                        record(kw.value, name, _float_const(kw.value), "field-default")
            elif name and num is not None:
                record(node.value, name, num, "assign")
    return hits


def scan_all() -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for path in sorted(APP.rglob("*.py")):
        out.extend(scan_file(path))
    # Apply per-file allowances: all hits in a file are allowed only when their count matches the
    # reviewed count exactly. A count that grew (a new literal) or shrank (a fix that the allowance
    # was not updated for) is reported so the entry is reviewed again.
    by_file: dict[str, list[dict[str, object]]] = {}
    for hit in out:
        by_file.setdefault(str(hit["file"]), []).append(hit)
    for rel, hits in by_file.items():
        if rel in ALLOWED_FILES:
            expected, reason = ALLOWED_FILES[rel]
            for hit in hits:
                hit["allowed"] = reason if len(hits) == expected else None
                if len(hits) != expected:
                    hit["note"] = f"file has {len(hits)} literal bindings, reviewed count is {expected}"
    return out


def main(argv: list[str]) -> int:
    hits = scan_all()
    unallowed = [h for h in hits if not h["allowed"]]
    if "--json" in argv:
        print(json.dumps(hits, indent=2))
    else:
        for h in hits:
            mark = "ok" if h["allowed"] else "!!"
            print(f"[{mark}] {h['file']}:{h['line']}  {h['kind']:6} {h['name']} = {h['value']}")
        print(f"total={len(hits)} unallowed={len(unallowed)}")
    return 1 if unallowed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
