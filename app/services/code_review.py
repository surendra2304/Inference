"""Code review for FORGE: deterministic static checks, plus a model review that must parse.

What this replaced. The earlier service ran four substring checks (``eval(``, ``exec(``, a
password literal, ``except:``), never read the model's review, defaulted the verdict to
``approve``, returned a constant ``consensus_confidence`` of 0.91, and wrote "Panel consensus
reached across Coder, Security Analyst, and Critic" into the summary although one provider call
ran. [FACT] measured by the operator tour: a SQL-injection snippet came back ``approve``, zero
issues, confidence 0.91.

The rules now:

* Static checks run on the **full** source. For Python they walk the AST, so an
  ``execute(f"... {user}")`` is found by its structure, not by a substring. Other languages get
  pattern checks, and the summary says they are pattern checks.
* The model is asked for a JSON object. Its output counts only if it parses and validates.
  Otherwise ``model_status`` is ``unparsed`` or ``unavailable``, and the review says so.
* ``approve`` is possible only when the model review ran and nothing at medium or higher was
  found. Without a model review the best verdict is ``needs_review``.
* ``consensus_confidence`` is a prior for the review that ran, not a calibrated accuracy:
  0.8 when the model review parsed, 0.5 when only static checks ran.
"""

from __future__ import annotations

import ast
import re
import time
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from app.providers.unified_manager import (
    UnifiedExecutionRequest,
    unified_provider_manager,
)
from app.utils.model_json import extract_json_object

Severity = Literal["critical", "high", "medium", "low", "info"]
Verdict = Literal["approve", "fix_required", "needs_review"]

_SEVERITY_RANK: dict[str, int] = {"critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0}
_VERDICT_RANK: dict[str, int] = {"approve": 0, "needs_review": 1, "fix_required": 2}
MODEL_INPUT_CHARS = 3000
MAX_MODEL_ISSUES = 50


class CodeReviewIssue(BaseModel):
    severity: Severity
    line_hint: str | None = None
    description: str
    suggested_fix: str


class CodeReviewRequest(BaseModel):
    code: str = Field(..., description="Source code to review")
    filename: str = Field(default="app/main.py", description="Filename")
    project_context: str | None = Field(default="", description="High-level project context")
    review_focus: list[Literal["bugs", "security", "performance", "style"]] = Field(
        default=["bugs", "security", "performance", "style"]
    )


class CodeReviewResponse(BaseModel):
    verdict: Verdict
    issues: list[CodeReviewIssue]
    consensus_confidence: float
    debate_summary: str
    latency_ms: float
    review_source: Literal["static_and_model", "static_only"]
    model_status: Literal["reviewed", "unparsed", "unavailable"]
    static_issue_count: int
    model_issue_count: int = 0
    analysis_scope: str


class _ModelReview(BaseModel):
    """What the model must return. Anything that does not validate is not a review."""

    verdict: Verdict
    issues: list[CodeReviewIssue] = Field(default_factory=list)


# -- static analysis --------------------------------------------------------------------

_SECRET_NAME = re.compile(r"(passw(or)?d|passwd|secret|api_?key|access_?token|private_?key)", re.IGNORECASE)
_PLACEHOLDERS = ("changeme", "change_me", "xxx", "your_", "<", "example", "placeholder", "todo")
_SQL_METHODS = frozenset({"execute", "executemany", "executescript", "raw"})

# Pattern checks for languages without a parser here. Each pattern is deliberately narrow.
_PATTERN_RULES: list[tuple[re.Pattern[str], Severity, str, str]] = [
    (re.compile(r"\b(eval|exec)\s*\("), "critical", "Dynamic code execution",
     "Remove eval/exec; dispatch through an explicit table of allowed operations."),
    (re.compile(r"(execute|query)\s*\(\s*f?[\"'][^\"']*(\{|%s|\+\s*\w)"), "critical",
     "SQL built by string interpolation", "Use parameterized queries with placeholders."),
    (re.compile(r"(passw(or)?d|secret|api_?key|token)\s*[:=]\s*[\"'][^\"'\s]{8,}[\"']", re.IGNORECASE), "critical",
     "Hard-coded credential", "Load the secret from the environment or a secret store."),
    (re.compile(r"\bverify\s*=\s*False\b"), "high", "TLS certificate verification disabled",
     "Remove verify=False and pin the CA bundle if needed."),
    (re.compile(r"\bexcept\s*:"), "medium", "Bare except clause",
     "Catch specific exceptions, e.g. `except Exception as exc:`."),
]


def _dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def _is_built_string(node: ast.AST) -> bool:
    """True when a string is assembled at runtime from non-constant parts."""
    if isinstance(node, ast.JoinedStr):
        return any(isinstance(part, ast.FormattedValue) for part in node.values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mod)):
        return not (_is_constant_str(node.left) and _is_constant_str(node.right))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "format":
        return _is_constant_str(node.func.value)
    return False


def _is_constant_str(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


class _PythonAuditor(ast.NodeVisitor):
    """Walks the AST and tracks which local names hold a string built at runtime.

    The taint is deliberately simple: a name is tainted when it is assigned a built string (or a
    tainted name), and cleared when it is reassigned to anything else. Scopes are per function.
    That is enough to catch the common shape ``query = "..." + user; cur.execute(query)``
    without pretending to be a full data-flow analysis.
    """

    def __init__(self) -> None:
        self.findings: list[CodeReviewIssue] = []
        self._tainted: dict[str, int] = {}

    def _is_built(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return node.id in self._tainted
        return _is_built_string(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._scoped(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._scoped(node)

    def _scoped(self, node: ast.AST) -> None:
        outer = self._tainted
        self._tainted = dict(outer)
        self.generic_visit(node)
        self._tainted = outer

    def visit_Assign(self, node: ast.Assign) -> None:
        self._check_secret(node.targets, node.value, node.lineno)
        self.generic_visit(node)
        built = self._is_built(node.value)
        for target in node.targets:
            if isinstance(target, ast.Name):
                if built:
                    self._tainted[target.id] = node.lineno
                else:
                    self._tainted.pop(target.id, None)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.generic_visit(node)
        if isinstance(node.target, ast.Name) and isinstance(node.op, (ast.Add, ast.Mod)):
            # ``q += <anything non-constant>`` assembles a string from non-constant parts.
            if not _is_constant_str(node.value) or node.target.id in self._tainted:
                self._tainted[node.target.id] = node.lineno

    def _add(self, line: int, severity: Severity, title: str, fix: str) -> None:
        self.findings.append(CodeReviewIssue(
            severity=severity, line_hint=f"line {line}", description=title, suggested_fix=fix,
        ))

    def visit_Call(self, node: ast.Call) -> None:
        name = _dotted(node.func)
        last = name.rsplit(".", 1)[-1]
        if name in ("eval", "exec"):
            self._add(node.lineno, "critical", f"Dynamic code execution via {name}()",
                      "Remove eval/exec; dispatch through an explicit table of allowed operations.")
        elif name == "os.system":
            self._add(node.lineno, "high", "Shell command via os.system()",
                      "Use subprocess.run([...]) with an argument list and no shell.")
        elif name.startswith("subprocess.") and any(
            kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True
            for kw in node.keywords
        ):
            self._add(node.lineno, "high", "subprocess called with shell=True",
                      "Pass an argument list and drop shell=True.")
        elif name in ("pickle.load", "pickle.loads", "marshal.loads", "dill.loads"):
            self._add(node.lineno, "high", f"Deserialising untrusted data with {name}()",
                      "Use JSON or another data-only format for anything that crosses a trust boundary.")
        elif name == "yaml.load" and not any(kw.arg == "Loader" for kw in node.keywords):
            self._add(node.lineno, "high", "yaml.load() without a safe Loader",
                      "Use yaml.safe_load().")
        elif name in ("hashlib.md5", "hashlib.sha1"):
            self._add(node.lineno, "low", f"Weak hash {name}()",
                      "Use hashlib.sha256 or a password KDF for credentials.")
        elif last in _SQL_METHODS and node.args and self._is_built(node.args[0]):
            via = " (via a variable)" if isinstance(node.args[0], ast.Name) else ""
            self._add(node.lineno, "critical", f"SQL built by string interpolation in .{last}(){via}",
                      "Pass the values as parameters: cursor.execute(sql, params).")

        for kw in node.keywords:
            if kw.arg == "verify" and isinstance(kw.value, ast.Constant) and kw.value.value is False:
                self._add(node.lineno, "high", "TLS certificate verification disabled (verify=False)",
                          "Remove verify=False and pin the CA bundle if needed.")
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self._check_secret([node.target], node.value, node.lineno)
        self.generic_visit(node)

    def _check_secret(self, targets: list[ast.expr], value: ast.expr, line: int) -> None:
        if not (isinstance(value, ast.Constant) and isinstance(value.value, str)):
            return
        literal = value.value
        if len(literal) < 8:
            return
        if any(marker in literal.lower() for marker in _PLACEHOLDERS):
            return
        for target in targets:
            name = target.id if isinstance(target, ast.Name) else (
                target.attr if isinstance(target, ast.Attribute) else "")
            if name and _SECRET_NAME.search(name):
                self._add(line, "critical", f"Hard-coded credential assigned to `{name}`",
                          "Load the secret from the environment or a secret store.")
                return

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is None:
            self._add(node.lineno, "medium", "Bare except clause catches SystemExit and KeyboardInterrupt",
                      "Catch specific exceptions, e.g. `except Exception as exc:`.")
        self.generic_visit(node)


def static_findings(code: str, filename: str) -> tuple[list[CodeReviewIssue], str]:
    """Return ``(issues, scope)``. ``scope`` says what kind of analysis produced them."""
    if filename.lower().endswith(".py"):
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return [CodeReviewIssue(
                severity="high", line_hint=f"line {exc.lineno}",
                description=f"Does not parse as Python: {exc.msg}",
                suggested_fix="Fix the syntax error before review or deployment.",
            )], "python AST analysis"
        auditor = _PythonAuditor()
        auditor.visit(tree)
        return auditor.findings, "python AST analysis"

    findings: list[CodeReviewIssue] = []
    for lineno, line in enumerate(code.splitlines(), start=1):
        for pattern, severity, title, fix in _PATTERN_RULES:
            if pattern.search(line):
                findings.append(CodeReviewIssue(
                    severity=severity, line_hint=f"line {lineno}", description=title, suggested_fix=fix,
                ))
    return findings, "pattern analysis only (no parser for this language)"


# -- model review ----------------------------------------------------------------------

def parse_model_review(text: str) -> _ModelReview | None:
    """Return the model's review, or None if the output is not a valid review."""
    if not text or not text.strip():
        return None
    try:
        data = extract_json_object(text)
        review = _ModelReview.model_validate(data)
    except (ValueError, ValidationError):
        return None
    review.issues = review.issues[:MAX_MODEL_ISSUES]
    return review


# -- service -----------------------------------------------------------------------------

class CodeReviewDebateService:
    """Static checks plus one structured model review. Not a multi-agent debate."""

    async def review_code(self, req: CodeReviewRequest) -> CodeReviewResponse:
        start_time = time.perf_counter()
        static_issues, scope = static_findings(req.code, req.filename)

        shown = req.code[:MODEL_INPUT_CHARS]
        truncated = len(req.code) > MODEL_INPUT_CHARS
        prompt = (
            f"Review this source code for `{req.filename}` with focus on {', '.join(req.review_focus)}.\n\n"
            f"```\n{shown}\n```\n\n"
            "Respond with ONLY a JSON object, no prose, in this shape:\n"
            '{"verdict": "approve" | "fix_required" | "needs_review", '
            '"issues": [{"severity": "critical|high|medium|low|info", "line_hint": "line N or null", '
            '"description": "...", "suggested_fix": "..."}]}\n'
            "Report real defects only. Use fix_required for any correctness or security defect."
        )
        exec_res = await unified_provider_manager.execute(UnifiedExecutionRequest(
            provider="auto",
            agent_role="code_reviewer",
            prompt=prompt,
            context={"filename": req.filename, "project_context": req.project_context},
            max_tokens=2500,
            temperature=0.2,
        ))
        elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        model_review: _ModelReview | None = None
        if exec_res is None or getattr(exec_res, "degraded", False):
            model_status: Literal["reviewed", "unparsed", "unavailable"] = "unavailable"
        else:
            model_review = parse_model_review(exec_res.content or "")
            model_status = "reviewed" if model_review is not None else "unparsed"

        model_issues = model_review.issues if model_review else []
        all_issues = static_issues + model_issues

        verdict = self._merge_verdict(static_issues, model_issues, model_review, model_status)
        review_source: Literal["static_and_model", "static_only"] = (
            "static_and_model" if model_review is not None else "static_only"
        )
        confidence = 0.8 if model_review is not None else 0.5

        return CodeReviewResponse(
            verdict=verdict,
            issues=all_issues,
            consensus_confidence=confidence,
            debate_summary=self._summary(req, verdict, static_issues, model_issues, model_status, truncated),
            latency_ms=elapsed_ms,
            review_source=review_source,
            model_status=model_status,
            static_issue_count=len(static_issues),
            model_issue_count=len(model_issues),
            analysis_scope=scope,
        )

    @staticmethod
    def _merge_verdict(
        static_issues: list[CodeReviewIssue],
        model_issues: list[CodeReviewIssue],
        model_review: _ModelReview | None,
        model_status: str,
    ) -> Verdict:
        all_issues = static_issues + model_issues
        if any(_SEVERITY_RANK[i.severity] >= _SEVERITY_RANK["high"] for i in all_issues):
            return "fix_required"
        if model_review is not None and _VERDICT_RANK[model_review.verdict] >= _VERDICT_RANK["fix_required"]:
            return "fix_required"
        if model_status != "reviewed":
            return "needs_review"
        if any(_SEVERITY_RANK[i.severity] >= _SEVERITY_RANK["medium"] for i in all_issues):
            return "needs_review"
        if model_review is not None and model_review.verdict == "needs_review":
            return "needs_review"
        return "approve"

    @staticmethod
    def _summary(
        req: CodeReviewRequest,
        verdict: str,
        static_issues: list[CodeReviewIssue],
        model_issues: list[CodeReviewIssue],
        model_status: str,
        truncated: bool,
    ) -> str:
        parts = [f"Review of `{req.filename}`: static checks found {len(static_issues)} issue(s)."]
        if model_status == "reviewed":
            parts.append(f"Model review found {len(model_issues)} issue(s).")
        elif model_status == "unparsed":
            parts.append("The model's review could not be parsed as a valid review; static checks only.")
        else:
            parts.append("No model review was available; static checks only.")
        if truncated:
            parts.append(f"The model saw the first {MODEL_INPUT_CHARS} of {len(req.code)} characters; "
                         "static checks covered all of it.")
        parts.append(f"Final verdict: {verdict}.")
        return " ".join(parts)


code_review_service = CodeReviewDebateService()
