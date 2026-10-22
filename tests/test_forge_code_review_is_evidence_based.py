"""FORGE's code review must not approve code it did not review.

Before the fix (``app/services/code_review.py``), the operator tour sent a SQL-injection snippet
and got back ``verdict: approve``, zero issues and ``consensus_confidence: 0.91``. The cause, read
from the source: the verdict defaulted to "approve", four substring checks were the only
analysis, the model's answer was never parsed, and the summary described a "panel consensus"
although one provider call had run. Each test below names the pre-fix behaviour it guards.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient

import app.services.code_review as code_review_module
from app.main import app
from app.providers import unified_manager
from app.services.code_review import CodeReviewRequest, code_review_service

# Looked up through the module at call time: a helper missing from an older module then fails
# one test instead of the whole file at import.
MODEL_INPUT_CHARS = 3000


def static_findings(code: str, filename: str):
    return code_review_module.static_findings(code, filename)


def parse_model_review(text: str):
    return code_review_module.parse_model_review(text)

CLEAN = "def add(a: int, b: int) -> int:\n    return a + b\n"
SQLI = (
    "import sqlite3\n"
    "def find(conn: sqlite3.Connection, user: str):\n"
    "    cur = conn.cursor()\n"
    "    cur.execute(f\"SELECT * FROM users WHERE name = '{user}'\")\n"
    "    return cur.fetchall()\n"
)


def _model(content: str, *, degraded: bool = False) -> SimpleNamespace:
    return SimpleNamespace(content=content, degraded=degraded, error=None, token_usage={})


@pytest.fixture
def model_reply(monkeypatch):
    """Set the reply the model review receives. Default: prose, which is not a review."""
    state = {"content": "Looks fine to me.", "degraded": False, "calls": 0}

    async def execute(exec_req, *args, **kwargs):  # noqa: ARG001
        state["calls"] += 1
        return _model(state["content"], degraded=state["degraded"])

    monkeypatch.setattr(unified_manager.unified_provider_manager, "execute", execute)
    return state


async def _review(code: str, filename: str = "module.py"):
    return await code_review_service.review_code(CodeReviewRequest(code=code, filename=filename))


# -- static analysis -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sql_built_by_fstring_is_critical_even_when_the_model_says_nothing(model_reply):
    """Pre-fix: verdict approve, zero issues, with the model's prose ignored."""
    response = await _review(SQLI)

    assert response.verdict == "fix_required"
    assert any(i.severity == "critical" and "SQL" in i.description for i in response.issues)
    assert response.static_issue_count >= 1


def test_parameterized_query_is_not_flagged():
    code = "def f(cur, user):\n    cur.execute('SELECT * FROM t WHERE name = ?', (user,))\n"
    issues, _ = static_findings(code, "p.py")
    assert not any("SQL" in i.description for i in issues)


def test_sql_built_by_concatenation_and_format_is_flagged():
    concat = "def f(cur, u):\n    cur.execute('SELECT * FROM t WHERE n = ' + u)\n"
    fmt = "def f(cur, u):\n    cur.execute('SELECT * FROM t WHERE n = {}'.format(u))\n"
    for code in (concat, fmt):
        issues, _ = static_findings(code, "q.py")
        assert any("SQL" in i.description for i in issues), code


@pytest.mark.parametrize("code,needle", [
    ("import subprocess\nsubprocess.run(cmd, shell=True)\n", "shell=True"),
    ("import pickle\nobj = pickle.loads(blob)\n", "pickle"),
    ("import yaml\ncfg = yaml.load(text)\n", "yaml.load"),
    ("import requests\nrequests.get(url, verify=False)\n", "verify=False"),
    ("import os\nos.system('rm -rf ' + path)\n", "os.system"),
    ("password = 'hunter2hunter2'\n", "Hard-coded credential"),
    ("try:\n    x()\nexcept:\n    pass\n", "Bare except"),
    ("import hashlib\nh = hashlib.md5(b'x')\n", "Weak hash"),
    ("v = eval(user_input)\n", "Dynamic code execution"),
])
def test_each_python_rule_fires(code, needle):
    issues, scope = static_findings(code, "rules.py")
    assert any(needle in i.description for i in issues), (needle, [i.description for i in issues])
    assert scope == "python AST analysis"


def test_placeholder_secret_is_not_flagged():
    issues, _ = static_findings("api_key = 'changeme-changeme'\n", "cfg.py")
    assert not any("credential" in i.description for i in issues)


def test_syntax_error_in_python_is_a_high_finding_not_a_pass():
    issues, _ = static_findings("def broken(:\n    pass\n", "bad.py")
    assert issues and issues[0].severity == "high"
    assert "Does not parse as Python" in issues[0].description


def test_non_python_uses_pattern_analysis_and_says_so():
    issues, scope = static_findings("const r = eval(userInput);\n", "app.js")
    assert any("Dynamic code execution" in i.description for i in issues)
    assert "pattern analysis" in scope


def test_static_analysis_covers_the_whole_source_not_the_model_window():
    """The model sees the first 3000 characters. A defect after that must still be found."""
    padding = "# filler line\n" * (MODEL_INPUT_CHARS // 10 + 50)
    code = padding + SQLI
    assert len(code) > MODEL_INPUT_CHARS
    issues, _ = static_findings(code, "long.py")
    assert any("SQL" in i.description for i in issues)


# -- the verdict comes from evidence ---------------------------------------------------


@pytest.mark.asyncio
async def test_clean_code_is_not_approved_when_the_model_review_did_not_parse(model_reply):
    """Pre-fix: any clean-looking code was 'approve'. With no parsed model review, it is needs_review."""
    model_reply["content"] = "Looks fine to me."
    response = await _review(CLEAN)

    assert response.model_status == "unparsed"
    assert response.review_source == "static_only"
    assert response.verdict == "needs_review"
    assert response.consensus_confidence == 0.5


@pytest.mark.asyncio
async def test_clean_code_is_approved_only_with_a_parsed_model_review(model_reply):
    model_reply["content"] = '{"verdict": "approve", "issues": []}'
    response = await _review(CLEAN)

    assert response.model_status == "reviewed"
    assert response.review_source == "static_and_model"
    assert response.verdict == "approve"
    assert response.consensus_confidence == 0.8


@pytest.mark.asyncio
async def test_model_approval_cannot_override_a_static_critical_finding(model_reply):
    model_reply["content"] = '{"verdict": "approve", "issues": []}'
    response = await _review(SQLI)
    assert response.verdict == "fix_required"


@pytest.mark.asyncio
async def test_model_defects_are_returned_as_structured_issues(model_reply):
    model_reply["content"] = (
        '```json\n{"verdict": "fix_required", "issues": [{"severity": "high", '
        '"line_hint": "line 7", "description": "off-by-one in pagination", '
        '"suggested_fix": "use < instead of <="}]}\n```'
    )
    response = await _review(CLEAN)

    assert response.model_status == "reviewed"
    assert response.verdict == "fix_required"
    assert response.model_issue_count == 1
    assert response.issues[0].description == "off-by-one in pagination"


@pytest.mark.asyncio
async def test_unavailable_model_is_reported_as_unavailable_not_as_a_review(model_reply):
    model_reply["degraded"] = True
    response = await _review(CLEAN)

    assert response.model_status == "unavailable"
    assert response.verdict == "needs_review"
    assert "No model review was available" in response.debate_summary


@pytest.mark.asyncio
async def test_summary_does_not_claim_a_panel_debate_happened(model_reply):
    """Pre-fix: 'Panel consensus reached across Coder, Security Analyst, and Critic.'"""
    model_reply["content"] = '{"verdict": "approve", "issues": []}'
    response = await _review(CLEAN)

    lowered = response.debate_summary.lower()
    assert "consensus" not in lowered
    assert "panel" not in lowered
    assert "static checks" in lowered


@pytest.mark.asyncio
async def test_summary_discloses_that_the_model_saw_a_truncated_source(model_reply):
    model_reply["content"] = '{"verdict": "approve", "issues": []}'
    code = CLEAN + ("# pad\n" * 1000)
    response = await _review(code)
    assert "saw the first 3000" in response.debate_summary


def test_model_output_must_validate_to_count_as_a_review():
    assert parse_model_review("Looks fine.") is None
    assert parse_model_review('{"verdict": "maybe", "issues": []}') is None
    assert parse_model_review('{"verdict": "approve"}') is not None
    assert parse_model_review("") is None


# -- over HTTP -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_review_route_does_not_approve_sql_injection(model_reply, auth):
    model_reply["content"] = "No issues found."
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver",
                           headers=auth, timeout=60) as client:
        response = await client.post("/v1/forge/review-code", json={"code": SQLI, "filename": "users.py"})

    assert response.status_code == 200
    body = response.json()
    assert body["verdict"] == "fix_required"
    assert body["review_source"] == "static_only"
    assert body["analysis_scope"] == "python AST analysis"


# -- data flow through a local variable -------------------------------------------------


PLANTED_VIA_VARIABLE = (
    "import sqlite3\n"
    "def find_user(conn: sqlite3.Connection, name: str):\n"
    "    query = \"SELECT id, email FROM users WHERE name = '\" + name + \"'\"\n"
    "    return conn.execute(query).fetchall()\n"
)


def test_sql_built_into_a_variable_then_executed_is_flagged():
    """Pre-fix: the planted bug in the operator tour built the query in a variable and passed the
    variable to execute(). The check only looked at inline arguments, so it passed the bug."""
    issues, _ = static_findings(PLANTED_VIA_VARIABLE, "users.py")
    assert any("SQL" in i.description and "via a variable" in i.description for i in issues)


def test_variable_reassigned_to_a_constant_is_no_longer_tainted():
    code = (
        "def f(cur, name):\n"
        "    q = 'SELECT 1 WHERE x = ' + name\n"
        "    q = 'SELECT 1'\n"
        "    cur.execute(q)\n"
    )
    issues, _ = static_findings(code, "clear.py")
    assert not any("SQL" in i.description for i in issues)


def test_taint_does_not_leak_out_of_its_function():
    code = (
        "def a(name):\n"
        "    q = 'SELECT ' + name\n"
        "    return q\n"
        "def b(cur):\n"
        "    q = 'SELECT 1'\n"
        "    cur.execute(q)\n"
    )
    issues, _ = static_findings(code, "scope.py")
    assert not any("SQL" in i.description for i in issues)


def test_augmented_concatenation_taints_the_variable():
    code = (
        "def f(cur, name):\n"
        "    q = 'SELECT * FROM t WHERE n = '\n"
        "    q += name\n"
        "    cur.execute(q)\n"
    )
    issues, _ = static_findings(code, "aug.py")
    assert any("SQL" in i.description for i in issues)


@pytest.mark.asyncio
async def test_planted_bug_through_the_route_is_not_approved(model_reply):
    """The exact shape the operator tour sent, with the model returning prose (not a review)."""
    response = await _review(PLANTED_VIA_VARIABLE, "users.py")
    assert response.verdict == "fix_required"
    assert response.issues[0].severity == "critical"
