"""Contract fuzzing: every documented route must survive a schema-valid request body.

This file exists because a passing suite proved nothing about these failures. The
defects below were all found by driving the running service with bodies its *own*
published schema accepts, and every one of them returned HTTP 500 — a server fault on
input the service declared valid:

* ``POST /v1/market/portfolio-analysis`` with ``{"positions": {}}``
  -> ``KeyError: 'concentration_risk_warning'`` (``app/debate/market_debate.py:34``).
  The engine's empty-portfolio branch returned two keys, one of them misspelled
  (``concentration_warning``), while the populated branch returned four.
* ``POST /v1/trading/live/stress-test`` with ``{"portfolio_equity": 0, ...}``
  -> ``ZeroDivisionError`` (``app/analysis/stress_intelligence.py:43``). The loss ratio
  was computed twice per scenario; only one of the two divisions carried a zero guard.
* ``POST /v1/sentinel/analyze`` (and ``/v1/nexus/intelligence``, ``/v1/intelx/research``,
  ``/v1/futuris/enhance``) after another of those endpoints cached a response under the
  same ``request_id``
  -> ``ValidationError: analysis Field required``
  (``app/services/sentinel_intelligence.py:190``). All four shared one deduplication cache
  keyed by the bare caller-supplied ``request_id``, so one endpoint could be served
  another endpoint's payload.

The generator below synthesises a body from each route's OpenAPI schema — empty objects,
empty lists, zeros, empty strings and "required fields only" — then asserts that no route
answers 5xx. An empty object is always legal for a field with no ``required`` children,
so a 500 here is never the client's fault.
"""

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from app.main import app

# Requests are sent to the in-process ASGI app, so this exercises the real middleware
# stack (security, rate limiting, error encoding) without needing a running server.


def _schema_of(spec: dict, op: dict) -> dict | None:
    body = op.get("requestBody")
    if not body:
        return None
    content = body.get("content", {})
    if not content:
        return None
    first = content.get("application/json") or next(iter(content.values()))
    return first.get("schema")


def _deref(schema: dict, spec: dict) -> dict:
    """Follow ``$ref`` headers so nested models are visible to the generator."""
    components = spec.get("components", {}).get("schemas", {})
    if "$ref" in schema:
        name = schema["$ref"].split("/")[-1]
        return components.get(name, {})
    if "allOf" in schema:
        merged: dict = {}
        for part in schema["allOf"]:
            merged.update(_deref(part, spec))
        return merged
    return schema


def _value(name: str, sub: dict, spec: dict, style: str, depth: int = 0):
    """One plausible value for ``sub`` in the requested ``style``.

    ``style`` is one of ``empty`` (objects/arrays collapse to empty), ``zeros``
    (numbers become 0), ``blank`` (strings become ""), or ``required_only``.
    """
    if depth > 8:
        return None
    sub = _deref(sub, spec)
    if "enum" in sub and sub["enum"]:
        return sub["enum"][0]
    if sub.get("anyOf"):
        for option in sub["anyOf"]:
            if option.get("type") != "null":
                return _value(name, option, spec, style, depth + 1)
        return None
    kind = sub.get("type")
    if kind == "array":
        return []
    if kind == "object" or "properties" in sub:
        return _object(sub, spec, style, depth + 1)
    if kind == "string":
        return "" if style in ("blank", "required_only") else "x"
    if kind in ("integer", "number"):
        return 0
    if kind == "boolean":
        return False
    return ""


def _object(schema: dict, spec: dict, style: str, depth: int = 0, required_only: bool = False) -> dict:
    schema = _deref(schema, spec)
    properties = schema.get("properties") or {}
    if not properties:
        return {}
    if required_only:
        names = [n for n in (schema.get("required") or []) if n in properties]
    else:
        names = list(properties)
    return {n: _value(n, properties[n], spec, style, depth) for n in names}


def _variants(schema: dict, spec: dict) -> list[tuple[str, dict]]:
    if not schema:
        return []
    out = [
        ("empty", _object(schema, spec, "empty")),
        ("zeros", _object(schema, spec, "zeros")),
        ("blank", _object(schema, spec, "blank")),
        ("required_only", _object(schema, spec, "required_only", required_only=True)),
    ]
    seen: set[str] = set()
    unique: list[tuple[str, dict]] = []
    for name, payload in out:
        key = json.dumps(payload, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        unique.append((name, payload))
    return unique


def _post_routes() -> list[tuple[str, dict]]:
    spec = app.openapi()
    routes: list[tuple[str, dict]] = []
    for path, operations in sorted(spec["paths"].items()):
        for method, op in operations.items():
            if method != "post":
                continue
            if "{" in path:  # path parameters need concrete values; covered elsewhere
                continue
            schema = _schema_of(spec, op)
            if schema:
                routes.append((path, schema))
    return routes


def _crash_verdict(response) -> str | None:
    """Classify a response, returning a reason if it represents an unhandled crash.

    A 5xx is not automatically a defect: refusing with ``503`` and a specific reason
    ("no model provider produced output (groq: ProviderUnconfiguredError...)") is honest
    API behaviour for a service that genuinely cannot serve the request. What must never
    happen is the *unhandled* path — status 500, or the generic encoder's
    ``INTERNAL_SERVER_ERROR`` body
    (``app/security/api_security.py:164``) — because that tells the caller nothing and
    means the service broke rather than declined.
    """
    if response.status_code == 500:
        return "unhandled 500"
    try:
        body = response.json()
    except Exception:
        body = None
    if isinstance(body, dict):
        if body.get("error_code") == "INTERNAL_SERVER_ERROR":
            return "generic INTERNAL_SERVER_ERROR body"
        if response.status_code >= 500:
            explained = body.get("detail") or body.get("message") or body.get("reason")
            if not explained:
                return f"unexplained {response.status_code} with no reason given"
    return None


@pytest.mark.parametrize("path,schema", _post_routes(), ids=lambda v: v if isinstance(v, str) else "")
def test_schema_valid_degenerate_body_never_returns_5xx(path: str, schema: dict, auth):
    """No route may answer 5xx to a body its own schema accepts.

    Each variant is a maximally degenerate but schema-legal body. A 4xx is a perfectly
    good answer (it is a client-facing contract); a 5xx means the service crashed on
    input it published as valid, which is what every defect in this module's docstring
    did before it was fixed.
    """
    spec = app.openapi()
    # The `auth` fixture points the server at a known test credential; without it every
    # route fails closed with 503 (correct behaviour, but it hides the body handling).
    headers = {**auth, "Content-Type": "application/json"}
    failures: list[str] = []

    # ``raise_server_exceptions=False`` returns a 500 response instead of re-raising, so
    # every crashing variant is collected and reported together rather than stopping at
    # the first one.
    client = TestClient(app, raise_server_exceptions=False)
    for variant, payload in _variants(schema, spec):
        response = client.post(path, json=payload, headers=headers)
        verdict = _crash_verdict(response)
        if verdict:
            failures.append(
                f"{path} [{variant}] -> {response.status_code} ({verdict}): "
                f"sent={json.dumps(payload)[:120]} got={response.text[:160]}"
            )

    assert not failures, (
        "these routes crashed on bodies their own OpenAPI schema accepts, so the "
        "published contract is a lie:\n  " + "\n  ".join(failures)
    )


def test_the_fuzzer_actually_covers_the_post_surface():
    """Guard against the suite silently shrinking to nothing.

    A parametrised test that generates zero cases passes vacuously. This asserts the
    generator still finds a meaningful number of documented POST routes, so a change to
    OpenAPI generation cannot quietly disable the whole file.
    """
    routes = _post_routes()
    assert len(routes) >= 20, f"expected the POST surface to be large, found {len(routes)}"
    assert "/v1/market/portfolio-analysis" in [p for p, _ in routes]
    assert "/v1/trading/live/stress-test" in [p for p, _ in routes]


def test_cross_endpoint_cache_collision_is_impossible_over_http(auth):
    """Two endpoints sharing a request_id must each get their own answer.

    This is the end-to-end form of the dedup defect: send the same request_id to
    /v1/nexus/intelligence and then to /v1/sentinel/analyze and require that the second
    endpoint answers with its *own* schema (or a clean client error), never with a
    validation failure caused by the first endpoint's cached payload.
    """
    spec = app.openapi()
    shared_id = "collision-probe-001"

    nexus_schema = _schema_of(spec, spec["paths"]["/v1/nexus/intelligence"]["post"])
    sentinel_schema = _schema_of(spec, spec["paths"]["/v1/sentinel/analyze"]["post"])

    nexus_body = _object(nexus_schema, spec, "blank")
    nexus_body["request_id"] = shared_id
    nexus_body["goal"] = "Decide whether to co-locate Postgres and SQLite in one service."
    sentinel_body = _object(sentinel_schema, spec, "blank")
    sentinel_body["request_id"] = shared_id
    sentinel_body["analysis_type"] = "risk_scoring"
    sentinel_body["target_context"] = {
        "asset_type": "web-service",
        "technologies_detected": ["nginx"],
        "exposure_level": "public_internet",
    }
    sentinel_body["findings"] = [
        {"finding_id": "f1", "title": "TLS 1.0 enabled", "severity": "MEDIUM", "description": "Old TLS."}
    ]

    headers = {**auth, "Content-Type": "application/json"}
    client = TestClient(app, raise_server_exceptions=False)
    first = client.post("/v1/nexus/intelligence", json=nexus_body, headers=headers)
    second = client.post("/v1/sentinel/analyze", json=sentinel_body, headers=headers)

    assert first.status_code < 500, f"nexus failed: {first.status_code} {first.text[:200]}"
    assert second.status_code < 500, (
        f"the sentinel endpoint was served the nexus entry for request_id={shared_id!r} "
        f"and crashed while validating it: {second.status_code} {second.text[:300]}"
    )


# ── non-finite floats, on every route ────────────────────────────────────────
#
# Python's ``json`` accepts ``NaN``/``Infinity`` and parses ``1e309`` as ``inf``, so a
# hostile or careless client can put a non-finite float into any numeric field. Pydantic
# rejects those values correctly, but the rejection message echoes the offending input,
# and Starlette's JSONResponse encodes with ``allow_nan=False`` — so the 422 failed to
# serialise and surfaced as a 500 on every validating route. That is a whole-class defect
# rather than a per-route bug, so it is swept across the whole surface here.

NON_FINITE_BODIES = [
    ('{"n": NaN}', "NaN"),
    ('{"n": Infinity}', "Infinity"),
    ('{"n": -Infinity}', "-Infinity"),
    ('{"value": 1e309}', "Infinity"),
    ('{"amount": 1e400, "n": 0}', "Infinity"),
]


@pytest.mark.parametrize("path,schema", _post_routes(), ids=lambda v: v if isinstance(v, str) else "")
def test_non_finite_floats_never_produce_a_5xx(path: str, schema: dict, auth):
    """A non-finite float in the body must yield a client error, never a server fault.

    The body is deliberately generic (``{"n": ...}``) so the value lands in a *different*
    field per route — or in an unknown field, which is ignored. The property under test is
    that no route can turn "the client sent a non-finite number" into an unhandled crash.
    """
    properties = _deref(schema, app.openapi()).get("properties", {})
    # Aim the non-finite value at a real numeric field where one exists, so the request
    # reaches the validator that rejects it rather than being ignored as an extra key.
    numeric_names = [
        name for name, spec in properties.items()
        if _deref(spec, app.openapi()).get("type") in ("integer", "number")
    ]
    headers = {**auth, "Content-Type": "application/json"}
    client = TestClient(app, raise_server_exceptions=False)
    failures: list[str] = []

    for raw, _label in NON_FINITE_BODIES:
        payload = json.loads(raw)
        body = dict(payload)
        if numeric_names:
            body = {numeric_names[0]: payload.get("n", payload.get("value", payload.get("amount")))}
        response = client.post(
            path,
            content=json.dumps(body, allow_nan=True).encode(),
            headers=headers,
        )
        verdict = _crash_verdict(response)
        if verdict:
            failures.append(
                f"{path} body={json.dumps(body)[:80]} -> {response.status_code} ({verdict})"
            )
        if response.status_code == 422:
            assert "detail" in response.json(), (
                f"{path}: a 422 must explain itself: {response.text[:200]}"
            )

    assert not failures, (
        "non-finite JSON floats must be a clean client error:\n  " + "\n  ".join(failures)
    )


# ── every router module in the repo must actually be mounted ─────────────────
# A router that is written but never ``include_router``-ed is invisible: it appears in no
# OpenAPI path, so no live sweep and no schema test can see it. It is also the most common
# FastAPI wiring mistake. This test enumerates the APIRouter prefixes declared anywhere in
# the application package and asserts each one is present among the app's route paths.


def _declared_router_prefixes() -> dict[str, str]:
    """prefix -> module, for every ``APIRouter(prefix=...)`` under app/."""
    import pathlib
    import re

    declared: dict[str, str] = {}
    root = pathlib.Path(__file__).resolve().parent.parent / "app"
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in re.finditer(r"APIRouter\(\s*prefix\s*=\s*[\"']([^\"']*)[\"']", text):
            declared.setdefault(match.group(1), str(path.relative_to(root.parent)))
    return declared


def test_every_declared_router_prefix_is_mounted():
    from app.main import app

    # ``app.routes`` holds ``_IncludedRouter`` wrappers whose ``path`` is None; the
    # resolved paths come from the OpenAPI schema, which is the same set a client sees.
    mounted_paths = set(app.openapi()["paths"].keys())
    prefixes = _declared_router_prefixes()
    assert len(prefixes) >= 15, f"expected the app to declare many routers, found {len(prefixes)}"

    missing = []
    for prefix, module in sorted(prefixes.items()):
        if not prefix:
            continue
        if not any(p.startswith(prefix) for p in mounted_paths):
            missing.append(f"{prefix} (declared in {module})")
    assert not missing, "routers declared but never mounted: " + "; ".join(missing)


def test_every_advertised_openapi_path_is_routed():
    """The schema a client reads must match the app that answers.

    A documented path that 404s at runtime is the disagreement this catches; 422/401/200/
    503 are all legitimate answers to a request with a dummy body.
    """
    import httpx

    from app.main import app

    spec = app.openapi()
    documented = {(path, method.upper()) for path, ops in spec["paths"].items() for method in ops}
    assert len(documented) > 80, f"the schema should document the API, found {len(documented)}"

    async def probe():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test",
                                     headers={"X-API-Key": "test-key"}) as client:
            unrouted = []
            for path, ops in spec["paths"].items():
                for method in ops:
                    if method.lower() in ("head", "options"):
                        continue
                    url = path
                    for segment in path.split("/"):
                        if segment.startswith("{"):
                            url = url.replace(segment, "nonexistent-id")
                    spec_ops = spec["paths"][path][method]
                    lookup = method.upper() == "GET" and "{" in path
                    params = [
                        p["name"] for p in spec_ops.get("parameters", [])
                        if p.get("in") == "query" and p.get("required")
                    ]
                    r = await client.request(
                        method.upper(), url,
                        params={name: 1 for name in params} or None,
                        json={} if method.upper() in ("POST", "PUT", "PATCH") else None,
                    )
                    if r.status_code == 404 and not lookup:
                        unrouted.append((method.upper(), url, r.status_code))
            return unrouted

    unrouted = asyncio.run(probe())
    # A GET with a path parameter is a resource lookup; 404 for an unknown id is correct.
    assert not unrouted, f"documented but unrouted operations: {unrouted}"
