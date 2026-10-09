"""Every path the OpenAPI schema advertises must actually be routed.

The schema is the contract: it is what a client generates code from and what an operator
reads. A router that is declared but never mounted (or mounted under a different prefix)
produces a schema entry that answers ``404 {"detail": "Not Found"}`` — the exact symptom
that made an earlier path sweep ambiguous, because ``404`` is also a legitimate business
answer (``GET /tasks/{id}`` for an unknown id).

The discriminator used here is precise: Starlette's *routing* miss returns the bare detail
``"Not Found"``, while every handler in this service that answers 404 does so with its own
message (e.g. ``"Task 'x' not found."`` or an eviction-aware retention explanation).
"""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

ROUTING_MISS_DETAIL = "Not Found"

#: Methods with no request body to build; anything else gets an empty JSON object, which the
#: contract is free to reject with 422.
_IDEMPOTENT = {"get", "delete", "head", "options"}


def _concrete(path: str) -> str:
    return re.sub(r"\{[^}]+\}", "probe_value", path)


@pytest.fixture(scope="module")
def schema() -> dict:
    from app.main import app

    return app.openapi()


def test_the_schema_is_not_empty(schema):
    assert len(schema["paths"]) > 50, "the schema shrank; this test would silently pass"


def _unrouted_operations(paths: dict, client: TestClient, auth: dict) -> tuple[list[str], int]:
    """Return the declared operations that answer with a bare routing miss."""
    unrouted: list[str] = []
    checked = 0
    for path, operations in sorted(paths.items()):
        for method in operations:
            if method not in ("get", "post", "put", "patch", "delete"):
                continue
            checked += 1
            kwargs: dict = {"headers": auth}
            if method not in _IDEMPOTENT:
                kwargs["json"] = {}
            response = client.request(method.upper(), _concrete(path), **kwargs)
            body = (
                response.json()
                if response.headers.get("content-type", "").startswith("application/json")
                else {}
            )
            detail = body.get("detail") if isinstance(body, dict) else None
            if response.status_code == 404 and detail == ROUTING_MISS_DETAIL:
                unrouted.append(f"{method.upper()} {path} -> 404 routing miss")
    return unrouted, checked


def test_every_declared_operation_is_routed(schema, auth):
    from app.main import app

    unrouted, checked = _unrouted_operations(schema["paths"], TestClient(app), auth)
    assert checked > 100, f"only {checked} operations checked; the schema may have failed to build"
    assert not unrouted, "declared but unreachable: " + "; ".join(unrouted)


def test_the_detector_flags_a_path_that_is_not_mounted(schema, auth):
    """Falsification: the invariant must fire for a declared-but-absent path.

    A router missing from ``include_router`` is the failure this guards against, and the
    only way to test the *detector* is to hand it a schema entry that has no route behind it
    (patching ``app.router.routes`` does not work: included routers are wrapped in objects
    whose ``path`` attribute is ``None``, so a path-based filter removes nothing).
    """
    from app.main import app

    augmented = {"paths": dict(schema["paths"]), **{k: v for k, v in schema.items() if k != "paths"}}
    augmented["paths"]["/v1/definitely_not_mounted/probe"] = {
        "get": {"summary": "Probe", "responses": {"200": {"description": "ok"}}}
    }

    unrouted, checked = _unrouted_operations(augmented["paths"], TestClient(app), auth)
    assert checked > 100
    assert any("/v1/definitely_not_mounted/probe" in row for row in unrouted), (
        f"the detector missed an unmounted path; unrouted={unrouted}"
    )
