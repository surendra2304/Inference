"""Make arbitrary Python values safe to serialise into an HTTP JSON response.

Why this exists
---------------
Starlette's ``JSONResponse.render`` calls ``json.dumps(..., allow_nan=False)``, so a
non-finite float anywhere in a response body raises::

    ValueError: Out of range float values are not JSON compliant

That turned a *correct validation error* into a server fault. Python's ``json`` module
accepts the non-standard literals ``NaN``, ``Infinity`` and ``-Infinity`` when parsing, so
a client can send ``{"max_agents": 1e309}`` and Pydantic will correctly reject it with
"Input should be a finite number" — while echoing the offending value back:

    {'type': 'finite_number', 'loc': ('body', 'max_agents'),
     'msg': 'Input should be a finite number', 'input': inf}

FastAPI's default ``RequestValidationError`` handler serialises that structure directly,
the echoed ``inf`` blows up the encoder, and the exception escapes into the global
handler. The caller asked a malformed question and was told the *server* was broken:

    POST /ask {"question": "q", "max_agents": 1e309}
      -> HTTP 500 {"error_code": "INTERNAL_SERVER_ERROR", ...}      (before)
      -> HTTP 422 {"detail": [... 'input': 'Infinity' ...]}         (after)

Every validating route was affected by any non-finite float, so the fix belongs at the
boundary where error payloads are built, not in each route.
"""

from __future__ import annotations

import math
from typing import Any

# Non-finite floats are named rather than silently dropped: the caller needs to see what
# was wrong with the request, and `null` would hide that the value was sent at all.
_NON_FINITE_NAMES = {
    "inf": "Infinity",
    "-inf": "-Infinity",
    "nan": "NaN",
}


def _name_non_finite(value: float) -> str:
    if math.isnan(value):
        return _NON_FINITE_NAMES["nan"]
    return _NON_FINITE_NAMES["inf"] if value > 0 else _NON_FINITE_NAMES["-inf"]


def make_safe_response_class() -> type:
    """Build the concrete SafeJSONResponse (kept here to avoid importing starlette early)."""
    from starlette.responses import JSONResponse

    class _SafeJSONResponse(SafeJSONResponse, JSONResponse):
        pass

    return _SafeJSONResponse


def to_json_safe(value: Any, _depth: int = 0) -> Any:
    """Return ``value`` with every member serialisable by a strict JSON encoder.

    Non-finite floats become their names (``"Infinity"``, ``"-Infinity"``, ``"NaN"``),
    which keeps the diagnostic content intact instead of discarding it. Objects that
    cannot be represented at all fall back to ``repr`` so a response can never fail to
    render because of one unexpected member.
    """
    if _depth > 25:
        return repr(value)

    # bool must be tested before int: bool is a subclass of int.
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return _name_non_finite(value) if not math.isfinite(value) else value
    if isinstance(value, dict):
        return {
            (k if isinstance(k, str) else str(k)): to_json_safe(v, _depth + 1)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_json_safe(v, _depth + 1) for v in value]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, BaseException):
        return f"{type(value).__name__}: {value}"
    try:
        import json as _json

        _json.dumps(value, allow_nan=False)
        return value
    except Exception:
        return repr(value)


def validation_errors_payload(errors: Any) -> list[Any]:
    """Sanitise a Pydantic/FastAPI error list for use as an HTTP ``detail`` value.

    Accepts any sequence (FastAPI types ``exc.errors()`` as a ``Sequence``), and never
    raises: the whole point is that building an error response must not itself fail.
    """
    return [to_json_safe(error) for error in errors]


class SafeJSONResponse:
    """Mixin that sanitises response content before the strict encoder sees it.

    Starlette's ``JSONResponse`` renders with ``json.dumps(..., allow_nan=False)``, so a
    single non-finite float anywhere in a response body raises
    ``ValueError: Out of range float values are not JSON compliant`` and the request
    becomes an unhandled 500. That is reachable through validation errors (the handler
    echoes the offending input) and through ordinary arithmetic: Pydantic accepts ``NaN``
    and ``inf`` for a plain ``float`` field, so ``{"portfolio_equity": NaN}`` passed
    validation, produced non-finite scenario values, and then the *response* failed to
    encode — measured on ``POST /v1/trading/live/stress-test``.

    Routing every response through this class makes "the response is encodable" a
    property of the framework boundary rather than of each of the ~120 routes.
    """

    def render(self, content: Any) -> bytes:  # pragma: no cover - exercised via HTTP
        return super().render(to_json_safe(content))  # type: ignore[misc]
