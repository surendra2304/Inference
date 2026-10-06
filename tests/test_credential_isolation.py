"""M12: credential detection must cover every pattern scrubbing handles.

`detect_credentials` checked 7 of the 9 compiled patterns - it never consulted
_HEX_PRIVATE_KEY_PATTERN or _GENERIC_SECRET_FIELD_PATTERN (lines 47-50) - so
the audit reported a clean result for inputs that scrubbing was actively
redacting. The module's own invariant #1 ("confidential credentials MUST
NEVER be sent to model providers") therefore rested on an unverified claim.

These tests assert symmetry (detect flags exactly what scrub redacts), that
verification is fail-closed, and that both scrub sites in /v1 enforce it.
"""

import pytest

from app.api import v1_core_routes
from app.security.prompt_isolation import (
    CredentialLeakError,
    detect_credentials,
    scrub_credentials,
    scrub_credentials_dict,
    scrub_credentials_verified,
)

HEX_64 = "ab" * 32

# name -> (raw text, expected detection labels)
CASES: list[tuple[str, str, list[str]]] = [
    ("pem", "-----BEGIN RSA PRIVATE KEY-----\nMIIE\n-----END RSA PRIVATE KEY-----", ["pem_private_key"]),
    ("hex_keyed", f"private_key: {HEX_64}", ["hex_private_key"]),
    ("hex_bare", HEX_64, ["hex_private_key"]),
    ("jwt", "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N", ["jwt_token"]),
    ("bearer", f"Bearer {'x' * 30}", ["bearer_token"]),
    # Both also satisfy the generic-secret pattern (they contain api_key=/password=),
    # and scrubbing applies both, so detection reports both.
    ("exchange", f"binance_api_key={'K' * 24}", ["exchange_key", "generic_secret_field"]),
    ("smtp", f"smtp_password={'S' * 20}", ["smtp_password", "generic_secret_field"]),
    ("adb", f"adb_key={'A' * 25}", ["adb_key"]),
    ("session", f"session_id={'S' * 20}", ["session_cookie"]),
    ("generic_api_key", f"api_key={'G' * 24}", ["generic_secret_field"]),
    ("generic_password", f"password={'P' * 24}", ["generic_secret_field"]),
    ("generic_token", f"auth_token={'T' * 24}", ["generic_secret_field"]),
    ("clean", "just a plain sentence about the weather today", []),
]


@pytest.mark.parametrize("name,text,expected", CASES, ids=[c[0] for c in CASES])
def test_detection_flags_everything_scrubbing_redacts(name, text, expected):
    detected = detect_credentials(text)
    assert sorted(set(detected)) == sorted(set(expected)), (
        f"{name}: detect_credentials returned {detected}, expected {expected}"
    )


@pytest.mark.parametrize("name,text,expected", CASES, ids=[c[0] for c in CASES])
def test_detection_and_scrubbing_are_symmetric(name, text, expected):
    """Detect must flag exactly the inputs scrubbing redacts - no more, no less."""
    redacted = "[REDACTED" in scrub_credentials(text)
    detected = detect_credentials(text)
    if expected:
        assert redacted, f"{name}: detected as {expected} but scrubbing left it in place"
        assert detected, f"{name}: scrubbed but not reported by detect_credentials"
    else:
        assert not redacted, f"{name}: clean input was redacted"
        assert not detected, f"{name}: clean input was flagged: {detected}"


@pytest.mark.parametrize("name,text,expected", CASES, ids=[c[0] for c in CASES])
def test_verification_leaves_no_residual(name, text, expected):
    cleaned = scrub_credentials_verified(text, field=name)
    assert detect_credentials(cleaned) == [], (
        f"{name}: credential survived verification: {detect_credentials(cleaned)}"
    )


def test_verification_raises_when_scrub_cannot_remove_credential(monkeypatch):
    """The guard must fail closed rather than forward a detected secret."""
    def _leaky(text, field="text"):
        raise CredentialLeakError(field, ["generic_secret_field"])

    monkeypatch.setattr(v1_core_routes, "scrub_credentials_verified", _leaky)

    import fastapi

    with pytest.raises(fastapi.HTTPException) as exc_info:
        v1_core_routes._scrub_and_verify("api_key=" + "G" * 24, {}, "trace_x")
    assert exc_info.value.status_code == 500
    assert "credential" in exc_info.value.detail.lower()


def test_route_refuses_request_when_residual_survives(monkeypatch):
    """_scrub_and_verify answers 500 instead of letting the secret through."""
    import fastapi

    # Residual path: context scrubbing leaves a raw credential behind.
    monkeypatch.setattr(
        v1_core_routes, "scrub_credentials_dict",
        lambda ctx: {"api_key": "rawsecretvalue123456789012"},
    )
    with pytest.raises(fastapi.HTTPException) as exc_info:
        v1_core_routes._scrub_and_verify("hello", {"api_key": "x"}, "trace_y")
    assert exc_info.value.status_code == 500
    assert "credential" in exc_info.value.detail.lower()


def test_dict_context_detection_covers_forbidden_keys():
    detected = detect_credentials({"api_key": "supersecretvalue123456"})
    assert "forbidden_key:api_key" in detected

    scrubbed = scrub_credentials_dict({"api_key": "supersecretvalue123456"})
    assert detect_credentials(scrubbed) == [], "redacted dict must audit clean"


def test_verified_helper_is_noop_for_empty_and_non_strings():
    assert scrub_credentials_verified("", field="p") == ""
    assert scrub_credentials_verified(None, field="p") is None  # type: ignore[arg-type]
    assert scrub_credentials_verified(123, field="p") == 123  # type: ignore[arg-type]
