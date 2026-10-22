"""Regression tests for the logging credential boundary."""

from __future__ import annotations

import io
import logging
import sys

import pytest

from app.utils.logger import SensitiveDataFilter, setup_logger


def _record(message: str, args: tuple[object, ...] = (), exc_info=None) -> logging.LogRecord:
    return logging.LogRecord(
        name="inference.test.redaction",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=args,
        exc_info=exc_info,
    )


@pytest.mark.parametrize(
    ("message", "secrets"),
    [
        ("provider failed password=topsecret-marker-123", ["topsecret-marker-123"]),
        ("request used api_key='key with spaces 123'", ["key with spaces 123"]),
        ('{"api_key": "json-secret-marker-123", "mode": "safe"}', ["json-secret-marker-123"]),
        ("Authorization: Bearer bearer-secret-marker-123", ["bearer-secret-marker-123"]),
        ("provider rejected sk-abcdefghijklmnop12345", ["sk-abcdefghijklmnop12345"]),
        ("provider rejected sk-or-v1-abcdefghijklmnopqrstuvwx", ["sk-or-v1-abcdefghijklmnopqrstuvwx"]),
        ("provider returned AIza" + "S" * 35, ["AIza" + "S" * 35]),
        ("provider returned gsk_abcdefghijklmnopqrstuvwxyz01", ["gsk_abcdefghijklmnopqrstuvwxyz01"]),
        ("provider returned nvapi-AbCdEfGhIjKlMnOpQrStUvWx", ["nvapi-AbCdEfGhIjKlMnOpQrStUvWx"]),
        ("provider returned xai-ABCDEFGHIJKLMNOPQRSTUV", ["xai-ABCDEFGHIJKLMNOPQRSTUV"]),
        ("provider returned pplx-abcdefghijklmnopqrstuv", ["pplx-abcdefghijklmnopqrstuv"]),
        ("provider returned r8_abcdefghijklmnopqrstuv", ["r8_abcdefghijklmnopqrstuv"]),
        ("provider returned mistral-abcdefghijklmnopqrstuv", ["mistral-abcdefghijklmnopqrstuv"]),
        ("provider returned AKIAIOSFODNN7EXAMPLE", ["AKIAIOSFODNN7EXAMPLE"]),
        ("POSTGRES_PASSWORD=database-secret-marker", ["database-secret-marker"]),
        ("AWS_SECRET_ACCESS_KEY=aws-secret-marker", ["aws-secret-marker"]),
        ("smtp_password='smtp with spaces secret'", ["smtp with spaces secret"]),
        (
            "private key: -----BEGIN PRIVATE KEY-----FAKE_PRIVATE_KEY_MATERIAL"
            "-----END PRIVATE KEY-----",
            ["FAKE_PRIVATE_KEY_MATERIAL"],
        ),
        (
            "JWT eyJabcdefghijk.abcdefghijklmnop.abcdefghijklmnop",
            ["eyJabcdefghijk.abcdefghijklmnop.abcdefghijklmnop"],
        ),
    ],
)
def test_filter_redacts_credential_shapes_from_plain_messages(message, secrets):
    record = _record(message)

    assert SensitiveDataFilter().filter(record) is True
    rendered = logging.Formatter("%(message)s").format(record)

    for secret in secrets:
        assert secret not in rendered
    assert "[REDACTED]" in rendered


def test_filter_redacts_secret_values_added_through_logging_arguments():
    secret = "argument-secret-marker-456"
    record = _record("provider authentication failed password=%s", (secret,))

    SensitiveDataFilter().filter(record)
    rendered = logging.Formatter("%(message)s").format(record)

    assert secret not in rendered
    assert "password=[REDACTED]" in rendered
    assert record.args == ()


def test_filter_redacts_credentials_from_exception_tracebacks():
    secret = "exception-secret-marker-789"
    try:
        raise RuntimeError(f"provider failed with api_key={secret}")
    except RuntimeError:
        record = _record("provider request failed", exc_info=sys.exc_info())

    SensitiveDataFilter().filter(record)
    rendered = logging.Formatter("%(levelname)s %(message)s").format(record)

    assert secret not in rendered
    assert "RuntimeError" in rendered
    assert "[REDACTED]" in rendered
    assert record.exc_info is None


@pytest.mark.parametrize(
    "message",
    [
        "task-1234567890abcdef is not a credential",
        "risk-assessment-of-the-write-path",
        "sk-learn model comparison",
        "The write path needs sharding by tenant.",
        "provider completed request in 0.25 seconds",
    ],
)
def test_filter_preserves_benign_diagnostic_text(message):
    record = _record(message)

    SensitiveDataFilter().filter(record)

    assert record.getMessage() == message


def test_filter_sanitizes_malformed_log_arguments_without_raising():
    record = _record("a mismatched format %s %s", ("one",))

    assert SensitiveDataFilter().filter(record) is True
    assert record.args == ()
    assert "[REDACTED: log formatting failed]" in record.getMessage()


@pytest.mark.parametrize(
    "message",
    [
        "Authorization: Bearer repeated-secret-123",
        "private key: -----BEGIN PRIVATE KEY-----FAKE_MATERIAL-----END PRIVATE KEY-----",
    ],
)
def test_filter_is_idempotent_for_already_redacted_fields(message):
    record = _record(message)
    filter_ = SensitiveDataFilter()

    filter_.filter(record)
    once = record.getMessage()
    filter_.filter(record)
    twice = record.getMessage()

    assert twice == once
    assert "[REDACTED]]" not in twice


def test_setup_logger_attaches_redaction_to_preexisting_handlers():
    name = "inference.test.existing-handler-redaction"
    test_logger = logging.getLogger(name)
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    test_logger.addHandler(handler)
    try:
        configured = setup_logger(name=name)

        assert any(isinstance(item, SensitiveDataFilter) for item in configured.filters)
        assert any(isinstance(item, SensitiveDataFilter) for item in handler.filters)

        configured.error("provider failed password=%s", "configured-handler-secret")
        assert "configured-handler-secret" not in stream.getvalue()
        assert "[REDACTED]" in stream.getvalue()
    finally:
        test_logger.removeHandler(handler)
