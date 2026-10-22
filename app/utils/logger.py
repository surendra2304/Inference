"""Structured logging utility for Inference.

Logging is a security boundary: upstream exceptions and request metadata can echo
credentials. The filter here intentionally uses only the standard library so it can
run before application modules (including the richer prompt-isolation scrubber) load.
"""

from __future__ import annotations

import logging
import re
import sys
import traceback

_REDACTED = "[REDACTED]"

# Recognise common credential field names in assignment-style logs, including JSON
# keys, environment variable names, HTTP headers, and compact key=value diagnostics.
_SECRET_FIELD_NAMES = (
    r"(?:[A-Za-z0-9]+[-_])*(?:"
    r"authorization|proxy[-_]?authorization|x[-_]?api[-_]?key|"
    r"api[-_ ]?(?:key|secret)|access[-_ ]?token|refresh[-_ ]?token|"
    r"auth[-_ ]?token|client[-_ ]?secret|private[-_ ]?key|"
    r"password|passwd|passphrase|secret|session[-_ ]?(?:id|token)|"
    r"cookie|set[-_ ]?cookie|api[-_ ]?token|bearer|key|token"
    r")"
)
_SECRET_PREFIX = rf"""(?P<prefix>["']?\b(?:{_SECRET_FIELD_NAMES})\b["']?\s*[:=]\s*(?!\[REDACTED\]))"""
_QUOTED_SECRET_FIELD = re.compile(
    _SECRET_PREFIX + r"""(?P<quote>["'])(?P<secret>.*?)(?P=quote)""",
    re.IGNORECASE | re.DOTALL,
)
_UNQUOTED_SECRET_FIELD = re.compile(
    _SECRET_PREFIX + r"""(?P<secret>[^\s,"';&}\]]+)""",
    re.IGNORECASE,
)

# A credential may be echoed as a bare value, with no "api_key=" label. Match
# well-known provider token formats without masking ordinary words or identifiers.
_PROVIDER_TOKEN = re.compile(
    r"\b(?:"
    r"sk-(?:proj-|or-v1-|live-|test-)?[A-Za-z0-9_-]{12,}|"
    r"AIza[0-9A-Za-z_-]{20,}|"
    r"gsk_[A-Za-z0-9_-]{16,}|"
    r"nvapi-[A-Za-z0-9_-]{16,}|"
    r"xai-[A-Za-z0-9_-]{16,}|"
    r"pplx-[A-Za-z0-9_-]{16,}|"
    r"r8_[A-Za-z0-9_-]{16,}|"
    r"mistral-[A-Za-z0-9_-]{16,}|"
    r"cohere-[A-Za-z0-9_-]{16,}|"
    r"xox[baprs]-[A-Za-z0-9-]{16,}|"
    r"gh[pousr]_[A-Za-z0-9]{20,}|"
    r"hf_[A-Za-z0-9]{20,}|"
    r"AKIA[A-Z0-9]{16}"
    r")\b"
)
_AUTH_SCHEME = re.compile(
    r"(?i)(\b(?:proxy-)?authorization\s*[:=]\s*)(?:bearer|basic|token)\s+"
    r"([^\s,\"']+)"
)
_BEARER_TOKEN = re.compile(r"(?i)\bBearer\s+([^\s,\"']+)")
_JWT = re.compile(
    r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"
)
_PRIVATE_KEY_BLOCK = re.compile(
    r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"
    r"[\s\S]*?"
    r"-----END (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
    re.IGNORECASE,
)


def _redact_text(value: str) -> str:
    """Redact credentials while preserving useful surrounding diagnostic text."""
    if not value:
        return value

    value = _PRIVATE_KEY_BLOCK.sub(_REDACTED, value)
    value = _AUTH_SCHEME.sub(lambda match: f"{match.group(1)}{_REDACTED}", value)
    value = _BEARER_TOKEN.sub(lambda match: f"Bearer {_REDACTED}", value)
    value = _QUOTED_SECRET_FIELD.sub(
        lambda match: f"{match.group('prefix')}{match.group('quote')}{_REDACTED}{match.group('quote')}",
        value,
    )
    value = _UNQUOTED_SECRET_FIELD.sub(
        lambda match: f"{match.group('prefix')}{_REDACTED}",
        value,
    )
    value = _JWT.sub(_REDACTED, value)
    return _PROVIDER_TOKEN.sub(_REDACTED, value)


class SensitiveDataFilter(logging.Filter):
    """Mask credentials in messages, formatting arguments, and exception tracebacks.

    The record is rewritten rather than only filtering the final rendered string so
    another handler cannot later format the original arguments or exception object.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            # Resolve %-formatting parameters before scrubbing, otherwise a secret carried
            # in record.args could be added back by the formatter after this filter.
            try:
                rendered_message = record.getMessage()
            except Exception:  # noqa: BLE001 - malformed format strings must fail closed
                rendered_message = "[REDACTED: log formatting failed]"
            record.msg = _redact_text(rendered_message)
            record.args = ()

            if record.stack_info:
                record.stack_info = _redact_text(record.stack_info)

            if record.exc_info:
                try:
                    rendered_exception = "".join(traceback.format_exception(*record.exc_info))
                except Exception:  # noqa: BLE001 - preserve a safe failure description
                    exc_type, exc_value, _ = record.exc_info
                    exception_name = exc_type.__name__ if exc_type is not None else "Exception"
                    rendered_exception = f"{exception_name}: {exc_value}"
                record.exc_text = _redact_text(rendered_exception)
                # Do not let later handlers or formatters re-render the raw exception.
                record.exc_info = None
            elif record.exc_text:
                record.exc_text = _redact_text(record.exc_text)
        except Exception:  # noqa: BLE001 - logging must never break request handling
            # Fail closed: if an unexpected object defeats formatting, don't expose it.
            record.msg = "[REDACTED: log formatting failed]"
            record.args = ()
            record.exc_info = None
            record.exc_text = None
            record.stack_info = None
        return True


def setup_logger(
    name: str = "inference",
    log_level: str | None = None,
) -> logging.Logger:
    """Configures and returns a structured application logger."""
    level_name = (log_level or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    sensitive_filter = next(
        (item for item in logger.filters if isinstance(item, SensitiveDataFilter)),
        None,
    )
    if sensitive_filter is None:
        sensitive_filter = SensitiveDataFilter()
        logger.addFilter(sensitive_filter)

    # Avoid duplicate handlers if setup_logger is called repeatedly.
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)

        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [inference]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # The logger-level filter covers its own records; handler-level filters also
    # cover records propagated from child loggers, which skip ancestor logger filters.
    for registered_handler in logger.handlers:
        if not any(isinstance(item, SensitiveDataFilter) for item in registered_handler.filters):
            registered_handler.addFilter(sensitive_filter)

    return logger


logger = setup_logger()
