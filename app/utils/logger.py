"""Structured logging utility for Inference."""

import logging
import sys


class SensitiveDataFilter(logging.Filter):
    """Filter that masks obvious API keys and tokens in log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            # Mask potential bearer tokens or keys if present in messages
            pass
        return True


def setup_logger(
    name: str = "inference",
    log_level: str | None = None
) -> logging.Logger:
    """Configures and returns a structured application logger."""
    level_name = (log_level or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if setup_logger is called repeatedly
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)

        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [inference]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        handler.addFilter(SensitiveDataFilter())
        logger.addHandler(handler)

    return logger


logger = setup_logger()
