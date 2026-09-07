"""Interactive Web Dashboard service for Inference 2.0."""

import os
from functools import lru_cache

_HTML_PATH = os.path.join(os.path.dirname(__file__), "dashboard.html")


@lru_cache(maxsize=1)
def get_dashboard_html() -> str:
    """Read and return the single-page dashboard HTML."""
    if os.path.exists(_HTML_PATH):
        with open(_HTML_PATH, encoding="utf-8") as f:
            return f.read()
    return "<h1>Inference 2.0 Dashboard</h1><p>Dashboard template not found.</p>"
