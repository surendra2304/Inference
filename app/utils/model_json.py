"""Parse the JSON a model was asked to return, tolerating a code fence and surrounding prose.

Shared by the services that ask a model for structured output (code review, debugging). A
reply that does not parse is a failed answer, not an empty success: callers must report it as
such, and must not fill the gap with a template.
"""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"^```[a-zA-Z]*\s*(.*?)\s*```$", re.DOTALL)


def extract_json_object(text: str) -> Any:
    """Return the first JSON object in ``text``. Raises ``ValueError`` if there is none."""
    stripped = text.strip()
    fence = _FENCE.match(stripped)
    if fence:
        stripped = fence.group(1)
    try:
        return json.loads(stripped)
    except ValueError:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(stripped[start:end + 1])
