from __future__ import annotations

import json
from typing import Any

from .errors import SchemaViolation


def validate_json(text: str, schema: dict[str, Any]) -> Any:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SchemaViolation(f"Provider returned invalid JSON: {exc}") from exc
    if schema.get("type") == "object" and not isinstance(obj, dict):
        raise SchemaViolation("Expected JSON object")
    required = schema.get("required", [])
    if isinstance(obj, dict):
        missing = [x for x in required if x not in obj]
        if missing:
            raise SchemaViolation(f"Missing required fields: {missing}")
        props = schema.get("properties", {})
        for name, desc in props.items():
            if name not in obj:
                continue
            typ = str(desc.get("type"))
            type_map: dict[str, type | tuple[type, ...]] = {
                "string": str,
                "integer": int,
                "number": (int, float),
                "boolean": bool,
                "array": list,
                "object": dict,
            }
            good = type_map.get(typ)
            if good is not None and not isinstance(obj[name], good):
                raise SchemaViolation(f"Field {name} has type {type(obj[name]).__name__}, expected {typ}")
    return obj
