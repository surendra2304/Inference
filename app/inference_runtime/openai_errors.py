from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class APIError:
    status_code: int
    code: str
    message: str
    retryable: bool = False
    provider: str | None = None


def normalize_http_error(status: int, body: str, provider: str) -> APIError:
    retry = status in {408, 409, 425, 429, 500, 502, 503, 504}
    code = {
        401: "invalid_api_key",
        403: "forbidden",
        404: "not_found",
        408: "timeout",
        409: "conflict",
        429: "rate_limit",
        500: "provider_error",
        502: "bad_gateway",
        503: "unavailable",
        504: "gateway_timeout",
    }.get(status, "provider_error")
    return APIError(status, code, body[:1000], retry, provider)


def to_openai_error(error: APIError) -> dict:
    return {"error": {"message": error.message, "type": "inference_error", "code": error.code, "param": None}}
