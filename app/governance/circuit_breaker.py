"""Graceful Degradation Matrix and Circuit Breaker Governance."""

import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.utils.logger import logger


class CircuitBreakerState(BaseModel):
    provider: str
    state: Literal["CLOSED", "OPEN", "HALF_OPEN"] = "CLOSED"
    consecutive_failures: int = 0
    failure_threshold: int = 5
    cooldown_seconds: float = 60.0
    last_state_change: float = Field(default_factory=time.time)


class ProviderCircuitBreakerManager:
    """Manages 5-failure circuit breaker trips, 60s cooldowns, and graceful degradation."""

    def __init__(self) -> None:
        self.breakers: dict[str, CircuitBreakerState] = {
            p: CircuitBreakerState(provider=p)
            for p in ["gemini", "groq", "mistral", "openrouter", "nvidia", "cohere", "huggingface"]
        }

    def record_failure(self, provider: str) -> None:
        brk = self.breakers.get(provider)
        if not brk:
            return
        brk.consecutive_failures += 1
        if brk.consecutive_failures >= brk.failure_threshold:
            brk.state = "OPEN"
            brk.last_state_change = time.time()
            logger.error("[CIRCUIT OPEN] Provider '%s' exceeded 5 consecutive failures. Tripping breaker for 60s.", provider)

    def record_success(self, provider: str) -> None:
        brk = self.breakers.get(provider)
        if not brk:
            return
        brk.consecutive_failures = 0
        brk.state = "CLOSED"

    def is_available(self, provider: str) -> bool:
        brk = self.breakers.get(provider)
        if not brk:
            return True
        if brk.state == "OPEN":
            if time.time() - brk.last_state_change > brk.cooldown_seconds:
                brk.state = "HALF_OPEN"
                logger.info("[CIRCUIT HALF-OPEN] Provider '%s' cooldown elapsed; entering half-open probe state.", provider)
                return True
            return False
        return True

    def get_circuit_statuses(self) -> dict[str, Any]:
        return {
            p: {
                "state": b.state,
                "consecutive_failures": b.consecutive_failures,
                "is_available": self.is_available(p)
            }
            for p, b in self.breakers.items()
        }


circuit_breaker_manager = ProviderCircuitBreakerManager()
