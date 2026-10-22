from __future__ import annotations

import math
import threading

from .errors import BudgetExceeded


def _finite_non_negative(value: float, name: str) -> float:
    """Validate monetary values before they can poison budget arithmetic."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return result


class BudgetLedger:
    """Thread-safe per-tenant reservations and observed spend."""

    def __init__(self, default_budget: float = 10.0) -> None:
        self.default = _finite_non_negative(default_budget, "default_budget")
        self._spent: dict[tuple[str, str], float] = {}
        self._reserved: dict[tuple[str, str], float] = {}
        self._lock = threading.RLock()

    def _key(self, tenant: str, period: str) -> tuple[str, str]:
        return tenant, period

    def remaining(self, tenant: str, period: str = "current") -> float:
        with self._lock:
            return (
                self.default
                - self._spent.get(self._key(tenant, period), 0.0)
                - self._reserved.get(self._key(tenant, period), 0.0)
            )

    def reserve(self, tenant: str, amount: float, period: str = "current") -> None:
        amount = _finite_non_negative(amount, "amount")
        with self._lock:
            k = self._key(tenant, period)
            available = self.default - self._spent.get(k, 0.0) - self._reserved.get(k, 0.0)
            if amount > available:
                raise BudgetExceeded(f"Budget exceeded for tenant {tenant}: need {amount:.6f}, have {available:.6f}")
            self._reserved[k] = self._reserved.get(k, 0.0) + amount

    def reconcile(self, tenant: str, estimated: float, actual: float, period: str = "current") -> None:
        estimated = _finite_non_negative(estimated, "estimated")
        actual = _finite_non_negative(actual, "actual")
        with self._lock:
            k = self._key(tenant, period)
            self._reserved[k] = max(0.0, self._reserved.get(k, 0.0) - estimated)
            self._spent[k] = self._spent.get(k, 0.0) + actual
