from __future__ import annotations

from .budget import BudgetLedger
from .rate_limit import AsyncTokenBucket, ConcurrencyGate


class AdmissionController:
    def __init__(self, rpm: int, max_concurrency: int, budget: BudgetLedger) -> None:
        self.rpm = rpm
        self.bucket = AsyncTokenBucket(rpm / 60.0, max(1.0, rpm / 10))
        self.gate = ConcurrencyGate(max_concurrency)
        self.budget = budget

    async def admit(self, tenant: str, reserved_cost: float):
        await self.bucket.acquire()
        self.budget.reserve(tenant, reserved_cost)
        return self.gate
