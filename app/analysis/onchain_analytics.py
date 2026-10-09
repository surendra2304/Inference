"""On-chain analytics: honest about the absence of a data source.

Before this change, ``get_onchain_metrics`` returned fixed network statistics (945,120 active
addresses, a 162.5 million dollar whale transfer with a made-up transaction id, net outflow of
165 million dollars) labelled as "real-world baselines". The numbers did not come from any chain
and did not change with the symbol. They then fed the trajectory forecast as if measured.

No on-chain provider is wired into this service. Until one is, every field is ``None`` and
``status`` says so; downstream code treats a missing field as a missing input.
"""

from __future__ import annotations

import time
from typing import Any


class OnChainAnalyticsEngine:
    """Returns on-chain metrics only when a provider has measured them. None are wired yet."""

    def get_onchain_metrics(self, symbol: str = "BTC") -> dict[str, Any]:
        return {
            "symbol": symbol.upper(),
            "timestamp": time.time(),
            "status": "not_measured",
            "source": None,
            "reason": "no on-chain data provider is configured for this service",
            "network_health": None,
            "whale_movements": [],
            "exchange_flows": None,
            "holder_distribution": None,
        }


onchain_engine = OnChainAnalyticsEngine()
