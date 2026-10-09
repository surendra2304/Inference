"""Market data with honest provenance.

Every series this module returns comes with a source label:

* ``binance``: fetched from the public Binance endpoints.
* ``synthetic_baseline``: generated locally because the fetch failed. It is a deterministic
  zig-zag, not market data. Callers must not forecast from it. Before this change the
  synthetic series was returned and cached as if it were market data.

Symbols and intervals are validated before they reach a URL. Before this change the symbol
was interpolated raw into the query string, so a crafted symbol could add query parameters.

News is not fetched by this module. Before this change three invented headlines attributed to
CryptoPanic and Reddit were returned on every call and fed the sentiment score. Now the feed is
empty until a real news source is implemented, and callers see an empty list.
"""

from __future__ import annotations

import re
import time
from typing import Any

import httpx

from app.utils.logger import logger

BINANCE_SOURCE = "binance"
SYNTHETIC_SOURCE = "synthetic_baseline"

_SYMBOL_RE = re.compile(r"^[A-Z0-9]{3,20}$")
ALLOWED_INTERVALS = frozenset({"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"})
MAX_LIMIT = 1000


class InvalidMarketParameter(ValueError):
    """A symbol, interval or limit that must not reach an upstream URL."""


def normalize_symbol(symbol: str) -> str:
    """Upper-cases and validates a trading symbol. Anything else is rejected, not escaped."""
    candidate = (symbol or "").strip().upper()
    if not _SYMBOL_RE.fullmatch(candidate):
        raise InvalidMarketParameter(
            "symbol must be 3-20 letters or digits, for example BTCUSDT"
        )
    return candidate


def _check_interval(interval: str) -> str:
    if interval not in ALLOWED_INTERVALS:
        raise InvalidMarketParameter(f"interval must be one of {sorted(ALLOWED_INTERVALS)}")
    return interval


def _check_limit(limit: int) -> int:
    if not 1 <= int(limit) <= MAX_LIMIT:
        raise InvalidMarketParameter(f"limit must be between 1 and {MAX_LIMIT}")
    return int(limit)


class MarketDataFetcher:
    """Fetches and normalizes market data with a provenance label on every series."""

    def __init__(self, cache_ttl_sec: float = 10.0) -> None:
        self.cache_ttl = cache_ttl_sec
        self._cache: dict[str, tuple[float, Any]] = {}

    async def get_ohlcv(self, symbol: str = "BTCUSDT", interval: str = "1h", limit: int = 100) -> list[dict[str, Any]]:
        """Candles only. Use ``get_ohlcv_sourced`` when the caller must know the provenance."""
        candles, _source = await self.get_ohlcv_sourced(symbol=symbol, interval=interval, limit=limit)
        return candles

    async def get_ohlcv_sourced(
        self, symbol: str = "BTCUSDT", interval: str = "1h", limit: int = 100
    ) -> tuple[list[dict[str, Any]], str]:
        """Returns ``(candles, source)``; source is ``binance`` or ``synthetic_baseline``."""
        sym = normalize_symbol(symbol)
        interval = _check_interval(interval)
        limit = _check_limit(limit)

        cache_key = f"ohlcv:{sym}:{interval}:{limit}"
        now = time.time()
        if cache_key in self._cache:
            ts, val = self._cache[cache_key]
            if now - ts < self.cache_ttl:
                return val

        url = f"https://api.binance.com/api/v3/klines?symbol={sym}&interval={interval}&limit={limit}"
        try:
            async with httpx.AsyncClient(timeout=4.0) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    candles = [
                        {
                            "timestamp": int(k[0]),
                            "open": float(k[1]),
                            "high": float(k[2]),
                            "low": float(k[3]),
                            "close": float(k[4]),
                            "volume": float(k[5]),
                            "quote_volume": float(k[7]),
                            "trades_count": int(k[8]),
                        }
                        for k in resp.json()
                    ]
                    result = (candles, BINANCE_SOURCE)
                    self._cache[cache_key] = (now, result)
                    return result
                logger.warning("Public market data returned HTTP %s for %s", resp.status_code, sym)
        except Exception as e:  # noqa: BLE001 - the fallback is labelled, never hidden
            logger.warning("Public market data fetch failed for %s (%s); series is synthetic", sym, type(e).__name__)

        result = (self._synthetic_candles(sym, limit, now), SYNTHETIC_SOURCE)
        self._cache[cache_key] = (now, result)
        return result

    @staticmethod
    def _synthetic_candles(sym: str, limit: int, now: float) -> list[dict[str, Any]]:
        """A deterministic zig-zag used only so that the offline path runs. Not market data."""
        base_price = 65000.0 if "BTC" in sym else 3400.0
        candles: list[dict[str, Any]] = []
        cur_price = base_price
        for i in range(limit):
            t_stamp = int((now - (limit - i) * 3600) * 1000)
            delta = ((i % 7 - 3) * 50.0) + (i % 3 - 1) * 20.0
            o = cur_price
            c = round(cur_price + delta, 2)
            h = round(max(o, c) + abs(delta) * 0.4 + 20.0, 2)
            low_val = round(min(o, c) - abs(delta) * 0.4 - 20.0, 2)
            v = round(120.0 + (i % 5) * 30.0, 2)
            candles.append({
                "timestamp": t_stamp,
                "open": o,
                "high": h,
                "low": low_val,
                "close": c,
                "volume": v,
                "quote_volume": round(v * c, 2),
                "trades_count": int(v * 15),
            })
            cur_price = c
        return candles

    async def get_orderbook(self, symbol: str = "BTCUSDT", limit: int = 20) -> dict[str, Any]:
        """Order book snapshot with a ``source`` label (``binance`` or ``synthetic_baseline``)."""
        sym = normalize_symbol(symbol)
        limit = _check_limit(limit)
        cache_key = f"depth:{sym}:{limit}"
        now = time.time()
        if cache_key in self._cache:
            ts, val = self._cache[cache_key]
            if now - ts < self.cache_ttl:
                return val

        url = f"https://api.binance.com/api/v3/depth?symbol={sym}&limit={limit}"
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    data = resp.json()
                    depth = {
                        "symbol": sym,
                        "bids": [[float(p), float(q)] for p, q in data.get("bids", [])],
                        "asks": [[float(p), float(q)] for p, q in data.get("asks", [])],
                        "timestamp": now,
                        "source": BINANCE_SOURCE,
                    }
                    self._cache[cache_key] = (now, depth)
                    return depth
        except Exception:  # noqa: BLE001
            logger.warning("Public order book fetch failed for %s; book is synthetic", sym)

        base = 65000.0 if "BTC" in sym else 3400.0
        depth = {
            "symbol": sym,
            "bids": [[base - i * 10, round(1.5 + i * 0.2, 2)] for i in range(1, limit + 1)],
            "asks": [[base + i * 10, round(1.4 + i * 0.2, 2)] for i in range(1, limit + 1)],
            "timestamp": now,
            "source": SYNTHETIC_SOURCE,
        }
        self._cache[cache_key] = (now, depth)
        return depth

    async def get_news_and_social_feed(self, symbol: str = "BTC") -> list[dict[str, Any]]:
        """News is not fetched yet. Returns no items rather than invented headlines."""
        return []


market_data_fetcher = MarketDataFetcher()
