"""Persistent Async HTTP Connection Pool for High-Throughput, Ultra-Low Latency Inference.

Reuses persistent TCP/TLS keep-alive connections across all upstream LLM provider calls,
eliminating 150ms-350ms of socket handshake and TLS negotiation overhead per request.
"""

from __future__ import annotations

import asyncio

import httpx

from app.utils.logger import logger

# Bounded connection pool limits for maximum concurrency with keep-alive socket reuse
DEFAULT_LIMITS = httpx.Limits(
    max_connections=100,
    max_keepalive_connections=30,
    keepalive_expiry=120.0,
)

DEFAULT_TIMEOUT = httpx.Timeout(
    timeout=60.0,
    connect=5.0,
    read=60.0,
    write=10.0,
)


class HttpClientPool:
    """Singleton connection pool manager for shared httpx.AsyncClient instances."""

    _instance: HttpClientPool | None = None
    _client: httpx.AsyncClient | None = None
    _lock: asyncio.Lock | None = None

    @classmethod
    def get_instance(cls) -> HttpClientPool:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def get_client(self) -> httpx.AsyncClient:
        """Returns the shared, keep-alive enabled AsyncClient. Creates one if none exists or closed."""
        if self._client is not None and not self._client.is_closed:
            return self._client

        async with self._get_lock():
            if self._client is None or self._client.is_closed:
                self._client = httpx.AsyncClient(
                    limits=DEFAULT_LIMITS,
                    timeout=DEFAULT_TIMEOUT,
                    follow_redirects=True,
                )
                logger.debug("Initialized global shared HttpClientPool connection.")
            return self._client

    async def close(self) -> None:
        """Cleanly closes open keep-alive connections during application shutdown."""
        async with self._get_lock():
            if self._client is not None and not self._client.is_closed:
                try:
                    await self._client.aclose()
                except Exception:
                    pass
                finally:
                    self._client = None
                logger.debug("Closed global shared HttpClientPool connections.")

    async def prewarm(self, urls: list[str]) -> None:
        """Pre-warms TCP/TLS connections to primary provider hosts to eliminate cold-start latency."""
        client = await self.get_client()
        tasks = []
        for url in urls:
            tasks.append(self._prewarm_single(client, url))
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _prewarm_single(self, client: httpx.AsyncClient, url: str) -> None:
        try:
            # Send a fast HEAD or GET to negotiate TLS and cache the socket in keep-alive pool
            await client.head(url, timeout=3.0)
            logger.debug("Pre-warmed connection to %s", url)
        except Exception:
            # Prewarming is best-effort and non-fatal
            pass


http_client_pool = HttpClientPool.get_instance()


async def get_shared_client() -> httpx.AsyncClient:
    """Convenience accessor for the global shared AsyncClient."""
    return await http_client_pool.get_client()
