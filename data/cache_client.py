"""Cache client protocol for API fetcher backends.

Defines a CacheClient interface that TradingDatabase and mock caches
implement, so API fetchers (e.g. SantimentFetcher) depend on an abstract
cache rather than on TradingDatabase directly.
"""

from typing import Optional, Protocol


class CacheClient(Protocol):
    """Interface for cache backends used by API fetchers.

    Implementations must be thread-safe (or at least safe for the caller's
    concurrency model). TradingDatabase (SQLite) is the production backend.
    """

    def get_cached(self, cache_key: str) -> Optional[dict]:
        """Return cached data dict, or None if missing or expired."""
        ...

    def set_cached(
        self, cache_key: str, data: dict, source: str, ttl_seconds: int
    ) -> None:
        """Insert or replace a cached entry with TTL."""
        ...
