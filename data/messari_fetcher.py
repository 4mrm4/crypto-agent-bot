"""Messari API client — institutional-grade asset fundamentals and on-chain metrics.

Three endpoints used:
- /v2/assets/{slug}/metrics  — price, market cap, on-chain data
- /v1/assets/{slug}/profile  — qualitative project info
- /v1/news/topics            — AI-analysed trending topics

Gracefully degrades: returns None on any error, never raises.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import httpx

from config import settings
from data.database import TradingDatabase
from data.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

MESSARI_BASE = "https://messari.io/api/v2"


@dataclass
class MessariMetrics:
    """Structured asset metrics from Messari."""
    asset: str
    price_usd: float
    market_cap: float
    volume_24h: float
    active_addresses_24h: Optional[int] = None
    transaction_volume_24h: Optional[float] = None
    developer_activity_30d: Optional[float] = None
    token_supply_circulating: Optional[float] = None
    token_supply_total: Optional[float] = None
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = "messari"


class MessariFetcher:
    """Async REST client for Messari API v1/v2.

    Caches:
    - get_metrics(): 15 min (SQLite api_cache)
    - get_profile(): 24 hours
    - get_trending_topics(): 30 min

    Rate limited: 20 req/min on free tier (enforced by RateLimiter).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        enabled: Optional[bool] = None,
        health_tracker=None,
    ):
        self._api_key = api_key or settings.MESSARI_API_KEY
        self._enabled = enabled if enabled is not None else settings.MESSARI_ENABLED
        self._health_tracker = health_tracker
        self._client: Optional[httpx.AsyncClient] = None
        self._rate_limiter = RateLimiter(rpm=settings.MESSARI_RATE_LIMIT_RPM)
        self._cache = TradingDatabase()

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=15,
                headers={"x-messari-api-key": self._api_key},
            )
        return self._client

    async def _get(self, path: str, cache_key: str = "", ttl: int = 900) -> Optional[dict]:
        """GET with rate limiting, caching, and health tracking."""
        if not self._enabled or not self._api_key:
            logger.debug("Messari disabled or no API key")
            return None

        # Check cache first
        if cache_key:
            cached = self._cache.get_cached(cache_key)
            if cached is not None:
                return cached

        # Rate limit
        if not self._rate_limiter.acquire(block=True, timeout=10):
            logger.warning("Messari rate limit hit, skipping %s", path)
            return None

        client = await self._get_client()
        url = f"{MESSARI_BASE}{path}"
        try:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
            if self._health_tracker:
                self._health_tracker.record_success("messari")
            # Cache result
            if cache_key:
                self._cache.set_cached(cache_key, data, "messari", ttl)
            return data
        except Exception as exc:
            logger.warning("Messari API error [%s]: %s", path, exc)
            if self._health_tracker:
                self._health_tracker.record_failure("messari", str(exc))
            return None

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    async def get_metrics(self, slug: str) -> Optional[MessariMetrics]:
        """Fetch current metrics for an asset slug (e.g. 'bitcoin', 'ethereum')."""
        cache_key = f"messari:metrics:{slug}"
        data = await self._get(
            f"/assets/{slug}/metrics",
            cache_key=cache_key,
            ttl=settings.MESSARI_CACHE_METRICS_TTL,
        )
        if not data or "data" not in data:
            return None
        d = data["data"]
        try:
            md = d.get("market_data", {})
            od = d.get("onchain_data", {})
            sup = d.get("supply", {})
            return MessariMetrics(
                asset=slug,
                price_usd=float(md.get("price_usd", 0)),
                market_cap=float(md.get("market_cap", 0)),
                volume_24h=float(md.get("volume_last_24_hours", 0)),
                active_addresses_24h=(
                    int(od["active_addresses_24h"]) if od.get("active_addresses_24h") else None
                ),
                transaction_volume_24h=(
                    float(od["transaction_volume_24h"]) if od.get("transaction_volume_24h") else None
                ),
                developer_activity_30d=(
                    float(od["developer_activity_30d"]) if od.get("developer_activity_30d") else None
                ),
                token_supply_circulating=(
                    float(sup["circulating"]) if sup.get("circulating") else None
                ),
                token_supply_total=(
                    float(sup["total"]) if sup.get("total") else None
                ),
            )
        except (ValueError, TypeError) as exc:
            logger.warning("Messari parse error for %s: %s", slug, exc)
            return None

    async def get_profile(self, slug: str) -> Optional[dict]:
        """Fetch qualitative project profile."""
        cache_key = f"messari:profile:{slug}"
        data = await self._get(
            f"/assets/{slug}/profile",
            cache_key=cache_key,
            ttl=settings.MESSARI_CACHE_PROFILE_TTL,
        )
        if data and "data" in data:
            return data["data"]
        return data

    async def get_trending_topics(self, limit: int = 10) -> List[dict]:
        """Fetch AI-analysed trending topics."""
        cache_key = f"messari:trending:{limit}"
        data = await self._get(
            f"/news/topics?limit={limit}",
            cache_key=cache_key,
            ttl=1800,
        )
        if data and isinstance(data, list):
            return data[:limit]
        return []

    async def get_batch_metrics(
        self, slugs: List[str],
    ) -> Dict[str, MessariMetrics]:
        """Fetch metrics for multiple assets concurrently."""
        import asyncio
        tasks = [self.get_metrics(slug) for slug in slugs]
        results = await asyncio.gather(*tasks)
        return {
            slug: result
            for slug, result in zip(slugs, results)
            if result is not None
        }
