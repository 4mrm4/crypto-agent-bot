"""Santiment API client — social volume, developer activity, and sentiment balance.

Uses Santiment's GraphQL API directly (via httpx) rather than sanpy to keep
dependencies minimal and control async behaviour.

Gracefully degrades: returns None on any error, never raises.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import httpx

from config import settings
from data.database import TradingDatabase

logger = logging.getLogger(__name__)

SANTIMENT_GRAPHQL = "https://api.santiment.net/graphql"


@dataclass
class SantimentSignal:
    """Social and development activity signal from Santiment."""
    asset_slug: str
    social_volume_24h: Optional[float] = None
    sentiment_balance_24h: Optional[float] = None
    dev_activity_30d: Optional[float] = None
    social_dominance_pct: Optional[float] = None
    daily_active_addresses: Optional[int] = None
    fetched_at: datetime = field(default_factory=datetime.utcnow)
    source: str = "santiment"


# ── GraphQL query templates ──

SINGLE_VALUE_QUERY = """
query($slug: String!, $metric: String!, $from: DateTime!, $to: DateTime!) {
  getMetric(metric: $metric) {
    timeseriesData(
      slug: $slug
      from: $from
      to: $to
      interval: "1d"
    ) {
      datetime
      value
    }
  }
}
"""

TRENDING_QUERY = """
{
  getTrendingAssets(size: 10) {
    slug
    score
  }
}
"""


class SantimentFetcher:
    """Async GraphQL client for Santiment API.

    Caches get_signal() in SQLite api_cache with 30 min TTL.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        enabled: Optional[bool] = None,
        health_tracker=None,
    ):
        self._api_key = api_key or settings.SANTIMENT_API_KEY
        self._enabled = enabled if enabled is not None else settings.SANTIMENT_ENABLED
        self._health_tracker = health_tracker
        self._client: Optional[httpx.AsyncClient] = None
        self._cache = TradingDatabase()

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=15,
                headers={"Authorization": f"Apikey {self._api_key}"},
            )
        return self._client

    async def _query(self, query: str, variables: dict = None) -> Optional[dict]:
        """Execute a GraphQL query with error handling."""
        if not self._enabled or not self._api_key:
            logger.debug("Santiment disabled or no API key")
            return None

        client = await self._get_client()
        payload = {"query": query}
        if variables:
            payload["variables"] = variables

        try:
            resp = await client.post(SANTIMENT_GRAPHQL, json=payload)
            resp.raise_for_status()
            data = resp.json()

            # Check for GraphQL errors (e.g. "outside allowed interval")
            if "errors" in data:
                for err in data["errors"]:
                    msg = err.get("message", "")
                    if "outside allowed interval" in msg.lower():
                        logger.debug("Santiment: %s — free tier data range limit", msg)
                    else:
                        logger.warning("Santiment GraphQL error: %s", msg)
                if self._health_tracker:
                    self._health_tracker.record_failure("santiment", str(data["errors"]))
                return None

            if self._health_tracker:
                self._health_tracker.record_success("santiment")
            return data
        except Exception as exc:
            logger.warning("Santiment API error: %s", exc)
            if self._health_tracker:
                self._health_tracker.record_failure("santiment", str(exc))
            return None

    async def _fetch_metric_value(
        self, slug: str, metric: str, days: int = 7,
    ) -> Optional[float]:
        """Fetch the latest value for a single metric.

        Free tier has ~30 day data lag. Caps `to` at 35 days ago to stay
        within the allowed interval.
        """
        now = datetime.utcnow()
        # Free tier only has data up to ~30 days ago — cap to stay in allowed range
        capped_to = now - timedelta(days=30)
        from_dt = (capped_to - timedelta(days=days)).isoformat() + "Z"
        to_dt = capped_to.isoformat() + "Z"

        cache_key = f"santiment:{slug}:{metric}:{days}"
        cached = self._cache.get_cached(cache_key)
        if cached is not None:
            return cached.get("value")

        data = await self._query(SINGLE_VALUE_QUERY, {
            "slug": slug, "metric": metric,
            "from": from_dt, "to": to_dt,
        })
        if not data:
            return None
        try:
            series = data["data"]["getMetric"]["timeseriesData"]
            if not series:
                return None
            value = float(series[-1]["value"])
            self._cache.set_cached(
                cache_key, {"value": value}, "santiment",
                settings.SANTIMENT_CACHE_TTL,
            )
            return value
        except (KeyError, IndexError, TypeError, ValueError):
            return None

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    async def get_signal(self, slug: str, days: int = 7) -> Optional[SantimentSignal]:
        """Fetch combined social + dev signal for an asset slug.

        Uses free tier data (last ~12 months). If query returns
        "outside allowed interval", returns None.
        """
        try:
            social_vol = await self._fetch_metric_value(slug, "social_volume_total", days)
            sentiment = await self._fetch_metric_value(slug, "sentiment_balance_total", days)
            dev_activity = await self._fetch_metric_value(slug, "dev_activity", days)
            dominance = await self._fetch_metric_value(slug, "social_dominance_total", days)
            addresses = await self._fetch_metric_value(slug, "daily_active_addresses", days)

            if not any([social_vol, sentiment, dev_activity, dominance, addresses]):
                return None

            return SantimentSignal(
                asset_slug=slug,
                social_volume_24h=social_vol,
                sentiment_balance_24h=sentiment,
                dev_activity_30d=dev_activity,
                social_dominance_pct=dominance,
                daily_active_addresses=int(addresses) if addresses else None,
            )
        except Exception as exc:
            logger.warning("Santiment get_signal failed for %s: %s", slug, exc)
            return None

    async def get_trending_assets(self) -> List[str]:
        """Fetch assets with surging social volume."""
        data = await self._query(TRENDING_QUERY)
        if not data:
            return []
        try:
            assets = data["data"]["getTrendingAssets"]
            return [a["slug"] for a in assets]
        except (KeyError, IndexError):
            return []

    async def get_batch_signals(
        self, slugs: List[str],
    ) -> Dict[str, SantimentSignal]:
        """Fetch signals for multiple assets concurrently."""
        import asyncio
        tasks = [self.get_signal(slug) for slug in slugs]
        results = await asyncio.gather(*tasks)
        return {
            slug: signal
            for slug, signal in zip(slugs, results)
            if signal is not None
        }
