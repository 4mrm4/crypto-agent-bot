# Santiment API Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add social volume, developer activity, and sentiment balance signals via Santiment GraphQL API.

**Architecture:** Create `data/santiment_fetcher.py` with `SantimentFetcher` class wrapping Santiment's GraphQL API (sanpy or direct httpx). Integrate into `SentimentFetcher` as a third source alongside Fear & Greed and CryptoPanic. Extend `CombinedSentiment` with Santiment fields. Integrate trending assets into autonomous research loop priority.

**Tech Stack:** httpx, GraphQL, SQLite api_cache, pandas (via sanpy)

**Prerequisites:** Shared infrastructure (api_cache table + APIHealthTracker) must be implemented first.

---
## Files

- Create: `data/santiment_fetcher.py` — SantimentFetcher, SantimentSignal dataclass
- Modify: `data/sentiment.py` — extend CombinedSentiment, integrate Santiment as third source
- Modify: `data/regime.py` — add social dominance signal
- Modify: `orchestration/autonomous_loop.py` — integrate trending assets into goal priority
- Modify: `config.py` — add SANTIMENT_* settings
- Test: `test_santiment.py`

---

### Task 1: Add config settings

- [ ] **Step 1: Add to config.py**

```python
# Santiment API (social volume + developer activity)
SANTIMENT_API_KEY: str = os.getenv("SANTIMENT_API_KEY", "")
SANTIMENT_ENABLED: bool = os.getenv("SANTIMENT_ENABLED", "false").lower() == "true"
SANTIMENT_CACHE_TTL: int = int(os.getenv("SANTIMENT_CACHE_TTL", "1800"))  # 30 min
SANTIMENT_SLUGS: str = os.getenv("SANTIMENT_SLUGS", "bitcoin,ethereum,solana")
```

- [ ] **Step 2: Add to .env.example**

```bash
# === Santiment API (social volume + developer activity) ===
# Get free key at: https://app.santiment.net/account#api-keys
SANTIMENT_API_KEY=
SANTIMENT_ENABLED=false
# Comma-separated list of asset slugs to track
SANTIMENT_SLUGS=bitcoin,ethereum,solana
```

- [ ] **Step 3: Commit**

```bash
git add config.py .env.example
git commit -m "chore: add SANTIMENT_* settings"
```

---

### Task 2: Create SantimentFetcher

- [ ] **Step 1: Write the failing tests**

Create `test_santiment.py`:

```python
"""Tests for Santiment API integration."""
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from data.santiment_fetcher import SantimentFetcher, SantimentSignal


@pytest.fixture
def fetcher():
    return SantimentFetcher(api_key="test_key")


@pytest.mark.asyncio
async def test_get_signal_returns_correct_dataclass(fetcher):
    """Test get_signal returns SantimentSignal from mocked API response."""
    mock_response = {
        "data": {
            "getMetric": {
                "timeseriesData": [
                    {"datetime": "2024-01-01T00:00:00Z", "value": 1500.0},
                ]
            }
        }
    }

    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.return_value = AsyncMock()
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = mock_response

        # Mock the second call for sentiment_balance
        mock_post.return_value.json.side_effect = [
            mock_response,  # social_volume
            {"data": {"getMetric": {"timeseriesData": [{"value": 0.35}]}}},  # sentiment
            {"data": {"getMetric": {"timeseriesData": [{"value": 250.0}]}}},  # dev
            {"data": {"getMetric": {"timeseriesData": [{"value": 2.5}]}}},  # dominance
            {"data": {"getMetric": {"timeseriesData": [{"value": 800000}]}}},  # addresses
        ]

        result = await fetcher.get_signal("bitcoin", days=7)

    assert isinstance(result, SantimentSignal)
    assert result.asset_slug == "bitcoin"
    assert result.social_volume_24h == 1500.0
    assert result.sentiment_balance_24h == 0.35
    assert result.source == "santiment"


@pytest.mark.asyncio
async def test_get_signal_returns_none_on_outside_interval_error(fetcher):
    """Free tier returns 'outside allowed interval' for old data — must not crash."""
    mock_error = {
        "errors": [{"message": "Outside allowed interval for metric social_volume"}]
    }

    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.return_value = AsyncMock()
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = mock_error

        result = await fetcher.get_signal("bitcoin", days=365)

    assert result is None


@pytest.mark.asyncio
async def test_get_signal_returns_none_on_api_error(fetcher):
    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.side_effect = Exception("Connection failed")
        result = await fetcher.get_signal("bitcoin")
    assert result is None


@pytest.mark.asyncio
async def test_get_signal_returns_none_when_disabled():
    fetcher = SantimentFetcher(enabled=False)
    result = await fetcher.get_signal("bitcoin")
    assert result is None


@pytest.mark.asyncio
async def test_get_trending_assets_returns_list(fetcher):
    mock_response = {
        "data": {
            "getTrendingAssets": [
                {"slug": "bitcoin", "score": 100},
                {"slug": "ethereum", "score": 85},
            ]
        }
    }
    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.return_value = AsyncMock()
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = mock_response

        trending = await fetcher.get_trending_assets()
    assert "bitcoin" in trending
    assert "ethereum" in trending


@pytest.mark.asyncio
async def test_get_batch_signals(fetcher):
    with patch.object(fetcher, "get_signal", new_callable=AsyncMock) as mock_gs:
        mock_gs.side_effect = [
            SantimentSignal(asset_slug="bitcoin", social_volume_24h=1000.0,
                            sentiment_balance_24h=0.5, fetched_at=datetime.utcnow()),
            SantimentSignal(asset_slug="ethereum", social_volume_24h=500.0,
                            sentiment_balance_24h=-0.2, fetched_at=datetime.utcnow()),
        ]
        results = await fetcher.get_batch_signals(["bitcoin", "ethereum"])

    assert "bitcoin" in results
    assert "ethereum" in results
    assert results["bitcoin"].sentiment_balance_24h == 0.5
```

- [ ] **Step 2: Implement SantimentFetcher**

Create `data/santiment_fetcher.py`:

```python
"""Santiment API client — social volume, developer activity, and sentiment balance.

Uses Santiment's GraphQL API directly (via httpx) rather than sanpy to keep
dependencies minimal and control async behaviour.

Gracefully degrades: returns None on any error, never raises.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
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

METRIC_QUERY = """
query($slug: String!, $metric: String!, $from: DateTime!, $to: DateTime!) {
  getMetric(metric: $metric) {
    timeseriesData(
      slug: $slug
      from: $from
      to: $to
      interval: "1d"
      transform: {type: "last", value: "1d"}
    ) {
      datetime
      value
    }
  }
}
"""

SINGLE_VALUE_QUERY = """
query($slug: String!, $metric: String!, $from: DateTime!, $to: DateTime!) {
  getMetric(metric: $metric) {
    timeseriesData(
      slug: $slug
      from: $from
      to: $to
      interval: "1d"
      limit: 1
      transform: {type: "last", value: "1d"}
    ) {
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
        """Fetch the latest value for a single metric."""
        from datetime import timedelta
        now = datetime.utcnow()
        from_dt = (now - timedelta(days=days)).isoformat() + "Z"
        to_dt = now.isoformat() + "Z"

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
            sentiment = await self._fetch_metric_value(slug, "sentiment_balance", days)
            dev_activity = await self._fetch_metric_value(slug, "dev_activity", days)
            dominance = await self._fetch_metric_value(slug, "social_dominance", days)
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
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest test_santiment.py -v --tb=short`
Expected: all tests PASS

- [ ] **Step 4: Commit**

```bash
git add data/santiment_fetcher.py test_santiment.py
git commit -m "feat: add SantimentFetcher with social volume, sentiment, dev activity"
```

---

### Task 3: Integrate Santiment into SentimentFetcher as third source

- [ ] **Step 1: Write the failing test**

Add to `test_santiment.py`:

```python
from data.sentiment import SentimentFetcher


@pytest.mark.asyncio
async def test_combined_sentiment_includes_santiment():
    """Test Santiment signals are merged into CombinedSentiment."""
    fetcher = SentimentFetcher()

    with patch.object(fetcher, "_fetch_santiment", new_callable=AsyncMock) as mock_s:
        mock_s.return_value = SantimentSignal(
            asset_slug="bitcoin",
            social_volume_24h=1500.0,
            sentiment_balance_24h=0.35,
            dev_activity_30d=250.0,
            fetched_at=datetime.utcnow(),
        )

        result = await fetcher.get_combined_sentiment()

    assert result.santiment_social_volume == 1500.0
    assert result.santiment_sentiment_balance == 0.35
    assert result.santiment_dev_activity == 250.0
    assert result.signal_count >= 2  # F&G + Santiment
    assert isinstance(result.overall_score, float)


@pytest.mark.asyncio
async def test_combined_sentiment_weight_redistribution():
    """When Santiment is down, weights redistribute to F&G + CryptoPanic."""
    fetcher = SentimentFetcher()

    with patch.object(fetcher, "_fetch_santiment", new_callable=AsyncMock) as mock_s:
        mock_s.return_value = None  # Santiment unavailable

        result = await fetcher.get_combined_sentiment()

    assert result.santiment_social_volume is None
    assert result.signal_count >= 1  # At least F&G
    assert isinstance(result.overall_score, float)


@pytest.mark.asyncio
async def test_combined_sentiment_all_sources_down():
    """When ALL sentiment sources fail, should still return a neutral score."""
    fetcher = SentimentFetcher()

    with patch.object(fetcher, "get_fear_greed_index", return_value=None):
        with patch.object(fetcher, "get_cryptopanic_news", return_value=[]):
            with patch.object(fetcher, "_fetch_santiment", new_callable=AsyncMock) as mock_s:
                mock_s.return_value = None

                result = await fetcher.get_combined_sentiment()

    assert result.overall_score == 0.0  # Neutral
    assert result.signal_count == 0
```

- [ ] **Step 2: Add CombinedSentiment dataclass and extend SentimentFetcher**

In `data/sentiment.py`, add new imports, the `CombinedSentiment` dataclass, and make `SentimentFetcher` async-aware:

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class CombinedSentiment:
    """Aggregated sentiment from all available sources."""
    fear_greed_index: Optional[int] = None
    cryptopanic_sentiment: Optional[float] = None
    santiment_social_volume: Optional[float] = None
    santiment_sentiment_balance: Optional[float] = None
    santiment_dev_activity: Optional[float] = None
    overall_score: float = 0.0       # -1.0 to +1.0
    signal_count: int = 0             # number of sources that contributed
    fetched_at: datetime = field(default_factory=datetime.utcnow)
```

Add to `SentimentFetcher`:

```python
async def _fetch_santiment(self, slug: str = "bitcoin") -> Optional[SantimentSignal]:
    """Fetch Santiment signal if enabled."""
    from data.santiment_fetcher import SantimentFetcher
    sf = SantimentFetcher()
    return await sf.get_signal(slug)


async def get_combined_sentiment(self, slug: str = "bitcoin") -> CombinedSentiment:
    """Aggregate sentiment from all available sources.

    Weight distribution:
    - Fear & Greed: 40%
    - CryptoPanic news sentiment: 20%
    - Santiment social volume: 25%
    - Santiment sentiment balance: 15%
    """
    fg = self.get_fear_greed_index()
    news = self.get_cryptopanic_news()
    santiment = await self._fetch_santiment(slug)

    result = CombinedSentiment()

    # Fear & Greed (sync, existing)
    if fg and fg.get("value") is not None:
        result.fear_greed_index = fg["value"]

    # CryptoPanic (sync, existing)
    if news:
        vote_scores = []
        for item in news:
            pos = item.get("votes_positive", 0)
            neg = item.get("votes_negative", 0)
            total = pos + neg
            if total > 0:
                vote_scores.append((pos - neg) / total)
        if vote_scores:
            result.cryptopanic_sentiment = sum(vote_scores) / len(vote_scores)

    # Santiment (async, new)
    if santiment:
        result.santiment_social_volume = santiment.social_volume_24h
        result.santiment_sentiment_balance = santiment.sentiment_balance_24h
        result.santiment_dev_activity = santiment.dev_activity_30d

    # Compute weighted overall_score
    weights = []
    scores = []

    if result.fear_greed_index is not None:
        # Map 0-100 to -1..+1
        scores.append((result.fear_greed_index - 50) / 50.0)
        weights.append(0.40)

    if result.cryptopanic_sentiment is not None:
        scores.append(result.cryptopanic_sentiment)
        weights.append(0.20)

    if result.santiment_social_volume is not None:
        # Normalise social volume to z-score-ish range, cap at -1..+1
        vol_z = (result.santiment_social_volume - 1000) / 2000
        scores.append(max(-1, min(1, vol_z)))
        weights.append(0.25)

    if result.santiment_sentiment_balance is not None:
        scores.append(result.santiment_sentiment_balance)
        weights.append(0.15)

    if scores and weights:
        # Redistribute weights proportionally if some sources are missing
        total_weight = sum(weights)
        if total_weight > 0:
            result.overall_score = sum(
                s * w / total_weight for s, w in zip(scores, weights)
            )
        result.signal_count = len(scores)

    return result
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest test_santiment.py -v --tb=short`
Expected: all tests PASS

- [ ] **Step 4: Update hermes.py to use CombinedSentiment**

In `orchestration/hermes.py`, update the sentiment fetching code (around line ~180) to use `get_combined_sentiment()` instead of just F&G index:

```python
sentiment_data = await SentimentFetcher().get_combined_sentiment()
if sentiment_data:
    self._current_sentiment_score = sentiment_data.overall_score
    logger.info(
        "Combined sentiment: overall=%.2f, sources=%d, F&G=%s",
        sentiment_data.overall_score,
        sentiment_data.signal_count,
        sentiment_data.fear_greed_index,
    )
```

- [ ] **Step 5: Commit**

```bash
git add data/sentiment.py orchestration/hermes.py test_santiment.py
git commit -m "feat: integrate Santiment into SentimentFetcher as third source with CombinedSentiment"
```

---

### Task 4: Integrate Santiment social dominance into MarketRegimeDetector

- [ ] **Step 1: Add method and modify RegimeSnapshot**

In `data/regime.py`, add fields to `RegimeSnapshot`:

```python
social_dominance_zscore: Optional[float] = None  # Santiment social dominance z-score
santiment_signal: str = "neutral"                 # Additional regime signal
```

Add to `MarketRegimeDetector`:

```python
async def _get_social_signal(self, slug: str = "bitcoin") -> dict:
    """Fetch Santiment social dominance and return regime signal."""
    from data.santiment_fetcher import SantimentFetcher
    sf = SantimentFetcher()
    signal = await sf.get_signal(slug)
    if not signal or signal.social_dominance_pct is None:
        return {"signal": "neutral", "zscore": 0}

    # Social dominance spike (>3 std dev above mean) = potential local top
    # We use a simple heuristic: >5% dominance is a spike
    dom = signal.social_dominance_pct
    zscore = (dom - 1.0) / 1.5  # approximate z-score assuming mean=1%, std=1.5%
    if zscore > 3:
        signal_label = "bearish"  # spike = potential top
    elif zscore < -1:
        signal_label = "bullish"  # very low dominance = accumulation
    else:
        signal_label = "neutral"
    return {"signal": signal_label, "zscore": round(zscore, 2)}
```

- [ ] **Step 2: Commit**

```bash
git add data/regime.py
git commit -m "feat: add Santiment social dominance signal to MarketRegimeDetector"
```

---

### Task 5: Integrate trending assets into autonomous research loop

- [ ] **Step 1: Modify generate_next_goal**

In `orchestration/autonomous_loop.py`, modify `_generate_next_goal` to check Santiment trending assets:

Add before the coverage gap check (around line 217):

```python
# 0. Check Santiment trending assets — boost priority if trending asset has no strategy
try:
    from data.santiment_fetcher import SantimentFetcher
    sf = SantimentFetcher()
    trending = await sf.get_trending_assets()
    if trending:
        logger.info("Santiment trending assets: %s", trending[:5])
        # Check if any trending asset lacks strategy coverage
        slug_map = {"bitcoin": "BTC/USDT", "ethereum": "ETH/USDT", "solana": "SOL/USDT"}
        for slug in trending[:3]:
            pair = slug_map.get(slug)
            if pair and self._vector_store:
                best = self._vector_store.get_best_strategies(
                    regime=current_regime, min_sharpe=0.5, k=1,
                )
                if not best:
                    logger.info(
                        "Trending asset %s has no strategy for %s — priority boost",
                        slug, current_regime,
                    )
                    # Continue with normal coverage gap logic but use this slug
                    break
except Exception as exc:
    logger.debug("Trending assets check skipped: %s", exc)
```

- [ ] **Step 2: Commit**

```bash
git add orchestration/autonomous_loop.py
git commit -m "feat: integrate Santiment trending assets into autonomous research priority"
```
