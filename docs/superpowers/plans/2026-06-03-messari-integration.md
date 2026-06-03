# Messari API Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add institutional-grade asset fundamentals and on-chain metrics via Messari API.

**Architecture:** Create `data/messari_fetcher.py` with `MessariFetcher` class wrapping Messari REST API. Integrate into OnChainFetcher as supplementary source alongside Whale Alert. Add `get_asset_fundamentals` tool to ResearcherAgent. Extend MarketRegimeDetector to consume on-chain signals.

**Tech Stack:** httpx, asyncio, SQLite api_cache, pandas

**Prerequisites:** Shared infrastructure (api_cache table + APIHealthTracker) must be implemented first.

---
## Files

- Create: `data/messari_fetcher.py` — MessariFetcher, MessariMetrics dataclass, RateLimiter
- Create: `data/rate_limiter.py` — Token bucket rate limiter
- Modify: `data/onchain.py` — add MessariFetcher to OnChainFetcher, create OnChainSignal dataclass
- Modify: `data/regime.py` — add on-chain signal consumption in RegimeSnapshot
- Modify: `agents/researcher.py` — add get_asset_fundamentals tool
- Modify: `config.py` — add MESSARI_* settings
- Test: `test_messari.py`, `test_rate_limiter.py`

---

### Task 1: Add config settings

- [ ] **Step 1: Add to config.py**

```python
# Messari API (fundamentals + on-chain metrics)
MESSARI_API_KEY: str = os.getenv("MESSARI_API_KEY", "")
MESSARI_ENABLED: bool = os.getenv("MESSARI_ENABLED", "false").lower() == "true"
MESSARI_RATE_LIMIT_RPM: int = int(os.getenv("MESSARI_RATE_LIMIT_RPM", "20"))
MESSARI_CACHE_METRICS_TTL: int = int(os.getenv("MESSARI_CACHE_METRICS_TTL", "900"))  # 15min
MESSARI_CACHE_PROFILE_TTL: int = int(os.getenv("MESSARI_CACHE_PROFILE_TTL", "86400"))  # 24h
```

- [ ] **Step 2: Add to .env.example**

```bash
# === Messari API (fundamentals + on-chain metrics) ===
# Get free key at: https://messari.io/account/api
MESSARI_API_KEY=
MESSARI_ENABLED=false
```

- [ ] **Step 3: Commit**

```bash
git add config.py .env.example
git commit -m "chore: add MESSARI_* settings"
```

---

### Task 2: Create RateLimiter

- [ ] **Step 1: Write the failing test**

Create `test_rate_limiter.py`:

```python
"""Tests for token bucket rate limiter."""
import time
import pytest
from data.rate_limiter import RateLimiter


def test_initial_tokens_equals_capacity():
    rl = RateLimiter(rpm=10)
    assert rl._tokens == 10.0


def test_consume_allows_within_limit():
    rl = RateLimiter(rpm=5)
    for _ in range(5):
        assert rl.acquire() is True


def test_consume_blocks_over_limit():
    rl = RateLimiter(rpm=3)
    for _ in range(3):
        assert rl.acquire() is True
    # 4th should block (or return False if non-blocking)
    assert rl.acquire(block=False) is False


def test_acquire_blocking_wait(self):
    rl = RateLimiter(rpm=60)  # 1 per second
    for _ in range(60):
        rl.acquire()
    start = time.time()
    rl.acquire(block=True, timeout=5)
    elapsed = time.time() - start
    assert elapsed >= 0.9  # Should have waited ~1 second
```

- [ ] **Step 2: Implement RateLimiter**

Create `data/rate_limiter.py`:

```python
"""Token bucket rate limiter for external API calls."""
import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)


class RateLimiter:
    """Token bucket rate limiter.

    Uses a background thread to refill tokens at the configured rate.
    Thread-safe via a lock.
    """

    def __init__(self, rpm: int = 20):
        self._capacity = rpm
        self._tokens = float(rpm)
        self._refill_rate = rpm / 60.0  # tokens per second
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self):
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._capacity, self._tokens + elapsed * self._refill_rate)
        self._last_refill = now

    def acquire(self, block: bool = True, timeout: Optional[float] = None) -> bool:
        """Acquire a token. Returns True if acquired, False if rate limited.

        If block=True, waits up to `timeout` seconds for a token.
        If block=False, returns immediately with False if no token available.
        """
        deadline = time.monotonic() + timeout if timeout else None

        while True:
            with self._lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return True
                if not block:
                    return False

            # Blocking wait
            wait = 1.0 / self._refill_rate  # time for 1 token
            if deadline:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                wait = min(wait, remaining)
            time.sleep(wait)

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *args):
        pass
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest test_rate_limiter.py -v --tb=short`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add data/rate_limiter.py test_rate_limiter.py
git commit -m "feat: add token bucket RateLimiter for external API rate limits"
```

---

### Task 3: Create MessariFetcher

- [ ] **Step 1: Write the failing test**

Create `test_messari.py`:

```python
"""Tests for Messari API integration."""
from datetime import datetime
from unittest.mock import AsyncMock, patch
import pytest
from data.messari_fetcher import MessariFetcher, MessariMetrics


@pytest.fixture
def fetcher():
    return MessariFetcher(api_key="test_key")


@pytest.mark.asyncio
async def test_get_metrics_returns_correct_dataclass(fetcher):
    mock_metrics = {
        "data": {
            "market_data": {
                "price_usd": 50000.0,
                "market_cap": 1e12,
                "volume_last_24_hours": 3e10,
            },
            "onchain_data": {
                "active_addresses_24h": 800000,
                "transaction_volume_24h": 5e9,
                "developer_activity_30d": 450.0,
            },
            "supply": {
                "circulating": 19000000,
                "total": 21000000,
            },
        }
    }

    with patch("httpx.AsyncClient.get") as mock_get:
        mock_get.return_value = AsyncMock()
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = mock_metrics

        result = await fetcher.get_metrics("bitcoin")

    assert isinstance(result, MessariMetrics)
    assert result.asset == "bitcoin"
    assert result.price_usd == 50000.0
    assert result.active_addresses_24h == 800000
    assert result.developer_activity_30d == 450.0
    assert result.source == "messari"


@pytest.mark.asyncio
async def test_get_metrics_returns_none_on_api_error(fetcher):
    with patch("httpx.AsyncClient.get") as mock_get:
        mock_get.side_effect = Exception("API error")
        result = await fetcher.get_metrics("bitcoin")
    assert result is None


@pytest.mark.asyncio
async def test_get_metrics_returns_none_when_disabled():
    fetcher = MessariFetcher(enabled=False)
    result = await fetcher.get_metrics("bitcoin")
    assert result is None


@pytest.mark.asyncio
async def test_get_profile_returns_dict(fetcher):
    mock_profile = {
        "data": {
            "id": "bitcoin",
            "name": "Bitcoin",
            "symbol": "BTC",
            "profile": {
                "consensus": "PoW",
                "algorithm": "SHA-256",
                "description": "Bitcoin is...",
            },
        }
    }
    with patch("httpx.AsyncClient.get") as mock_get:
        mock_get.return_value = AsyncMock()
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = mock_profile

        result = await fetcher.get_profile("bitcoin")
    assert result is not None
    assert result.get("id") == "bitcoin"
    assert result.get("profile", {}).get("consensus") == "PoW"


@pytest.mark.asyncio
async def test_get_batch_metrics(fetcher):
    with patch.object(fetcher, "get_metrics", new_callable=AsyncMock) as mock_gm:
        mock_gm.side_effect = [
            MessariMetrics(asset="bitcoin", price_usd=50000.0, market_cap=1e12,
                           volume_24h=3e10, fetched_at=datetime.utcnow()),
            MessariMetrics(asset="ethereum", price_usd=3000.0, market_cap=3e11,
                           volume_24h=1e10, fetched_at=datetime.utcnow()),
        ]
        results = await fetcher.get_batch_metrics(["bitcoin", "ethereum"])

    assert "bitcoin" in results
    assert "ethereum" in results
    assert results["bitcoin"].price_usd == 50000.0
```

- [ ] **Step 2: Create the MessariFetcher**

Create `data/messari_fetcher.py`:

```python
"""Messari API client — institutional-grade asset fundamentals and on-chain metrics.

Three endpoints used:
- /v2/assets/{slug}/metrics  — price, market cap, on-chain data
- /v1/assets/{slug}/profile  — qualitative project info
- /v1/news/topics            — AI-analysed trending topics

Gracefully degrades: returns None on any error, never raises.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import httpx

from config import settings
from data.database import TradingDatabase
from data.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

MESSARI_BASE = "https://api.messari.io"


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
    fetched_at: datetime = field(default_factory=datetime.utcnow)
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
            f"/api/v2/assets/{slug}/metrics",
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
            f"/api/v1/assets/{slug}/profile",
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
            f"/api/v1/news/topics?limit={limit}",
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
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest test_messari.py -v --tb=short`
Expected: all tests PASS

- [ ] **Step 4: Commit**

```bash
git add data/messari_fetcher.py test_messari.py
git commit -m "feat: add MessariFetcher with metrics, profile, trending topics"
```

---

### Task 4: Integrate Messari into OnChainFetcher

- [ ] **Step 1: Add combined OnChainSignal dataclass**

In `data/onchain.py`, add before `OnChainFetcher`:

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class OnChainSignal:
    """Combined on-chain signal from Whale Alert + Messari."""
    regime_signal: str  # "bullish" | "bearish" | "neutral"
    whale_alert_events: list  # from Whale Alert
    messari_metrics: Optional[dict] = None  # from Messari
    confidence: float = 0.0
    fetched_at: datetime = field(default_factory=datetime.utcnow)
```

- [ ] **Step 2: Write test**

In `test_messari.py`:

```python
@pytest.mark.asyncio
async def test_onchain_get_combined_signal_merges_sources():
    from data.onchain import OnChainFetcher
    fetcher = OnChainFetcher()
    # Whitelist Messari + Whale Alert; mock both
    ...
```

- [ ] **Step 3: Implement get_combined_signal**

In `data/onchain.py`, add method to `OnChainFetcher`:

```python
async def get_combined_signal(self, slug: str = "bitcoin") -> OnChainSignal:
    """Merge Whale Alert events with Messari on-chain metrics into a single signal."""
    # 1. Get whale alerts (existing, sync)
    whale = self.get_whale_transactions(min_usd=1_000_000, limit=5)

    # 2. Get Messari metrics (new, async)
    messari_source = None
    messari_metrics = {}
    from data.messari_fetcher import MessariFetcher
    mf = MessariFetcher()
    metrics = await mf.get_metrics(slug)
    if metrics:
        messari_source = metrics.source
        messari_metrics = {
            "active_addresses_24h": metrics.active_addresses_24h,
            "transaction_volume_24h": metrics.transaction_volume_24h,
            "developer_activity_30d": metrics.developer_activity_30d,
        }

    # 3. Compute combined signal
    whale_bearish = sum(1 for t in whale if t.get("from_owner", "") == "exchange")
    whale_bullish = sum(1 for t in whale if t.get("to_owner", "") == "exchange")

    # High dev activity + rising active addresses = fundamental strength
    dev_signal = 0
    if metrics and metrics.developer_activity_30d and metrics.developer_activity_30d > 300:
        dev_signal = 1
    if metrics and metrics.active_addresses_24h and metrics.active_addresses_24h > 500000:
        dev_signal += 1

    net_whale = whale_bullish - whale_bearish
    total_signals = 0
    bullish_signals = 0

    if whale:
        total_signals += 1
        if net_whale > 0:
            bullish_signals += 1
    if dev_signal >= 2:
        total_signals += 1
        bullish_signals += 1

    if total_signals == 0:
        regime_signal = "neutral"
    else:
        ratio = bullish_signals / total_signals
        regime_signal = "bullish" if ratio >= 0.5 else "bearish"

    return OnChainSignal(
        regime_signal=regime_signal,
        whale_alert_events=whale,
        messari_metrics=messari_metrics if messari_metrics else None,
        confidence=total_signals / 2.0,
    )
```

- [ ] **Step 4: Commit**

```bash
git add data/onchain.py test_messari.py
git commit -m "feat: integrate Messari on-chain metrics into OnChainFetcher get_combined_signal"
```

---

### Task 5: Add get_asset_fundamentals tool to ResearcherAgent

- [ ] **Step 1: Write the failing test**

Add to `test_messari.py`:

```python
from agents.researcher import ResearcherAgent


def test_get_asset_fundamentals_tool_registered():
    """ResearcherAgent should have get_asset_fundamentals tool."""
    agent = ResearcherAgent()
    tool_names = [t.name for t in agent.tool_list]
    assert "get_asset_fundamentals" in tool_names


def test_get_asset_fundamentals_returns_error_message_when_no_api_key():
    """Without API key, should return a graceful error, not crash."""
    agent = ResearcherAgent()
    # Find the tool
    tool = agent.tools.get("get_asset_fundamentals")
    assert tool is not None
    result = tool.func("bitcoin")
    assert isinstance(result, str)
    assert len(result) > 0
```

- [ ] **Step 2: Add tool to ResearcherAgent._build_tools()**

In `agents/researcher.py`, in `_build_tools`:

```python
def get_asset_fundamentals(asset_slug: str = "bitcoin") -> str:
    """Fetch fundamental and on-chain data for an asset to evaluate strategy viability.
    Pass the asset slug (e.g. 'bitcoin', 'ethereum', 'solana').
    Returns structured fundamentals including price, market cap, active addresses,
    developer activity, and project profile information.
    """
    import asyncio
    try:
        from data.messari_fetcher import MessariFetcher
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        mf = MessariFetcher()
        metrics = loop.run_until_complete(mf.get_metrics(asset_slug))
        profile = loop.run_until_complete(mf.get_profile(asset_slug))
        loop.close()

        lines = [f"=== Asset Fundamentals: {asset_slug} ==="]
        if metrics:
            lines.append(f"Price: ${metrics.price_usd:,.2f}")
            lines.append(f"Market Cap: ${metrics.market_cap:,.0f}")
            lines.append(f"24h Volume: ${metrics.volume_24h:,.0f}")
            if metrics.active_addresses_24h:
                lines.append(f"Active Addresses (24h): {metrics.active_addresses_24h:,}")
            if metrics.transaction_volume_24h:
                lines.append(f"Transaction Volume (24h): ${metrics.transaction_volume_24h:,.0f}")
            if metrics.developer_activity_30d:
                lines.append(f"Dev Activity (30d): {metrics.developer_activity_30d}")
        if profile:
            desc = profile.get("profile", {}).get("description", "")[:200]
            if desc:
                lines.append(f"Description: {desc}")
        return "\n".join(lines) if len(lines) > 1 else f"No fundamentals data for {asset_slug}"
    except Exception as exc:
        return f"Fundamentals lookup failed: {exc}"
```

Register as a tool:

```python
Tool(
    name="get_asset_fundamentals",
    func=get_asset_fundamentals,
    description="Fetch fundamental and on-chain data for an asset slug (bitcoin, ethereum, solana). Returns price, market cap, active addresses, dev activity, and profile.",
),
```

- [ ] **Step 3: Commit**

```bash
git add agents/researcher.py
git commit -m "feat: add get_asset_fundamentals tool to ResearcherAgent via Messari"
```

---

### Task 6: Integrate Messari on-chain signals into MarketRegimeDetector

- [ ] **Step 1: Modify RegimeSnapshot**

Add fields to `RegimeSnapshot` in `data/regime.py`:

```python
# New fields to add:
on_chain_dev_activity: Optional[float] = None       # Normalised z-score
on_chain_active_addresses: Optional[float] = None    # Normalised z-score
on_chain_signal: str = "neutral"                     # "bullish" | "bearish" | "neutral"
```

- [ ] **Step 2: Add method to check on-chain signal**

In `MarketRegimeDetector`:

```python
async def _get_on_chain_signal(self, slug: str = "bitcoin") -> dict:
    """Fetch and normalise on-chain signals from Messari."""
    from data.messari_fetcher import MessariFetcher
    mf = MessariFetcher()
    metrics = await mf.get_metrics(slug)
    if not metrics:
        return {"signal": "neutral", "dev_zscore": 0, "address_zscore": 0}
    # Normalise: assume baseline of 200 dev commits, 300k addresses
    dev_z = (metrics.developer_activity_30d - 200) / 100 if metrics.developer_activity_30d else 0
    addr_z = (metrics.active_addresses_24h - 300000) / 150000 if metrics.active_addresses_24h else 0
    dev_signal = "bullish" if dev_z > 1 else "bearish" if dev_z < -1 else "neutral"
    addr_signal = "bullish" if addr_z > 1 else "bearish" if addr_z < -1 else "neutral"
    return {"signal": dev_signal, "dev_zscore": dev_z, "address_zscore": addr_z}
```

- [ ] **Step 3: Commit**

```bash
git add data/regime.py
git commit -m "feat: add on-chain signal consumption to MarketRegimeDetector via Messari"
```
