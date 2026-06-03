# CoinCap v3 API Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add CoinCap v3 as a tertiary backup price feed when Binance and Bybit are both unavailable.

**Architecture:** Create `data/coincap_fetcher.py` with `CoinCapFetcher` class (REST client). Integrate into `MultiExchangeFetcher.fetch_ohlcv()` as third fallback. Add degraded-mode backup to `MarketDataStream` WebSocket fallback. Add price source health check to `AnomalyDetector`.

**Tech Stack:** httpx, asyncio, pandas, FastAPI EventBus

**Prerequisites:** Shared infrastructure (api_cache table + APIHealthTracker) must be implemented first.

---
## Files

- Create: `data/coincap_fetcher.py` — CoinCapFetcher class
- Modify: `data/fetcher.py` — extend MultiExchangeFetcher with CoinCap fallback
- Modify: `data/stream.py` — add degraded-mode CoinCap polling
- Modify: `monitoring/anomaly_detector.py` — add price_source_check
- Modify: `api/server.py` — wire CoinCap health into APIHealthTracker
- Modify: `config.py` — add COINCAP_* settings
- Modify: `.env.example` — add COINCAP vars
- Test: `test_coincap.py`

---

### Task 1: Add config settings + .env.example

- [ ] **Step 1: Add to config.py**

Add after line ~88 (REDIS_URL):

```python
# CoinCap v3 API (backup price feed)
COINCAP_API_KEY: str = os.getenv("COINCAP_API_KEY", "")
COINCAP_ENABLED: bool = os.getenv("COINCAP_ENABLED", "false").lower() == "true"
COINCAP_FALLBACK_ONLY: bool = os.getenv("COINCAP_FALLBACK_ONLY", "true").lower() == "true"
COINCAP_POLL_INTERVAL_SECONDS: int = int(os.getenv("COINCAP_POLL_INTERVAL_SECONDS", "10"))
```

- [ ] **Step 2: Add to .env.example**

Append:

```bash
# === CoinCap v3 API (backup price feed) ===
# Get free key at: https://coincap.io
COINCAP_API_KEY=
COINCAP_ENABLED=false
COINCAP_FALLBACK_ONLY=true
```

- [ ] **Step 3: Commit**

```bash
git add config.py .env.example
git commit -m "chore: add COINCAP_* settings"
```

---

### Task 2: Create CoinCapFetcher

- [ ] **Step 1: Write the failing test first**

Create `test_coincap.py`:

```python
"""Tests for CoinCap v3 API integration."""
from datetime import datetime
from unittest.mock import AsyncMock, patch
import pytest
import pandas as pd
from data.coincap_fetcher import CoinCapFetcher, CoinCapPrice


@pytest.fixture
def fetcher():
    return CoinCapFetcher(api_key="test_key")


@pytest.mark.asyncio
async def test_get_price_returns_correct_dataclass(fetcher):
    mock_response = {
        "data": {
            "id": "bitcoin",
            "symbol": "BTC",
            "priceUsd": "50000.00",
            "marketCapUsd": "1000000000000",
            "volumeUsd24Hr": "30000000000",
            "changePercent24Hr": "2.5",
            "supply": "19000000",
        },
        "timestamp": 1700000000000,
    }

    with patch("httpx.AsyncClient.get") as mock_get:
        mock_get.return_value = AsyncMock()
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = mock_response

        result = await fetcher.get_price("bitcoin")

    assert isinstance(result, CoinCapPrice)
    assert result.asset_id == "bitcoin"
    assert result.symbol == "BTC"
    assert result.price_usd == 50000.00
    assert result.market_cap_usd == 1000000000000.0
    assert result.volume_24h_usd == 30000000000.0
    assert result.change_pct_24h == 2.5
    assert result.supply_circulating == 19000000.0
    assert result.source == "coincap"


@pytest.mark.asyncio
async def test_get_price_returns_none_on_api_error(fetcher):
    with patch("httpx.AsyncClient.get") as mock_get:
        mock_get.side_effect = Exception("Connection error")
        result = await fetcher.get_price("bitcoin")
    assert result is None


@pytest.mark.asyncio
async def test_get_price_returns_none_when_disabled():
    fetcher = CoinCapFetcher(enabled=False)
    result = await fetcher.get_price("bitcoin")
    assert result is None


@pytest.mark.asyncio
async def test_get_batch_prices_returns_dict(fetcher):
    assets = ["bitcoin", "ethereum"]

    with patch.object(fetcher, "get_price", new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = [
            CoinCapPrice(asset_id="bitcoin", symbol="BTC", price_usd=50000.0,
                         fetched_at=datetime.utcnow()),
            CoinCapPrice(asset_id="ethereum", symbol="ETH", price_usd=3000.0,
                         fetched_at=datetime.utcnow()),
        ]
        results = await fetcher.get_batch_prices(assets)

    assert "bitcoin" in results
    assert "ethereum" in results
    assert results["bitcoin"].price_usd == 50000.0


@pytest.mark.asyncio
async def test_get_ohlcv_fallback_returns_dataframe(fetcher):
    mock_history = {
        "data": [
            {"time": 1700000000000, "open": "50000", "high": "51000",
             "low": "49000", "close": "50500", "volume": "1000"},
            {"time": 1700003600000, "open": "50500", "high": "51500",
             "low": "50000", "close": "51000", "volume": "1200"},
        ]
    }

    with patch("httpx.AsyncClient.get") as mock_get:
        mock_get.return_value = AsyncMock()
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = mock_history

        df = await fetcher.get_ohlcv_fallback(
            "bitcoin", "h1",
            datetime(2024, 1, 1), datetime(2024, 1, 2),
        )

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert len(df) == 2
    assert df["close"].iloc[-1] == 51000.0


@pytest.mark.asyncio
async def test_get_ohlcv_fallback_returns_none_on_error(fetcher):
    with patch("httpx.AsyncClient.get") as mock_get:
        mock_get.side_effect = Exception("API error")
        df = await fetcher.get_ohlcv_fallback(
            "bitcoin", "h1",
            datetime(2024, 1, 1), datetime(2024, 1, 2),
        )
    assert df is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest test_coincap.py -v --tb=short`
Expected: FAIL with `ModuleNotFoundError: data.coincap_fetcher`

- [ ] **Step 3: Implement CoinCapFetcher**

Create `data/coincap_fetcher.py`:

```python
"""CoinCap v3 REST API client — backup price feed and OHLCV fallback.

Designed as a tertiary source when Binance and Bybit are both unavailable.
Only enabled when COINCAP_ENABLED=true and COINCAP_API_KEY is set.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import httpx
import pandas as pd

from config import settings

logger = logging.getLogger(__name__)

COINCAP_BASE = "https://rest.coincap.io/v3"


@dataclass
class CoinCapPrice:
    """Current price snapshot from CoinCap."""
    asset_id: str
    symbol: str
    price_usd: float
    market_cap_usd: Optional[float] = None
    volume_24h_usd: Optional[float] = None
    change_pct_24h: Optional[float] = None
    supply_circulating: Optional[float] = None
    fetched_at: datetime = field(default_factory=datetime.utcnow)
    source: str = "coincap"


class CoinCapFetcher:
    """Async REST client for CoinCap v3 API.

    Gracefully degrades: returns None on any error, never raises.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        enabled: Optional[bool] = None,
        health_tracker=None,
    ):
        self._api_key = api_key or settings.COINCAP_API_KEY
        self._enabled = enabled if enabled is not None else settings.COINCAP_ENABLED
        self._health_tracker = health_tracker
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=10)
        return self._client

    async def _get(self, path: str) -> Optional[dict]:
        if not self._enabled or not self._api_key:
            return None
        client = await self._get_client()
        url = f"{COINCAP_BASE}{path}"
        params = {"apiKey": self._api_key}
        try:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            if self._health_tracker:
                self._health_tracker.record_success("coincap")
            return resp.json()
        except Exception as exc:
            logger.warning("CoinCap API error [%s]: %s", path, exc)
            if self._health_tracker:
                self._health_tracker.record_failure("coincap", str(exc))
            return None

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    async def get_price(self, asset_id: str) -> Optional[CoinCapPrice]:
        """Fetch current price for a single asset."""
        data = await self._get(f"/assets/{asset_id}")
        if not data or "data" not in data:
            return None
        d = data["data"]
        try:
            return CoinCapPrice(
                asset_id=d.get("id", asset_id),
                symbol=d.get("symbol", ""),
                price_usd=float(d.get("priceUsd", 0)),
                market_cap_usd=(
                    float(d["marketCapUsd"]) if d.get("marketCapUsd") else None
                ),
                volume_24h_usd=(
                    float(d["volumeUsd24Hr"]) if d.get("volumeUsd24Hr") else None
                ),
                change_pct_24h=(
                    float(d["changePercent24Hr"])
                    if d.get("changePercent24Hr") else None
                ),
                supply_circulating=(
                    float(d["supply"]) if d.get("supply") else None
                ),
            )
        except (ValueError, TypeError) as exc:
            logger.warning("CoinCap parse error for %s: %s", asset_id, exc)
            return None

    async def get_batch_prices(
        self, asset_ids: List[str],
    ) -> Dict[str, CoinCapPrice]:
        """Fetch prices for multiple assets concurrently."""
        import asyncio
        tasks = [self.get_price(aid) for aid in asset_ids]
        results = await asyncio.gather(*tasks)
        return {
            aid: result
            for aid, result in zip(asset_ids, results)
            if result is not None
        }

    async def get_ohlcv_fallback(
        self,
        asset_id: str,
        interval: str = "h1",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> Optional[pd.DataFrame]:
        """Fetch OHLCV history from CoinCap as a fallback data source.

        Returns a DataFrame with columns: timestamp, open, high, low, close, volume.
        Returns None on any error.
        """
        path = f"/assets/{asset_id}/history"
        params = {"interval": interval}
        if start:
            params["start"] = str(int(start.timestamp() * 1000))
        if end:
            params["end"] = str(int(end.timestamp() * 1000))

        client = await self._get_client()
        url = f"{COINCAP_BASE}{path}"
        try:
            resp = await client.get(url, params={**params, "apiKey": self._api_key})
            resp.raise_for_status()
            data = resp.json().get("data", [])
            if not data:
                return None
            rows = []
            for d in data:
                rows.append({
                    "timestamp": d["time"],
                    "open": float(d["open"]),
                    "high": float(d["high"]),
                    "low": float(d["low"]),
                    "close": float(d["close"]),
                    "volume": float(d["volume"]),
                })
            df = pd.DataFrame(rows)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            if self._health_tracker:
                self._health_tracker.record_success("coincap")
            return df
        except Exception as exc:
            logger.warning("CoinCap OHLCV fallback failed [%s]: %s", asset_id, exc)
            if self._health_tracker:
                self._health_tracker.record_failure("coincap", str(exc))
            return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest test_coincap.py -v --tb=short`
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add data/coincap_fetcher.py test_coincap.py
git commit -m "feat: add CoinCapFetcher with get_price, batch_prices, OHLCV fallback"
```

---

### Task 3: Wire symbol_to_coincap_id helper

- [ ] **Step 1: Add the symbol mapping to coincap_fetcher.py**

Append to `data/coincap_fetcher.py`:

```python
# ── Symbol mapping: CCXT pair format -> CoinCap asset ID ──

SYMBOL_TO_COINCAP: Dict[str, str] = {
    "BTC/USDT": "bitcoin",
    "ETH/USDT": "ethereum",
    "SOL/USDT": "solana",
    "BNB/USDT": "binancecoin",
    "XRP/USDT": "xrp",
    "ADA/USDT": "cardano",
    "DOGE/USDT": "dogecoin",
    "DOT/USDT": "polkadot",
    "MATIC/USDT": "matic",
    "LINK/USDT": "chainlink",
    "UNI/USDT": "uniswap",
    "ATOM/USDT": "cosmos",
    "LTC/USDT": "litecoin",
    "BCH/USDT": "bitcoin-cash",
    "AVAX/USDT": "avalanche",
}

# Inverse: CoinCap asset ID -> CCXT symbol
COINCAP_TO_SYMBOL: Dict[str, str] = {v: k for k, v in SYMBOL_TO_COINCAP.items()}


def symbol_to_coincap_id(symbol: str) -> Optional[str]:
    """Convert CCXT pair like 'BTC/USDT' to CoinCap asset ID like 'bitcoin'.

    Returns None if the symbol is not in the known mapping.
    """
    return SYMBOL_TO_COINCAP.get(symbol)


def coincap_id_to_symbol(asset_id: str) -> Optional[str]:
    """Reverse lookup: CoinCap asset ID -> CCXT symbol."""
    return COINCAP_TO_SYMBOL.get(asset_id)
```

- [ ] **Step 2: Write test for the mapping**

Add to `test_coincap.py`:

```python
from data.coincap_fetcher import symbol_to_coincap_id, coincap_id_to_symbol


def test_symbol_to_coincap_id_known():
    assert symbol_to_coincap_id("BTC/USDT") == "bitcoin"
    assert symbol_to_coincap_id("ETH/USDT") == "ethereum"


def test_symbol_to_coincap_id_unknown():
    assert symbol_to_coincap_id("UNKNOWN/USDT") is None


def test_coincap_id_to_symbol_reverse():
    assert coincap_id_to_symbol("bitcoin") == "BTC/USDT"
    assert coincap_id_to_symbol("ethereum") == "ETH/USDT"
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest test_coincap.py::test_symbol_to_coincap_id_known test_coincap.py::test_symbol_to_coincap_id_unknown test_coincap.py::test_coincap_id_to_symbol_reverse -v`
Expected: 3 PASS

- [ ] **Step 4: Commit**

```bash
git add data/coincap_fetcher.py test_coincap.py
git commit -m "feat: add symbol_to_coincap_id mapping"
```

---

### Task 4: Extend MultiExchangeFetcher with CoinCap fallback

- [ ] **Step 1: Add DataUnavailableError**

Add to `data/exceptions.py` (create):

```python
"""Custom exceptions for the data layer."""


class DataUnavailableError(Exception):
    """Raised when ALL configured data sources fail to return data."""
    pass
```

- [ ] **Step 2: Write test for the fallback chain**

Add to `test_coincap.py`:

```python
from unittest.mock import AsyncMock, patch
from data.fetcher import MultiExchangeFetcher


@pytest.mark.asyncio
async def test_coincap_fallback_when_binance_and_bybit_fail():
    """When Binance and Bybit both fail, CoinCap should be tried as tertiary fallback."""
    fetcher = MultiExchangeFetcher(exchange_ids=["binance", "bybit"])

    with patch.object(fetcher._fetchers["binance"], "fetch_ohlcv",
                      side_effect=Exception("Binance down")):
        with patch.object(fetcher._fetchers["bybit"], "fetch_ohlcv",
                          side_effect=Exception("Bybit down")):
            with patch("data.coincap_fetcher.CoinCapFetcher.get_ohlcv_fallback",
                       new_callable=AsyncMock) as mock_coincap:
                mock_df = pd.DataFrame({
                    "timestamp": [1700000000000],
                    "open": [50000.0], "high": [51000.0],
                    "low": [49000.0], "close": [50500.0],
                    "volume": [1000.0],
                })
                mock_coincap.return_value = mock_df

                df = await fetcher.fetch_ohlcv("BTC/USDT", "1h", limit=1)

    assert df is not None
    assert not df.empty
    # Verify CoinCap was called with the correct asset_id
    mock_coincap.assert_called_once()


@pytest.mark.asyncio
async def test_all_sources_fail_returns_empty_df():
    """When ALL sources fail, MultiExchangeFetcher should return empty DataFrame."""
    fetcher = MultiExchangeFetcher(exchange_ids=["binance", "bybit"])

    with patch.object(fetcher._fetchers["binance"], "fetch_ohlcv",
                      side_effect=Exception("Binance down")):
        with patch.object(fetcher._fetchers["bybit"], "fetch_ohlcv",
                          side_effect=Exception("Bybit down")):
            with patch("data.coincap_fetcher.CoinCapFetcher.get_ohlcv_fallback",
                       new_callable=AsyncMock) as mock_coincap:
                mock_coincap.return_value = None  # CoinCap also fails

                df = await fetcher.fetch_ohlcv("BTC/USDT", "1h", limit=1)

    assert isinstance(df, pd.DataFrame)
    assert df.empty
```

- [ ] **Step 3: Modify MultiExchangeFetcher.fetch_ohlcv to use CoinCap**

In `data/fetcher.py`, modify `fetch_ohlcv_merged`:

```python
async def fetch_ohlcv_merged(
    self, symbol: str, timeframe: str = "1h", limit: int = 500
) -> pd.DataFrame:
    """Return OHLCV from primary exchange with fallback chain: Binance -> Bybit -> CoinCap."""
    # Primary
    primary = self._fetchers.get("binance")
    if primary:
        try:
            return await primary.fetch_ohlcv(symbol, timeframe, limit)
        except Exception:
            pass

    # Fallback 1: Bybit
    fallback = self._fetchers.get("bybit")
    if fallback:
        try:
            return await fallback.fetch_ohlcv(symbol, timeframe, limit)
        except Exception:
            pass

    # Fallback 2: CoinCap (tertiary)
    coincap_id = symbol_to_coincap_id(symbol)
    if coincap_id:
        from data.coincap_fetcher import CoinCapFetcher
        cf = CoinCapFetcher(health_tracker=_health_tracker)
        df = await cf.get_ohlcv_fallback(coincap_id, _interval_map(timeframe))
        if df is not None and not df.empty:
            logger.info("Using CoinCap fallback for %s", symbol)
            return df

    logger.error("All data sources failed for %s", symbol)
    return pd.DataFrame()
```

Note: the existing `fetch_ohlcv_merged` is synchronous. It needs to be made async, OR we make `CoinCapFetcher` have a sync path. The simplest change: convert `fetch_ohlcv_merged` to `async def` and have callers `await` it. Check existing callers first.

Also add `_interval_map` helper:

```python
def _interval_map(timeframe: str) -> str:
    """Convert CCXT timeframe format to CoinCap interval format."""
    mapping = {
        "1m": "m1", "5m": "m5", "15m": "m15",
        "30m": "m30", "1h": "h1", "4h": "h4",
        "1d": "d1", "1w": "w1",
    }
    return mapping.get(timeframe, "h1")
```

And add `_health_tracker` import.

- [ ] **Step 4: Update existing callers of fetch_ohlcv_merged**

Check if any code calls `fetch_ohlcv_merged()` synchronously. If so, add `await` or create a synchronous wrapper.

- [ ] **Step 5: Run all coin cap tests**

Run: `python -m pytest test_coincap.py -v --tb=short`
Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add data/fetcher.py data/exceptions.py data/coincap_fetcher.py test_coincap.py
git commit -m "feat: integrate CoinCap as tertiary OHLCV fallback in MultiExchangeFetcher"
```

---

### Task 5: Degraded-mode WebSocket backup

- [ ] **Step 1: Write test for degraded mode**

Add to `test_coincap.py`:

```python
@pytest.mark.asyncio
async def test_market_data_stream_degraded_mode():
    """When WebSocket reconnects fail, switch to CoinCap REST polling."""
    from data.stream import MarketDataStream
    from unittest.mock import AsyncMock, patch

    stream = MarketDataStream(symbols=["BTC/USDT"])

    with patch.object(stream, "_ws_connect", new_callable=AsyncMock) as mock_connect:
        mock_connect.side_effect = [Exception("fail"), Exception("fail"),
                                     Exception("fail")]  # 3 failures
        with patch.object(stream, "_fallback_to_coincap", new_callable=AsyncMock) as mock_fb:
            await stream._try_reconnect()
            assert mock_fb.called
```

- [ ] **Step 2: Modify MarketDataStream**

In `data/stream.py`:

- Add reconnect counter (track consecutive fails)
- On 3rd consecutive reconnect fail, call `_fallback_to_coincap()`
- `_fallback_to_coincap()` polls `CoinCapFetcher.get_price()` every 10s
- Emit `PRICE_SOURCE_DEGRADED` event to EventBus
- When WebSocket reconnects, restore primary mode

- [ ] **Step 3: Commit**

```bash
git add data/stream.py test_coincap.py
git commit -m "feat: add CoinCap degraded-mode backup for WebSocket disconnects"
```

---

### Task 6: Add price_source_check to AnomalyDetector

- [ ] **Step 1: Write the failing test**

Add to `test_coincap.py`:

```python
from monitoring.anomaly_detector import AnomalyDetector
from agents.risk_manager import CircuitBreakerState


@pytest.mark.asyncio
async def test_price_source_check_trips_circuit_breaker_when_all_down():
    """When both Binance and CoinCap fail for 30+ seconds, trip circuit breaker."""
    CircuitBreakerState.reset()

    detector = AnomalyDetector()

    with patch.object(detector, "_check_price_source", return_value=False):
        result = await detector._check_price_source()

    assert result is False
    assert CircuitBreakerState.is_halted()
    CircuitBreakerState.reset()


@pytest.mark.asyncio
async def test_price_source_check_passes_when_one_source_up():
    """When at least one price source responds, circuit breaker stays open."""
    CircuitBreakerState.reset()

    detector = AnomalyDetector()

    with patch.object(detector, "_check_price_source", return_value=True):
        result = await detector._check_price_source()

    assert result is True
    assert not CircuitBreakerState.is_halted()
```

- [ ] **Step 2: Implement the price source check**

In `monitoring/anomaly_detector.py`, add check method (insert after existing checks around line ~120):

```python
async def _check_price_source(self) -> bool:
    """Verify at least one price source is responding.

    Returns True if healthy, False if all sources failed.
    Trips circuit breaker if all sources are down.
    """
    binance_ok = False
    coincap_ok = False

    # Check Binance WebSocket
    if self._market_data_stream:
        try:
            binance_ok = self._market_data_stream.is_connected
        except Exception:
            pass

    # Check CoinCap REST
    from data.coincap_fetcher import CoinCapFetcher
    cf = CoinCapFetcher()
    try:
        price = await cf.get_price("bitcoin")
        coincap_ok = price is not None
    except Exception:
        pass

    if not binance_ok:
        logger.warning("Binance WebSocket not connected")
    if not coincap_ok:
        logger.warning("CoinCap REST not responding")

    all_down = not binance_ok and not coincap_ok
    if all_down:
        logger.critical("ALL PRICE SOURCES DOWN — tripping circuit breaker")
        CircuitBreakerState.halt(reason="All price sources unavailable")
        if self._event_bus:
            await self._event_bus.publish("price_source_critical", {
                "error": "All price sources failed",
                "binance_ok": binance_ok,
                "coincap_ok": coincap_ok,
            })
    return not all_down
```

Add call to this check in the main `run_checks` loop (around line ~60):

```python
async def run_checks(self):
    """Run all anomaly checks every MONITOR_INTERVAL seconds."""
    while True:
        try:
            await self._check_price_source()  # NEW
            # ... existing checks ...
        except Exception as exc:
            logger.error("Anomaly check failed: %s", exc)
        await asyncio.sleep(MONITOR_INTERVAL)
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest test_coincap.py -v --tb=short`
Expected: price_source_check tests PASS

- [ ] **Step 4: Commit**

```bash
git add monitoring/anomaly_detector.py test_coincap.py
git commit -m "feat: add price_source_check to AnomalyDetector with CoinCap fallback monitoring"
```
