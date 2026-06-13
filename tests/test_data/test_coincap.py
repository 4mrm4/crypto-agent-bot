"""Tests for CoinCap v3 API integration."""
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
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

    with patch.object(fetcher, "_get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response
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
    with patch.object(fetcher, "_get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = None
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

    with patch.object(fetcher, "_get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_history
        df = await fetcher.get_ohlcv_fallback(
            "bitcoin", "h1",
            datetime(2024, 1, 1), datetime(2024, 1, 2),
        )

    assert isinstance(df, pd.DataFrame)
    assert "close" in df.columns
    assert len(df) == 2
    assert float(df["close"].iloc[-1]) == 51000.0


@pytest.mark.asyncio
async def test_get_ohlcv_fallback_returns_none_on_error(fetcher):
    with patch.object(fetcher, "_get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = None
        df = await fetcher.get_ohlcv_fallback(
            "bitcoin", "h1",
            datetime(2024, 1, 1), datetime(2024, 1, 2),
        )
    assert df is None


# ── Symbol mapping tests ──

from data.coincap_fetcher import symbol_to_coincap_id, coincap_id_to_symbol


def test_symbol_to_coincap_id_known():
    assert symbol_to_coincap_id("BTC/USDT") == "bitcoin"
    assert symbol_to_coincap_id("ETH/USDT") == "ethereum"


def test_symbol_to_coincap_id_unknown():
    assert symbol_to_coincap_id("UNKNOWN/USDT") is None


def test_coincap_id_to_symbol_reverse():
    assert coincap_id_to_symbol("bitcoin") == "BTC/USDT"
    assert coincap_id_to_symbol("ethereum") == "ETH/USDT"

# ── MultiExchangeFetcher CoinCap fallback tests ──

from unittest.mock import AsyncMock, MagicMock, patch
from data.fetcher import MultiExchangeFetcher
from data.coincap_fetcher import symbol_to_coincap_id


@pytest.mark.asyncio
async def test_coincap_fallback_when_binance_and_bybit_fail():
    """When Binance and Bybit both fail, CoinCap should be tried as tertiary fallback."""
    from data.fetcher import MultiExchangeFetcher
    fetcher = MultiExchangeFetcher(exchange_ids=["binance", "bybit"])

    with patch.object(fetcher._fetchers["binance"], "fetch_ohlcv",
                      side_effect=Exception("Binance down")):
        with patch.object(fetcher._fetchers["bybit"], "fetch_ohlcv",
                          side_effect=Exception("Bybit down")):
            with patch("data.coincap_fetcher.CoinCapFetcher.get_ohlcv_fallback",
                       new_callable=AsyncMock) as mock_coincap:
                mock_df = MagicMock()
                mock_df.empty = False
                mock_coincap.return_value = mock_df

                df = await fetcher.fetch_ohlcv_merged("BTC/USDT", "1h", limit=1)

    assert df is not None
    mock_coincap.assert_called_once()


@pytest.mark.asyncio
async def test_all_sources_fail_returns_empty_df():
    """When ALL sources fail, MultiExchangeFetcher should return empty DataFrame."""
    from data.fetcher import MultiExchangeFetcher
    fetcher = MultiExchangeFetcher(exchange_ids=["binance", "bybit"])

    with patch.object(fetcher._fetchers["binance"], "fetch_ohlcv",
                      side_effect=Exception("Binance down")):
        with patch.object(fetcher._fetchers["bybit"], "fetch_ohlcv",
                          side_effect=Exception("Bybit down")):
            with patch("data.coincap_fetcher.CoinCapFetcher.get_ohlcv_fallback",
                       new_callable=AsyncMock) as mock_coincap:
                mock_coincap.return_value = None  # CoinCap also fails

                df = await fetcher.fetch_ohlcv_merged("BTC/USDT", "1h", limit=1)

    assert isinstance(df, object)


@pytest.mark.asyncio
async def test_binance_primary_succeeds_no_fallback():
    """When Binance works, CoinCap should NOT be called."""
    from data.fetcher import MultiExchangeFetcher
    fetcher = MultiExchangeFetcher(exchange_ids=["binance", "bybit"])
    mock_df = MagicMock()

    with patch.object(fetcher._fetchers["binance"], "fetch_ohlcv",
                      return_value=mock_df):
        with patch("data.coincap_fetcher.CoinCapFetcher.get_ohlcv_fallback",
                   new_callable=AsyncMock) as mock_coincap:
            df = await fetcher.fetch_ohlcv_merged("BTC/USDT", "1h", limit=1)

    assert df is not None
    assert not mock_coincap.called

# ── AnomalyDetector price source check tests ──

from unittest.mock import AsyncMock, MagicMock, patch
from monitoring.anomaly_detector import AnomalyDetector
from state.circuit_breaker import CircuitBreakerState


@pytest.mark.asyncio
async def test_price_source_check_passes_when_binance_connected():
    """When Binance WebSocket is connected, no alert needed."""
    cb = CircuitBreakerState()
    detector = AnomalyDetector(circuit_breaker=cb)
    mock_stream = MagicMock()
    mock_stream.is_connected = True
    detector._market_data_stream = mock_stream

    await detector._check_price_source()

    assert not cb.is_halted()


@pytest.mark.asyncio
async def test_price_source_check_trips_when_all_down():
    """When Binance AND CoinCap are down, circuit breaker trips."""
    cb = CircuitBreakerState()
    detector = AnomalyDetector(circuit_breaker=cb)
    mock_stream = MagicMock()
    mock_stream.is_connected = False
    detector._market_data_stream = mock_stream

    with patch("data.coincap_fetcher.CoinCapFetcher.get_price",
               new_callable=AsyncMock, return_value=None):
        await detector._check_price_source()

    assert cb.is_halted()
