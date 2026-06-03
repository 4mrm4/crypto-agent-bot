"""Tests for Messari API integration."""
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from data.messari_fetcher import MessariFetcher, MessariMetrics


@pytest.fixture
def fetcher():
    return MessariFetcher(api_key="test_key", enabled=True)


def _mock_response(data: dict, status: int = 200) -> MagicMock:
    """Create a sync mock response whose .json() returns data directly."""
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = data
    return resp


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

    client = await fetcher._get_client()
    with patch.object(client, "get") as mock_get:
        mock_get.return_value = _mock_response(mock_metrics)
        result = await fetcher.get_metrics("bitcoin")

    assert isinstance(result, MessariMetrics)
    assert result.asset == "bitcoin"
    assert result.price_usd == 50000.0
    assert result.active_addresses_24h == 800000
    assert result.developer_activity_30d == 450.0
    assert result.source == "messari"


@pytest.mark.asyncio
async def test_get_metrics_returns_none_on_api_error(fetcher):
    """Use unique slug to avoid cache hit from previous test."""
    client = await fetcher._get_client()
    with patch.object(client, "get") as mock_get:
        mock_get.side_effect = Exception("API error")
        result = await fetcher.get_metrics("unique_error_test_asset")
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
    client = await fetcher._get_client()
    with patch.object(client, "get") as mock_get:
        mock_get.return_value = _mock_response(mock_profile)
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


@pytest.mark.asyncio
async def test_get_trending_topics_returns_list(fetcher):
    mock_topics = [
        {"topic": "Bitcoin ETF", "mentions": 1500},
        {"topic": "DeFi Summer", "mentions": 800},
    ]
    client = await fetcher._get_client()
    with patch.object(client, "get") as mock_get:
        mock_get.return_value = _mock_response(mock_topics)
        result = await fetcher.get_trending_topics(limit=2)
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["topic"] == "Bitcoin ETF"


@pytest.mark.asyncio
async def test_get_profile_returns_none_on_api_error(fetcher):
    """Use unique slug to avoid cache hit from previous test."""
    client = await fetcher._get_client()
    with patch.object(client, "get") as mock_get:
        mock_get.side_effect = Exception("API error")
        result = await fetcher.get_profile("unique_profile_error_test")
    assert result is None


@pytest.mark.asyncio
async def test_close_cleans_up_client(fetcher):
    with patch("httpx.AsyncClient.aclose", new_callable=AsyncMock) as mock_close:
        client = await fetcher._get_client()
        await fetcher.close()
        mock_close.assert_called_once()
        assert fetcher._client is None
