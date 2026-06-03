"""Tests for Santiment API integration."""
from datetime import datetime
from unittest.mock import AsyncMock, patch
import pytest
from data.santiment_fetcher import SantimentFetcher, SantimentSignal


@pytest.fixture
def fetcher():
    return SantimentFetcher(api_key="test_key", enabled=True)


@pytest.mark.asyncio
async def test_get_signal_returns_correct_dataclass(fetcher):
    """Test get_signal returns SantimentSignal from mocked API response."""
    with patch.object(fetcher, "_fetch_metric_value", new_callable=AsyncMock) as mock_fmv:
        mock_fmv.side_effect = [1500.0, 0.35, 250.0, 2.5, 800000]
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

    with patch.object(fetcher, "_query", new_callable=AsyncMock) as mock_q:
        mock_q.return_value = mock_error

        result = await fetcher.get_signal("bitcoin", days=365)

    assert result is None


@pytest.mark.asyncio
async def test_get_signal_returns_none_on_api_error(fetcher):
    with patch.object(fetcher, "_fetch_metric_value", new_callable=AsyncMock) as mock_fmv:
        mock_fmv.side_effect = Exception("Connection failed")
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
    with patch.object(fetcher, "_query", new_callable=AsyncMock) as mock_q:
        mock_q.return_value = mock_response
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


@pytest.mark.asyncio
async def test_get_trending_assets_returns_empty_on_error(fetcher):
    with patch.object(fetcher, "_query", new_callable=AsyncMock) as mock_q:
        mock_q.return_value = None
        trending = await fetcher.get_trending_assets()
    assert trending == []
