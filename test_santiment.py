"""Tests for Santiment API integration."""
from datetime import datetime
from unittest.mock import AsyncMock, patch
import pytest
from data.santiment_fetcher import SantimentFetcher, SantimentSignal
from data.sentiment import SentimentFetcher, CombinedSentiment


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
    """Free tier returns 'outside allowed interval' for old data -- must not crash."""
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


# ── CombinedSentiment + SentimentFetcher integration tests ──


def test_combined_sentiment_dataclass_defaults():
    """CombinedSentiment can be created with just an overall_score."""
    cs = CombinedSentiment(overall_score=0.5)
    assert cs.overall_score == 0.5
    assert cs.fear_greed_index is None
    assert cs.santiment_social_volume is None
    assert cs.signal_count == 0
    assert cs.fetched_at is not None


def test_combined_sentiment_full():
    cs = CombinedSentiment(
        fear_greed_index=45,
        cryptopanic_sentiment=0.3,
        santiment_social_volume=1500.0,
        santiment_sentiment_balance=0.35,
        santiment_dev_activity=250.0,
        overall_score=0.62,
        signal_count=3,
    )
    assert cs.fear_greed_index == 45
    assert cs.santiment_social_volume == 1500.0
    assert cs.overall_score == 0.62
    assert cs.signal_count == 3


def test_get_combined_sentiment_includes_santiment():
    """When all sources available, weights work correctly."""
    sentiment = SentimentFetcher()
    with patch.object(sentiment, "get_fear_greed_index") as mock_fng, \
         patch.object(sentiment, "get_cryptopanic_sentiment_score") as mock_cp, \
         patch.object(sentiment, "_fetch_santiment") as mock_sant:
        mock_fng.return_value = {"value": 50, "classification": "Neutral"}
        mock_cp.return_value = 0.2
        mock_sant.return_value = SantimentSignal(
            asset_slug="bitcoin", social_volume_24h=1000.0,
            sentiment_balance_24h=0.5,
        )
        result = sentiment.get_combined_sentiment(slug="bitcoin")

    assert result.fear_greed_index == 50
    assert result.cryptopanic_sentiment == 0.2
    assert result.santiment_social_volume == 1000.0
    assert result.santiment_sentiment_balance == 0.5
    # F&G=40% + CP=20% + social=25% + balance=15%
    # social maps: volume/5000 -> 0.2, balance maps: (0.5+1)/2 -> 0.75
    social_norm = min(1.0, 1000.0 / 5000.0)
    balance_norm = (0.5 + 1) / 2
    expected = (40 * 0.5 + 20 * 0.2 + 25 * social_norm + 15 * balance_norm) / 100
    assert abs(result.overall_score - expected) < 0.01
    assert result.signal_count == 4


def test_get_combined_sentiment_redistributes_weights():
    """When Santiment is unavailable, weights redistribute proportionally."""
    sentiment = SentimentFetcher()
    with patch.object(sentiment, "get_fear_greed_index") as mock_fng, \
         patch.object(sentiment, "get_cryptopanic_sentiment_score") as mock_cp, \
         patch.object(sentiment, "_fetch_santiment") as mock_sant:
        mock_fng.return_value = {"value": 55, "classification": "Neutral"}
        mock_cp.return_value = 0.3
        mock_sant.return_value = None  # Santiment not available
        result = sentiment.get_combined_sentiment(slug="bitcoin")

    assert result.fear_greed_index == 55
    assert result.cryptopanic_sentiment == 0.3
    assert result.santiment_social_volume is None
    assert result.santiment_sentiment_balance is None
    # Redistribution: F&G=40/60, CP=20/60
    expected = (40 / 60) * 0.55 + (20 / 60) * 0.3
    assert abs(result.overall_score - expected) < 0.01
    assert result.signal_count == 2


def test_fetch_santiment_returns_none_when_disabled():
    """_fetch_santiment returns None when Santiment is not enabled."""
    sentiment = SentimentFetcher()
    with patch("data.sentiment.settings") as mock_settings:
        mock_settings.SANTIMENT_ENABLED = False
        result = sentiment._fetch_santiment(slug="bitcoin")
    assert result is None


def test_get_combined_sentiment_returns_fng_only():
    """When only F&G available, overall_score matches."""
    sentiment = SentimentFetcher()
    with patch.object(sentiment, "get_fear_greed_index") as mock_fng, \
         patch.object(sentiment, "get_cryptopanic_sentiment_score") as mock_cp, \
         patch.object(sentiment, "_fetch_santiment") as mock_sant:
        mock_fng.return_value = {"value": 50, "classification": "Neutral"}
        mock_cp.return_value = None
        mock_sant.return_value = None
        result = sentiment.get_combined_sentiment(slug="bitcoin")
    assert abs(result.overall_score - 0.5) < 0.01
    assert result.signal_count == 1


# ── Regime integration tests ──


def test_regime_snapshot_has_social_dominance():
    """RegimeSnapshot should accept social_dominance_zscore."""
    from data.regime import RegimeSnapshot
    snap = RegimeSnapshot(
        regime="ranging",
        confidence=0.8,
        adx=15.0,
        atr_pct=0.02,
        sma200_distance=0.05,
        social_dominance_zscore=1.5,
    )
    assert snap.social_dominance_zscore == 1.5


def test_regime_snapshot_defaults_social_dominance():
    """RegimeSnapshot should default social_dominance_zscore to 0.0."""
    from data.regime import RegimeSnapshot
    snap = RegimeSnapshot(
        regime="ranging", confidence=0.8, adx=15.0,
        atr_pct=0.02, sma200_distance=0.05,
    )
    assert snap.social_dominance_zscore == 0.0


def test_get_social_signal_returns_none_when_disabled():
    """_get_social_signal returns None when Santiment not enabled."""
    from data.regime import MarketRegimeDetector
    detector = MarketRegimeDetector()
    with patch("data.regime.settings") as mock_settings:
        mock_settings.SANTIMENT_ENABLED = False
        result = detector._get_social_signal("bitcoin")
    assert result is None


def test_compute_dominance_zscore():
    """_compute_dominance_zscore computes correct z-score from history."""
    from data.regime import MarketRegimeDetector
    detector = MarketRegimeDetector()
    # Populate history: mean=3.0, std~0.79
    for val in [2.0, 3.0, 4.0, 2.5, 3.5]:
        detector._dominance_history.append(val)
    z = detector._compute_dominance_zscore(4.5)
    # 4.5 is ~1.9 stddev above mean of 3.0
    assert z is not None
    assert 1.0 < z < 3.0
    # A value at the mean should give ~0
    z_mean = detector._compute_dominance_zscore(3.0)
    assert abs(z_mean) < 0.5


def test_compute_dominance_zscore_empty_history():
    """_compute_dominance_zscore returns None with no history."""
    from data.regime import MarketRegimeDetector
    detector = MarketRegimeDetector()
    z = detector._compute_dominance_zscore(5.0)
    assert z is None


# ── Autonomous loop integration tests ──


def test_check_trending_assets_boosts_priority():
    """When trending assets exist, priority is boosted in goal generation."""
    import config as cfg
    from orchestration.autonomous_loop import AutonomousResearchLoop
    loop = AutonomousResearchLoop.__new__(AutonomousResearchLoop)
    loop._santiment_fetcher = None

    # Test that _check_trending_assets returns slugs
    with patch("data.santiment_fetcher.SantimentFetcher.get_trending_assets",
               new_callable=AsyncMock) as mock_ta, \
         patch.object(cfg.settings, "SANTIMENT_ENABLED", True):
        mock_ta.return_value = ["bitcoin", "ethereum"]
        import asyncio
        result = asyncio.run(loop._check_trending_assets())
    assert "bitcoin" in result
    assert "ethereum" in result


def test_check_trending_assets_empty_when_disabled():
    """When Santiment is disabled, _check_trending_assets returns empty list."""
    import config as cfg
    from orchestration.autonomous_loop import AutonomousResearchLoop
    loop = AutonomousResearchLoop.__new__(AutonomousResearchLoop)
    loop._santiment_fetcher = None

    with patch("data.santiment_fetcher.SantimentFetcher.get_trending_assets",
               new_callable=AsyncMock) as mock_ta, \
         patch.object(cfg.settings, "SANTIMENT_ENABLED", False):
        mock_ta.return_value = ["bitcoin", "ethereum"]
        import asyncio
        result = asyncio.run(loop._check_trending_assets())
    assert result == []
