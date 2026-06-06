"""Sentiment and news data fetchers for crypto markets."""
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import httpx
from config import settings

logger = logging.getLogger(__name__)

# Weight distribution for combined sentiment scoring
# Weights are expressed as percentages (sum = 100 when all sources active)
SENTIMENT_WEIGHTS = {
    "fear_greed": 40,
    "cryptopanic": 20,
    "santiment_social": 25,
    "santiment_balance": 15,
}

_SANTIMENT_WEIGHT_KEYS = ["santiment_social", "santiment_balance"]


@dataclass
class CombinedSentiment:
    """Aggregated sentiment from multiple sources."""
    fear_greed_index: Optional[int] = None
    cryptopanic_sentiment: Optional[float] = None
    santiment_social_volume: Optional[float] = None
    santiment_sentiment_balance: Optional[float] = None
    santiment_dev_activity: Optional[float] = None
    overall_score: float = 0.5  # 0.0–1.0, default neutral
    signal_count: int = 0
    fetched_at: datetime = field(default_factory=datetime.utcnow)


class SentimentFetcher:

    def get_fear_greed_index(self) -> dict:
        """Returns {'value': int, 'classification': str} from alternative.me"""
        try:
            r = httpx.get("https://api.alternative.me/fng/?limit=1", timeout=10)
            data = r.json()["data"][0]
            return {
                "value": int(data["value"]),
                "classification": data["value_classification"],
            }
        except Exception as e:
            logger.warning("Fear/Greed fetch failed: %s", e)
            return {"value": 50, "classification": "Neutral"}

    def get_cryptopanic_news(self, currency: str = "BTC", limit: int = 10) -> list:
        """Fetch recent news. Requires CRYPTOPANIC_API_KEY in .env (optional)."""
        api_key = getattr(settings, "CRYPTOPANIC_API_KEY", None)
        if not api_key:
            logger.info(
                "CryptoPanic news skipped — set CRYPTOPANIC_API_KEY in .env to enable"
            )
            return []
        try:
            url = (
                f"https://cryptopanic.com/api/v1/posts/"
                f"?auth_token={api_key}&currencies={currency}&kind=news&limit={limit}"
            )
            r = httpx.get(url, timeout=10)
            items = r.json().get("results", [])
            return [
                {
                    "title": i.get("title", ""),
                    "published_at": i.get("published_at", ""),
                    "url": i.get("url", ""),
                    "votes_positive": i.get("votes", {}).get("positive", 0),
                    "votes_negative": i.get("votes", {}).get("negative", 0),
                }
                for i in items
            ]
        except Exception as e:
            logger.warning("CryptoPanic fetch failed: %s", e)
            return []

    def score_sentiment(self, news_items: list, fear_greed: Optional[dict] = None) -> float:
        """
        Returns sentiment score from -1.0 (very bearish) to +1.0 (very bullish).
        Combines fear/greed index with news vote ratios using SENTIMENT_WEIGHTS.
        """
        score = 0.0
        total_weight = 0

        if fear_greed:
            total_weight += SENTIMENT_WEIGHTS["fear_greed"]
            fg_val = fear_greed.get("value", 50)
            score += (fg_val - 50) / 50.0 * (SENTIMENT_WEIGHTS["fear_greed"] / 100.0)

        if news_items:
            total_weight += SENTIMENT_WEIGHTS["cryptopanic"]
            vote_scores = []
            for item in news_items:
                pos = item.get("votes_positive", 0)
                neg = item.get("votes_negative", 0)
                total = pos + neg
                if total > 0:
                    vote_scores.append((pos - neg) / total)
            if vote_scores:
                score += (sum(vote_scores) / len(vote_scores)) * (SENTIMENT_WEIGHTS["cryptopanic"] / 100.0)

        # Redistribute proportionally if some sources missing
        if total_weight > 0 and total_weight < 100:
            score /= (total_weight / 100.0)

        return max(-1.0, min(1.0, score))

    def get_full_sentiment_report(self, symbol: str = "BTC") -> dict:
        """Get complete sentiment snapshot."""
        currency = symbol.replace("/USDT", "").replace("/BTC", "")
        fg = self.get_fear_greed_index()
        news = self.get_cryptopanic_news(currency)
        score = self.score_sentiment(news, fg)
        return {
            "symbol": symbol,
            "fear_greed": fg,
            "news": news[:5],
            "score": round(score, 3),
            "bias": "bullish" if score > 0.2 else "bearish" if score < -0.2 else "neutral",
        }

    def get_cryptopanic_sentiment_score(self, currency: str = "BTC") -> Optional[float]:
        """Return float 0.0-1.0 from CryptoPanic news, or None if unavailable."""
        news = self.get_cryptopanic_news(currency, limit=20)
        if not news:
            return None
        pos = sum(
            n.get("votes_positive", 0) for n in news
        )
        neg = sum(
            n.get("votes_negative", 0) for n in news
        )
        total = pos + neg
        if total == 0:
            return None
        ratio = pos / total  # 0.0-1.0
        return round(ratio, 4)

    async def _fetch_santiment_async(self, slug: str = "bitcoin") -> Optional[object]:
        """Fetch Santiment signal asynchronously."""
        if not getattr(settings, "SANTIMENT_ENABLED", False):
            return None
        try:
            from data.santiment_fetcher import SantimentFetcher
            fetcher = SantimentFetcher()
            signal = await fetcher.get_signal(slug)
            return signal
        except Exception as exc:
            logger.warning("Santiment fetch failed: %s", exc)
            return None

    def _fetch_santiment(self, slug: str = "bitcoin") -> Optional[object]:
        """Fetch Santiment signal synchronously (wraps async)."""
        try:
            loop = asyncio.get_running_loop()
            future = asyncio.run_coroutine_threadsafe(
                self._fetch_santiment_async(slug), loop
            )
            return future.result(timeout=30)
        except RuntimeError:
            # No running loop — use asyncio.run()
            return asyncio.run(self._fetch_santiment_async(slug))
        except Exception as exc:
            logger.warning("Santiment fetch failed: %s", exc)
            return None

    async def get_combined_sentiment(
        self, slug: str = "bitcoin", currency: str = "BTC"
    ) -> CombinedSentiment:
        """
        Aggregate all sentiment sources into a single CombinedSentiment.

        Weight distribution (when all sources active):
          - Fear & Greed:  40%
          - CryptoPanic:   20%
          - Santiment social: 25%
          - Santiment balance: 15%

        Weights redistribute proportionally when a source is unavailable.
        """
        # Fetch all sources
        fg = self.get_fear_greed_index()
        cp_score = self.get_cryptopanic_sentiment_score(currency)
        santiment_signal = await self._fetch_santiment_async(slug)

        fg_value = fg.get("value", 50) if fg else 50
        fg_normalized = fg_value / 100.0  # 0.0-1.0

        # Build active weights list
        active_weights = {}
        if fg:
            active_weights["fear_greed"] = SENTIMENT_WEIGHTS["fear_greed"]
        if cp_score is not None:
            active_weights["cryptopanic"] = SENTIMENT_WEIGHTS["cryptopanic"]
        if santiment_signal is not None:
            if santiment_signal.social_volume_24h is not None:
                active_weights["santiment_social"] = SENTIMENT_WEIGHTS["santiment_social"]
            if santiment_signal.sentiment_balance_24h is not None:
                active_weights["santiment_balance"] = SENTIMENT_WEIGHTS["santiment_balance"]

        total_weight = sum(active_weights.values())
        signal_count = len(active_weights)

        # Compute overall score
        overall = 0.5  # default neutral
        if total_weight > 0:
            score = 0.0
            for key, weight in active_weights.items():
                normalized_weight = weight / total_weight
                if key == "fear_greed":
                    score += normalized_weight * fg_normalized
                elif key == "cryptopanic":
                    score += normalized_weight * (cp_score or 0.5)
                elif key == "santiment_social":
                    # Map social volume to 0-1 using sigmoid-like scaling
                    vol = santiment_signal.social_volume_24h or 0
                    social_score = min(1.0, vol / 5000.0)
                    score += normalized_weight * social_score
                elif key == "santiment_balance":
                    # sentiment_balance ranges roughly -1 to 1, map to 0-1
                    bal = santiment_signal.sentiment_balance_24h or 0
                    bal_score = (bal + 1) / 2
                    score += normalized_weight * bal_score
            overall = round(max(0.0, min(1.0, score)), 4)

        return CombinedSentiment(
            fear_greed_index=fg_value if fg else None,
            cryptopanic_sentiment=cp_score,
            santiment_social_volume=(
                santiment_signal.social_volume_24h
                if santiment_signal else None
            ),
            santiment_sentiment_balance=(
                santiment_signal.sentiment_balance_24h
                if santiment_signal else None
            ),
            santiment_dev_activity=(
                santiment_signal.dev_activity_30d
                if santiment_signal else None
            ),
            overall_score=overall,
            signal_count=signal_count,
        )

    def get_combined_sentiment_sync(
        self, slug: str = "bitcoin", currency: str = "BTC"
    ) -> CombinedSentiment:
        """Synchronous wrapper for get_combined_sentiment."""
        try:
            loop = asyncio.get_running_loop()
            future = asyncio.run_coroutine_threadsafe(
                self.get_combined_sentiment(slug, currency), loop
            )
            return future.result(timeout=30)
        except RuntimeError:
            # No running loop — use asyncio.run()
            return asyncio.run(self.get_combined_sentiment(slug, currency))