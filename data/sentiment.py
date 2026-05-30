"""Sentiment and news data fetchers for crypto markets."""
import logging
from typing import Optional
import httpx
from config import settings

logger = logging.getLogger(__name__)


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
            return self._get_coingecko_news(currency)
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

    def _get_coingecko_news(self, currency: str = "BTC") -> list:
        """Fallback: CoinGecko news feed (no key needed)."""
        try:
            symbol_map = {"BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana"}
            coin_id = symbol_map.get(currency.upper(), currency.lower())
            r = httpx.get(
                f"https://api.coingecko.com/api/v3/news",
                params={"category": coin_id},
                timeout=10
            )
            items = r.json().get("data", [])[:10]
            return [
                {"title": i.get("title", ""), "published_at": i.get("created_at", ""), "url": i.get("url", "")}
                for i in items
            ]
        except Exception as e:
            logger.warning("CoinGecko news fetch failed: %s", e)
            return []

    def score_sentiment(self, news_items: list, fear_greed: Optional[dict] = None) -> float:
        """
        Returns sentiment score from -1.0 (very bearish) to +1.0 (very bullish).
        Combines fear/greed index with news vote ratios.
        """
        score = 0.0

        if fear_greed:
            fg_val = fear_greed.get("value", 50)
            # Map 0-100 to -1 to +1
            score += (fg_val - 50) / 50.0 * 0.6  # 60% weight

        if news_items:
            vote_scores = []
            for item in news_items:
                pos = item.get("votes_positive", 0)
                neg = item.get("votes_negative", 0)
                total = pos + neg
                if total > 0:
                    vote_scores.append((pos - neg) / total)
            if vote_scores:
                score += (sum(vote_scores) / len(vote_scores)) * 0.4  # 40% weight

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