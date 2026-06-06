"""Analyst agent — gathers market data and sentiment, produces market analysis."""

import logging
from typing import Optional

import pandas as pd
from langchain_core.tools import Tool

from agents.base import BaseAgent
from data.fetcher import MarketDataFetcher

logger = logging.getLogger(__name__)

ANALYST_SYSTEM_PROMPT = """You are a skilled crypto market analyst. Your job is to:
1. Fetch current OHLCV data for the given symbol
2. Fetch the current price
3. Analyse price action — note trend direction, volatility, volume patterns
4. Check sentiment (if available)
5. Produce a concise market analysis with key levels and bias

Be specific with numbers and percentages. Always reference the data you fetched.
IMPORTANT: Use ONLY plain ASCII text. No emoji, no Unicode symbols."""


class AnalystAgent(BaseAgent):
    """Specialised agent that analyses market conditions using live data."""

    def __init__(self, fetcher: Optional[MarketDataFetcher] = None):
        self._fetcher = fetcher or MarketDataFetcher()
        tools = self._build_tools()
        super().__init__(
            name="analyst",
            tools=tools,
            system_prompt=ANALYST_SYSTEM_PROMPT,
        )

    def _build_tools(self):
        def fetch_ohlcv_fn(symbol_timeframe_limit: str = "") -> str:
            """Fetch OHLCV candles. Pass as 'SYMBOL TIMEFRAME LIMIT' or blank for defaults."""
            parts = symbol_timeframe_limit.strip().split()
            symbol = parts[0] if len(parts) > 0 else None
            timeframe = parts[1] if len(parts) > 1 else None
            limit = int(parts[2]) if len(parts) > 2 else None
            df = self._fetcher.fetch_ohlcv(symbol, timeframe, limit)
            return df.tail(20).to_string()

        def fetch_price_fn(symbol: str = "") -> str:
            """Fetch current spot price. Pass a symbol like 'BTC/USDT' or blank for default."""
            s = symbol.strip() or None
            price = self._fetcher.fetch_current_price(s)
            return f"Current price: ${price:,.2f}"

        def sentiment_fn(_dummy: str = "") -> str:
            """Get crypto market sentiment using real sentiment data."""
            try:
                from data.sentiment import SentimentFetcher
                sf = SentimentFetcher()
                result = sf.get_combined_sentiment_sync()
                if result is None:
                    return (
                        "Sentiment analysis: NEUTRAL (score 0.0/1.0). "
                        "Sentiment data temporarily unavailable."
                    )
                return (
                    f"Sentiment analysis: overall score {result.overall_score:.2f}/1.0. "
                    f"Fear & Greed Index: {result.fear_greed_index or 'N/A'}. "
                    f"CryptoPanic sentiment: {result.cryptopanic_sentiment or 'N/A'}. "
                    f"Signal count: {result.signal_count}."
                )
            except Exception as exc:
                return (
                    f"Sentiment analysis: UNAVAILABLE ({exc}). "
                    "Using neutral default."
                )

        return [
            Tool(name="fetch_ohlcv", func=fetch_ohlcv_fn,
                 description="Fetch OHLCV candles. Args: 'SYMBOL TIMEFRAME LIMIT' e.g. 'BTC/USDT 1h 100'"),
            Tool(name="fetch_current_price", func=fetch_price_fn,
                 description="Fetch current spot price. Args: 'SYMBOL' e.g. 'BTC/USDT'"),
            Tool(name="get_market_sentiment", func=sentiment_fn,
                 description="Get crypto market sentiment score"),
        ]