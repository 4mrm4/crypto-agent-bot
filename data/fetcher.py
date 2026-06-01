"""Live market data fetcher using the CCXT exchange library."""

import logging
from typing import Optional

import ccxt
import pandas as pd

from config import settings

logger = logging.getLogger(__name__)


class MarketDataFetcher:
    """Fetches OHLCV and ticker data from a cryptocurrency exchange via CCXT."""

    def __init__(self, exchange_id: Optional[str] = None):
        self.exchange_id = exchange_id or settings.EXCHANGE_ID
        self._exchange: Optional[ccxt.Exchange] = None

    @property
    def exchange(self) -> ccxt.Exchange:
        """Lazy-initialised CCXT exchange instance with rate-limit awareness."""
        if self._exchange is None:
            exchange_class = getattr(ccxt, self.exchange_id)
            self._exchange = exchange_class(
                {
                    "enableRateLimit": True,
                    "options": {"defaultType": "spot"},
                }
            )
            self._exchange.load_markets()
            logger.info(
                "Connected to %s — %d markets loaded.",
                self.exchange_id,
                len(self._exchange.markets),
            )
        return self._exchange

    def fetch_ohlcv(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV candles and return a pandas DataFrame.

        Columns: timestamp, open, high, low, close, volume.
        """
        symbol = symbol or settings.SYMBOL
        timeframe = timeframe or settings.TIMEFRAME
        limit = limit or settings.DATA_LIMIT

        logger.info(
            "Fetching %d %s candles for %s from %s …",
            limit,
            timeframe,
            symbol,
            self.exchange_id,
        )

        try:
            raw = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            self.exchange.sleep(self.exchange.rateLimit / 1000)  # be polite
        except ccxt.NetworkError as exc:
            logger.error("Network error fetching OHLCV: %s", exc)
            raise
        except ccxt.ExchangeError as exc:
            logger.error("Exchange error fetching OHLCV: %s", exc)
            raise

        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df

    def fetch_current_price(self, symbol: Optional[str] = None) -> float:
        """Return the latest ticker price for *symbol*."""
        symbol = symbol or settings.SYMBOL
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            self.exchange.sleep(self.exchange.rateLimit / 1000)
            price = ticker["last"]
            logger.info("Current %s price: %.2f", symbol, price)
            return price
        except ccxt.NetworkError as exc:
            logger.error("Network error fetching price: %s", exc)
            raise
        except ccxt.ExchangeError as exc:
            logger.error("Exchange error fetching price: %s", exc)
            raise


# ── Multi-exchange support ──


class MultiExchangeFetcher:
    """Wraps multiple CCXT instances for best-price routing and fallback."""

    def __init__(self, exchange_ids: Optional[list] = None):
        self._exchange_ids = exchange_ids or ["binance", "bybit"]
        self._fetchers: dict = {}
        for eid in self._exchange_ids:
            try:
                self._fetchers[eid] = MarketDataFetcher(exchange_id=eid)
            except Exception as exc:
                logger.warning("Could not initialise %s: %s", eid, exc)

    def fetch_best_price(self, symbol: str) -> dict:
        """Return the exchange with the best bid/ask spread."""
        prices = {}
        for name, fetcher in self._fetchers.items():
            try:
                exch = fetcher.exchange
                ticker = exch.fetch_ticker(symbol)
                ask = ticker.get("ask", 0)
                bid = ticker.get("bid", 0)
                spread = ((ask - bid) / bid) if bid > 0 else float("inf")
                prices[name] = {"bid": bid, "ask": ask, "spread_pct": spread}
            except Exception:
                pass
        if not prices:
            return {"exchange": "binance", "price": 0, "error": "No exchange available"}
        best = min(prices.items(), key=lambda x: x[1].get("spread_pct", float("inf")))
        return {"exchange": best[0], **best[1]}

    def fetch_ohlcv_merged(self, symbol: str, timeframe: str = "1h", limit: int = 500) -> pd.DataFrame:
        """Return OHLCV from primary exchange with fallback."""
        primary = self._fetchers.get("binance")
        if primary:
            try:
                return primary.fetch_ohlcv(symbol, timeframe, limit)
            except Exception:
                pass
        fallback = self._fetchers.get("bybit")
        if fallback:
            try:
                return fallback.fetch_ohlcv(symbol, timeframe, limit)
            except Exception:
                pass
        return pd.DataFrame()