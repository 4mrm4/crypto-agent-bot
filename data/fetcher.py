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