"""Live market data fetcher using the CCXT exchange library.

Provides MarketDataFetcher (single exchange) and MultiExchangeFetcher
(multiple exchanges) wrappers around CCXT. Supports OHLCV historical data
fetching, real-time ticker polling, exchange config from settings, and
configurable rate-limiting. Primary data source for backtesting and live
trading pipelines.
"""

import logging
from typing import Optional

import ccxt
import pandas as pd

from config import settings

logger = logging.getLogger(__name__)


class MarketDataFetcher:
    """Fetches OHLCV and ticker data from a cryptocurrency exchange via CCXT."""

    def __init__(self, exchange_id: Optional[str] = None) -> None:
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

# CoinCap interval mapping
_INTERVAL_MAP = {
    "1m": "m1", "5m": "m5", "15m": "m15",
    "30m": "m30", "1h": "h1", "4h": "h4",
    "1d": "d1", "1w": "w1",
}


def _interval_map(timeframe: str) -> str:
    """Convert CCXT timeframe format to CoinCap interval format."""
    return _INTERVAL_MAP.get(timeframe, "h1")


def symbol_to_coincap_id(symbol: str) -> Optional[str]:
    """Convert CCXT pair like 'BTC/USDT' to CoinCap asset ID like 'bitcoin'.

    Uses the mapping from coincap_fetcher if available, else falls back to
    stripping /USDT and lowercasing.
    """
    try:
        from data.coincap_fetcher import SYMBOL_TO_COINCAP
        return SYMBOL_TO_COINCAP.get(symbol)
    except ImportError:
        pass
    return symbol.split("/")[0].lower() if "/" in symbol else symbol.lower()


class MultiExchangeFetcher:
    """Wraps multiple CCXT instances for best-price routing and fallback."""

    def __init__(self, exchange_ids: Optional[list] = None) -> None:
        self._exchange_ids = exchange_ids or ["kraken", "binance"]
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

    async def fetch_ohlcv_merged(self, symbol: str, timeframe: str = "1h", limit: int = 500) -> pd.DataFrame:
        """Return OHLCV with fallback chain: Binance -> Bybit -> CoinCap.

        CoinCap is the tertiary fallback when both CCXT exchanges fail.
        """
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

        # Tertiary fallback: CoinCap REST API
        coincap_id = symbol_to_coincap_id(symbol)
        if coincap_id:
            from data.coincap_fetcher import CoinCapFetcher
            cf = CoinCapFetcher()
            df = await cf.get_ohlcv_fallback(
                coincap_id, interval=_interval_map(timeframe),
            )
            if df is not None and not df.empty:
                logger.info("Using CoinCap fallback for %s", symbol)
                return df

        logger.error("All data sources failed for %s", symbol)
        return pd.DataFrame()