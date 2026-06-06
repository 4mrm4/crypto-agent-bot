"""Price feed protocol for trade execution.

Defines a PriceFeed interface that MarketDataFetcher and mock price
feeds implement, so execution modules (PaperTrader, etc.) depend on an
abstract price source rather than on MarketDataFetcher directly.
"""

from typing import Optional, Protocol

import pandas as pd


class PriceFeed(Protocol):
    """Interface for price data sources used by execution modules.

    MarketDataFetcher is the production implementation. Mocks and test
    fixtures implement this protocol structurally.
    """

    def fetch_ohlcv(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV candles and return a pandas DataFrame.

        Columns: timestamp, open, high, low, close, volume (at minimum).
        """
        ...
