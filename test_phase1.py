"""Phase 1 test — verify data fetcher works with live exchange data."""

import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def main():
    # Allow running from project root or tests/
    sys.path.insert(0, ".")

    from data.fetcher import MarketDataFetcher

    fetcher = MarketDataFetcher()

    print("\n=== Fetching 100 BTC/USDT 1h candles ===")
    df = fetcher.fetch_ohlcv(symbol="BTC/USDT", timeframe="1h", limit=100)
    print(df.head(10))
    print(f"\nShape: {df.shape}")
    print(f"Date range: {df.index[0]} -> {df.index[-1]}")

    print("\n=== Fetching current spot price ===")
    price = fetcher.fetch_current_price("BTC/USDT")
    print(f"BTC/USDT current price: ${price:,.2f}")

    print("\n[OK] Phase 1 test passed!")


if __name__ == "__main__":
    main()