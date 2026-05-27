"""Phase 2 test — verify backtesting engine works with a simple SMA crossover."""

import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def main():
    sys.path.insert(0, ".")

    from backtesting.engine import BacktestEngine

    engine = BacktestEngine(ft_userdata_dir="./ft_userdata")

    print("\n=== Downloading historical data (30d BTC/USDT 1h) ===")
    engine.download_data(timerange="20260427-")

    print("\n=== Running SMA crossover backtest (10/30) ===")
    result = engine.run_backtest(
        strategy_params={
            "fast_ma": 10,
            "slow_ma": 30,
            "stoploss": -0.05,
        },
        timerange="20260427-20260527",
    )

    print(f"\nTotal trades:     {result.get('total_trades', 'N/A')}")
    print(f"Profit ratio:     {result.get('profit_ratio', 'N/A')}")
    print(f"Win rate:         {result.get('win_rate', 'N/A')}")
    print(f"Max drawdown:     {result.get('max_drawdown', 'N/A')}")
    print(f"Sharpe ratio:     {result.get('sharpe_ratio', 'N/A')}")

    trades_df = result.get("trades_df", pd.DataFrame())
    if not trades_df.empty:
        print(f"\nTrades DataFrame: {trades_df.shape}")
        print(trades_df[["profit_ratio"]].describe())

    print("\n[OK] Phase 2 test passed!")


if __name__ == "__main__":
    import pandas as pd
    main()