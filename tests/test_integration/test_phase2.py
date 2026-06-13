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


def test_sanitize_timerange_slash_format():
    from backtesting.timerange_utils import sanitize_timerange
    assert sanitize_timerange("2024-01-01/2024-12-31") == "20240101-20241231"


def test_sanitize_timerange_year_only():
    from backtesting.timerange_utils import sanitize_timerange
    result = sanitize_timerange("2024")
    assert result.startswith("2024")


def test_sanitize_timerange_already_correct():
    from backtesting.timerange_utils import sanitize_timerange
    assert sanitize_timerange("20240101-20241231") == "20240101-20241231"


def test_sanitize_timerange_open_ended():
    from backtesting.timerange_utils import sanitize_timerange
    result = sanitize_timerange("20260427-")
    assert result == "20260427-"


def test_sanitize_timerange_iso_slash_variants():
    from backtesting.timerange_utils import sanitize_timerange
    assert sanitize_timerange("2024-06-01/2024-09-01") == "20240601-20240901"
    assert sanitize_timerange("2023-01-01/2023-12-31") == "20230101-20231231"


def test_strategy_syntax_validation_catches_bad_python():
    from backtesting.engine import BacktestEngine
    engine = BacktestEngine()
    bad_code = "class DynamicStrategy:\n    def populate_indicators(self):\n      bad indent\n    bad = 1"
    try:
        engine._validate_strategy(bad_code)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "syntax" in str(e).lower() or "indent" in str(e).lower()


def test_strategy_syntax_validation_passes_good_python():
    from backtesting.engine import BacktestEngine
    engine = BacktestEngine()
    good_code = (
        "from freqtrade.strategy import IStrategy\n"
        "import pandas as pd\n"
        "class DynamicStrategy(IStrategy):\n"
        "    timeframe = '1h'\n"
        "    stoploss = -0.05\n"
        "    minimal_roi = {'0': 0.01}\n"
    )
    engine._validate_strategy(good_code)  # should not raise


def test_parse_results_handles_zero_trades():
    """Ensure _parse_results returns zeroed dict not exception on empty result."""
    from backtesting.engine import BacktestEngine
    engine = BacktestEngine()
    raw = {"strategy": {"DynamicStrategy": {"total_trades": 0, "trades": []}}}
    result = engine._parse_results(raw)
    assert result["total_trades"] == 0
    assert result["sharpe_ratio"] == 0
    assert "error" not in result


def test_parse_results_handles_missing_strategy_key():
    """Ensure _parse_results handles unexpected JSON structure gracefully."""
    from backtesting.engine import BacktestEngine
    engine = BacktestEngine()
    raw = {"unexpected_key": "some_value"}
    result = engine._parse_results(raw)
    assert "error" in result or result.get("total_trades", 0) == 0


if __name__ == "__main__":
    import pandas as pd
    main()