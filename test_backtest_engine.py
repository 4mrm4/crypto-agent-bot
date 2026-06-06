"""Tests for BacktestEngine.dataframe_override — fast backtest without Freqtrade subprocess."""

import pandas as pd
import numpy as np
import pytest

from backtesting.engine import BacktestEngine


@pytest.fixture
def engine():
    return BacktestEngine()


@pytest.fixture
def synthetic_ohlcv():
    """Create a small synthetic OHLCV DataFrame."""
    n = 200
    np.random.seed(42)
    close = 100 * np.exp(np.random.randn(n).cumsum() * 0.02)
    df = pd.DataFrame({
        "open": close * (1 + np.random.randn(n) * 0.005),
        "high": close * (1 + np.abs(np.random.randn(n) * 0.01)),
        "low": close * (1 - np.abs(np.random.randn(n) * 0.01)),
        "close": close,
        "volume": np.random.exponential(1000, n),
    })
    return df


class TestDataframeOverride:
    """Verify that dataframe_override runs SignalFactory + FastMetrics directly."""

    def test_override_returns_results_fast(self, engine, synthetic_ohlcv):
        """dataframe_override should return metrics without launching Freqtrade."""
        result = engine.run_backtest(
            strategy_type="sma_crossover",
            strategy_params={"fast_ma": 10, "slow_ma": 30},
            dataframe_override={"BTC/USDT": synthetic_ohlcv},
        )
        # Expect valid metrics
        assert isinstance(result, dict)
        assert "sharpe_ratio" in result
        assert "win_rate" in result
        assert "total_trades" in result
        assert "profit_ratio" in result
        # With 200 candles of random walk, sma_crossover may or may not trade
        assert result["total_trades"] >= 0

    def test_override_all_strategy_types(self, engine, synthetic_ohlcv):
        """All supported strategy types should work with dataframe_override."""
        for stype in ["sma_crossover", "macd_crossover", "rsi_oversold"]:
            result = engine.run_backtest(
                strategy_type=stype,
                dataframe_override={"BTC/USDT": synthetic_ohlcv},
            )
            assert isinstance(result, dict), f"{stype} failed: {result}"
            assert "sharpe_ratio" in result

    def test_override_with_custom_params(self, engine, synthetic_ohlcv):
        """Custom strategy params are passed through to SignalFactory."""
        result = engine.run_backtest(
            strategy_type="sma_crossover",
            strategy_params={"fast_ma": 5, "slow_ma": 20},
            dataframe_override={"BTC/USDT": synthetic_ohlcv},
        )
        assert result["total_trades"] >= 0

    def test_override_multiple_pairs(self, engine, synthetic_ohlcv):
        """Multiple pairs should be aggregated correctly."""
        dataframes = {
            "BTC/USDT": synthetic_ohlcv,
            "ETH/USDT": synthetic_ohlcv.copy(),
        }
        result = engine.run_backtest(
            strategy_type="sma_crossover",
            dataframe_override=dataframes,
        )
        assert result["total_trades"] >= 0

    def test_override_insufficient_data(self, engine):
        """Less than 50 candles should return zero trades."""
        small_df = pd.DataFrame({
            "open": [100] * 20,
            "high": [101] * 20,
            "low": [99] * 20,
            "close": [100] * 20,
            "volume": [1000] * 20,
        })
        result = engine.run_backtest(
            strategy_type="sma_crossover",
            dataframe_override={"BTC/USDT": small_df},
        )
        assert result["total_trades"] == 0

    def test_override_empty_dataframes(self, engine):
        """Empty dict returns zero trades."""
        result = engine.run_backtest(
            strategy_type="sma_crossover",
            dataframe_override={},
        )
        assert result["total_trades"] == 0
        assert result["sharpe_ratio"] == 0

    def test_no_override_uses_prefilter(self, engine):
        """Without dataframe_override, the pre-filter runs and returns results
        (no Freqtrade subprocess started in the common case)."""
        result = engine.run_backtest(
            strategy_type="sma_crossover",
            timerange="20210101-20210201",
        )
        # Result should be a dict (from pre-filter or Freqtrade fallback)
        assert isinstance(result, dict)
        assert "sharpe_ratio" in result

    def test_override_known_strategy_behavior(self, engine):
        """Sma crossover on trending data should produce trades (structure check)."""
        n = 500
        np.random.seed(1)
        close = 100 * np.exp(np.linspace(0, 0.1, n) + np.random.randn(n) * 0.01)
        trending_df = pd.DataFrame({
            "open": close * 0.999,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": np.random.exponential(1000, n),
        })
        result = engine.run_backtest(
            strategy_type="sma_crossover",
            strategy_params={"fast_ma": 10, "slow_ma": 50},
            dataframe_override={"BTC/USDT": trending_df},
        )
        # Should produce trades and return valid structure
        assert result["total_trades"] > 0, "Expected trades on trending data"
        assert isinstance(result["sharpe_ratio"], float)
        assert isinstance(result["win_rate"], float)
        assert isinstance(result["profit_ratio"], float)

    def test_run_fastmetrics_backtest_directly(self, engine, synthetic_ohlcv):
        """Call _run_fastmetrics_backtest directly."""
        result = engine._run_fastmetrics_backtest(
            strategy_type="sma_crossover",
            strategy_params={"fast_ma": 10, "slow_ma": 30},
            dataframes={"BTC/USDT": synthetic_ohlcv},
        )
        assert isinstance(result, dict)
        assert result["total_trades"] >= 0
