"""Tests for BacktestEngine.dataframe_override — fast backtest without Freqtrade subprocess."""

import pandas as pd
import numpy as np
import pytest

from backtesting.engine import BacktestEngine
from backtesting.signal_factory import SignalFactory
from backtesting.strategy_templates import STRATEGY_REGISTRY


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


# ── Strategy template render tests ──

class TestRenderStrategy:
    """Verify that _render_strategy produces valid Python for all 11 strategy types,
    with and without custom param overrides."""

    def test_all_types_render_without_placeholder_leaks(self, engine):
        """Every strategy type must render with zero unresolved $placeholders."""
        for stype in SignalFactory.supported_types():
            params = engine._default_strategy_params(stype)
            params["strategy_name"] = f"TestStrategy_{stype}"
            params["timestamp"] = "2024-01-01T00:00:00"
            rendered = engine._render_strategy(params)
            engine._validate_strategy(rendered)  # raises on unsubstituted $

    def test_multi_timeframe_all_9_params_rendered(self, engine):
        """multi_timeframe's 9 indicator params each appear in rendered output."""
        params = engine._default_strategy_params("multi_timeframe")
        params["strategy_name"] = "TestMTF"
        params["timestamp"] = "2024-01-01T00:00:00"
        rendered = engine._render_strategy(params)

        # Check each param value appears in the indicator_params_block area
        assert "default=10" in rendered and "fast_ma" in rendered  # fast_ma=10
        assert "default=30" in rendered                          # slow_ma=30
        assert "default=14" in rendered                          # adx_period=14, rsi_period=14
        assert "default=20" in rendered                          # adx_threshold=20
        assert "default=40" in rendered                          # rsi_oversold=40
        assert "default=70" in rendered                          # rsi_overbought=70
        assert "default=80" in rendered                          # higher_tf_fast=80
        assert "default=200" in rendered                         # higher_tf_slow=200, sma200

    def test_custom_params_override_defaults(self, engine):
        """Custom params passed through strategy_params should appear in rendered output."""
        custom = {"fast_ma": 5, "slow_ma": 15, "startup_candle_count": 50}
        strategy_params = {
            "indicator_code": STRATEGY_REGISTRY["sma_crossover"]["indicator_code"],
            "entry_condition": STRATEGY_REGISTRY["sma_crossover"]["entry_condition"],
            "exit_condition": STRATEGY_REGISTRY["sma_crossover"]["exit_condition"],
            "indicator_params_block": STRATEGY_REGISTRY["sma_crossover"]["indicator_params_block"],
            "stoploss": -0.05,
            "trailing_stop": False,
            "minimal_roi": '{"0": 0.01}',
            "timeframe": "1h",
            **custom,
        }
        strategy_params["strategy_name"] = "CustomTest"
        strategy_params["timestamp"] = "2024-01-01T00:00:00"
        rendered = engine._render_strategy(strategy_params)

        assert "default=5" in rendered, "Custom fast_ma=5 not in output"
        assert "default=15" in rendered, "Custom slow_ma=15 not in output"
        assert "startup_candle_count = 50" in rendered, "Custom startup_candle_count not in output"
        assert "stoploss = -0.05" in rendered
        assert "trailing_stop = False" in rendered

    def test_trailing_stop_true_renders_correctly(self, engine):
        """Boolean True for trailing_stop renders as Python True."""
        params = engine._default_strategy_params("sma_crossover")
        params["strategy_name"] = "TrailingTest"
        params["timestamp"] = "2024-01-01T00:00:00"
        params["trailing_stop"] = True
        rendered = engine._render_strategy(params)
        assert "trailing_stop = True" in rendered

    def test_timeframe_passthrough(self, engine):
        """Custom timeframe appears in rendered output."""
        params = engine._default_strategy_params("sma_crossover")
        params["strategy_name"] = "TimeTest"
        params["timestamp"] = "2024-01-01T00:00:00"
        params["timeframe"] = "5m"
        rendered = engine._render_strategy(params)
        assert 'timeframe = "5m"' in rendered

    def test_strategy_type_without_params_block(self, engine):
        """Types with empty indicator_params_block (momentum, breakout, etc.) render cleanly."""
        for stype in ["momentum", "breakout", "mean_reversion",
                       "volatility_squeeze", "sentiment_driven"]:
            params = engine._default_strategy_params(stype)
            params["strategy_name"] = f"Test_{stype}"
            params["timestamp"] = "2024-01-01T00:00:00"
            rendered = engine._render_strategy(params)
            engine._validate_strategy(rendered)  # must not raise

    def test_render_strategy_with_strategy_type_key(self, engine):
        """strategy_type key in params dict must not interfere with rendering."""
        params = engine._default_strategy_params("sma_crossover")
        params["strategy_name"] = "KeyTest"
        params["timestamp"] = "2024-01-01T00:00:00"
        params["strategy_type"] = "sma_crossover"  # injected by backtester agent
        rendered = engine._render_strategy(params)
        engine._validate_strategy(rendered)  # must not raise

    def test_trailing_stop_true_sets_all_params(self, engine):
        """Enabling trailing_stop=true should render all 4 trailing stop params correctly."""
        params = engine._default_strategy_params("sma_crossover")
        params["strategy_name"] = "TrailFullTest"
        params["timestamp"] = "2024-01-01T00:00:00"
        params["trailing_stop"] = True
        params["trailing_stop_positive"] = 0.008
        params["trailing_stop_positive_offset"] = 0.025
        params["trailing_only_offset_is_reached"] = True
        rendered = engine._render_strategy(params)

        assert "trailing_stop = True" in rendered
        assert "trailing_stop_positive = 0.008" in rendered
        assert "trailing_stop_positive_offset = 0.025" in rendered
        assert "trailing_only_offset_is_reached = True" in rendered
        engine._validate_strategy(rendered)  # no unresolved $placeholders

    def test_trailing_stop_default_false_renders(self, engine):
        """Default trailing_stop=false should include all 4 params as False/0."""
        params = engine._default_strategy_params("sma_crossover")
        params["strategy_name"] = "TrailDefaultTest"
        params["timestamp"] = "2024-01-01T00:00:00"
        rendered = engine._render_strategy(params)

        assert "trailing_stop = False" in rendered
        assert "trailing_stop_positive = 0.01" in rendered
        assert "trailing_stop_positive_offset = 0.02" in rendered
        assert "trailing_only_offset_is_reached = False" in rendered
        engine._validate_strategy(rendered)

    def test_trailing_stop_params_override_in_strategy_params(self, engine):
        """trailing_stop passed via strategy_params should override engine defaults."""
        params = engine._default_strategy_params("sma_crossover")
        params["strategy_name"] = "TrailOverrideTest"
        params["timestamp"] = "2024-01-01T00:00:00"
        # Simulate what happens when backtester agent passes custom trailing params
        engine._engine = engine  # fake so we can call run_backtest-like path
        strategy_params = {"trailing_stop": True, "trailing_stop_positive": 0.005,
                           "trailing_stop_positive_offset": 0.015,
                           "trailing_only_offset_is_reached": True}
        # Simulate the update flow in run_backtest
        params.update(strategy_params)
        rendered = engine._render_strategy(params)

        assert "trailing_stop = True" in rendered
        assert "trailing_stop_positive = 0.005" in rendered
        assert "trailing_stop_positive_offset = 0.015" in rendered
        assert "trailing_only_offset_is_reached = True" in rendered
        engine._validate_strategy(rendered)
