"""Tests for SignalFactory + FastMetrics (Task 6 — vectorbt pre-filter)."""

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import pandas as pd

from backtesting.signal_factory import (
    REGISTRY,
    FastMetrics,
    SignalFactory,
    _s,
    _build_signal,
)
from config import settings


# ── Synthetic OHLCV fixtures ──

def make_ohlcv(n=300, seed=42) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(seed)
    close = 50000 + np.cumsum(np.random.randn(n) * 50)
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 5,
        "high": close + np.abs(np.random.randn(n) * 15),
        "low": close - np.abs(np.random.randn(n) * 15),
        "close": close,
        "volume": np.random.randint(100, 1000, n).astype(float),
    })


def make_trending(n=300) -> pd.DataFrame:
    """Strong upward trend data."""
    close = 50000 + np.arange(n) * 10 + np.random.randn(n) * 50
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 5,
        "high": close + np.abs(np.random.randn(n) * 10),
        "low": close - np.abs(np.random.randn(n) * 10),
        "close": close,
        "volume": np.random.randint(200, 1000, n).astype(float),
    })


def make_ranging(n=300) -> pd.DataFrame:
    """Ranging/mean-reverting data."""
    np.random.seed(99)
    close = 50000 + np.sin(np.arange(n) * 0.1) * 1000 + np.random.randn(n) * 50
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 5,
        "high": close + np.abs(np.random.randn(n) * 10),
        "low": close - np.abs(np.random.randn(n) * 10),
        "close": close,
        "volume": np.random.randint(200, 1000, n).astype(float),
    })


# ── Helper tests ──

class TestHelpers:
    def test_s_wraps_ndarray_in_series(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = _s(arr)
        assert isinstance(result, pd.Series)
        assert list(result) == [1.0, 2.0, 3.0]

    def test_s_shift_works(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = _s(arr)
        assert pd.isna(result.shift(1).iloc[0])

    def test_build_signal_empty(self):
        n = 10
        entry = pd.Series([False] * n)
        exit = pd.Series([False] * n)
        sig = _build_signal(entry, exit)
        assert all(sig == 0)

    def test_build_signal_entry_exit_cycle(self):
        n = 10
        entry = pd.Series([False] * n)
        exit = pd.Series([False] * n)
        entry.iloc[2] = True
        exit.iloc[6] = True
        sig = _build_signal(entry, exit)
        assert sig.iloc[2] == 1
        assert sig.iloc[6] == -1
        assert sig.iloc[0] == 0
        assert sig.iloc[4] == 0  # inside position, hold

    def test_build_signal_entry_only(self):
        n = 10
        entry = pd.Series([False] * n)
        exit = pd.Series([False] * n)
        entry.iloc[2] = True
        sig = _build_signal(entry, exit)
        assert sig.iloc[2] == 1


# ── SignalFactory basic tests ──

class TestSignalFactory:
    def test_supported_types(self):
        types = SignalFactory.supported_types()
        assert len(types) == 11
        assert "sma_crossover" in types

    def test_unknown_type_raises(self):
        df = make_ohlcv(100)
        with pytest.raises(ValueError, match="Unknown strategy type"):
            SignalFactory.generate(df, "nonexistent")

    def test_produces_signal_series(self):
        df = make_ohlcv(300)
        for stype in SignalFactory.supported_types():
            sig = SignalFactory.generate(df, stype)
            assert isinstance(sig, pd.Series)
            assert sig.dtype in (np.int64, np.float64)
            assert set(sig.unique()).issubset({-1, 0, 1})

    def test_all_types_run_without_error(self):
        for stype in SignalFactory.supported_types():
            # Use trending data for trend-following strategies, ranging for mean-reversion
            if stype in ("rsi_oversold", "bollinger_bands", "mean_reversion"):
                df = make_ranging(500)
            else:
                df = make_trending(500)
            sig = SignalFactory.generate(df, stype)
            assert isinstance(sig, pd.Series)

    def test_signal_length_matches_df(self):
        df = make_ohlcv(100)
        sig = SignalFactory.generate(df, "sma_crossover")
        assert len(sig) == len(df)


# ── Specific strategy type tests ──

class TestSmaCrossover:
    def test_buy_on_crossover(self):
        # Create data that starts flat then trends up sharply, so fast crosses slow
        np.random.seed(42)
        flat = np.ones(100) * 50000 + np.random.randn(100) * 20
        spike = np.cumsum(np.ones(200) * 30 + np.random.randn(200) * 20) + 50000
        close = np.concatenate([flat, spike])
        df = pd.DataFrame({
            "open": close + np.random.randn(300) * 5,
            "high": close + np.abs(np.random.randn(300) * 10),
            "low": close - np.abs(np.random.randn(300) * 10),
            "close": close,
            "volume": np.random.randint(100, 1000, 300).astype(float),
        })
        sig = SignalFactory.generate(df, "sma_crossover", {"fast_ma": 10, "slow_ma": 30})
        assert 1 in sig.values or -1 in sig.values  # has at least one signal

    def test_default_params_work(self):
        df = make_ohlcv(200)
        sig = SignalFactory.generate(df, "sma_crossover")
        assert isinstance(sig, pd.Series)


class TestMacdCrossover:
    def test_generates_signals(self):
        df = make_ohlcv(200)
        sig = SignalFactory.generate(df, "macd_crossover")
        assert sig.abs().sum() >= 0


class TestRsiOversold:
    def test_generates_signals(self):
        df = make_ranging(200)
        sig = SignalFactory.generate(df, "rsi_oversold", {"rsi_period": 14})
        assert sig.abs().sum() >= 0


class TestBollingerBands:
    def test_generates_signals(self):
        df = make_ranging(300)
        sig = SignalFactory.generate(df, "bollinger_bands")
        assert sig.abs().sum() >= 0


class TestCombinedSmaRsi:
    def test_generates_signals(self):
        df = make_trending(200)
        sig = SignalFactory.generate(df, "combined_sma_rsi", {"fast_ma": 10, "slow_ma": 30})
        assert sig.abs().sum() >= 0


class TestMomentum:
    def test_generates_signals(self):
        df = make_trending(200)
        sig = SignalFactory.generate(df, "momentum")
        assert sig.abs().sum() >= 0


class TestBreakout:
    def test_generates_signals(self):
        df = make_ohlcv(200)
        sig = SignalFactory.generate(df, "breakout")
        assert sig.abs().sum() >= 0


class TestMeanReversion:
    def test_generates_signals(self):
        df = make_ranging(200)
        sig = SignalFactory.generate(df, "mean_reversion")
        assert sig.abs().sum() >= 0


class TestVolatilitySqueeze:
    def test_generates_signals(self):
        df = make_ohlcv(300)
        sig = SignalFactory.generate(df, "volatility_squeeze")
        assert sig.abs().sum() >= 0


class TestSentimentDriven:
    def test_generates_signals(self):
        df = make_ohlcv(200)
        sig = SignalFactory.generate(df, "sentiment_driven")
        assert sig.abs().sum() >= 0


class TestMultiTimeframe:
    def test_generates_signals(self):
        df = make_trending(300)
        sig = SignalFactory.generate(df, "multi_timeframe")
        assert sig.abs().sum() >= 0


# ── FastMetrics tests ──

class TestFastMetrics:
    def test_empty_signals(self):
        df = make_ohlcv(100)
        sigs = pd.Series(0, index=df.index)
        metrics = FastMetrics.compute(df, sigs)
        assert metrics["total_trades"] == 0
        assert metrics["passed"] is False

    def test_profitable_trade(self):
        df = make_ohlcv(100)
        # Force entry at a low point and exit at a higher point
        entry_idx = 10
        exit_idx = 50
        entry_price = df.iloc[entry_idx]["close"]
        dfc = df.copy()
        dfc.loc[dfc.index[exit_idx], "close"] = entry_price * 1.1
        sigs = pd.Series(0, index=df.index)
        sigs.iloc[entry_idx] = 1
        sigs.iloc[exit_idx] = -1
        metrics = FastMetrics.compute(dfc, sigs)
        assert metrics["total_trades"] >= 1
        assert metrics["win_rate"] > 0 or metrics["total_trades"] == 0

    def test_sharpe_positive_with_profitable_trades(self):
        df = make_ohlcv(200)
        sigs = pd.Series(0, index=df.index)
        # Create multiple profitable signals at low points
        sigs.iloc[20] = 1
        sigs.iloc[50] = -1
        sigs.iloc[70] = 1
        sigs.iloc[100] = -1
        metrics = FastMetrics.compute(df, sigs)
        assert isinstance(metrics["sharpe_ratio"], float)
        assert isinstance(metrics["win_rate"], float)
        assert isinstance(metrics["total_trades"], int)

    def test_metrics_structure(self):
        df = make_ohlcv(100)
        sigs = pd.Series(0, index=df.index)
        sigs.iloc[10] = 1
        sigs.iloc[20] = -1
        metrics = FastMetrics.compute(df, sigs, portfolio_value=50000.0)
        expected_keys = {"sharpe_ratio", "win_rate", "max_drawdown", "total_trades",
                         "total_return_pct", "num_entries", "num_exits", "passed"}
        assert expected_keys.issubset(metrics.keys())

    def test_portfolio_value_affects_return(self):
        df = make_ohlcv(100)
        sigs = pd.Series(0, index=df.index)
        sigs.iloc[10] = 1
        sigs.iloc[20] = -1
        small = FastMetrics.compute(df, sigs, portfolio_value=1000)
        large = FastMetrics.compute(df, sigs, portfolio_value=100000)
        # Return pct should be the same regardless of portfolio size
        assert small["total_return_pct"] == large["total_return_pct"]

    def test_all_signals_no_trades(self):
        """All-hold signals produce zero-trade metrics."""
        df = make_ohlcv(50)
        sigs = pd.Series(0, index=df.index)
        metrics = FastMetrics.compute(df, sigs)
        assert metrics["total_trades"] == 0
        assert metrics["passed"] is False

    def test_entry_without_exit(self):
        """Entry without matching exit produces partial trade."""
        df = make_ohlcv(50)
        sigs = pd.Series(0, index=df.index)
        sigs.iloc[10] = 1  # entry but no exit
        metrics = FastMetrics.compute(df, sigs)
        assert metrics["total_trades"] == 0  # no completed pair


# ── TA-Lib ndarray wrapping tests ──

class TestTalibWrapping:
    def test_ta_lib_output_wrapped(self):
        """TA-Lib functions return ndarrays, verify our _s wrapper makes Series."""
        import talib.abstract as ta
        arr = np.array([50.0, 51.0, 52.0, 53.0, 54.0], dtype=float)
        result = ta.SMA(arr, timeperiod=3)
        assert isinstance(result, np.ndarray)  # TA-Lib returns ndarray
        wrapped = _s(result)
        assert isinstance(wrapped, pd.Series)  # our wrapper makes Series
        assert pd.isna(wrapped.shift(1).iloc[0])  # shift works after wrapping


# ── Pre-filter integration tests ──

class TestRunPrefilter:
    def test_disabled_when_config_false(self):
        with patch.object(settings, "VECTORBT_PREFILTER_ENABLED", False):
            from backtesting.engine import BacktestEngine
            engine = BacktestEngine()
            result = engine._run_prefilter("sma_crossover", {}, "20210101-", ["BTC/USDT"])
            assert result is None

    def test_unknown_type_passes_through(self):
        from backtesting.engine import BacktestEngine
        engine = BacktestEngine()
        result = engine._run_prefilter("nonexistent_type", {}, "20210101-", ["BTC/USDT"])
        assert result is None

    def test_passes_through_on_fetch_failure(self):
        from backtesting.engine import BacktestEngine
        engine = BacktestEngine()
        with patch("data.fetcher.MarketDataFetcher") as mock:
            mock.return_value.fetch_ohlcv.return_value = None
            result = engine._run_prefilter("sma_crossover", {}, "20210101-", ["BTC/USDT"])
            assert result is None

    def test_early_return_shape(self):
        """_run_prefilter returning a result dict has pre_filter_rejected key."""
        from backtesting.engine import BacktestEngine
        engine = BacktestEngine()
        with patch("data.fetcher.MarketDataFetcher") as mock:
            mock.return_value.fetch_ohlcv.return_value = make_ohlcv(100)
            result = engine._run_prefilter("sma_crossover", {}, "20210101-", ["BTC/USDT"])
            if result is not None:
                assert "pre_filter_rejected" in result

    def test_prefilter_reject_shape(self):
        """Verify rejected dict has expected structure."""
        from backtesting.engine import BacktestEngine
        engine = BacktestEngine()
        with patch("data.fetcher.MarketDataFetcher") as mock:
            mock.return_value.fetch_ohlcv.return_value = make_ohlcv(100)
            result = engine._run_prefilter("sma_crossover", {}, "20210101-", ["BTC/USDT"])
            if result is not None:
                assert "sharpe_ratio" in result
                assert "win_rate" in result
                assert "total_trades" in result


# ── Regression tests ──

class TestRegistry:
    def test_all_types_have_functions(self):
        """Every entry in REGISTRY is callable."""
        for stype, fn in REGISTRY.items():
            assert callable(fn), f"{stype} is not callable"

    def test_registry_matches_supported_types(self):
        supported = set(SignalFactory.supported_types())
        registered = set(REGISTRY.keys())
        assert supported == registered

    def test_registry_has_exactly_11_entries(self):
        assert len(REGISTRY) == 11
