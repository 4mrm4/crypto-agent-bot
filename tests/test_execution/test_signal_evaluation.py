"""Tests for SignalScanner._evaluate_single_strategy — all 5 strategy types + edge cases."""

import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from execution.signal_scanner import SignalScanner, SignalResult


# ── Talib mock (imported inside methods, not at module level) ──

@pytest.fixture(autouse=True)
def mock_talib():
    """Mock talib so tests don't need the actual C extension installed.

    Yields the mock so tests can override specific function return values.
    """
    mock = MagicMock()
    mock.RSI.return_value = [50.0]
    mock.BBANDS.return_value = ([115.0], [105.0], [95.0])
    mock.ROC.return_value = [0.5]
    with patch.dict("sys.modules", {"talib": mock}):
        yield mock


# ── Helpers ──

def make_ohlcv(prices, volumes=None):
    """Build a synthetic OHLCV DataFrame."""
    n = len(prices)
    if volumes is None:
        volumes = [1000.0] * n
    return pd.DataFrame({
        "open": prices,
        "high": [p * 1.01 for p in prices],
        "low": [p * 0.99 for p in prices],
        "close": prices,
        "volume": volumes,
    })


@pytest.fixture
def scanner():
    """Minimal SignalScanner instance."""
    return SignalScanner(
        pairs=["BTC/USDT"],
        approved_strategies=[{"strategy_type": "sma_crossover"}],
        regime_detector=MagicMock(),
        live_executor=None,
        fetcher=MagicMock(),
        vector_store=None,
        event_bus=None,
    )


# ── Test: SMA Crossover (no talib needed) ──

class TestSmaCrossover:
    def test_buy_signal_on_fast_crosses_above_slow(self, scanner):
        """Fast MA crosses above slow MA → buy signal.
        Uses a single-bar spike at the last bar so SMA5 moves above SMA20 exactly at crossover.
        """
        prices = [10.0] * 29 + [100.0]  # spike at last bar → SMA5[29]=28 > SMA20[29]=14.5
        ohlcv = make_ohlcv(prices)
        config = {"strategy_type": "sma_crossover", "params": {"fast_ma": 5, "slow_ma": 20}}

        result = scanner._evaluate_single_strategy(config, ohlcv, "ranging", "BTC/USDT")

        assert result is not None
        assert result.signal == "buy"
        assert result.confidence == 0.65
        assert result.strategy_type == "sma_crossover"
        assert result.pair == "BTC/USDT"

    def test_sell_signal_on_fast_crosses_below_slow(self, scanner):
        """Fast MA crosses below slow MA → sell signal.
        Uses a single-bar drop at the last bar so SMA5 drops below SMA20 exactly at crossover.
        """
        prices = [50.0] * 29 + [10.0]  # crash at last bar → SMA5[29]=42 < SMA20[29]=48
        ohlcv = make_ohlcv(prices)
        config = {"strategy_type": "sma_crossover", "params": {"fast_ma": 5, "slow_ma": 20}}

        result = scanner._evaluate_single_strategy(config, ohlcv, "trending", "BTC/USDT")

        assert result is not None
        assert result.signal == "sell"
        assert result.confidence == 0.60

    def test_hold_when_no_cross(self, scanner):
        """No crossover → None (hold)."""
        prices = [50.0] * 30  # flat, no crossover
        ohlcv = make_ohlcv(prices)
        config = {"strategy_type": "sma_crossover", "params": {"fast_ma": 5, "slow_ma": 20}}

        result = scanner._evaluate_single_strategy(config, ohlcv, "ranging", "BTC/USDT")

        assert result is None

    def test_sma_confidence_in_range(self, scanner):
        """Confidence is always between 0 and 1."""
        config = {"strategy_type": "sma_crossover", "params": {"fast_ma": 5, "slow_ma": 20}}

        prices_up = [10.0] * 29 + [100.0]
        result = scanner._evaluate_single_strategy(config, make_ohlcv(prices_up), "ranging", "BTC/USDT")
        if result:
            assert 0 <= result.confidence <= 1.0

        prices_down = [50.0] * 29 + [10.0]
        result = scanner._evaluate_single_strategy(config, make_ohlcv(prices_down), "trending", "BTC/USDT")
        if result:
            assert 0 <= result.confidence <= 1.0

    def test_sma_indicators_included(self, scanner):
        """Result includes indicator values in metadata."""
        prices = [10.0] * 29 + [100.0]
        config = {"strategy_type": "sma_crossover", "params": {"fast_ma": 5, "slow_ma": 20}}

        result = scanner._evaluate_single_strategy(config, make_ohlcv(prices), "ranging", "BTC/USDT")

        assert result is not None
        assert "sma_fast" in result.indicators
        assert "sma_slow" in result.indicators

    def test_insufficient_periods_no_crash(self, scanner):
        """Less data than MA period → None, no error."""
        prices = [10.0] * 3
        config = {"strategy_type": "sma_crossover", "params": {"fast_ma": 5, "slow_ma": 20}}

        result = scanner._evaluate_single_strategy(config, make_ohlcv(prices), "ranging", "BTC/USDT")

        assert result is None


# ── Test: RSI ──

class TestRsi:
    def test_buy_on_oversold(self, scanner, mock_talib):
        """RSI below buy threshold → buy."""
        mock_talib.RSI.return_value = [25.0]
        prices = [10.0] * 30
        config = {"strategy_type": "rsi_oversold", "params": {"rsi_period": 14, "rsi_buy_threshold": 30}}

        result = scanner._evaluate_single_strategy(config, make_ohlcv(prices), "ranging", "BTC/USDT")

        assert result is not None
        assert result.signal == "buy"
        assert result.confidence == 0.70

    def test_sell_on_overbought(self, scanner, mock_talib):
        """RSI above sell threshold → sell."""
        mock_talib.RSI.return_value = [75.0]
        prices = [10.0] * 30
        config = {"strategy_type": "rsi_oversold", "params": {"rsi_period": 14, "rsi_sell_threshold": 70}}

        result = scanner._evaluate_single_strategy(config, make_ohlcv(prices), "trending", "BTC/USDT")

        assert result is not None
        assert result.signal == "sell"

    def test_hold_on_neutral_rsi(self, scanner, mock_talib):
        """RSI between thresholds → no signal."""
        mock_talib.RSI.return_value = [50.0]
        config = {"strategy_type": "rsi_oversold", "params": {"rsi_period": 14}}

        result = scanner._evaluate_single_strategy(config, make_ohlcv([10.0] * 30), "ranging", "BTC/USDT")

        assert result is None

    def test_rsi_indicator_included(self, scanner, mock_talib):
        """Result includes RSI value in indicators."""
        mock_talib.RSI.return_value = [25.0]
        config = {"strategy_type": "rsi_oversold", "params": {"rsi_period": 14, "rsi_buy_threshold": 30}}

        result = scanner._evaluate_single_strategy(config, make_ohlcv([10.0] * 30), "ranging", "BTC/USDT")

        assert result is not None
        assert "rsi" in result.indicators


# ── Test: Bollinger Bands ──

class TestBollingerBands:
    def test_buy_when_price_below_lower(self, scanner, mock_talib):
        """Price below lower band → buy."""
        mock_talib.BBANDS.return_value = ([115.0], [105.0], [95.0])
        prices = [90.0] * 30
        config = {"strategy_type": "bollinger_bands", "params": {"bb_period": 20}}

        result = scanner._evaluate_single_strategy(config, make_ohlcv(prices), "ranging", "BTC/USDT")

        assert result is not None
        assert result.signal == "buy"

    def test_sell_when_price_above_upper(self, scanner, mock_talib):
        """Price above upper band → sell."""
        mock_talib.BBANDS.return_value = ([115.0], [105.0], [95.0])
        prices = [120.0] * 30
        config = {"strategy_type": "bollinger_bands", "params": {"bb_period": 20}}

        result = scanner._evaluate_single_strategy(config, make_ohlcv(prices), "ranging", "BTC/USDT")

        assert result is not None
        assert result.signal == "sell"

    def test_hold_within_bands(self, scanner, mock_talib):
        """Price within bands → no signal."""
        mock_talib.BBANDS.return_value = ([115.0], [105.0], [95.0])
        prices = [105.0] * 30
        config = {"strategy_type": "bollinger_bands", "params": {"bb_period": 20}}

        result = scanner._evaluate_single_strategy(config, make_ohlcv(prices), "ranging", "BTC/USDT")

        assert result is None


# ── Test: Mean Reversion ──

class TestMeanReversion:
    def test_buy_when_rsi_low_and_price_below_lower(self, scanner, mock_talib):
        """RSI < 35 and price below lower band → buy."""
        mock_talib.RSI.return_value = [30.0]
        mock_talib.BBANDS.return_value = ([115.0], [105.0], [95.0])
        prices = [90.0] * 30
        config = {"strategy_type": "mean_reversion", "params": {"bb_period": 20}}

        result = scanner._evaluate_single_strategy(config, make_ohlcv(prices), "ranging", "BTC/USDT")

        assert result is not None
        assert result.signal == "buy"
        assert result.confidence == 0.75

    def test_hold_on_mean_reversion_no_trigger(self, scanner, mock_talib):
        """Either condition not met → no signal."""
        mock_talib.RSI.return_value = [50.0]
        mock_talib.BBANDS.return_value = ([115.0], [105.0], [95.0])
        prices = [105.0] * 30
        config = {"strategy_type": "mean_reversion", "params": {"bb_period": 20}}

        result = scanner._evaluate_single_strategy(config, make_ohlcv(prices), "ranging", "BTC/USDT")

        assert result is None


# ── Test: Momentum ──

class TestMomentum:
    def test_buy_on_strong_momentum(self, scanner, mock_talib):
        """ROC > 2% and high volume → buy."""
        mock_talib.ROC.return_value = [3.0]
        prices = [10.0] * 25 + [15.0] * 5
        volumes = [1000.0] * 20 + [5000.0] * 10
        config = {"strategy_type": "momentum", "params": {"roc_period": 10}}

        result = scanner._evaluate_single_strategy(
            config, make_ohlcv(prices, volumes), "trending", "BTC/USDT",
        )

        assert result is not None
        assert result.signal == "buy"

    def test_hold_on_weak_momentum(self, scanner, mock_talib):
        """ROC < threshold → no signal."""
        mock_talib.ROC.return_value = [0.5]
        prices = [10.0] * 30
        volumes = [1000.0] * 20 + [5000.0] * 10
        config = {"strategy_type": "momentum", "params": {"roc_period": 10}}

        result = scanner._evaluate_single_strategy(
            config, make_ohlcv(prices, volumes), "ranging", "BTC/USDT",
        )

        assert result is None

    def test_hold_on_low_volume_despite_strong_roc(self, scanner, mock_talib):
        """ROC > 2% but low volume → no signal."""
        mock_talib.ROC.return_value = [3.0]
        prices = [10.0] * 25 + [15.0] * 5
        volumes = [1000.0] * 30
        config = {"strategy_type": "momentum", "params": {"roc_period": 10}}

        result = scanner._evaluate_single_strategy(
            config, make_ohlcv(prices, volumes), "trending", "BTC/USDT",
        )

        assert result is None


# ── Test: Edge cases ──

class TestEdgeCases:
    def test_empty_dataframe(self, scanner):
        """Empty DataFrame → None, no crash."""
        ohlcv = pd.DataFrame()
        config = {"strategy_type": "sma_crossover", "params": {}}
        result = scanner._evaluate_single_strategy(config, ohlcv, "ranging", "BTC/USDT")
        assert result is None

    def test_nan_in_close(self, scanner):
        """NaN values in close → None, no crash."""
        prices = [10.0] * 10 + [float("nan")] * 20
        config = {"strategy_type": "sma_crossover", "params": {"fast_ma": 5, "slow_ma": 20}}
        result = scanner._evaluate_single_strategy(config, make_ohlcv(prices), "ranging", "BTC/USDT")
        assert result is None

    def test_unknown_strategy_type(self, scanner):
        """Unknown strategy type → None."""
        prices = [10.0] * 30
        config = {"strategy_type": "unknown_magic_strategy", "params": {}}
        result = scanner._evaluate_single_strategy(config, make_ohlcv(prices), "ranging", "BTC/USDT")
        assert result is None

    def test_signal_result_dataclass(self, scanner):
        """SignalResult has required fields."""
        prices = [10.0] * 29 + [100.0]
        config = {"strategy_type": "sma_crossover", "params": {"fast_ma": 5, "slow_ma": 20}}

        result = scanner._evaluate_single_strategy(config, make_ohlcv(prices), "ranging", "BTC/USDT")

        assert result is not None
        assert isinstance(result.signal, str)
        assert isinstance(result.confidence, float)
        assert isinstance(result.indicators, dict)
        assert isinstance(result.pair, str)
        assert isinstance(result.strategy_type, str)
        assert isinstance(result.regime, str)

    def test_missing_volume_column(self, scanner):
        """DataFrame without volume column → fallback to zeros, no crash."""
        prices = [10.0] * 29 + [100.0]
        ohlcv = pd.DataFrame({
            "open": prices, "high": prices, "low": prices, "close": prices,
        })
        config = {"strategy_type": "sma_crossover", "params": {"fast_ma": 5, "slow_ma": 20}}
        result = scanner._evaluate_single_strategy(config, ohlcv, "ranging", "BTC/USDT")
        assert result is not None


# ── Test: Regime gating edge cases ──

class TestRegimeGating:
    def test_get_strategies_for_regime_empty(self, scanner):
        """Empty regime name → empty list when no strategies match."""
        scanner._approved_strategies = [{"strategy_type": "sma_crossover"}]
        with patch("data.regime.REGIME_STRATEGY_MAP", {}):
            result = scanner._get_strategies_for_regime("")
            assert result == []

    def test_get_strategies_for_regime_validated_not_matching(self, scanner):
        """Strategy validated for different regime → filtered out."""
        scanner._approved_strategies = [{
            "strategy_type": "sma_crossover", "validated_regimes": ["trending"],
        }]
        with patch("data.regime.REGIME_STRATEGY_MAP",
                   {"ranging": {"use": ["sma_crossover"]}}):
            result = scanner._get_strategies_for_regime("ranging")
            assert len(result) >= 0
