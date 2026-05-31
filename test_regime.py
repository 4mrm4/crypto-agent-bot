"""Tests for data/regime.py"""
import pandas as pd
import numpy as np
from data.regime import MarketRegimeDetector


def _make_trending_df(n=250, direction=1):
    """Synthetic strongly trending data."""
    np.random.seed(1)
    close = 40000 + direction * np.arange(n) * 100 + np.random.randn(n) * 200
    high = close + np.abs(np.random.randn(n) * 150)
    low = close - np.abs(np.random.randn(n) * 150)
    open_ = close - direction * 50
    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
        "volume": np.ones(n) * 1000
    })


def _make_ranging_df(n=250):
    """Synthetic flat/ranging data."""
    np.random.seed(2)
    close = 50000 + np.random.randn(n) * 300
    high = close + np.abs(np.random.randn(n) * 100)
    low = close - np.abs(np.random.randn(n) * 100)
    open_ = close + np.random.randn(n) * 50
    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
        "volume": np.ones(n) * 1000
    })


def test_regime_returns_valid_string():
    det = MarketRegimeDetector()
    df = _make_trending_df()
    result = det.classify_regime(df)
    valid = {"strong_uptrend", "strong_downtrend", "ranging", "volatile", "weak_trend"}
    assert result in valid, f"Got unexpected regime: {result}"


def test_uptrend_detected():
    det = MarketRegimeDetector()
    df = _make_trending_df(direction=1)
    result = det.classify_regime(df)
    # Strong uptrend should be classified as trending (not ranging)
    assert result != "ranging"


def test_get_best_strategy_types_coverage():
    det = MarketRegimeDetector()
    regimes = ["strong_uptrend", "strong_downtrend", "ranging", "volatile", "weak_trend"]
    for regime in regimes:
        types = det.get_best_strategy_types(regime)
        assert isinstance(types, list)
        assert len(types) > 0


def test_unknown_regime_fallback():
    det = MarketRegimeDetector()
    types = det.get_best_strategy_types("unknown_regime_xyz")
    assert types == ["sma_crossover"]


if __name__ == "__main__":
    test_regime_returns_valid_string()
    test_uptrend_detected()
    test_get_best_strategy_types_coverage()
    test_unknown_regime_fallback()
    print("All regime tests passed.")
