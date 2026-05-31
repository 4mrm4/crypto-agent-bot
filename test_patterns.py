"""Tests for data/patterns.py"""
import pandas as pd
import numpy as np
from data.patterns import PatternDetector, BULLISH_PATTERNS, BEARISH_PATTERNS


def _make_df(n=50):
    """Generate synthetic OHLCV data."""
    np.random.seed(42)
    close = 50000 + np.cumsum(np.random.randn(n) * 500)
    high = close + np.abs(np.random.randn(n) * 200)
    low = close - np.abs(np.random.randn(n) * 200)
    open_ = close + np.random.randn(n) * 100
    volume = np.random.randint(100, 1000, n).astype(float)
    return pd.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": close, "volume": volume
    })


def test_detect_patterns_returns_dict():
    df = _make_df()
    pd_det = PatternDetector()
    result = pd_det.detect_patterns(df)
    assert isinstance(result, dict)
    assert len(result) > 0
    for v in result.values():
        assert v in (-100, 0, 100)


def test_get_active_patterns_subset():
    df = _make_df()
    pd_det = PatternDetector()
    active = pd_det.get_active_patterns(df)
    assert isinstance(active, list)
    # All active patterns must be in the known list
    for p in active:
        assert p in BULLISH_PATTERNS or p in BEARISH_PATTERNS


def test_pattern_to_signal_values():
    pd_det = PatternDetector()
    assert pd_det.pattern_to_signal([]) == "neutral"
    assert pd_det.pattern_to_signal(["CDLHAMMER", "CDL3WHITESOLDIERS"]) == "bullish"
    assert pd_det.pattern_to_signal(["CDLSHOOTINGSTAR", "CDL3BLACKCROWS"]) == "bearish"


def test_get_pattern_report_structure():
    df = _make_df()
    pd_det = PatternDetector()
    report = pd_det.get_pattern_report(df)
    assert "active_patterns" in report
    assert "bias" in report
    assert "bullish_count" in report
    assert "bearish_count" in report
    assert report["bias"] in ("bullish", "bearish", "neutral")


if __name__ == "__main__":
    test_detect_patterns_returns_dict()
    test_get_active_patterns_subset()
    test_pattern_to_signal_values()
    test_get_pattern_report_structure()
    print("All pattern tests passed.")
