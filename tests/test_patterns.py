"""Tests for pattern detection module."""
import pytest
from data.patterns import (
    BULLISH_PATTERNS, BEARISH_PATTERNS, PatternDetector,
)
import pandas as pd
import numpy as np


class TestPatternLists:

    def test_cdlengulfing_not_in_bearish(self):
        """CDLENGUFING should not appear in both lists (double-count bug C8)."""
        assert "CDLENGULFING" not in BEARISH_PATTERNS

    def test_cdlengulfing_in_bullish(self):
        """CDLENGUFING should remain in BULLISH_PATTERNS (signed value from TA-Lib)."""
        assert "CDLENGULFING" in BULLISH_PATTERNS

    def test_no_overlap_between_lists(self):
        """No pattern should appear in both BULLISH_PATTERNS and BEARISH_PATTERNS."""
        overlap = set(BULLISH_PATTERNS) & set(BEARISH_PATTERNS)
        assert len(overlap) == 0, f"Overlapping patterns: {overlap}"


class TestPatternToSignal:

    def test_engulfing_only_returns_bullish(self):
        """pattern_to_signal with only CDLENGULFING returns bullish."""
        detector = PatternDetector()
        result = detector.pattern_to_signal(["CDLENGULFING"])
        assert result == "bullish"

    def test_engulfing_and_bearish_returns_neutral(self):
        """One bullish + one bearish pattern returns neutral."""
        detector = PatternDetector()
        result = detector.pattern_to_signal(["CDLENGULFING", "CDLSHOOTINGSTAR"])
        assert result == "neutral"

    def test_empty_patterns_returns_neutral(self):
        detector = PatternDetector()
        assert detector.pattern_to_signal([]) == "neutral"

    def test_all_bullish_returns_bullish(self):
        detector = PatternDetector()
        result = detector.pattern_to_signal(BULLISH_PATTERNS[:2])
        assert result == "bullish"

    def test_all_bearish_returns_bearish(self):
        detector = PatternDetector()
        result = detector.pattern_to_signal(BEARISH_PATTERNS[:2])
        assert result == "bearish"


class TestPatternDetector:

    def test_detect_patterns_returns_dict(self):
        detector = PatternDetector()
        df = pd.DataFrame({
            "open": np.random.rand(50) + 100,
            "high": np.random.rand(50) + 102,
            "low": np.random.rand(50) + 98,
            "close": np.random.rand(50) + 100,
        })
        result = detector.detect_patterns(df)
        assert isinstance(result, dict)
        for pattern in result:
            assert pattern in BULLISH_PATTERNS or pattern in BEARISH_PATTERNS

    def test_get_active_patterns_returns_list(self):
        detector = PatternDetector()
        df = pd.DataFrame({
            "open": np.random.rand(50) + 100,
            "high": np.random.rand(50) + 102,
            "low": np.random.rand(50) + 98,
            "close": np.random.rand(50) + 100,
        })
        active = detector.get_active_patterns(df)
        assert isinstance(active, list)

    def test_get_pattern_report_structure(self):
        detector = PatternDetector()
        df = pd.DataFrame({
            "open": np.random.rand(50) + 100,
            "high": np.random.rand(50) + 102,
            "low": np.random.rand(50) + 98,
            "close": np.random.rand(50) + 100,
        })
        report = detector.get_pattern_report(df)
        assert "active_patterns" in report
        assert "bias" in report
        assert "bullish_count" in report
        assert "bearish_count" in report

