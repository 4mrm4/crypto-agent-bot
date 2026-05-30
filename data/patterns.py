"""Candlestick pattern detection using TA-Lib."""
import logging
import pandas as pd
import talib
from typing import Dict, List

logger = logging.getLogger(__name__)

BULLISH_PATTERNS = [
    "CDLHAMMER", "CDLENGULFING", "CDLMORNINGSTAR", "CDLPIERCING",
    "CDL3WHITESOLDIERS", "CDLINVERTEDHAMMER", "CDLDRAGONFLYDOJI",
]
BEARISH_PATTERNS = [
    "CDLSHOOTINGSTAR", "CDLENGULFING", "CDLEVENINGSTAR",
    "CDLDARKCLOUCOVER", "CDL3BLACKCROWS", "CDLHANGINGMAN", "CDLGRAVESTONEDOJI",
]
ALL_PATTERNS = list(set(BULLISH_PATTERNS + BEARISH_PATTERNS))


class PatternDetector:

    def detect_patterns(self, df: pd.DataFrame) -> Dict[str, int]:
        """Run all CDL patterns on the dataframe.
        Returns {pattern_name: signal} where signal is -100, 0, or 100 on last candle."""
        results = {}
        o = df["open"].astype(float).values
        h = df["high"].astype(float).values
        l = df["low"].astype(float).values
        c = df["close"].astype(float).values
        for pattern in ALL_PATTERNS:
            try:
                func = getattr(talib, pattern)
                signals = func(o, h, l, c)
                results[pattern] = int(signals[-1])
            except Exception as e:
                logger.debug("Pattern %s failed: %s", pattern, e)
        return results

    def get_active_patterns(self, df: pd.DataFrame) -> List[str]:
        """Return only patterns with non-zero signal on last candle."""
        patterns = self.detect_patterns(df)
        return [name for name, sig in patterns.items() if sig != 0]

    def pattern_to_signal(self, active_patterns: List[str]) -> str:
        """Return overall bias from active patterns."""
        bullish = sum(1 for p in active_patterns if p in BULLISH_PATTERNS)
        bearish = sum(1 for p in active_patterns if p in BEARISH_PATTERNS)
        if bullish > bearish:
            return "bullish"
        elif bearish > bullish:
            return "bearish"
        return "neutral"

    def get_pattern_report(self, df: pd.DataFrame) -> dict:
        """Full pattern analysis report."""
        active = self.get_active_patterns(df)
        bias = self.pattern_to_signal(active)
        return {
            "active_patterns": active,
            "bias": bias,
            "bullish_count": sum(1 for p in active if p in BULLISH_PATTERNS),
            "bearish_count": sum(1 for p in active if p in BEARISH_PATTERNS),
        }