"""Market regime detection."""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import talib

logger = logging.getLogger(__name__)

# ── Regime-strategy compatibility map ──
# Used by the autonomous loop and dispatch_task to route strategies correctly.
REGIME_STRATEGY_MAP: Dict[str, Dict[str, List[str]]] = {
    "strong_uptrend":  {"use": ["momentum", "multi_timeframe", "sma_crossover", "breakout"],
                        "avoid": ["mean_reversion", "bollinger_bands", "rsi_oversold"]},
    "strong_downtrend":{"use": ["multi_timeframe", "macd_crossover"],
                        "avoid": ["momentum", "breakout", "sma_crossover"]},
    "weak_trend":      {"use": ["macd_crossover", "combined_sma_rsi"],
                        "avoid": ["momentum", "breakout"]},
    "ranging":         {"use": ["mean_reversion", "bollinger_bands", "rsi_oversold", "volatility_squeeze"],
                        "avoid": ["momentum", "sma_crossover"]},
    "volatile":        {"use": ["volatility_squeeze", "breakout", "bollinger_bands"],
                        "avoid": ["sma_crossover", "multi_timeframe"]},
    "low_liquidity":   {"use": [], "avoid": ["all"]},
}


@dataclass
class RegimeSnapshot:
    """Rich regime classification with supporting metrics."""
    regime: str                         # "strong_uptrend" | "weak_trend" | "ranging" | "volatile" | "low_liquidity"
    confidence: float                   # 0.0–1.0
    adx: float                          # ADX value
    atr_pct: float                      # ATR as % of price
    sma200_distance: float              # % above/below SMA200
    recommended_strategies: List[str] = field(default_factory=list)
    discouraged_strategies: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class MarketRegimeDetector:

    def classify_regime(self, df: pd.DataFrame) -> str:
        """
        Classify market regime using ADX, ATR, and price vs SMA200.
        Returns: 'strong_uptrend', 'strong_downtrend', 'ranging', 'volatile', 'weak_trend'
        """
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)

        adx = talib.ADX(high.values, low.values, close.values, timeperiod=14)
        plus_di = talib.PLUS_DI(high.values, low.values, close.values, timeperiod=14)
        minus_di = talib.MINUS_DI(high.values, low.values, close.values, timeperiod=14)
        atr = talib.ATR(high.values, low.values, close.values, timeperiod=14)
        sma200 = talib.SMA(close.values, timeperiod=200)

        last_adx = float(adx[-1]) if adx[-1] == adx[-1] else 0
        last_plus = float(plus_di[-1]) if plus_di[-1] == plus_di[-1] else 0
        last_minus = float(minus_di[-1]) if minus_di[-1] == minus_di[-1] else 0
        last_atr = float(atr[-1]) if atr[-1] == atr[-1] else 0
        last_price = float(close.iloc[-1])
        last_sma200 = float(sma200[-1]) if sma200[-1] == sma200[-1] else last_price

        # Volatility check: ATR vs 30-period average ATR
        avg_atr = float(atr[-30:].mean()) if len(atr) >= 30 else last_atr
        if last_atr > 2 * avg_atr:
            return "volatile"

        if last_adx > 25:
            if last_plus > last_minus and last_price > last_sma200:
                return "strong_uptrend"
            elif last_minus > last_plus and last_price < last_sma200:
                return "strong_downtrend"

        if last_adx < 20:
            return "ranging"

        return "weak_trend"

    def classify_regime_snapshot(self, df: pd.DataFrame) -> "RegimeSnapshot":
        """Return a full RegimeSnapshot with metrics and strategy recommendations."""
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)

        adx = talib.ADX(high.values, low.values, close.values, timeperiod=14)
        plus_di = talib.PLUS_DI(high.values, low.values, close.values, timeperiod=14)
        minus_di = talib.MINUS_DI(high.values, low.values, close.values, timeperiod=14)
        atr = talib.ATR(high.values, low.values, close.values, timeperiod=14)
        sma200 = talib.SMA(close.values, timeperiod=200)

        last_adx = float(adx[-1]) if adx[-1] == adx[-1] else 0
        last_plus = float(plus_di[-1]) if plus_di[-1] == plus_di[-1] else 0
        last_minus = float(minus_di[-1]) if minus_di[-1] == minus_di[-1] else 0
        last_atr = float(atr[-1]) if atr[-1] == atr[-1] else 0
        last_price = float(close.iloc[-1])
        last_sma200 = float(sma200[-1]) if sma200[-1] == sma200[-1] else last_price
        atr_pct = (last_atr / last_price) if last_price > 0 else 0.0
        sma200_distance = ((last_price - last_sma200) / last_sma200) if last_sma200 > 0 else 0.0

        regime = self.classify_regime(df)

        recommended = REGIME_STRATEGY_MAP.get(regime, {}).get("use", [])
        discouraged = REGIME_STRATEGY_MAP.get(regime, {}).get("avoid", [])

        # Confidence based on ADX clarity
        if regime in ("strong_uptrend", "strong_downtrend") and last_adx > 30:
            confidence = 0.9
        elif regime in ("strong_uptrend", "strong_downtrend"):
            confidence = 0.7
        elif regime == "volatile":
            confidence = 0.6
        elif regime == "ranging" and last_adx < 15:
            confidence = 0.8
        else:
            confidence = 0.5

        return RegimeSnapshot(
            regime=regime,
            confidence=confidence,
            adx=last_adx,
            atr_pct=atr_pct,
            sma200_distance=sma200_distance,
            recommended_strategies=recommended,
            discouraged_strategies=discouraged,
        )

    def get_best_strategy_types(self, regime: str) -> List[str]:
        """Map regime to suitable strategy types."""
        mapping = {
            "strong_uptrend":   ["sma_crossover", "combined_sma_rsi", "momentum"],
            "strong_downtrend": ["rsi_oversold", "mean_reversion"],
            "ranging":          ["bollinger_bands", "rsi_oversold", "mean_reversion"],
            "volatile":         ["bollinger_bands", "breakout"],
            "weak_trend":       ["combined_sma_rsi", "macd_crossover"],
        }
        return mapping.get(regime, ["sma_crossover"])