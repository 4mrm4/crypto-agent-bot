"""Market regime detection."""
import logging
import pandas as pd
import talib
from typing import List

logger = logging.getLogger(__name__)


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