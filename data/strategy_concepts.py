"""
Library of trading strategy concepts in structured JSON.
The LLM maps these to concrete implementations rather than
writing raw code from scratch. Add new concepts here to
expand what the system can discover and test.
"""

STRATEGY_CONCEPTS = [
    {
        "name": "Golden Cross",
        "category": "trend_following",
        "description": "50 SMA crosses above 200 SMA — classic bull signal",
        "indicators": ["SMA_50", "SMA_200"],
        "entry": "SMA_50 crosses above SMA_200",
        "exit": "SMA_50 crosses below SMA_200",
        "best_regime": "strong_uptrend",
        "freqtrade_type": "sma_crossover",
        "suggested_params": {"fast_ma": 50, "slow_ma": 200},
    },
    {
        "name": "RSI Divergence",
        "category": "mean_reversion",
        "description": "Price makes lower low but RSI makes higher low — bullish divergence",
        "indicators": ["RSI_14", "price"],
        "entry": "Price lower low + RSI higher low",
        "exit": "RSI > 70 or price returns to mean",
        "best_regime": "ranging",
        "freqtrade_type": "rsi_oversold",
        "suggested_params": {},
    },
    {
        "name": "Volatility Breakout",
        "category": "breakout",
        "description": "Price breaks above N-day high after period of low volatility",
        "indicators": ["ATR", "highest_high_20"],
        "entry": "Close > 20-period high AND ATR expanding",
        "exit": "Close < 10-period low OR ATR contracting",
        "best_regime": "volatile",
        "freqtrade_type": "breakout",
        "suggested_params": {},
    },
    {
        "name": "MACD Histogram Reversal",
        "category": "momentum",
        "description": "MACD histogram switches from negative to positive",
        "indicators": ["MACD_12_26_9"],
        "entry": "MACD histogram crosses zero from below",
        "exit": "MACD histogram crosses zero from above",
        "best_regime": "weak_trend",
        "freqtrade_type": "macd_crossover",
        "suggested_params": {},
    },
    {
        "name": "Bollinger Band Squeeze",
        "category": "volatility",
        "description": "BB width contracts to 6-month low then expands — big move coming",
        "indicators": ["BB_20_2", "BB_width"],
        "entry": "BB width at minimum then expanding + MACD positive",
        "exit": "BB width > 3x minimum",
        "best_regime": "volatile",
        "freqtrade_type": "volatility_squeeze",
        "suggested_params": {},
    },
    {
        "name": "Fear and Greed Contrarian",
        "category": "sentiment",
        "description": "Buy extreme fear, sell extreme greed",
        "indicators": ["Fear_Greed_Index", "RSI_14"],
        "entry": "Fear/Greed < 25 (extreme fear) + RSI < 40",
        "exit": "Fear/Greed > 75 (extreme greed) OR RSI > 65",
        "best_regime": "ranging",
        "freqtrade_type": "sentiment_driven",
        "suggested_params": {},
    },
    {
        "name": "Multi-Timeframe Trend",
        "category": "trend_following",
        "description": "Short-term crossover confirmed by long-term trend",
        "indicators": ["SMA_20", "SMA_50", "SMA_200", "ADX_14"],
        "entry": "SMA_20 > SMA_50 AND price > SMA_200 AND ADX > 20",
        "exit": "SMA_20 < SMA_50 OR price < SMA_200",
        "best_regime": "strong_uptrend",
        "freqtrade_type": "multi_timeframe",
        "suggested_params": {},
    },
    {
        "name": "Opening Range Breakout",
        "category": "breakout",
        "description": "Break above/below first candle high/low of the day",
        "indicators": ["daily_open", "first_candle_high", "first_candle_low"],
        "entry": "Price breaks first 4h candle high with volume",
        "exit": "End of day or 2x ATR stop",
        "best_regime": "volatile",
        "freqtrade_type": "breakout",
        "suggested_params": {},
    },
    {
        "name": "Mean Reversion to VWAP",
        "category": "mean_reversion",
        "description": "Price deviates far from VWAP and reverts",
        "indicators": ["BB_20_2", "RSI_14"],
        "entry": "Price > 2 std from BB middle + RSI < 35",
        "exit": "Price returns to BB middle",
        "best_regime": "ranging",
        "freqtrade_type": "mean_reversion",
        "suggested_params": {},
    },
    {
        "name": "Momentum with Volume",
        "category": "momentum",
        "description": "Strong price momentum confirmed by above-average volume",
        "indicators": ["ROC_10", "volume_SMA_20", "RSI_14"],
        "entry": "ROC > 2% AND volume > 1.5x average AND RSI 50-75",
        "exit": "ROC < 0 OR RSI > 75",
        "best_regime": "strong_uptrend",
        "freqtrade_type": "momentum",
        "suggested_params": {},
    },
]


def get_concepts_for_regime(regime: str) -> list:
    """Return strategy concepts suitable for the given regime."""
    return [c for c in STRATEGY_CONCEPTS if c["best_regime"] == regime
            or c["best_regime"] == "any"]


def get_concept_by_name(name: str) -> dict:
    """Look up a concept by name."""
    for c in STRATEGY_CONCEPTS:
        if c["name"].lower() == name.lower():
            return c
    return {}


def get_all_concept_names() -> list:
    return [c["name"] for c in STRATEGY_CONCEPTS]
