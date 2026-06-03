"""
SignalFactory — fast vectorized pre-filter for strategy performance estimation.

Each strategy type has a dedicated function that mirrors the corresponding
Freqtrade strategy template in engine.py **exactly** — same TA-Lib calls, same
parameters, same .shift(1) patterns, same constants.

TA-Lib returns numpy ndarrays. Each result is wrapped in pd.Series() immediately
so .shift(1) and boolean masking work identically to the template.
"""

import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import talib.abstract as ta

from config import settings

logger = logging.getLogger(__name__)


# ── Helpers ──

def _s(arr: np.ndarray) -> pd.Series:
    """Wrap a numpy array (TA-Lib output) in a pandas Series."""
    return pd.Series(arr)


def _build_signal(entry: pd.Series, exit: pd.Series) -> pd.Series:
    """Build a combined signal series from entry and exit boolean series.

    Returns:
        1 for entry, -1 for exit, 0 otherwise. Forward-fills position.
    """
    signals = pd.Series(0, index=entry.index)
    in_position = False
    for i in range(len(signals)):
        if not in_position and entry.iloc[i]:
            signals.iloc[i] = 1
            in_position = True
        elif in_position and exit.iloc[i]:
            signals.iloc[i] = -1
            in_position = False
    return signals


# ── Individual signal generators ──
# Each mirrors the corresponding template in engine.py

def _signal_sma_crossover(df: pd.DataFrame, params: dict) -> pd.Series:
    fast_ma = _s(ta.SMA(df["close"].values, timeperiod=params.get("fast_ma", 10)))
    slow_ma = _s(ta.SMA(df["close"].values, timeperiod=params.get("slow_ma", 30)))
    entry = (fast_ma.shift(1) <= slow_ma.shift(1)) & (fast_ma > slow_ma)
    exit = (fast_ma.shift(1) >= slow_ma.shift(1)) & (fast_ma < slow_ma)
    return _build_signal(entry, exit)


def _signal_macd_crossover(df: pd.DataFrame, params: dict) -> pd.Series:
    macd_data = ta.MACD(
        df["close"].values,
        fastperiod=params.get("macd_fast", 12),
        slowperiod=params.get("macd_slow", 26),
        signalperiod=params.get("macd_signal", 9),
    )
    macd = _s(macd_data[0])
    signal = _s(macd_data[1])
    hist = _s(macd_data[2])
    entry = (hist.shift(1) <= 0) & (hist > 0)
    exit = (hist.shift(1) >= 0) & (hist < 0)
    return _build_signal(entry, exit)


def _signal_rsi_oversold(df: pd.DataFrame, params: dict) -> pd.Series:
    rsi = _s(ta.RSI(df["close"].values, timeperiod=params.get("rsi_period", 14)))
    buy_thresh = params.get("rsi_buy_threshold", 30)
    sell_thresh = params.get("rsi_sell_threshold", 70)
    entry = (rsi < buy_thresh) & (rsi.shift(1) >= buy_thresh)
    exit = (rsi > sell_thresh) & (rsi.shift(1) <= sell_thresh)
    return _build_signal(entry, exit)


def _signal_bollinger_bands(df: pd.DataFrame, params: dict) -> pd.Series:
    period = params.get("bb_period", 20)
    upper, middle, lower = ta.BBANDS(
        df["close"].values.astype(float),
        timeperiod=period, nbdevup=2.0, nbdevdn=2.0,
    )
    bb_upper = _s(upper)
    bb_lower = _s(lower)
    entry = (df["close"] < bb_lower) & (df["close"].shift(1) >= bb_lower.shift(1))
    exit = (df["close"] > bb_upper) & (df["close"].shift(1) <= bb_upper.shift(1))
    return _build_signal(entry, exit)


def _signal_combined_sma_rsi(df: pd.DataFrame, params: dict) -> pd.Series:
    fast_ma = _s(ta.SMA(df["close"].values, timeperiod=params.get("fast_ma", 10)))
    slow_ma = _s(ta.SMA(df["close"].values, timeperiod=params.get("slow_ma", 30)))
    rsi = _s(ta.RSI(df["close"].values, timeperiod=14))
    entry = (
        (fast_ma.shift(1) <= slow_ma.shift(1)) &
        (fast_ma > slow_ma) &
        (rsi > 30) & (rsi < 70)
    )
    exit = (fast_ma.shift(1) >= slow_ma.shift(1)) & (fast_ma < slow_ma)
    return _build_signal(entry, exit)


def _signal_momentum(df: pd.DataFrame, params: dict) -> pd.Series:
    roc = _s(ta.ROC(df["close"].values, timeperiod=10))
    volume_ma = _s(ta.SMA(df["volume"].values, timeperiod=20))
    rsi = _s(ta.RSI(df["close"].values, timeperiod=14))
    entry = (
        (roc > 2.0) &
        (df["volume"] > volume_ma * 1.5) &
        (rsi > 50) & (rsi < 75)
    )
    exit = (roc < 0) | (rsi > 75)
    return _build_signal(entry, exit)


def _signal_breakout(df: pd.DataFrame, params: dict) -> pd.Series:
    highest_high = _s(df["high"].rolling(20).max().shift(1))
    volume_ma = _s(ta.SMA(df["volume"].values, timeperiod=20))
    atr = _s(ta.ATR(df["high"].values, df["low"].values, df["close"].values, timeperiod=14))
    entry = (
        (df["close"] > highest_high) &
        (df["volume"] > volume_ma * 1.3)
    )
    exit = (df["close"] < highest_high - atr * 2)
    return _build_signal(entry, exit)


def _signal_mean_reversion(df: pd.DataFrame, params: dict) -> pd.Series:
    upper, middle, lower = ta.BBANDS(
        df["close"].values, timeperiod=20, nbdevup=2.0, nbdevdn=2.0,
    )
    bb_upper = _s(upper)
    bb_middle = _s(middle)
    bb_lower = _s(lower)
    rsi = _s(ta.RSI(df["close"].values, timeperiod=14))
    distance = (df["close"] - bb_middle) / bb_middle
    entry = (
        (df["close"] < bb_lower) &
        (rsi < 35) &
        (distance < -0.02)
    )
    exit = (df["close"] > bb_middle) | (rsi > 60)
    return _build_signal(entry, exit)


def _signal_volatility_squeeze(df: pd.DataFrame, params: dict) -> pd.Series:
    upper, middle, lower = ta.BBANDS(
        df["close"].values, timeperiod=20, nbdevup=2.0, nbdevdn=2.0,
    )
    bb_upper = _s(upper)
    bb_middle = _s(middle)
    bb_lower = _s(lower)
    bb_width = (bb_upper - bb_lower) / bb_middle
    bb_width_min = _s(bb_width.rolling(120).min())

    macd_line, signal_line, _ = ta.MACD(df["close"].values.astype(float))
    macd = _s(macd_line)
    macdsignal = _s(signal_line)

    entry = (
        (bb_width <= bb_width_min * 1.05) &
        (macd > macdsignal)
    )
    exit = (
        (bb_width > bb_width_min * 3) |
        (macd < macdsignal)
    )
    return _build_signal(entry, exit)


def _signal_sentiment_driven(df: pd.DataFrame, params: dict) -> pd.Series:
    rsi = _s(ta.RSI(df["close"].values, timeperiod=14))
    sma50 = _s(ta.SMA(df["close"].values, timeperiod=50))
    entry = (rsi < 40) & (df["close"] > sma50)
    exit = (rsi > 65) | (df["close"] < sma50)
    return _build_signal(entry, exit)


def _signal_multi_timeframe(df: pd.DataFrame, params: dict) -> pd.Series:
    sma20 = _s(ta.SMA(df["close"].values, timeperiod=20))
    sma50 = _s(ta.SMA(df["close"].values, timeperiod=50))
    rsi = _s(ta.RSI(df["close"].values, timeperiod=14))
    sma80 = _s(ta.SMA(df["close"].values, timeperiod=80))
    sma200 = _s(ta.SMA(df["close"].values, timeperiod=200))
    adx = _s(ta.ADX(df["high"].values, df["low"].values, df["close"].values, timeperiod=14))
    entry = (
        (sma20.shift(1) <= sma50.shift(1)) &
        (sma20 > sma50) &
        (df["close"] > sma200) &
        (adx > 20) &
        (rsi > 40) & (rsi < 70)
    )
    exit = (
        (sma20.shift(1) >= sma50.shift(1)) &
        (sma20 < sma50)
    ) | (df["close"] < sma200)
    return _build_signal(entry, exit)


# ── Strategy Registry ──

REGISTRY: Dict[str, Any] = {
    "sma_crossover": _signal_sma_crossover,
    "macd_crossover": _signal_macd_crossover,
    "rsi_oversold": _signal_rsi_oversold,
    "bollinger_bands": _signal_bollinger_bands,
    "combined_sma_rsi": _signal_combined_sma_rsi,
    "momentum": _signal_momentum,
    "breakout": _signal_breakout,
    "mean_reversion": _signal_mean_reversion,
    "volatility_squeeze": _signal_volatility_squeeze,
    "sentiment_driven": _signal_sentiment_driven,
    "multi_timeframe": _signal_multi_timeframe,
}


class SignalFactory:
    """Fast vectorized signal generation for all supported strategy types.

    Each strategy type maps to a dedicated function that mirrors the
    Freqtrade template exactly. Use :meth:`generate` to get a signal series.
    """

    @staticmethod
    def generate(df: pd.DataFrame, strategy_type: str,
                 params: Optional[dict] = None) -> pd.Series:
        """Generate entry/exit signals for the given strategy.

        Args:
            df: OHLCV DataFrame with at least 'open', 'high', 'low', 'close', 'volume'.
            strategy_type: One of the keys in REGISTRY.
            params: Strategy parameters (dict of param_name -> value).

        Returns:
            pd.Series: 1 for entry, -1 for exit, 0 otherwise.

        Raises:
            ValueError: If strategy_type is unknown.
        """
        fn = REGISTRY.get(strategy_type)
        if fn is None:
            raise ValueError(f"Unknown strategy type: {strategy_type}. "
                             f"Known: {list(REGISTRY.keys())}")
        return fn(df, params or {})

    @staticmethod
    def supported_types() -> List[str]:
        return list(REGISTRY.keys())


# ── Fast Metrics ──

class FastMetrics:
    """Vectorized performance metrics from a signal series.

    Computes Sharpe ratio, win rate, max drawdown, and trade count
    from entry/exit signals and OHLCV data — no subprocess needed.
    """

    @staticmethod
    def compute(df: pd.DataFrame, signals: pd.Series,
                portfolio_value: float = 10000.0) -> Dict[str, Any]:
        """Compute fast performance metrics from signals.

        Args:
            df: OHLCV DataFrame (same as passed to SignalFactory).
            signals: Signal series from SignalFactory.generate().
            portfolio_value: Starting portfolio value (default 10000).

        Returns:
            dict with sharpe_ratio, win_rate, max_drawdown, total_trades,
            total_return_pct, num_entries, num_exits, passed.
        """
        entry_idx = signals[signals == 1].index
        exit_idx = signals[signals == -1].index

        num_entries = len(entry_idx)
        num_exits = len(exit_idx)

        # Match entries to exits (simplified: first entry paired with first exit)
        trade_returns = []
        for i in range(min(num_entries, num_exits)):
            entry_price = float(df.loc[entry_idx[i], "close"])
            exit_price = float(df.loc[exit_idx[i], "close"])
            ret = (exit_price - entry_price) / entry_price
            trade_returns.append(ret)

        total_trades = len(trade_returns)

        if total_trades < 1:
            return FastMetrics._empty(portfolio_value)

        # Win rate
        wins = sum(1 for r in trade_returns if r > 0)
        win_rate = wins / total_trades

        # Sharpe (annualized from trade returns)
        returns_arr = np.array(trade_returns)
        sharpe = 0.0
        if len(returns_arr) > 1 and returns_arr.std() > 0:
            sharpe = (returns_arr.mean() / returns_arr.std()) * math.sqrt(252)

        # Equity curve
        equity = portfolio_value
        running = [equity]
        for r in trade_returns:
            equity *= (1 + r)
            running.append(equity)
        total_return_pct = (running[-1] - portfolio_value) / portfolio_value

        # Max drawdown
        running_arr = np.array(running)
        peak = np.maximum.accumulate(running_arr)
        dd = (running_arr - peak) / peak
        max_dd = float(abs(dd.min())) if len(dd) > 0 else 0.0

        # Threshold check
        passed = (
            sharpe >= settings.VECTORBT_PREFILTER_MIN_SHARPE
            and win_rate >= settings.VECTORBT_PREFILTER_MIN_WIN_RATE
            and total_trades >= settings.VECTORBT_PREFILTER_MIN_TRADES
        )

        return {
            "sharpe_ratio": round(float(sharpe), 4),
            "win_rate": round(float(win_rate), 4),
            "max_drawdown": round(float(max_dd), 4),
            "total_trades": total_trades,
            "total_return_pct": round(float(total_return_pct), 4),
            "num_entries": num_entries,
            "num_exits": num_exits,
            "passed": bool(passed),
        }

    @staticmethod
    def _empty(portfolio_value: float) -> dict:
        return {
            "sharpe_ratio": 0.0,
            "win_rate": 0.0,
            "max_drawdown": 0.0,
            "total_trades": 0,
            "total_return_pct": 0.0,
            "num_entries": 0,
            "num_exits": 0,
            "passed": False,
        }
