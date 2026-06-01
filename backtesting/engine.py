"""Backtesting engine wrapping Freqtrade subprocess execution.

Generates temporary strategy files on the fly, runs backtests via the
Freqtrade CLI, and parses results into pandas DataFrames.
"""

import ast
import json
import logging
import uuid
import os
import re
import shutil
import string
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from config import settings
from backtesting.data_split import DATA_SPLIT

logger = logging.getLogger(__name__)


# ── Timerange sanitizer ──

def _sanitize_timerange(raw: str) -> str:
    """Convert any LLM-invented date format to freqtrade's ``YYYYMMDD-YYYYMMDD``.

    Handles all common variants:
      "2024-01-01/2024-12-31" -> "20240101-20241231"
      "2024-01-01-2024-12-31" -> "20240101-20241231"
      "2024-01-01"            -> "20240101-"
      "20240101-20241231"     -> unchanged (already valid)
      "20240101-"             -> unchanged
      "2024"                  -> "2024"
    """
    raw = raw.strip()
    if not raw:
        return "20210101-"
    # Separate by / or whitespace first, then by dash
    # Extract all digit groups: "2024-01-01/2024-12-31" -> [2024,01,01,2024,12,31]
    groups = re.findall(r'\d+', raw)
    if not groups:
        return "20210101-"
    # If there's a "/" or "-" separator between dates, groups are split into two dates
    # Detect: if groups have 6+ entries, treat as two dates of 3 groups each (YYYY MM DD)
    if len(groups) >= 6:
        # Two dates: first 3 groups = date1, next 3 = date2
        d1 = "".join(groups[:3])[:8]
        d2 = "".join(groups[3:6])[:8]
        return f"{d1}-{d2}"
    if len(groups) == 5:
        # Two dates, first has 3 groups, second has 2 (YYYY MM -> YYYYMM)
        d1 = "".join(groups[:3])[:8]
        d2 = "".join(groups[3:])[:8]
        return f"{d1}-{d2}"
    if len(groups) == 4:
        # Could be YYYYMMDD-YYYYMMDD split, or YYYY MM DD YYYY
        # If any group has length 2, it's likely YYYY MM DD YYYY
        if any(len(g) <= 2 for g in groups):
            d1 = "".join(groups[:2])[:8]
            d2 = "".join(groups[2:])[:8]
            return f"{d1}-{d2}"
        # Otherwise it's already two 8-digit dates
        return f"{groups[0][:8]}-{groups[1][:8]}"
    if len(groups) == 3:
        # Single date in YYYY MM DD format
        d = "".join(groups)[:8]
        return f"{d}-"
    if len(groups) == 2:
        # Could be YYYYMMDD-YYYYMMDD without hyphen, or YYYY MM alone
        if all(len(g) == 8 for g in groups):
            return f"{groups[0][:8]}-{groups[1][:8]}"
        # Two groups: likely YYYY and MM -> pad
        d = "".join(groups)[:8]
        return f"{d}-" if len(d) == 8 else d
    # Single group: "20240101" or "2024" or "20240101-"
    d = groups[0][:8]
    if len(d) == 8 and raw.endswith('-'):
        return f"{d}-"
    return f"{d}-" if len(d) == 8 else d

# ── Strategy template injected as a temp .py file ──

STRATEGY_TEMPLATE = '''"""
Auto-generated strategy by crypto_agent_bot.
Do not edit manually — generated on $timestamp.
"""
from freqtrade.strategy import IStrategy, IntParameter
import pandas as pd
import talib.abstract as ta


class $strategy_name(IStrategy):
    # --- User-defined parameters (set by agent) ---
    timeframe = "$timeframe"
    minimal_roi = $minimal_roi
    stoploss = $stoploss
    trailing_stop = $trailing_stop
    startup_candle_count = $startup_candle_count
    process_only_new_candles = True
    use_exit_signal = True
    can_short = False

    # --- Indicator parameters ---
$indicator_params_block
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # Coerce string-typed columns upfront (covers PyArrow backend which stores
        # strings as pd.ArrowDtype(pa.string()), not caught by simple 'string' check)
        import pandas.api.types as ptypes
        for col in dataframe.columns:
            if ptypes.is_string_dtype(dataframe[col]):
                dataframe[col] = pd.to_numeric(dataframe[col], errors='coerce')
        $indicator_code
        # Second pass: catch any new columns created by indicator code
        for col in dataframe.columns:
            if ptypes.is_string_dtype(dataframe[col]):
                dataframe[col] = pd.to_numeric(dataframe[col], errors='coerce')
        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[
            (
                $entry_condition
            ),
            "enter_long"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[
            (
                $exit_condition
            ),
            "exit_long"] = 1
        return dataframe
'''


# ── Default SMA crossover indicator/entry/exit snippets ──

SMA_CROSSOVER_INDICATOR = """
        dataframe['fast_ma'] = ta.SMA(dataframe, timeperiod=self.fast_ma.value)
        dataframe['slow_ma'] = ta.SMA(dataframe, timeperiod=self.slow_ma.value)
"""

SMA_CROSSOVER_ENTRY = """
        (dataframe['fast_ma'].shift(1) <= dataframe['slow_ma'].shift(1)) &
        (dataframe['fast_ma'] > dataframe['slow_ma'])
"""

SMA_CROSSOVER_EXIT = """
        (dataframe['fast_ma'].shift(1) >= dataframe['slow_ma'].shift(1)) &
        (dataframe['fast_ma'] < dataframe['slow_ma'])
"""

# ── MACD Crossover snippets ──

MACD_CROSSOVER_INDICATOR = """
        macd_data = ta.MACD(
            dataframe,
            fastperiod=self.macd_fast.value,
            slowperiod=self.macd_slow.value,
            signalperiod=self.macd_signal.value,
        )
        dataframe['macd'] = macd_data['macd'].astype(float)
        dataframe['macdsignal'] = macd_data['macdsignal'].astype(float)
        dataframe['macd_hist'] = (dataframe['macd'] - dataframe['macdsignal']).astype(float)
"""

MACD_CROSSOVER_ENTRY = """
        (dataframe['macd_hist'].shift(1) <= 0) & (dataframe['macd_hist'] > 0)
"""

MACD_CROSSOVER_EXIT = """
        (dataframe['macd_hist'].shift(1) >= 0) & (dataframe['macd_hist'] < 0)
"""

# ── RSI Oversold/Overbought snippets ──

RSI_INDICATOR = """
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.rsi_period.value)
"""

RSI_OVERSOLD_ENTRY = """
        (dataframe['rsi'] < self.rsi_buy_threshold.value) & (dataframe['rsi'].shift(1) >= self.rsi_buy_threshold.value)
"""

RSI_OVERSOLD_EXIT = """
        (dataframe['rsi'] > self.rsi_sell_threshold.value) & (dataframe['rsi'].shift(1) <= self.rsi_sell_threshold.value)
"""

# ── Bollinger Bands snippets ──

BB_INDICATOR = """
        upper, middle, lower = ta.BBANDS(
            dataframe['close'].astype(float),
            timeperiod=self.bb_period.value,
            nbdevup=2.0,
            nbdevdn=2.0,
        )
        dataframe['bb_upper'] = upper.astype(float)
        dataframe['bb_middle'] = middle.astype(float)
        dataframe['bb_lower'] = lower.astype(float)
"""

BB_ENTRY = """
        (dataframe['close'] < dataframe['bb_lower']) & (dataframe['close'].shift(1) >= dataframe['bb_lower'].shift(1))
"""

BB_EXIT = """
        (dataframe['close'] > dataframe['bb_upper']) & (dataframe['close'].shift(1) <= dataframe['bb_upper'].shift(1))
"""

# ── Combined SMA + RSI filter snippets ──

SMA_RSI_INDICATOR = """
        dataframe['fast_ma'] = ta.SMA(dataframe, timeperiod=self.fast_ma.value)
        dataframe['slow_ma'] = ta.SMA(dataframe, timeperiod=self.slow_ma.value)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
"""

SMA_RSI_ENTRY = """
        (dataframe['fast_ma'].shift(1) <= dataframe['slow_ma'].shift(1)) &
        (dataframe['fast_ma'] > dataframe['slow_ma']) &
        (dataframe['rsi'] > 30) & (dataframe['rsi'] < 70)
"""

SMA_RSI_EXIT = """
        (dataframe['fast_ma'].shift(1) >= dataframe['slow_ma'].shift(1)) &
        (dataframe['fast_ma'] < dataframe['slow_ma'])
"""

# ── Strategy registry: maps type to its snippets and defaults ──

STRATEGY_REGISTRY: Dict[str, Dict[str, Any]] = {
    "sma_crossover": {
        "indicator_code": SMA_CROSSOVER_INDICATOR,
        "entry_condition": SMA_CROSSOVER_ENTRY,
        "exit_condition": SMA_CROSSOVER_EXIT,
        "indicator_params_block": """
    fast_ma = IntParameter(5, 50, default=$fast_ma, space="buy")
    slow_ma = IntParameter(20, 200, default=$slow_ma, space="buy")
""",
        "default_params": {"fast_ma": 10, "slow_ma": 30, "startup_candle_count": 30},
    },
    "macd_crossover": {
        "indicator_code": MACD_CROSSOVER_INDICATOR,
        "entry_condition": MACD_CROSSOVER_ENTRY,
        "exit_condition": MACD_CROSSOVER_EXIT,
        "indicator_params_block": """
    macd_fast = IntParameter(8, 20, default=12, space="buy")
    macd_slow = IntParameter(20, 40, default=26, space="buy")
    macd_signal = IntParameter(6, 14, default=9, space="buy")
""",
        "default_params": {"startup_candle_count": 33},
    },
    "rsi_oversold": {
        "indicator_code": RSI_INDICATOR,
        "entry_condition": RSI_OVERSOLD_ENTRY,
        "exit_condition": RSI_OVERSOLD_EXIT,
        "indicator_params_block": """
    rsi_period = IntParameter(10, 21, default=14, space="buy")
    rsi_buy_threshold = IntParameter(25, 35, default=30, space="buy")
    rsi_sell_threshold = IntParameter(65, 80, default=70, space="sell")
""",
        "default_params": {"startup_candle_count": 20},
    },
    "bollinger_bands": {
        "indicator_code": BB_INDICATOR,
        "entry_condition": BB_ENTRY,
        "exit_condition": BB_EXIT,
        "indicator_params_block": """
    bb_period = IntParameter(15, 30, default=20, space="buy")
""",
        "default_params": {"startup_candle_count": 26},
    },
    "combined_sma_rsi": {
        "indicator_code": SMA_RSI_INDICATOR,
        "entry_condition": SMA_RSI_ENTRY,
        "exit_condition": SMA_RSI_EXIT,
        "indicator_params_block": """
    fast_ma = IntParameter(5, 50, default=$fast_ma, space="buy")
    slow_ma = IntParameter(20, 200, default=$slow_ma, space="buy")
""",
        "default_params": {"fast_ma": 10, "slow_ma": 30, "startup_candle_count": 30},
    },
    "custom": {
        "indicator_code": "",
        "entry_condition": "",
        "exit_condition": "",
        "indicator_params_block": "",
        "default_params": {"startup_candle_count": 20},
    },

    "momentum": {
        "indicator_code": """
        dataframe['roc'] = ta.ROC(dataframe, timeperiod=10)
        dataframe['volume_ma'] = ta.SMA(dataframe['volume'], timeperiod=20)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
    """,
        "entry_condition": """
        (dataframe['roc'] > 2.0) &
        (dataframe['volume'] > dataframe['volume_ma'] * 1.5) &
        (dataframe['rsi'] > 50) & (dataframe['rsi'] < 75)
    """,
        "exit_condition": """
        (dataframe['roc'] < 0) | (dataframe['rsi'] > 75)
    """,
        "indicator_params_block": "",
        "default_params": {"startup_candle_count": 25},
    },

    "breakout": {
        "indicator_code": """
        dataframe['highest_high'] = dataframe['high'].rolling(20).max().shift(1)
        dataframe['volume_ma'] = ta.SMA(dataframe['volume'], timeperiod=20)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
    """,
        "entry_condition": """
        (dataframe['close'] > dataframe['highest_high']) &
        (dataframe['volume'] > dataframe['volume_ma'] * 1.3)
    """,
        "exit_condition": """
        (dataframe['close'] < dataframe['highest_high'] - dataframe['atr'] * 2)
    """,
        "indicator_params_block": "",
        "default_params": {"startup_candle_count": 25},
    },

    "mean_reversion": {
        "indicator_code": """
        bb_upper, bb_middle, bb_lower = ta.BBANDS(
            dataframe['close'], timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe['bb_upper'] = bb_upper.astype(float)
        dataframe['bb_middle'] = bb_middle.astype(float)
        dataframe['bb_lower'] = bb_lower.astype(float)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['distance_from_mean'] = (dataframe['close'] - dataframe['bb_middle']) / dataframe['bb_middle']
    """,
        "entry_condition": """
        (dataframe['close'] < dataframe['bb_lower']) &
        (dataframe['rsi'] < 35) &
        (dataframe['distance_from_mean'] < -0.02)
    """,
        "exit_condition": """
        (dataframe['close'] > dataframe['bb_middle']) | (dataframe['rsi'] > 60)
    """,
        "indicator_params_block": "",
        "default_params": {"startup_candle_count": 25},
    },

    "volatility_squeeze": {
        "indicator_code": """
        bb_upper, bb_middle, bb_lower = ta.BBANDS(
            dataframe['close'], timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe['bb_upper'] = bb_upper.astype(float)
        dataframe['bb_middle'] = bb_middle.astype(float)
        dataframe['bb_lower'] = bb_lower.astype(float)
        dataframe['bb_width'] = (dataframe['bb_upper'] - dataframe['bb_lower']) / dataframe['bb_middle']
        dataframe['bb_width_min'] = dataframe['bb_width'].rolling(120).min()
        dataframe['macd'], dataframe['macdsignal'], _ = [
            x.astype(float) for x in ta.MACD(dataframe['close'].astype(float))]
    """,
        "entry_condition": """
        (dataframe['bb_width'] <= dataframe['bb_width_min'] * 1.05) &
        (dataframe['macd'] > dataframe['macdsignal'])
    """,
        "exit_condition": """
        (dataframe['bb_width'] > dataframe['bb_width_min'] * 3) |
        (dataframe['macd'] < dataframe['macdsignal'])
    """,
        "indicator_params_block": "",
        "default_params": {"startup_candle_count": 130},
    },

    "sentiment_driven": {
        "indicator_code": """
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['sma50'] = ta.SMA(dataframe, timeperiod=50)
    """,
        "entry_condition": """
        (dataframe['rsi'] < 40) &
        (dataframe['close'] > dataframe['sma50'])
    """,
        "exit_condition": """
        (dataframe['rsi'] > 65) | (dataframe['close'] < dataframe['sma50'])
    """,
        "indicator_params_block": "",
        "default_params": {"startup_candle_count": 55},
    },

    "multi_timeframe": {
        "indicator_code": """
        # 1h timeframe indicators (primary)
        dataframe['sma20'] = ta.SMA(dataframe, timeperiod=20)
        dataframe['sma50'] = ta.SMA(dataframe, timeperiod=50)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        # Proxy for 4h trend: use longer SMAs on 1h data (approx 4x periods)
        dataframe['sma80'] = ta.SMA(dataframe, timeperiod=80)
        dataframe['sma200'] = ta.SMA(dataframe, timeperiod=200)
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
    """,
        "entry_condition": """
        # Short-term signal: 20 crosses above 50
        (dataframe['sma20'].shift(1) <= dataframe['sma50'].shift(1)) &
        (dataframe['sma20'] > dataframe['sma50']) &
        # Long-term confirmation: price above 200 SMA (higher timeframe proxy)
        (dataframe['close'] > dataframe['sma200']) &
        # Trend strength: ADX > 20
        (dataframe['adx'] > 20) &
        # RSI not overbought
        (dataframe['rsi'] > 40) & (dataframe['rsi'] < 70)
    """,
        "exit_condition": """
        (dataframe['sma20'].shift(1) >= dataframe['sma50'].shift(1)) &
        (dataframe['sma20'] < dataframe['sma50']) |
        (dataframe['close'] < dataframe['sma200'])
    """,
        "indicator_params_block": "",
        "default_params": {"startup_candle_count": 205},
    },
}


class BacktestEngine:
    """Runs Freqtrade backtests by generating temporary strategy files."""

    def __init__(self, ft_userdata_dir: str = "./ft_userdata"):
        self.ft_userdata_dir = Path(ft_userdata_dir).resolve()
        self._config: Optional[Dict[str, Any]] = None
        # Locate freqtrade executable in the venv
        self._freqtrade_cmd = self._find_freqtrade()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _find_freqtrade(self) -> str:
        """Find the freqtrade executable, preferring the venv Scripts dir."""
        # Check for freqtrade in common locations
        candidates = [
            Path(__file__).parent.parent / "venv" / "Scripts" / "freqtrade.exe",
            Path(__file__).parent.parent / "venv" / "Scripts" / "freqtrade",
            shutil.which("freqtrade"),
        ]
        for path in candidates:
            if path and Path(path).exists():
                logger.debug("Using freqtrade at: %s", path)
                return str(path)
        return "freqtrade"  # fallback — hope it's on PATH

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_backtest(
        self,
        strategy_params: Optional[Dict[str, Any]] = None,
        strategy_type: str = "sma_crossover",
        timerange: str = "20210101-",
        pairs: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Generate a strategy from *params*, run a Freqtrade backtest,
        and return parsed results."""
        timerange = _sanitize_timerange(timerange)

        # HOLDOUT GUARD — never allow research backtests to touch holdout data
        if DATA_SPLIT.is_in_holdout(timerange):
            raise ValueError(
                f"HOLDOUT VIOLATION: Attempted to run research backtest on holdout window. "
                f"Timerange '{timerange}' includes data from {DATA_SPLIT.holdout_start} or later."
                "Holdout data is reserved for OOSValidator.final_oos_validation() only."
            )

        # Force research window if timerange is the default (open-ended)
        if timerange == "20210101-" or timerange.startswith("20210101"):
            timerange = DATA_SPLIT.research_timerange()
            logger.info("Forcing timerange to research window: %s", timerange)

        params = self._default_strategy_params(strategy_type)
        if strategy_params:
            # Pop timerange and pairs if they were stored in strategy params
            params.update(strategy_params)
        params.setdefault("timeframe", settings.TIMEFRAME)

        # 1. Write temporary strategy file with UUID to avoid collision
        strategy_name = f"Strategy_{uuid.uuid4().hex[:8]}"
        params["strategy_name"] = strategy_name
        strategy_code = self._render_strategy(params)
        strategy_dir = self.ft_userdata_dir / "strategies"
        strategy_dir.mkdir(parents=True, exist_ok=True)
        strategy_path = strategy_dir / f"{strategy_name}.py"
        strategy_path.write_text(strategy_code, encoding="utf-8")
        # Clean up old generated strategy files
        for f in strategy_dir.glob("Strategy_*.py"):
            if f.name != strategy_path.name:
                try: f.unlink()
                except Exception: pass

        # 2a. Validate the generated Python is syntactically valid
        self._validate_strategy(strategy_code)

        pairs = pairs or [settings.SYMBOL]

        # 2. Build config
        config = self._build_config(pairs, timerange)

        # 3. Run via subprocess
        result = self._run_freqtrade_backtest(config, strategy_path)

        # 4. Parse result JSON
        return self._parse_results(result, strategy_name=strategy_name)

    def get_performance_metrics(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute standard performance metrics from a trades DataFrame.

        Expected columns: profit_ratio, enter_tag, exit_tag, open_date, close_date.
        """
        if trades_df.empty:
            return {"error": "No trades", "win_rate": 0.0, "sharpe": 0.0}

        wins = trades_df[trades_df["profit_ratio"] > 0]
        losses = trades_df[trades_df["profit_ratio"] <= 0]
        win_rate = len(wins) / len(trades_df) if len(trades_df) > 0 else 0.0

        total_profit = trades_df["profit_ratio"].sum()
        profit_factor = (
            wins["profit_ratio"].sum() / abs(losses["profit_ratio"].sum())
            if not losses.empty and losses["profit_ratio"].sum() != 0
            else float("inf")
        )

        returns = trades_df["profit_ratio"]
        sharpe = 0.0
        if len(returns) > 1 and returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * (365 * 24) ** 0.5  # annualised for 1h

        cummax = trades_df["profit_ratio"].cummax()
        drawdown = (trades_df["profit_ratio"] - cummax).min()

        return {
            "total_trades": len(trades_df),
            "win_rate": round(win_rate, 4),
            "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else None,
            "total_profit": round(total_profit, 4),
            "max_drawdown": round(drawdown, 4),
            "sharpe_ratio": round(sharpe, 2),
            "avg_profit_win": round(wins["profit_ratio"].mean(), 4) if not wins.empty else 0.0,
            "avg_profit_loss": round(losses["profit_ratio"].mean(), 4) if not losses.empty else 0.0,
        }

    def download_data(self, pairs: Optional[List[str]] = None, timerange: str = "20210101-"):
        """Download historical data via ``freqtrade download-data``."""
        timerange = _sanitize_timerange(timerange)

        # HOLDOUT GUARD — never download holdout data for research
        if DATA_SPLIT.is_in_holdout(timerange):
            raise ValueError(
                f"HOLDOUT VIOLATION: Refusing to download holdout data for research. "
                f"Timerange '{timerange}' overlaps holdout window starting "
                f"{DATA_SPLIT.holdout_start}. Holdout data is reserved for "
                f"OOSValidator.final_oos_validation() only."
            )

        pairs = pairs or [settings.SYMBOL]

        def _build_cmd(prepend=False):
            cmd = [
                self._freqtrade_cmd,
                "download-data",
                "--userdir", str(self.ft_userdata_dir),
                "--exchange", settings.EXCHANGE_ID,
                "-p", *pairs,
                "--timerange", timerange,
                "--timeframes", settings.TIMEFRAME,
                "--datadir", str(self.ft_userdata_dir / "data"),
            ]
            if prepend:
                cmd.append("--prepend")
            return cmd

        logger.info("Downloading data: %s", " ".join(_build_cmd()))
        result = subprocess.run(
            _build_cmd(),
            capture_output=True, text=True,
            timeout=settings.BACKTEST_TIMEOUT
        )

        # Detect the prepend warning
        combined = (result.stdout or "") + (result.stderr or "")
        if "Use `--prepend`" in combined or "--prepend" in combined:
            logger.info("Local data exists but requested range is earlier — retrying with --prepend")
            result = subprocess.run(
                _build_cmd(prepend=True),
                capture_output=True, text=True,
                timeout=settings.BACKTEST_TIMEOUT
            )

        if result.returncode != 0:
            raise RuntimeError(f"Data download failed:\n{result.stderr}")

        # Check for zero-length download
        if "length 0" in (result.stdout + result.stderr):
            raise RuntimeError(
                f"Download returned 0 candles for timerange {timerange}. "
                "The requested date range may not be available from the exchange. "
                "Try a more recent timerange or use --prepend."
            )

        logger.info("Data download complete.")

    def walk_forward_validate(
        self,
        strategy_params: Dict[str, Any],
        strategy_type: str = "sma_crossover",
        start_date: str = "",
        end_date: str = "",
        windows: int = 3,
        pairs: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Split timerange into N windows and backtest each independently.
        Returns per-window metrics and an overall consistency score.
        A strategy is only valid if it performs reasonably across ALL windows.

        Uses DATA_SPLIT research window. Never touches holdout data.
        """
        from datetime import datetime, timedelta

        # Use DATA_SPLIT's predefined research window and WFV splits
        splits = DATA_SPLIT.wfv_splits(n_splits=windows)
        if not splits:
            return {
                "error": "No WFV splits could be generated from research window",
                "windows": [], "consistency_score": 0,
                "avg_sharpe": 0, "avg_win_rate": 0, "is_robust": False,
            }

        window_results = []
        for i, (train_tr, test_tr) in enumerate(splits):
            try:
                # Run on test window only (validation within research window)
                result = self.run_backtest(
                    strategy_params=dict(strategy_params),
                    strategy_type=strategy_type,
                    timerange=test_tr,
                    pairs=pairs,
                )
                window_results.append({
                    "window": i + 1,
                    "train_timerange": train_tr,
                    "test_timerange": test_tr,
                    "sharpe": result.get("sharpe_ratio", 0),
                    "win_rate": result.get("win_rate", 0),
                    "max_drawdown": result.get("max_drawdown", 0),
                    "total_trades": result.get("total_trades", 0),
                    "error": result.get("error", ""),
                })
            except Exception as exc:
                window_results.append({
                    "window": i + 1,
                    "train_timerange": train_tr,
                    "test_timerange": test_tr,
                    "error": str(exc),
                    "sharpe": 0, "win_rate": 0,
                    "max_drawdown": 0, "total_trades": 0,
                })

        # Consistency score: % of windows with positive Sharpe
        valid_windows = [w for w in window_results if w["sharpe"] > 0 and w["total_trades"] >= 3]
        consistency = len(valid_windows) / windows if windows > 0 else 0

        avg_sharpe = sum(w["sharpe"] for w in window_results) / windows if windows > 0 else 0
        avg_win_rate = sum(w["win_rate"] for w in window_results) / windows if windows > 0 else 0

        return {
            "windows": window_results,
            "consistency_score": round(consistency, 2),
            "avg_sharpe": round(avg_sharpe, 3),
            "avg_win_rate": round(avg_win_rate, 3),
            "is_robust": consistency >= 0.67,  # passes 2/3 windows minimum
        }

    # ------------------------------------------------------------------
    # Strategy validation
    # ------------------------------------------------------------------

    def _validate_strategy(self, source: str) -> None:
        """Parse the generated strategy Python for syntax errors before
        handing to Freqtrade. Raises ``ValueError`` on bad syntax so the
        strategist agent can catch and regenerate."""
        try:
            ast.parse(source, filename="DynamicStrategy.py")
        except SyntaxError as e:
            msg = f"Generated strategy has invalid Python syntax (line {e.lineno}): {e.msg}"
            logger.error(msg)
            raise ValueError(msg) from e

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _default_strategy_params(self, strategy_type: str = "sma_crossover") -> Dict[str, Any]:
        registry_entry = STRATEGY_REGISTRY.get(strategy_type, STRATEGY_REGISTRY["sma_crossover"])
        params = {
            "indicator_code": registry_entry["indicator_code"],
            "entry_condition": registry_entry["entry_condition"],
            "exit_condition": registry_entry["exit_condition"],
            "indicator_params_block": registry_entry["indicator_params_block"],
            "stoploss": -0.05,
            "trailing_stop": False,
            "minimal_roi": '{"0": 0.01}',
            "timeframe": settings.TIMEFRAME,
        }
        params.update(registry_entry["default_params"])
        return params

    def _render_strategy(self, params: Dict[str, Any]) -> str:
        import datetime
        params["timestamp"] = datetime.datetime.utcnow().isoformat()
        # Nested substitution for indicator_params_block which may contain $fast_ma/$slow_ma
        if "indicator_params_block" in params and "$" in params.get("indicator_params_block", ""):
            params["indicator_params_block"] = string.Template(params["indicator_params_block"]).safe_substitute(**params)
        return string.Template(STRATEGY_TEMPLATE).safe_substitute(**params)

    def _build_config(self, pairs: List[str], timerange: str) -> Dict[str, Any]:
        config_path = self.ft_userdata_dir / "config.json"
        with open(config_path) as f:
            config = json.load(f)

        config["exchange"]["pair_whitelist"] = pairs
        config["timerange"] = timerange
        config["timeframe"] = settings.TIMEFRAME
        config["dry_run"] = True
        # Required pricing sections for Freqtrade 2026.4
        config.setdefault("entry_pricing", {"price_side": "same", "use_order_book": False})
        config.setdefault("exit_pricing", {"price_side": "same", "use_order_book": False})
        # Ensure data dir is absolute
        config["datadir"] = str(self.ft_userdata_dir / "data")
        config["user_data_dir"] = str(self.ft_userdata_dir)
        return config

    def _run_freqtrade_backtest(self, config: Dict[str, Any], strategy_path: Path) -> Dict:
        """Execute ``freqtrade backtesting`` via subprocess and capture ZIP-packed JSON output."""
        export_path = self.ft_userdata_dir / "backtest_results"
        export_path.mkdir(parents=True, exist_ok=True)
        strategy_name = strategy_path.stem  # Derive strategy name from the filename

        # Write temp config
        temp_config = self.ft_userdata_dir / "temp_backtest_config.json"
        with open(temp_config, "w") as f:
            json.dump(config, f, indent=2)

        cmd = [
            self._freqtrade_cmd,
            "backtesting",
            "--userdir", str(self.ft_userdata_dir),
            "--config", str(temp_config),
            "--strategy-path", str(strategy_path.parent),
            "--strategy", strategy_name,
            "--export", "trades",
            "--backtest-directory", str(export_path),
        ]

        logger.info("Running backtest: %s", " ".join(cmd))
        proc_result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=settings.BACKTEST_TIMEOUT
        )
        stderr = proc_result.stderr or ""
        stdout = proc_result.stdout or ""

        if proc_result.returncode != 0:
            logger.error("Backtest stderr:\n%s", stderr)
            raise RuntimeError(f"Freqtrade backtest failed: {stderr}")

        if "No data found" in stderr or "No data found" in stdout:
            raise RuntimeError(
                "Freqtrade found no data for the requested timerange. "
                f"Stderr: {stderr[-500:]}"
            )

        logger.info("Backtest completed.")

        # Find result ZIP
        last_result_file = export_path / ".last_result.json"
        if last_result_file.exists():
            with open(last_result_file) as lrf:
                meta = json.load(lrf)
            zip_path = export_path / meta.get("latest_backtest", "")
        else:
            zips = sorted(export_path.glob("*.zip"), key=os.path.getmtime, reverse=True)
            if not zips:
                logger.warning("No result ZIP found — backtest likely produced 0 trades")
                return {"strategy": {strategy_name: {"total_trades": 0, "trades": []}}}
            zip_path = zips[0]

        if not zip_path.exists():
            logger.warning("Result ZIP missing — backtest likely produced 0 trades")
            return {"strategy": {strategy_name: {"total_trades": 0, "trades": []}}}

        import zipfile
        with zipfile.ZipFile(zip_path) as zf:
            json_name = [n for n in zf.namelist() if n.endswith(".json")][0]
            result_data = json.load(zf.open(json_name))
        # Dump for debugging (overwritten each run)
        debug_path = export_path / "last_result_debug.json"
        with open(debug_path, "w") as dbf:
            json.dump(result_data, dbf, indent=2, default=str)
        logger.debug("Raw result dumped to %s", debug_path)
        return result_data

    def _parse_results(self, raw: Dict[str, Any], strategy_name: str = "DynamicStrategy") -> Dict[str, Any]:
        """Parse Freqtrade result JSON into flat summary."""
        # Log raw structure for debugging
        if isinstance(raw, dict):
            logger.debug("Raw result top-level keys: %s", list(raw.keys()))

        try:
            strategy_block = raw.get("strategy", {})
            if not strategy_block:
                # Some versions put results directly at top level
                strategy_block = {strategy_name: raw}

            strat_name = next(iter(strategy_block))
            sd = strategy_block[strat_name]

            logger.debug("Parsing strategy block for '%s', keys: %s",
                         strat_name, list(sd.keys()))

            trades_raw = sd.get("trades", [])
            trades = pd.DataFrame(trades_raw) if trades_raw else pd.DataFrame()

            # Try multiple field name variants across Freqtrade versions
            def _get(d, *keys, default=0):
                for k in keys:
                    if k in d and d[k] is not None:
                        try:
                            return float(d[k])
                        except (TypeError, ValueError):
                            pass
                return default

            total_trades = int(_get(sd, "total_trades", "trades_count", default=len(trades)))
            sharpe = _get(sd, "sharpe", "sharpe_ratio", "sharpe_ratio_account")
            win_rate = _get(sd, "winrate", "win_rate", "wins_per_trade")
            drawdown = _get(sd, "max_drawdown_account", "max_drawdown", "max_drawdown_abs")
            profit = _get(sd, "profit_mean", "profit_total", "profit_factor",
                          "profit_total_abs")

            logger.info(
                "Parsed metrics: trades=%d sharpe=%.3f wr=%.3f dd=%.3f profit=%.4f",
                total_trades, sharpe, win_rate, drawdown, profit
            )

            metrics = {}
            if not trades.empty:
                metrics = self.get_performance_metrics(trades)

            return {
                "summary": sd,
                "total_trades": total_trades,
                "profit_ratio": profit or metrics.get("total_profit", 0),
                "max_drawdown": drawdown or metrics.get("max_drawdown", 0),
                "sharpe_ratio": sharpe or metrics.get("sharpe_ratio", 0),
                "win_rate": win_rate or metrics.get("win_rate", 0),
                "trades_df": trades,
                "raw": raw,
            }

        except (StopIteration, KeyError, TypeError, AttributeError) as exc:
            logger.warning(
                "Could not parse backtest result: %s\nRaw type: %s, keys: %s",
                exc,
                type(raw),
                list(raw.keys()) if isinstance(raw, dict) else "N/A"
            )
            return {
                "error": str(exc),
                "raw": raw,
                "total_trades": 0,
                "profit_ratio": 0,
                "max_drawdown": 0,
                "sharpe_ratio": 0,
                "win_rate": 0,
            }