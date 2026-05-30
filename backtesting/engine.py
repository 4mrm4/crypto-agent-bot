"""Backtesting engine wrapping Freqtrade subprocess execution.

Generates temporary strategy files on the fly, runs backtests via the
Freqtrade CLI, and parses results into pandas DataFrames.
"""

import ast
import json
import logging
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


class DynamicStrategy(IStrategy):
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
        $indicator_code
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
        macd_data = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe['macd'] = macd_data['macd'].astype(float)
        dataframe['macdsignal'] = macd_data['macdsignal'].astype(float)
        dataframe['macd'] = dataframe['macd'] - dataframe['macdsignal']
"""

MACD_CROSSOVER_ENTRY = """
        (dataframe['macd'].shift(1) <= 0) & (dataframe['macd'] > 0)
"""

MACD_CROSSOVER_EXIT = """
        (dataframe['macd'].shift(1) >= 0) & (dataframe['macd'] < 0)
"""

# ── RSI Oversold/Overbought snippets ──

RSI_INDICATOR = """
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
"""

RSI_OVERSOLD_ENTRY = """
        (dataframe['rsi'] < 30) & (dataframe['rsi'].shift(1) >= 30)
"""

RSI_OVERSOLD_EXIT = """
        (dataframe['rsi'] > 70) & (dataframe['rsi'].shift(1) <= 70)
"""

# ── Bollinger Bands snippets ──

BB_INDICATOR = """
        dataframe['bb_upper'], dataframe['bb_middle'], dataframe['bb_lower'] = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
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
        "indicator_params_block": "",
        "default_params": {"startup_candle_count": 33},
    },
    "rsi_oversold": {
        "indicator_code": RSI_INDICATOR,
        "entry_condition": RSI_OVERSOLD_ENTRY,
        "exit_condition": RSI_OVERSOLD_EXIT,
        "indicator_params_block": "",
        "default_params": {"startup_candle_count": 20},
    },
    "bollinger_bands": {
        "indicator_code": BB_INDICATOR,
        "entry_condition": BB_ENTRY,
        "exit_condition": BB_EXIT,
        "indicator_params_block": "",
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
        dataframe['bb_upper'], dataframe['bb_middle'], dataframe['bb_lower'] = ta.BBANDS(
            dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
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
        dataframe['bb_upper'], dataframe['bb_middle'], dataframe['bb_lower'] = ta.BBANDS(
            dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
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
        params = self._default_strategy_params(strategy_type)
        if strategy_params:
            # Pop timerange and pairs if they were stored in strategy params
            params.update(strategy_params)
        params.setdefault("timeframe", settings.TIMEFRAME)

        # 1. Write temporary strategy file
        strategy_code = self._render_strategy(params)
        strategy_dir = self.ft_userdata_dir / "strategies"
        strategy_dir.mkdir(parents=True, exist_ok=True)
        strategy_path = strategy_dir / "DynamicStrategy.py"
        strategy_path.write_text(strategy_code, encoding="utf-8")

        # 2a. Validate the generated Python is syntactically valid
        self._validate_strategy(strategy_code)

        pairs = pairs or [settings.SYMBOL]

        # 2. Build config
        config = self._build_config(pairs, timerange)

        # 3. Run via subprocess
        result = self._run_freqtrade_backtest(config, strategy_path)

        # 4. Parse result JSON
        return self._parse_results(result)

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
            "--strategy", "DynamicStrategy",
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
                return {"strategy": {"DynamicStrategy": {"total_trades": 0, "trades": []}}}
            zip_path = zips[0]

        if not zip_path.exists():
            logger.warning("Result ZIP missing — backtest likely produced 0 trades")
            return {"strategy": {"DynamicStrategy": {"total_trades": 0, "trades": []}}}

        import zipfile
        with zipfile.ZipFile(zip_path) as zf:
            json_name = [n for n in zf.namelist() if n.endswith(".json")][0]
            return json.load(zf.open(json_name))

    def _parse_results(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Freqtrade 2026.4's nested result into a flat summary + trades DataFrame."""
        try:
            # Navigate: raw["strategy"]["DynamicStrategy"] -> flat result dict
            strategy_block = raw.get("strategy", {})
            strat_name = next(iter(strategy_block))
            strategy_data = strategy_block[strat_name]

            trades_raw = strategy_data.get("trades", [])
            trades = pd.DataFrame(trades_raw) if trades_raw else pd.DataFrame()

            metrics = self.get_performance_metrics(trades) if not trades.empty else {}

            return {
                "summary": strategy_data,
                "total_trades": strategy_data.get("total_trades", len(trades)),
                "profit_ratio": strategy_data.get("profit_mean", 0),
                "max_drawdown": strategy_data.get("max_drawdown_account", 0),
                "sharpe_ratio": strategy_data.get("sharpe", 0),
                "win_rate": strategy_data.get("winrate", metrics.get("win_rate", 0)),
                "trades_df": trades,
                "raw": raw,
            }
        except (StopIteration, KeyError, TypeError, AttributeError) as exc:
            logger.warning("Could not parse backtest result: %s — raw keys: %s", exc, list(raw.keys()) if isinstance(raw, dict) else type(raw))
            # Return zeroed metrics so the agent knows it failed
            return {
                "error": str(exc),
                "raw": raw,
                "total_trades": 0,
                "profit_ratio": 0,
                "max_drawdown": 0,
                "sharpe_ratio": 0,
                "win_rate": 0,
            }