"""Backtesting engine wrapping Freqtrade subprocess execution.

Generates temporary strategy files on the fly, runs backtests via the
Freqtrade CLI, and parses results into pandas DataFrames.
"""

import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from config import settings

logger = logging.getLogger(__name__)

# ── Strategy template injected as a temp .py file ──

STRATEGY_TEMPLATE = '''"""
Auto-generated strategy by crypto_agent_bot.
Do not edit manually — generated on {timestamp}.
"""
from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter
import pandas as pd
import talib.abstract as ta


class DynamicStrategy(IStrategy):
    # --- User-defined parameters (set by agent) ---
    timeframe = "{timeframe}"
    minimal_roi = {minimal_roi}
    stoploss = {stoploss}
    trailing_stop = {trailing_stop}
    startup_candle_count = {startup_candle_count}
    process_only_new_candles = True
    use_exit_signal = True
    can_short = False

    # --- Indicator parameters ---
    fast_ma = IntParameter(5, 50, default={fast_ma}, space="buy")
    slow_ma = IntParameter(20, 200, default={slow_ma}, space="buy")

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        {indicator_code}
        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[
            (
                {entry_condition}
            ),
            "enter_long"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[
            (
                {exit_condition}
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
        timerange: str = "20260101-",
        pairs: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Generate a strategy from *params*, run a Freqtrade backtest,
        and return parsed results."""
        params = self._default_strategy_params()
        if strategy_params:
            params.update(strategy_params)
        params.setdefault("timeframe", settings.TIMEFRAME)

        # 1. Write temporary strategy file
        strategy_code = self._render_strategy(params)
        strategy_dir = self.ft_userdata_dir / "strategies"
        strategy_dir.mkdir(parents=True, exist_ok=True)
        strategy_path = strategy_dir / "DynamicStrategy.py"
        strategy_path.write_text(strategy_code, encoding="utf-8")

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

    def download_data(self, pairs: Optional[List[str]] = None, timerange: str = "20260101-"):
        """Download historical data via ``freqtrade download-data``."""
        pairs = pairs or [settings.SYMBOL]
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
        logger.info("Downloading data: %s", " ".join(cmd))
        subprocess.run(cmd, check=True, timeout=settings.BACKTEST_TIMEOUT)
        logger.info("Data download complete.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _default_strategy_params(self) -> Dict[str, Any]:
        return {
            "fast_ma": 10,
            "slow_ma": 30,
            "stoploss": -0.05,
            "trailing_stop": False,
            "minimal_roi": '{"0": 0.01}',
            "startup_candle_count": 30,
            "indicator_code": SMA_CROSSOVER_INDICATOR,
            "entry_condition": SMA_CROSSOVER_ENTRY,
            "exit_condition": SMA_CROSSOVER_EXIT,
            "timeframe": settings.TIMEFRAME,
        }

    def _render_strategy(self, params: Dict[str, Any]) -> str:
        import datetime
        params["timestamp"] = datetime.datetime.utcnow().isoformat()
        return STRATEGY_TEMPLATE.format(**params)

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
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=settings.BACKTEST_TIMEOUT)
            logger.info("Backtest completed.")
        except subprocess.CalledProcessError as exc:
            logger.error("Backtest stderr:\n%s", exc.stderr)
            raise RuntimeError(f"Freqtrade backtest failed: {exc.stderr}") from exc

        # In Freqtrade 2026.4 results are stored in a .zip inside export_path
        # with an associated .last_result.json pointing to the latest zip
        last_result_file = export_path / ".last_result.json"
        if not last_result_file.exists():
            # fallback: pick the newest .zip
            zips = sorted(export_path.glob("*.zip"), key=os.path.getmtime, reverse=True)
            if not zips:
                raise FileNotFoundError(f"No result ZIP found in {export_path}")
            zip_path = zips[0]
        else:
            with open(last_result_file) as lrf:
                meta = json.load(lrf)
            zip_path = export_path / meta.get("latest_backtest", "")

        import zipfile
        with zipfile.ZipFile(zip_path) as zf:
            # The zip contains one JSON entry (e.g. "backtest-result.json")
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
            logger.warning("Could not parse backtest result: %s", exc)
            return {"error": str(exc), "raw": raw}