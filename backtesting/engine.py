"""Backtesting engine wrapping Freqtrade subprocess execution.

Generates temporary strategy files on the fly, runs backtests via the
Freqtrade CLI, and parses results into pandas DataFrames.

Includes a vectorized pre-filter (SignalFactory + FastMetrics) that runs
before the Freqtrade subprocess to reject obviously worthless strategies
in <1 second.

Architecture note -- extracted sibling modules (re-exported for compat):
  TransactionCostModel -> backtesting/cost_model.py
  sanitize_timerange    -> backtesting/timerange_utils.py
  Strategy templates   -> backtesting/strategy_templates.py
  SignalFactory         -> backtesting/signal_factory.py
  DataSplit             -> backtesting/data_split.py
"""

import ast
import json
import logging
import re
import uuid
import os
import shutil
import string
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from config import settings
from backtesting.data_split import DATA_SPLIT
from backtesting.signal_factory import FastMetrics, SignalFactory
from backtesting.cost_model import TransactionCostModel
from backtesting.timerange_utils import sanitize_timerange
from backtesting.strategy_templates import (
    STRATEGY_TEMPLATE,
    SMA_CROSSOVER_INDICATOR, SMA_CROSSOVER_ENTRY, SMA_CROSSOVER_EXIT,
    MACD_CROSSOVER_INDICATOR, MACD_CROSSOVER_ENTRY, MACD_CROSSOVER_EXIT,
    RSI_INDICATOR, RSI_OVERSOLD_ENTRY, RSI_OVERSOLD_EXIT,
    BB_INDICATOR, BB_ENTRY, BB_EXIT,
    SMA_RSI_INDICATOR, SMA_RSI_ENTRY, SMA_RSI_EXIT,
    STRATEGY_REGISTRY,
)

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Runs Freqtrade backtests by generating temporary strategy files."""

    def __init__(self, ft_userdata_dir: str = "./ft_userdata") -> None:
        self.ft_userdata_dir = Path(ft_userdata_dir).resolve()
        self._config: Optional[Dict[str, Any]] = None
        # Locate freqtrade executable in the venv
        self._freqtrade_cmd = self._find_freqtrade()

    # ------------------------------------------------------------------
    # Pre-filter
    # ------------------------------------------------------------------

    def _run_prefilter(
        self,
        strategy_type: str,
        strategy_params: dict,
        timerange: str,
        pairs: list,
    ) -> Optional[dict]:
        """Run the vectorized pre-filter. Returns None if pass-through (no filter)."""
        if not settings.VECTORBT_PREFILTER_ENABLED:
            return None
        if strategy_type not in SignalFactory.supported_types():
            logger.debug("Pre-filter: unknown type %s, passing through", strategy_type)
            return None
        try:
            from data.fetcher import MarketDataFetcher
            fetcher = MarketDataFetcher()
            tf = strategy_params.get("timeframe", settings.TIMEFRAME)
            pair = pairs[0] if pairs else settings.SYMBOL
            raw = fetcher.fetch_ohlcv(pair, tf, limit=500)
            if raw is None or (isinstance(raw, pd.DataFrame) and len(raw) < 100):
                logger.debug("Pre-filter: insufficient data, passing through")
                return None
            df = raw.copy() if isinstance(raw, pd.DataFrame) else pd.DataFrame(raw)
            signals = SignalFactory.generate(df, strategy_type, strategy_params)
            metrics = FastMetrics.compute(df, signals)
            # Pass through if 0 trades: the pre-filter uses only 500 candles of live data,
            # which is often too little data for strategies with long indicator windows
            # (e.g. multi_timeframe needs SMA200). Zero trades does not mean the strategy
            # is bad — it means the sample is too small. Let Freqtrade evaluate properly.
            if metrics.get("total_trades", 0) < 1:
                logger.debug(
                    "Pre-filter: 0 trades for %s on sample data, passing through to Freqtrade",
                    strategy_type,
                )
                return None
            if metrics.get("passed", False):
                logger.debug(
                    "Pre-filter passed %s: Sharpe=%.2f WR=%.1f%% trades=%d",
                    strategy_type, metrics["sharpe_ratio"],
                    metrics["win_rate"] * 100, metrics["total_trades"],
                )
                return None  # pass through to Freqtrade
            logger.info(
                "Pre-filter REJECTED %s: Sharpe=%.2f WR=%.1f%% trades=%d",
                strategy_type, metrics["sharpe_ratio"],
                metrics["win_rate"] * 100, metrics["total_trades"],
            )
            return {
                "pre_filter_rejected": True,
                **metrics,
            }
        except Exception as exc:
            logger.warning("Pre-filter error (passing through): %s", exc)
            return None

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
        dataframe_override: Optional[Dict[str, pd.DataFrame]] = None,  # NEW
    ) -> Dict[str, Any]:
        """Generate a strategy from *params*, run a Freqtrade backtest,
        and return parsed results.

        If *dataframe_override* is provided (dict of pair -> DataFrame),
        skip the Freqtrade subprocess and use SignalFactory + FastMetrics
        directly on the supplied DataFrames. This is used for synthetic
        data validation and permutation testing.
        """
        # If dataframe_override is provided, skip Freqtrade entirely
        if dataframe_override is not None:
            return self._run_fastmetrics_backtest(strategy_type, strategy_params, dataframe_override)

        timerange = sanitize_timerange(timerange)

        # Fast pre-filter: reject obviously worthless strategies before Freqtrade
        prefilter_result = self._run_prefilter(strategy_type, strategy_params or {},
                                                timerange, pairs or [settings.SYMBOL])
        if prefilter_result is not None:
            return prefilter_result

        logger.info(
            "Starting Freqtrade subprocess for strategy_type=%s params=%s",
            strategy_type, strategy_params,
        )

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

        # 2a. Validate the generated Python is syntactically valid
        self._validate_strategy(strategy_code)

        pairs = pairs or [settings.SYMBOL]

        # 2. Build config
        config = self._build_config(pairs, timerange)

        # 3. Run via subprocess
        result = self._run_freqtrade_backtest(config, strategy_path)

        # 4. Clean up OLD generated strategy files (after subprocess completes
        #    so Freqtrade's backtest caching can still read strategy.__file__)
        for f in strategy_dir.glob("Strategy_*.py"):
            if f.name != strategy_path.name:
                try: f.unlink()
                except Exception: pass

        # 5. Parse result JSON
        return self._parse_results(result, strategy_name=strategy_name)

    def _run_fastmetrics_backtest(
        self,
        strategy_type: str,
        strategy_params: Optional[Dict[str, Any]],
        dataframes: Dict[str, pd.DataFrame],
    ) -> Dict[str, Any]:
        """Run backtest using SignalFactory + FastMetrics on provided DataFrames
        (no Freqtrade subprocess). Used by synthetic data validation."""
        params = strategy_params or {}
        tf = params.get("timeframe", settings.TIMEFRAME)

        results = {}
        for pair, df in dataframes.items():
            if df is None or (isinstance(df, pd.DataFrame) and len(df) < 50):
                results[pair] = {"error": "Insufficient data", "total_trades": 0, "sharpe_ratio": 0, "win_rate": 0, "profit_ratio": 0}
                continue
            df = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
            signals = SignalFactory.generate(df, strategy_type, params)
            metrics = FastMetrics.compute(df, signals)
            results[pair] = metrics

        # Aggregate across pairs
        total_trades = sum(r.get("total_trades", 0) for r in results.values())
        if total_trades == 0:
            return {"sharpe_ratio": 0, "win_rate": 0, "total_trades": 0, "profit_ratio": 0}

        weighted_sharpe = sum(
            r.get("sharpe_ratio", 0) * r.get("total_trades", 0)
            for r in results.values()
        ) / total_trades

        weighted_win_rate = sum(
            r.get("win_rate", 0) * r.get("total_trades", 0)
            for r in results.values()
        ) / total_trades

        total_profit = sum(r.get("total_return_pct", 0) * r.get("total_trades", 0) for r in results.values()) / total_trades

        return {
            "sharpe_ratio": round(weighted_sharpe, 4),
            "win_rate": round(weighted_win_rate, 4),
            "total_trades": total_trades,
            "profit_ratio": round(total_profit, 4),
            "net_sharpe_ratio": round(max(weighted_sharpe, 0), 2),
        }

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
            # Crypto trades 24/7, annualize assuming ~8760 hourly periods per year
            _annual_factor = (365 * 24) ** 0.5
            sharpe = (returns.mean() / returns.std()) * _annual_factor if len(returns) > 1 else 0.0

        cumulative = trades_df["profit_ratio"].cumsum()
        cummax = cumulative.cummax()
        drawdown = (cumulative - cummax).min()

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

    def download_data(self, pairs: Optional[List[str]] = None, timerange: str = "20210101-") -> None:
        """Download historical data via ``freqtrade download-data``."""
        timerange = sanitize_timerange(timerange)

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
        try:
            result = subprocess.run(
                _build_cmd(),
                capture_output=True, text=True,
                timeout=settings.BACKTEST_TIMEOUT
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Data download timed out after {settings.BACKTEST_TIMEOUT}s")

        # Detect the prepend warning
        combined = (result.stdout or "") + (result.stderr or "")
        if "Use `--prepend`" in combined or "--prepend" in combined:
            logger.info("Local data exists but requested range is earlier — retrying with --prepend")
            try:
                result = subprocess.run(
                    _build_cmd(prepend=True),
                    capture_output=True, text=True,
                    timeout=settings.BACKTEST_TIMEOUT
                )
            except subprocess.TimeoutExpired:
                raise RuntimeError(f"Data download timed out after {settings.BACKTEST_TIMEOUT}s")

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
        from datetime import datetime, timezone, timedelta

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
        strategist agent can catch and regenerate.

        Also detects unsubstituted $ placeholders from string.Template
        that would corrupt the generated file."""
        # Check for unsubstituted $ placeholders (but allow $$ which is
        # string.Template's escape for a literal $)
        placeholders = re.findall(r'(?<!\$)\$[a-zA-Z_]\w*', source)
        if placeholders:
            msg = (f"Generated strategy has {len(placeholders)} unsubstituted "
                   f"template placeholders: {placeholders[:5]} — "
                   f"LLM output contains '$' that safe_substitute couldn't resolve")
            logger.error(msg)
            raise ValueError(msg)

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
            "trailing_stop_positive": 0.01,
            "trailing_stop_positive_offset": 0.02,
            "trailing_only_offset_is_reached": False,
            "minimal_roi": '{"0": 0.01}',
            "timeframe": settings.TIMEFRAME,
        }
        params.update(registry_entry["default_params"])
        return params

    def _render_strategy(self, params: Dict[str, Any]) -> str:
        import datetime
        params["timestamp"] = datetime.datetime.now(timezone.utc).isoformat()
        # Nested substitution for indicator_params_block which may contain $fast_ma/$slow_ma
        if "indicator_params_block" in params and "$" in params.get("indicator_params_block", ""):
            params["indicator_params_block"] = string.Template(params["indicator_params_block"]).safe_substitute(**params)
        return string.Template(STRATEGY_TEMPLATE).safe_substitute(**params)

    def _build_config(self, pairs: List[str], timerange: str) -> Dict[str, Any]:
        config_path = self.ft_userdata_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
        else:
            # Minimal default config for environments without ft_userdata (e.g. CI)
            config = {
                "exchange": {"name": "binance", "pair_whitelist": pairs},
                "entry_pricing": {"price_side": "same", "use_order_book": False},
                "exit_pricing": {"price_side": "same", "use_order_book": False},
            }

        config["exchange"]["pair_whitelist"] = pairs
        config["timerange"] = timerange
        config["timeframe"] = settings.TIMEFRAME
        config["dry_run"] = True
        # Inject transaction cost model
        cost_model = TransactionCostModel.from_settings()
        config["fee"] = cost_model.to_freqtrade_fee()
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
            "--fee", str(TransactionCostModel.from_settings().to_freqtrade_fee()),
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

            # ── Net-of-costs metrics ──
            cost_model = TransactionCostModel.from_settings()
            # Estimate avg return per trade from profit ratio
            avg_return = profit / max(total_trades, 1)
            net_sharpe = cost_model.net_sharpe(sharpe, avg_return)
            net_win_rate = win_rate  # Win rate doesn't change with proportional costs

            metrics = {}
            if not trades.empty:
                metrics = self.get_performance_metrics(trades)

            return {
                "summary": sd,
                "total_trades": total_trades,
                "profit_ratio": profit or metrics.get("total_profit", 0),
                "max_drawdown": drawdown or metrics.get("max_drawdown", 0),
                "sharpe_ratio": sharpe or metrics.get("sharpe_ratio", 0),
                "net_sharpe_ratio": round(max(net_sharpe, 0), 2),
                "win_rate": win_rate or metrics.get("win_rate", 0),
                "cost_model": asdict(cost_model),
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