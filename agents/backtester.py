"""Backtester agent — executes backtests, hyperopt, walk-forward validation,
and data operations for trading strategies. Does NOT design strategies."""

import json
import logging
from typing import Any, Dict, List, Optional

from langchain_core.tools import Tool

from agents.base import BaseAgent
from backtesting.engine import BacktestEngine

logger = logging.getLogger(__name__)

BACKTESTER_SYSTEM_PROMPT = """You are a backtesting specialist. Your job is to execute backtests,
hyperopt optimization, walk-forward validation, and data operations
for trading strategies. You don't design strategies — you test them.

Available tools:
- run_backtest: Run a single backtest
- run_hyperopt: Parameter optimization
- walk_forward_validate: Robustness check across time windows
- blind_search: Automatic parameter search
- compare_strategies: Compare 2+ strategies side-by-side
- set_backtest_config: Set global timerange, pairs, timeframe
- download_data: Download historical data

IMPORTANT: Use ONLY plain ASCII text. No emoji, no Unicode symbols."""


class BacktesterAgent(BaseAgent):
    """Specialised agent that executes backtests, hyperopt, walk-forward
    validation, and data operations. Does not design strategies."""

    def __init__(self, engine: Optional[BacktestEngine] = None):
        self._engine = engine or BacktestEngine()
        # Strategy specs passed in for backtesting
        self._generated_strategies: Dict[str, Dict[str, Any]] = {}
        # Simple iteration results (dicts, not IterationRecord objects)
        self._iteration_history: List[Dict[str, Any]] = []
        # Global backtest configuration
        self._backtest_config: Dict[str, Any] = {}
        # Structured experiment tracker
        from orchestration.experiment_tracker import ExperimentTracker
        self._tracker = ExperimentTracker()
        # Current market context (set by Hermes before task dispatch)
        self._current_regime: str = ""
        self._current_sentiment: float = 0.0
        tools = self._build_tools()
        super().__init__(
            name="backtester",
            tools=tools,
            system_prompt=BACKTESTER_SYSTEM_PROMPT,
        )

    # ------------------------------------------------------------------
    # Run override — skip LLM for backtest commands
    # ------------------------------------------------------------------

    def run(self, input_text: str) -> Dict[str, Any]:
        """Override BaseAgent.run: execute backtest commands directly,
        bypassing the LLM entirely. Falls back to the LLM agent for
        non-backtest commands (hyperopt, compare, walk-forward, etc.)."""
        import re
        from config import settings

        stripped = input_text.strip()
        if stripped.startswith("backtest "):
            # Parse: backtest strategy_type=X params={...}
            m = re.search(r'strategy_type[=:]\s*(\w+)', stripped)
            if m:
                strategy_type = m.group(1)
                params_match = re.search(r'params=(\{.*\})', stripped, re.DOTALL)
                strat_params: Dict[str, Any] = {}
                if params_match:
                    try:
                        strat_params = json.loads(params_match.group(1))
                    except (json.JSONDecodeError, ValueError):
                        pass

                global_cfg = getattr(self, "_backtest_config", {})
                timerange = strat_params.pop("timerange",
                                             global_cfg.get("timerange", "20210101-"))
                pairs = strat_params.pop("pairs", global_cfg.get("pairs", None))
                strat_params.setdefault("timeframe",
                                        global_cfg.get("timeframe", settings.TIMEFRAME))

                try:
                    result = self._engine.run_backtest(
                        strat_params,
                        strategy_type=strategy_type,
                        timerange=timerange,
                        pairs=pairs,
                    )
                except Exception as exc:
                    logger.error("Backtest failed: %s", exc)
                    return {"output": f"Error running backtest: {exc}",
                            "intermediate_steps": []}

                # Evaluate and store in iteration history
                verdict, reason = self._evaluate_metrics(result)
                self._iteration_history.append({
                    "params": {**strat_params, "_strategy_id": strategy_type},
                    "metrics": result,
                    "verdict": verdict,
                    "reason": reason,
                })

                metrics = {k: result.get(k, "N/A") for k in
                           ["total_trades", "profit_ratio", "win_rate",
                            "sharpe_ratio", "max_drawdown"]}
                lines = [f"Backtest result for [{strategy_type}]:",
                         f"  Total trades: {metrics['total_trades']}"]
                for k, v in metrics.items():
                    if k != "total_trades":
                        lines.append(f"  {k}: {v}")
                lines.append(f"  Verdict: {verdict}")
                if reason:
                    lines.append(f"  Reason: {reason}")

                logger.info(
                    "Backtest result for %s: sharpe=%.2f trades=%d verdict=%s",
                    strategy_type, result.get("sharpe_ratio", 0),
                    result.get("total_trades", 0), verdict,
                )
                return {"output": "\n".join(lines), "intermediate_steps": []}

        # Non-backtest commands → use the LLM agent as before
        return super().run(input_text)

    @staticmethod
    def _evaluate_metrics(metrics: Dict[str, Any]) -> tuple:
        """Return (verdict, reason) based on metric thresholds.
        Mirrors IterationRecord.evaluate() without the class."""
        issues = []
        sharpe = metrics.get("sharpe_ratio", 0)
        win_rate = metrics.get("win_rate", 0)
        drawdown = abs(metrics.get("max_drawdown", 0))
        profit = metrics.get("profit_ratio", metrics.get("total_profit", 0))

        MIN_SHARPE = 1.0
        MIN_WIN_RATE = 0.40
        MAX_DRAWDOWN = 0.05

        if sharpe < MIN_SHARPE:
            issues.append(f"Sharpe {sharpe:.2f} < {MIN_SHARPE}")
        if win_rate < MIN_WIN_RATE:
            issues.append(f"Win rate {win_rate:.0%} < {MIN_WIN_RATE}")
        if drawdown > MAX_DRAWDOWN:
            issues.append(f"Drawdown {drawdown:.2%} > {MAX_DRAWDOWN}")
        if profit <= 0:
            issues.append(f"Non-positive profit ({profit})")

        if issues:
            return ("discarded", "; ".join(issues))
        return ("kept", "All targets met")

    # ------------------------------------------------------------------
    # Tool builder
    # ------------------------------------------------------------------

    def _build_tools(self):
        # ------------------------------------------------------------------
        # Tool: set_backtest_config
        # ------------------------------------------------------------------

        def set_backtest_config(config_json: str = "{}") -> str:
            """Set global backtest configuration that applies to all subsequent runs.
            Pass JSON with any of: timerange, pairs, timeframe, stoploss, trailing_stop.

            Examples:
              {"timerange": "20250101-20251231", "timeframe": "15m", "pairs": ["BTC/USDT", "ETH/USDT"]}
              {"timerange": "20230101-20251231", "timeframe": "1h"}
            """
            import json
            try:
                cfg = json.loads(config_json)
            except json.JSONDecodeError:
                return "Error: pass valid JSON."
            if not hasattr(self, "_backtest_config"):
                self._backtest_config = {}
            self._backtest_config.update(cfg)
            parts = [f"{k}={v}" for k, v in cfg.items()]
            return f"Backtest config updated: {', '.join(parts)}"

        # ------------------------------------------------------------------
        # Tool: run_backtest
        # ------------------------------------------------------------------

        def run_backtest(strategy_type: str = "sma_crossover", params: str = "{}") -> str:
            """Backtest a strategy by type and parameters.
            Args:
                strategy_type: One of sma_crossover, macd_crossover, rsi_oversold, etc.
                params: JSON string of strategy parameters, e.g. '{"fast_ma": 10, "slow_ma": 30}'
            Returns performance metrics with keep/discard verdict."""
            import json
            from config import settings
            try:
                strat_params = json.loads(params)
            except json.JSONDecodeError:
                strat_params = {}
            if not isinstance(strat_params, dict):
                strat_params = {}
            strat_params["strategy_type"] = strategy_type
            # Use global config for defaults
            global_cfg = getattr(self, "_backtest_config", {})
            timerange = strat_params.pop("timerange", global_cfg.get("timerange", "20210101-"))
            pairs = strat_params.pop("pairs", global_cfg.get("pairs", None))
            strat_params.setdefault("timeframe", global_cfg.get("timeframe", settings.TIMEFRAME))
            try:
                result = self._engine.run_backtest(
                    strat_params,
                    strategy_type=strategy_type,
                    timerange=timerange,
                    pairs=pairs,
                )
            except Exception as exc:
                return f"Error running backtest: {exc}"

            metrics = {k: result.get(k, "N/A") for k in
                       ["total_trades", "profit_ratio", "win_rate", "sharpe_ratio", "max_drawdown"]}
            lines = [f"Backtest result for [{strategy_type}]:", f"  Total trades: {metrics['total_trades']}"]
            for k, v in metrics.items():
                if k != "total_trades":
                    lines.append(f"  {k}: {v}")
            lines.append(f"  Timerange: {timerange}")
            if pairs:
                lines.append(f"  Pairs: {pairs}")

            # Auto-create iteration record
            verdict, reason = self._evaluate_metrics(result)
            record = {
                "params": {**strat_params, "_strategy_id": strategy_type},
                "metrics": result,
                "verdict": verdict,
                "reason": reason,
            }
            self._iteration_history.append(record)

            # Persist if kept
            if verdict == "kept":
                try:
                    from memory.vector_store import VectorStore
                    vs = VectorStore()
                    vs.store_strategy_result(
                        strategy_type=strategy_type,
                        params={k: v for k, v in strat_params.items()
                                if k not in ("indicator_code", "entry_condition",
                                             "exit_condition", "indicator_params_block")},
                        metrics=result,
                        timerange=timerange,
                    )
                except Exception as exc:
                    logger.debug("Could not persist strategy to memory: %s", exc)

            # Record experiment
            try:
                from orchestration.experiment_tracker import Experiment
                exp = Experiment(
                    strategy_type=strategy_type,
                    params={k: v for k, v in strat_params.items()
                            if k not in ("indicator_code", "entry_condition",
                                         "exit_condition", "indicator_params_block")},
                    timerange=timerange,
                    regime=getattr(self, "_current_regime", ""),
                    sentiment_score=getattr(self, "_current_sentiment", 0.0),
                    sharpe=result.get("sharpe_ratio", 0),
                    win_rate=result.get("win_rate", 0),
                    max_drawdown=result.get("max_drawdown", 0),
                    total_trades=result.get("total_trades", 0),
                    verdict=verdict,
                    iteration=len(self._iteration_history),
                )
                self._tracker.record(exp)
            except Exception as exc:
                logger.debug("Could not record experiment: %s", exc)

            lines.append(f"  Verdict: {verdict.upper()} -- {reason}")
            return "\n".join(lines)

        # ------------------------------------------------------------------
        # Tool: download_data
        # ------------------------------------------------------------------

        def download_data(params_json: str = "{}") -> str:
            """Download historical market data.
            ONLY call this if you get a data error. BTC/USDT, ETH/USDT on 5m/15m/1h
            from 2017-2023 are ALREADY cached. DO NOT download holdout data (2024+).
            Pass JSON: {"pairs": ["BTC/USDT"], "timeframe": "1h", "timerange": "20260101-"}
            Defaults: pairs=["BTC/USDT"], timeframe=config.TIMEFRAME, timerange="20210101-"
            Returns download status and row counts."""
            import json
            try:
                params = json.loads(params_json)
            except json.JSONDecodeError:
                params = {}
            from config import settings
            pairs = params.get("pairs", [settings.SYMBOL])
            timerange = params.get("timerange", "20210101-")
            timeframe = params.get("timeframe", settings.TIMEFRAME)
            try:
                self._engine.download_data(pairs=pairs, timerange=timerange)
                return f"Data downloaded: pairs={pairs}, timeframe={timeframe}, timerange={timerange}"
            except Exception as exc:
                return f"Error downloading data: {exc}"

        # ------------------------------------------------------------------
        # Tool: compare_strategies
        # ------------------------------------------------------------------

        def compare_strategies(ids_json: str = "{}") -> str:
            """Compare two or more strategies side-by-side by their IDs.
            Pass JSON: {"ids": ["abc123", "def456"]}
            Returns a metric table with a recommended winner."""
            import json
            try:
                data = json.loads(ids_json)
            except json.JSONDecodeError:
                return "Error: pass valid JSON with {'ids': ['id1', 'id2', ...]}"
            ids = data.get("ids", [])
            if len(ids) < 2:
                return "Provide at least two strategy IDs to compare."
            records = {r["params"].get("_strategy_id", ""): r for r in self._iteration_history}
            matched = []
            for sid in ids:
                found = None
                # Check records map first
                if sid in records:
                    found = records[sid]
                else:
                    # Search iteration history for records whose params match
                    for rec in self._iteration_history:
                        if rec["params"].get("_strategy_id") == sid:
                            found = rec
                            break
                if found:
                    matched.append((sid, found))
                else:
                    # Check if it's in generated strategies
                    if sid in self._generated_strategies:
                        matched.append((sid, None))
            if len(matched) < 2:
                return "Could not find enough strategies. Use run_backtest first."
            lines = ["### Strategy Comparison",
                     f"{'Metric':<20} |"]
            for sid, _ in matched:
                lines[0] += f" {sid:<15} |"
            lines.append("-" * len(lines[0]))
            # Predefined metrics
            metric_keys = ["total_trades", "profit_ratio", "win_rate", "sharpe_ratio", "max_drawdown"]
            for mk in metric_keys:
                row = f"{mk:<20} |"
                for sid, rec in matched:
                    val = rec["metrics"].get(mk, "N/A") if rec and rec["metrics"].get(mk) is not None else "N/A"
                    row += f" {str(val):<15} |"
                lines.append(row)
            # Determine winner by Sharpe
            best_sharpe = -999
            best_sid = None
            for sid, rec in matched:
                if rec:
                    s = rec["metrics"].get("sharpe_ratio", -999)
                    if isinstance(s, (int, float)) and s > best_sharpe:
                        best_sharpe = s
                        best_sid = sid
            if best_sid:
                lines.append(f"")
                lines.append(f"**Recommended**: [{best_sid}] with Sharpe={best_sharpe}")
            return "\n".join(lines)

        # ------------------------------------------------------------------
        # Tool: run_hyperopt
        # ------------------------------------------------------------------

        def run_hyperopt(params_json: str = "{}") -> str:
            """
            Run Freqtrade hyperopt to automatically find optimal parameters.
            MUCH more powerful than manual parameter tweaking.
            Pass JSON: {"strategy_id": "abc123", "epochs": 50, "loss": "SharpeHyperOptLoss"}
            Loss options: SharpeHyperOptLoss, WinRatioAndProfitRatioLoss, OnlyProfitHyperOptLoss
            Returns best parameters found by hyperopt.
            """
            import json, subprocess
            try:
                params = json.loads(params_json)
            except json.JSONDecodeError:
                params = {}

            sid = params.get("strategy_id", "")
            epochs = int(params.get("epochs", 50))
            loss = params.get("loss", "SharpeHyperOptLoss")

            if sid and sid not in self._generated_strategies:
                return f"Error: unknown strategy_id '{sid}'. Use generate_strategy first."

            # -- Check if this strategy type supports hyperopt --
            if sid and sid in self._generated_strategies:
                strat_type = self._generated_strategies[sid].get("strategy_type", "")
                HYPEROPT_SUPPORTED = {
                    "sma_crossover", "combined_sma_rsi", "macd_crossover",
                    "rsi_oversold", "bollinger_bands", "multi_timeframe"
                }
                if strat_type and strat_type not in HYPEROPT_SUPPORTED:
                    return (
                        f"Note: {strat_type} does not have IntParameter declarations "
                        f"so hyperopt cannot tune its parameters. "
                        f"Try one of: {', '.join(sorted(HYPEROPT_SUPPORTED))}"
                    )

            # Use latest generated strategy if no ID given
            if not sid and self._generated_strategies:
                sid = list(self._generated_strategies.keys())[-1]

            global_cfg = getattr(self, "_backtest_config", {})
            timerange = global_cfg.get("timerange", "20260427-")
            pairs = global_cfg.get("pairs", ["BTC/USDT"])

            cmd = [
                self._engine._freqtrade_cmd,
                "hyperopt",
                "--userdir", str(self._engine.ft_userdata_dir),
                "--config", str(self._engine.ft_userdata_dir / "config.json"),
                "--strategy-path", str(self._engine.ft_userdata_dir / "strategies"),
                "--strategy", "DynamicStrategy",
                "--hyperopt-loss", loss,
                "--epochs", str(epochs),
                "--timerange", timerange,
                "--spaces", "buy", "sell", "roi", "stoploss",
            ]

            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=300
                )
                output = result.stdout[-2000:] if result.stdout else ""
                stderr = result.stderr[-500:] if result.stderr else ""
                if result.returncode != 0:
                    return f"Hyperopt failed:\n{stderr}"
                return f"Hyperopt complete ({epochs} epochs):\n{output}"
            except subprocess.TimeoutExpired:
                return "Hyperopt timed out after 5 minutes. Try fewer epochs."
            except Exception as exc:
                return f"Hyperopt error: {exc}"

        # ------------------------------------------------------------------
        # Tool: walk_forward_validate
        # ------------------------------------------------------------------

        def walk_forward_validate(params_json: str = "{}") -> str:
            """
            Validate a strategy across multiple time windows to check robustness.
            A strategy that only works in one period is likely overfit.
            Pass JSON: {"strategy_id": "abc123", "windows": 3}
            Returns per-window metrics and overall consistency score.
            Call this before declaring a strategy as 'kept'.
            """
            import json
            try:
                params = json.loads(params_json)
            except json.JSONDecodeError:
                params = {}

            sid = params.get("strategy_id", "")
            windows = int(params.get("windows", 3))

            if not sid or sid not in self._generated_strategies:
                return f"Error: unknown strategy_id '{sid}'"

            strat_params = self._generated_strategies[sid].copy()
            strategy_type = strat_params.pop("strategy_type", "sma_crossover")
            strat_params.pop("timerange", None)
            strat_params.pop("pairs", None)

            try:
                result = self._engine.walk_forward_validate(
                    strategy_params=strat_params,
                    strategy_type=strategy_type,
                    windows=windows,
                )
            except Exception as exc:
                return f"Walk-forward failed: {exc}"

            lines = [
                f"Walk-forward validation ({windows} windows):",
                f"  Consistency: {result['consistency_score']:.0%} of windows profitable",
                f"  Average Sharpe: {result['avg_sharpe']:.2f}",
                f"  Average Win Rate: {result['avg_win_rate']:.0%}",
                f"  Robust: {'YES' if result['is_robust'] else 'NO - likely overfit'}",
                "",
                "Per-window results:",
            ]
            for w in result["windows"]:
                tr = w.get("test_timerange", w.get("train_timerange", "unknown"))
                if w.get("error"):
                    lines.append(f"  Window {w['window']} ({tr}): ERROR - {w['error']}")
                else:
                    lines.append(
                        f"  Window {w['window']} ({tr}): "
                        f"Sharpe={w['sharpe']:.2f} WR={w['win_rate']:.0%} "
                        f"Trades={w['total_trades']}"
                    )
            return "\n".join(lines)

        # ------------------------------------------------------------------
        # Tool: blind_search
        # ------------------------------------------------------------------

        def blind_search(params_json: str = "{}") -> str:
            """Automatic blind parameter search across N variants.
            Pass JSON: {"strategy_type": "sma_crossover", "n_variants": 20,
                        "pairs": ["BTC/USDT"]}
            Runs all variants blind (LLM does not see individual results),
            then returns aggregate statistics and the best variant.
            """
            import json
            try:
                params = json.loads(params_json)
            except json.JSONDecodeError:
                params = {}

            strategy_type = params.get("strategy_type", "")
            if not strategy_type:
                return (
                    "Error: 'strategy_type' is required. "
                    "Supported: sma_crossover, macd_crossover, rsi_oversold, "
                    "bollinger_bands, combined_sma_rsi, momentum, breakout, "
                    "mean_reversion, volatility_squeeze, sentiment_driven, "
                    "multi_timeframe"
                )

            n_variants = int(params.get("n_variants", 20))
            pairs = params.get("pairs", None)

            from backtesting.blind_search import BlindParameterSearch
            bps = BlindParameterSearch(engine=self._engine)

            # Generate default variants (no LLM involvement)
            variants = bps._generate_default_variants(strategy_type, n=n_variants)
            if not variants:
                return f"Error: could not generate variants for '{strategy_type}'"

            # Run batch backtest
            batch_results = bps.batch_backtest(
                variants=variants, strategy_type=strategy_type, pairs=pairs,
            )

            # Compute aggregate stats
            aggregate = bps.compute_aggregate_stats(batch_results)

            # Select best variant
            best = bps.select_best_for_wfv(batch_results)

            # Format output
            lines = [
                f"Blind parameter search for {strategy_type}:",
                f"  Variants tested: {len(variants)}",
                f"  Valid results: {aggregate['n_valid']}/{aggregate['n_total']}",
                "",
                "Aggregate statistics:",
                f"  Median Sharpe: {aggregate['median_sharpe']:.3f}",
                f"  Median Win Rate: {aggregate['median_win_rate']:.1%}",
                f"  Median Drawdown: {aggregate['median_drawdown']:.1%}",
                f"  Sharpe spread (std): {aggregate['std_sharpe']:.3f}",
                f"  % passing minimum criteria: {aggregate['pct_passing']:.1f}%",
                f"  Sharpe range: {aggregate['sharpe_range']}",
                f"  Best parameter region: {aggregate['best_region']}",
                f"  Worst parameter region: {aggregate['worst_region']}",
            ]

            if best:
                lines.append("")
                lines.append("Best variant (quantitative selection):")
                lines.append(f"  Params: {json.dumps(best['params'])}")
                for k, v in best['metrics'].items():
                    lines.append(f"  {k}: {v}")

                # Store as a generated strategy for follow-up use
                import uuid
                strategy_id = uuid.uuid4().hex[:8]
                best_params = dict(best["params"])
                best_params["strategy_type"] = strategy_type
                self._generated_strategies[strategy_id] = best_params
                lines.append("")
                lines.append(
                    f"Stored as strategy_id='{strategy_id}'. "
                    f"Use run_backtest, run_hyperopt, or walk_forward_validate with this ID."
                )
            else:
                lines.append("")
                lines.append("No viable variant found.")

            return "\n".join(lines)

        # ------------------------------------------------------------------
        # Return tool list
        # ------------------------------------------------------------------

        return [
            Tool(name="set_backtest_config", func=set_backtest_config,
                 description="Set global backtest config: timerange, pairs, timeframe. Applies to all subsequent backtests."),
            Tool(name="run_backtest", func=run_backtest,
                 description="Backtest a strategy type with JSON parameters. "
                 "Args: strategy_type (str), params (str JSON). "
                 "Example params: '{\"fast_ma\": 10, \"slow_ma\": 30}'"),
            Tool(name="download_data", func=download_data,
                 description="Download historical data for backtesting. Args: JSON with pairs, timeframe, timerange."),
            Tool(name="compare_strategies", func=compare_strategies,
                 description="Compare 2+ strategies side-by-side by IDs. Args: JSON with ids list."),
            Tool(name="run_hyperopt", func=run_hyperopt,
                 description="Run Freqtrade hyperopt for automatic parameter optimization. "
                             "Args: JSON with strategy_id, epochs (default 50), loss function. "
                             "Use after generating a strategy to find optimal parameters automatically."),
            Tool(name="walk_forward_validate", func=walk_forward_validate,
                 description="Validate strategy robustness across multiple time windows. "
                             "Args: JSON with strategy_id and windows (default 3). "
                             "Use before declaring a strategy as kept to check it is not overfit."),
            Tool(name="blind_search", func=blind_search,
                 description="Automatic blind parameter search across N variants. "
                             "Args: JSON with strategy_type, n_variants (default 20), pairs. "
                             "Returns aggregate stats and best variant."),
        ]
