"""Strategist agent — creates, backtests, and iterates on trading strategies."""

import json
import logging
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.tools import Tool

from agents.base import BaseAgent
from backtesting.engine import BacktestEngine

logger = logging.getLogger(__name__)

STRATEGIST_SYSTEM_PROMPT = """You are a quantitative trading strategist. Your job is to:
1. Create a SINGLE trading strategy by specifying type and parameters
2. Backtest that strategy immediately
3. Interpret the metrics
4. Use suggest_next_params to decide what to tweak next
5. Repeat until you find a strategy that meets targets

Available strategy types:
- sma_crossover: SMA fast/slow crossover (fast_ma, slow_ma)
- macd_crossover: MACD line crossing signal line
- rsi_oversold: Buy RSI < 30, sell RSI > 70
- bollinger_bands: Buy lower band touch, sell upper band
- combined_sma_rsi: SMA crossover with RSI filter
- momentum: ROC + volume confirmation (trending markets)
- breakout: N-period high breakout with volume spike
- mean_reversion: BB + RSI oversold (ranging markets)
- volatility_squeeze: BB width contraction then expansion
- sentiment_driven: RSI + SMA (use when fear/greed < 30)
- multi_timeframe: 1h SMA crossover confirmed by 200 SMA (higher TF proxy) + ADX filter

REGIME GUIDANCE:
- strong_uptrend -> prefer sma_crossover, momentum, combined_sma_rsi
- ranging -> prefer bollinger_bands, rsi_oversold, mean_reversion
- volatile -> prefer breakout, volatility_squeeze
- weak_trend -> prefer macd_crossover, combined_sma_rsi

WORK ITERATIVELY — one strategy at a time. Do NOT enumerate many variants at once.
Use this cycle:
  generate_strategy -> run_backtest -> interpret_metrics -> suggest_next_params -> repeat

New tools available:
- set_backtest_config: Set timerange, pairs, or timeframe globally before running strategies
- download_data: Download historical data for custom timeranges/timeframes not yet cached
- compare_strategies: Compare two or more strategies side-by-side to pick the best
- get_iteration_history: View all past attempts with keep/discard verdicts
- get_best_strategy: Get the current best strategy from iteration history
- get_research_history: Query ChromaDB for past research iterations to avoid repeating failed approaches

Target metrics:
- Sharpe ratio > 1.0
- Win rate > 50%
- Max drawdown < 5%
- Positive profit ratio

STRATEGY CONCEPT LIBRARY:
When choosing a strategy, first consider these proven concepts:
- Golden Cross: SMA50/200 crossover -> use sma_crossover(fast_ma=50, slow_ma=200)
- RSI Divergence: oversold bounce -> use rsi_oversold
- Volatility Breakout: N-day high + volume -> use breakout
- MACD Histogram Reversal: zero-cross -> use macd_crossover
- BB Squeeze: contraction then expansion -> use volatility_squeeze
- Fear/Greed Contrarian: extreme fear buy -> use sentiment_driven
- Multi-TF Trend: short cross + long confirm -> use multi_timeframe
- Momentum + Volume: ROC + volume spike -> use momentum
- Mean Reversion: BB + RSI oversold -> use mean_reversion

Map the research goal to the most suitable concept, then pick its
freqtrade_type and suggested_params as your starting point.

NEW TOOLS AVAILABLE:
- run_hyperopt: After generating a strategy, use this to automatically find
  optimal parameters via Freqtrade's built-in optimizer. Faster and more
  thorough than manual suggest_next_params. Use epochs=50 for a quick search,
  epochs=200 for thorough optimization.
- walk_forward_validate: Before declaring a strategy 'kept', validate it across
  multiple time windows to check it's not overfit.

VALIDATION WORKFLOW (use this order):
1. generate_strategy
2. run_backtest (quick check)
3. If metrics look promising: walk_forward_validate (robustness check)
4. If robust: run_hyperopt (optimize parameters)
5. run_backtest again with optimized params
6. If passes: record as kept

DATA AVAILABILITY:
- Local data starts from 2026-04-27 for BTC/USDT and ETH/USDT on 1h
- For earlier dates, call download_data first (it will use --prepend automatically)
- Always use timerange format YYYYMMDD-YYYYMMDD (e.g. 20260427-20260530)

IMPORTANT: Use ONLY plain ASCII text. No emoji, no Unicode symbols, no special characters.

"""

# Minimum acceptable metrics — any result below these is flagged.
_MIN_SHARPE = 1.0
_MIN_WIN_RATE = 0.40
_MAX_DRAWDOWN = 0.05


class IterationRecord:
    """Tracks one backtest attempt in the optimization loop."""

    def __init__(self, params: Dict[str, Any], metrics: Dict[str, Any]):
        self.params = dict(params)
        self.metrics = dict(metrics)
        self.verdict: str = "unknown"
        self.reason: str = ""

    def evaluate(self) -> "IterationRecord":
        """Assign keep/discard verdict based on metric thresholds."""
        issues = []
        sharpe = self.metrics.get("sharpe_ratio", 0)
        win_rate = self.metrics.get("win_rate", 0)
        drawdown = abs(self.metrics.get("max_drawdown", 0))
        profit = self.metrics.get("profit_ratio", self.metrics.get("total_profit", 0))

        if sharpe < _MIN_SHARPE:
            issues.append(f"Sharpe {sharpe:.2f} < {_MIN_SHARPE}")
        if win_rate < _MIN_WIN_RATE:
            issues.append(f"Win rate {win_rate:.0%} < {_MIN_WIN_RATE}")
        if drawdown > _MAX_DRAWDOWN:
            issues.append(f"Drawdown {drawdown:.2%} > {_MAX_DRAWDOWN}")
        if profit <= 0:
            issues.append(f"Non-positive profit ({profit})")

        if issues:
            self.verdict = "discarded"
            self.reason = "; ".join(issues)
        else:
            self.verdict = "kept"
            self.reason = "All targets met"
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "params": self.params,
            "metrics": self.metrics,
            "verdict": self.verdict,
            "reason": self.reason,
        }


class StrategistAgent(BaseAgent):
    """Specialised agent that creates, backtests and optimises trading strategies."""

    def __init__(self, engine: Optional[BacktestEngine] = None):
        self._engine = engine or BacktestEngine()
        # Track strategies we've generated
        self._generated_strategies: Dict[str, Dict[str, Any]] = {}
        # Track iteration history (keep/discard records)
        self._iteration_history: List[IterationRecord] = []
        # Track the best strategy found
        self._best_strategy: Optional[Dict[str, Any]] = None
        self._best_params: Optional[Dict[str, Any]] = None
        # Structured experiment tracker
        from orchestration.experiment_tracker import ExperimentTracker
        self._tracker = ExperimentTracker()
        # Current market context (set by Hermes before task dispatch)
        self._current_regime: str = ""
        self._current_sentiment: float = 0.0
        tools = self._build_tools()
        super().__init__(
            name="strategist",
            tools=tools,
            system_prompt=STRATEGIST_SYSTEM_PROMPT,
        )

    # ------------------------------------------------------------------
    # Tool: generate_strategy (generic)
    # ------------------------------------------------------------------

    def _build_tools(self):
        def generate_strategy(params_json: str = "{}") -> str:
            """Generate a strategy of any supported type.
            Pass JSON with: strategy_type, and type-specific params.

            Types: sma_crossover, macd_crossover, rsi_oversold, bollinger_bands,
            combined_sma_rsi, momentum, breakout, mean_reversion,
            volatility_squeeze, sentiment_driven, multi_timeframe

            AVOID 'custom' type — use one of the predefined types above.
            Do NOT set timerange — the research window is automatically applied.

            Examples:
              SMA:  {"strategy_type": "sma_crossover", "fast_ma": 10, "slow_ma": 30}
              MACD: {"strategy_type": "macd_crossover"}
              RSI:  {"strategy_type": "rsi_oversold"}

            Returns the strategy ID."""
            import json, uuid
            try:
                params = json.loads(params_json)
            except json.JSONDecodeError:
                params = {}

            strategy_type = params.pop("strategy_type", "sma_crossover")
            valid_types = {"sma_crossover", "macd_crossover", "rsi_oversold",
                           "bollinger_bands", "combined_sma_rsi", "custom",
                           "momentum", "breakout", "mean_reversion",
                           "volatility_squeeze", "sentiment_driven",
                           "multi_timeframe"}
            if strategy_type not in valid_types:
                return f"Error: unknown strategy_type '{strategy_type}'. Valid: {', '.join(sorted(valid_types))}"

            # Custom type requires all three code blocks
            if strategy_type == "custom":
                if not params.get("indicator_code") or not params.get("entry_condition") or not params.get("exit_condition"):
                    return "Error: custom strategy requires 'indicator_code', 'entry_condition', and 'exit_condition' in params."

            params["strategy_type"] = strategy_type
            params.setdefault("stoploss", -0.05)

            # Set type-specific defaults
            if strategy_type in ("sma_crossover", "combined_sma_rsi"):
                params.setdefault("fast_ma", 10)
                params.setdefault("slow_ma", 30)

            # Retrieve past winning strategies from memory for context
            try:
                from memory.vector_store import VectorStore
                vs = VectorStore()
                past_winners = vs.get_best_strategies(min_sharpe=0.8, k=3)
                if past_winners:
                    past_text = "\n".join([f"  - {r['text'][:150]}" for r in past_winners])
                    memory_hint = f"\n[Memory: {len(past_winners)} past winning strategies found:\n{past_text}\n]"
                else:
                    memory_hint = ""
            except Exception:
                memory_hint = ""

            strategy_id = uuid.uuid4().hex[:8]
            self._generated_strategies[strategy_id] = params

            summary = f"Strategy [{strategy_id}] created: type={strategy_type}"
            if strategy_type in ("sma_crossover", "combined_sma_rsi"):
                summary += f", fast_ma={params.get('fast_ma')}, slow_ma={params.get('slow_ma')}"
            summary += f", stoploss={params['stoploss']}"
            if params.get("timerange"):
                summary += f", timerange={params['timerange']}"
            if params.get("timeframe"):
                summary += f", timeframe={params['timeframe']}"
            if params.get("pairs"):
                summary += f", pairs={params['pairs']}"
            return summary + memory_hint

        # Tool: run_backtest

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

        def run_backtest(backtest_json: str = "{}") -> str:
            """Backtest a previously-generated strategy.
            Pass JSON: {"strategy_id": "abc123"} or just the strategy_id string.
            Uses timerange and pairs from the strategy params or global backtest config.
            Returns performance metrics and records keep/discard verdict."""
            import json
            from config import settings
            try:
                params = json.loads(backtest_json)
            except json.JSONDecodeError:
                params = {"strategy_id": backtest_json}

            sid = params.get("strategy_id", "")
            if not sid or sid not in self._generated_strategies:
                return f"Error: unknown strategy_id '{sid}'. Use generate_strategy first."

            strat_params = self._generated_strategies[sid].copy()
            # Apply global config as defaults, then strategy-level overrides
            global_cfg = getattr(self, "_backtest_config", {})
            timerange = strat_params.pop("timerange", global_cfg.get("timerange", "20210101-"))
            pairs = strat_params.pop("pairs", global_cfg.get("pairs", None))
            strat_params.setdefault("timeframe", global_cfg.get("timeframe", settings.TIMEFRAME))
            strategy_type = strat_params.pop("strategy_type", "sma_crossover")
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
            lines = [f"Backtest result for [{sid}]:", f"  Total trades: {metrics['total_trades']}"]
            for k, v in metrics.items():
                if k != "total_trades":
                    lines.append(f"  {k}: {v}")
            lines.append(f"  Timerange: {timerange}")
            if pairs:
                lines.append(f"  Pairs: {pairs}")

            # Auto-create iteration record for keep/discard tracking
            strat_params["_strategy_id"] = sid
            rec = IterationRecord(strat_params, result)
            rec.evaluate()
            self._iteration_history.append(rec)

            # Persist result to strategy memory if verdict is kept
            if rec.verdict == "kept":
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
            # Record structured experiment for data-driven iteration
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
                    verdict=rec.verdict,
                    iteration=len(self._iteration_history),
                )
                self._tracker.record(exp)
            except Exception as exc:
                logger.debug("Could not record experiment: %s", exc)
            lines.append(f"  Verdict: {rec.verdict.upper()} -- {rec.reason}")

            return "\n".join(lines)

        def suggest_next_params(_: str = "") -> str:
            """Analyse past attempts and recommend next parameters to try.
            Uses the structured ExperimentTracker for data-driven suggestions.
            Call this after run_backtest + interpret_metrics."""

            if not self._iteration_history:
                return ("No previous attempts found. Start with:\n"
                        "  generate_strategy with strategy_type='sma_crossover', fast_ma=10, slow_ma=30, stoploss=-0.05\n"
                        "Then backtest and iterate.")

            tracker_summary = self._tracker.summary()

            last = self._iteration_history[-1]
            strategy_type = last.params.get("strategy_type", "sma_crossover")
            current_params = {k: v for k, v in last.params.items()
                              if k not in ("strategy_type", "indicator_code",
                                           "entry_condition", "exit_condition")}

            next_params = self._tracker.suggest_next_params(strategy_type, current_params)
            next_params["strategy_type"] = strategy_type

            return (
                f"{tracker_summary}\n\n"
                f"Suggested next params: {json.dumps(next_params, indent=2)}\n"
                f"Use generate_strategy with these params, then run_backtest."
            )

        # ------------------------------------------------------------------
        # New tool: get_best_strategy
        # ------------------------------------------------------------------

        def get_best_strategy(_: str = "") -> str:
            """Return the best strategy found so far from iteration history."""
            if not self._best_strategy:
                return "No optimization run completed yet. Run some backtests first."
            lines = [
                "Best strategy from last optimization:",
                f"  Params: {json.dumps(self._best_params)}",
            ]
            for k, v in self._best_strategy.items():
                lines.append(f"  {k}: {v}")
            return "\n".join(lines)

        # ------------------------------------------------------------------
        # New tool: get_iteration_history
        # ------------------------------------------------------------------

        def get_iteration_history(filter_str: str = "") -> str:
            """Return the full keep/discard history.
            Pass optional filter: 'kept', 'discarded', or empty for all.
            """
            if not self._iteration_history:
                return "No optimization history yet. Use generate_strategy + run_backtest first."

            keep_only = filter_str.strip().lower()
            records = self._iteration_history
            if keep_only == "kept":
                records = [r for r in records if r.verdict == "kept"]
            elif keep_only == "discarded":
                records = [r for r in records if r.verdict == "discarded"]

            lines = [f"Iteration history ({len(records)} records):"]
            for i, r in enumerate(records):
                tag = "KEPT" if r.verdict == "kept" else "DISCARDED" if r.verdict == "discarded" else "SKIP"
                st = r.params.get("strategy_type", "sma_crossover")
                param_summary = ", ".join(f"{k}={v}" for k, v in sorted(r.params.items())
                                          if k not in ("indicator_code", "entry_condition", "exit_condition", "indicator_params_block"))
                sharpe = r.metrics.get("sharpe_ratio", "?")
                wr = r.metrics.get("win_rate", "?")
                trades = r.metrics.get("total_trades", "?")
                if r.metrics.get("error"):
                    lines.append(f"  [{tag}] type={st} [{param_summary}] — {r.metrics['error']}")
                else:
                    lines.append(f"  [{tag}] type={st} [{param_summary}] "
                                 f"Sharpe={sharpe} WR={wr} trades={trades} — {r.reason}")
            return "\n".join(lines)

        def interpret_metrics(metrics_json: str = "{}") -> str:
            """Interpret backtest metrics and suggest improvements.
            Pass JSON with: total_trades, profit_ratio, win_rate, sharpe_ratio, max_drawdown.
            Returns human-readable assessment."""
            try:
                m = json.loads(metrics_json)
            except (json.JSONDecodeError, ValueError):
                return "Error: pass valid JSON with metrics (total_trades, profit_ratio, win_rate, sharpe_ratio, max_drawdown)"

            issues = []
            tips = []

            total = m.get("total_trades", 0)
            wr = m.get("win_rate", 0)
            sharpe = m.get("sharpe_ratio", 0)
            dd = abs(m.get("max_drawdown", 0))
            profit = m.get("profit_ratio", m.get("total_profit", 0))

            if total < 10:
                issues.append(f"Only {total} trades — low sample size")
                tips.append("Use longer timerange or shorter timeframe (e.g. 1h instead of 1d)")
            if wr < 0.4:
                issues.append(f"Low win rate ({wr:.0%})")
                tips.append("Try stricter entry conditions or add volume filter")
            if sharpe < 1.0:
                issues.append(f"Poor risk-adjusted return (Sharpe {sharpe:.2f})")
                tips.append("Tighten stoploss or add exit indicator (RSI/ADX)")
            if dd > 0.05:
                issues.append(f"Excessive drawdown ({dd:.2%})")
                tips.append("Reduce position size or add trailing stop")
            if profit <= 0:
                issues.append(f"Non-positive profit ({profit})")
                tips.append("Reverse the crossover or try a different pair")

            lines = ["Metrics assessment:"]
            lines.append(f"  Trades: {total}, Win rate: {wr:.0%}, Sharpe: {sharpe:.2f}, Drawdown: {dd:.2%}, Profit ratio: {profit}")
            if issues:
                lines.append("  Issues found:")
                for i in issues:
                    lines.append(f"    - {i}")
            else:
                lines.append("  No issues — strategy looks solid.")

            if tips:
                lines.append("  Suggestions:")
                for t in tips:
                    lines.append(f"    - {t}")
            else:
                lines.append("  No suggestions.")
            return "\n".join(lines)

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
            records = {r.params.get("_strategy_id", ""): r for r in self._iteration_history}
            matched = []
            for sid in ids:
                found = None
                # Check records map first
                if sid in records:
                    found = records[sid]
                else:
                    # Search iteration history for records whose params match
                    for rec in self._iteration_history:
                        if rec.params.get("_strategy_id") == sid:
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
                    val = rec.metrics.get(mk, "N/A") if rec and rec.metrics.get(mk) is not None else "N/A"
                    row += f" {str(val):<15} |"
                lines.append(row)
            # Determine winner by Sharpe
            best_sharpe = -999
            best_sid = None
            for sid, rec in matched:
                if rec:
                    s = rec.metrics.get("sharpe_ratio", -999)
                    if isinstance(s, (int, float)) and s > best_sharpe:
                        best_sharpe = s
                        best_sid = sid
            if best_sid:
                lines.append(f"")
                lines.append(f"**Recommended**: [{best_sid}] with Sharpe={best_sharpe}")
            return "\n".join(lines)

        def generate_strategy_for_regime(params_json: str = "{}") -> str:
            """Generate a strategy tailored to the current market regime.
            Pass JSON with optional 'regime' and/or 'strategy_type_hint'.
            If regime is omitted, auto-detects it from live market data.
            If strategy_type_hint is omitted, picks the best fit for the regime."""
            import json
            try:
                params = json.loads(params_json)
            except json.JSONDecodeError:
                params = {}

            from data.regime import MarketRegimeDetector, REGIME_STRATEGY_MAP
            from data.fetcher import MarketDataFetcher
            from config import settings

            regime = params.get("regime", "")
            strategy_type_hint = params.get("strategy_type_hint", "")

            # Auto-detect regime if not provided
            if not regime:
                try:
                    fetcher = MarketDataFetcher()
                    df = fetcher.fetch_ohlcv(settings.SYMBOL, "1h", limit=250)
                    if df is not None and len(df) > 200:
                        detector = MarketRegimeDetector()
                        regime = detector.classify_regime(df)
                except Exception as exc:
                    return f"Error detecting regime: {exc}"

            # Get regime-strategy mapping
            regime_map = REGIME_STRATEGY_MAP.get(regime, {})
            recommended = regime_map.get("use", [])
            avoided = regime_map.get("avoid", [])

            if not recommended:
                return (
                    f"No strategies recommended for regime '{regime}'. "
                    f"Consider switching pairs or waiting for a different market condition."
                )

            # If hint is given, validate it against regime
            if strategy_type_hint:
                if strategy_type_hint in avoided:
                    return (
                        f"Strategy type '{strategy_type_hint}' is discouraged in "
                        f"regime '{regime}'. Recommended: {', '.join(recommended)}. "
                        f"Use generate_strategy instead if you want to override."
                    )
                chosen_type = strategy_type_hint
            else:
                chosen_type = recommended[0]

            # Build params dict for the chosen strategy type
            strategy_params = {"strategy_type": chosen_type, "stoploss": -0.05}

            # Set sensible defaults per type
            if chosen_type in ("sma_crossover", "combined_sma_rsi"):
                strategy_params.update({"fast_ma": 10, "slow_ma": 30})
            elif chosen_type == "macd_crossover":
                strategy_params.update({"macd_fast": 12, "macd_slow": 26, "macd_signal": 9})
            elif chosen_type == "rsi_oversold":
                strategy_params.update({"rsi_period": 14, "rsi_buy_threshold": 30, "rsi_sell_threshold": 70})
            elif chosen_type == "bollinger_bands":
                strategy_params.update({"bb_period": 20})
            elif chosen_type == "momentum":
                strategy_params.update({"roc_period": 10})
            elif chosen_type == "breakout":
                strategy_params.update({"lookback": 20})
            elif chosen_type == "mean_reversion":
                strategy_params.update({"bb_period": 20, "rsi_period": 14})
            elif chosen_type == "volatility_squeeze":
                strategy_params.update({"bb_period": 20})
            elif chosen_type == "sentiment_driven":
                strategy_params.update({"rsi_period": 14})
            elif chosen_type == "multi_timeframe":
                strategy_params.update({"fast_ma": 20, "slow_ma": 50})

            # Generate the strategy
            strategy_id = uuid.uuid4().hex[:8]
            self._generated_strategies[strategy_id] = strategy_params

            return (
                f"Regime-aware strategy [{strategy_id}] created:\n"
                f"  Regime: {regime}\n"
                f"  Type: {chosen_type}\n"
                f"  Params: {json.dumps(strategy_params)}\n"
                f"  Regime recommended types: {', '.join(recommended)}\n"
                f"  Regime avoided types: {', '.join(avoided)}"
            )

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

            # ── 2B: Check if this strategy type supports hyperopt ──
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

        def get_research_history(query_json: str = '{"goal":""}') -> str:
            """Query past research iterations from memory.
            Pass JSON: {"goal": "SMA crossover BTC", "k": 5}
            Returns list of past hypotheses, metrics, and critiques.
            Useful for avoiding previously-failed approaches."""
            import json
            from memory.vector_store import VectorStore
            try:
                params = json.loads(query_json) if query_json.strip() else {}
            except json.JSONDecodeError:
                params = {"goal": query_json}
            query = params.get("goal", "")
            k = int(params.get("k", 5))
            if not query:
                return "Error: empty query"
            vs = VectorStore()
            if vs.count() == 0:
                return "No research history yet."
            results = vs.query_similar(query, k=k)
            # Filter to research_iteration type
            research = [r for r in results if r["metadata"].get("type") == "research_iteration"]
            if not research:
                return "No past research iterations found."
            lines = [f"Found {len(research)} past research iterations:"]
            for r in research:
                lines.append(f"  - [{r['metadata'].get('iteration', '?')}] {r['metadata'].get('verdict', '?')}: {r['text'][:150]}")
            return "\n".join(lines)

        return [
            Tool(name="generate_strategy", func=generate_strategy,
                 description="Create a strategy of any type. Args: JSON with strategy_type and type-specific params. "
                             "Types: sma_crossover, macd_crossover, rsi_oversold, bollinger_bands, combined_sma_rsi. "
                             "AVOID 'custom' type — it often produces broken Python."),
            # Backward-compat alias
            Tool(name="generate_sma_strategy", func=generate_strategy,
                 description="[DEPRECATED] Use generate_strategy instead."),
            Tool(name="set_backtest_config", func=set_backtest_config,
                 description="Set global backtest config: timerange, pairs, timeframe. Applies to all subsequent backtests."),
            Tool(name="run_backtest", func=run_backtest,
                 description="Backtest a generated strategy by ID. Args: JSON with strategy_id."),
            Tool(name="download_data", func=download_data,
                 description="Download historical data for backtesting. Args: JSON with pairs, timeframe, timerange."),
            Tool(name="compare_strategies", func=compare_strategies,
                 description="Compare 2+ strategies side-by-side by IDs. Args: JSON with ids list."),
            Tool(name="interpret_metrics", func=interpret_metrics,
                 description="Interpret metrics and suggest improvements. Args: JSON with total_trades, profit_ratio, win_rate, sharpe_ratio, max_drawdown."),
            Tool(name="suggest_next_params", func=suggest_next_params,
                 description="Analyse past attempts and recommend next parameters to try. Call after run_backtest + interpret_metrics."),
            Tool(name="get_best_strategy", func=get_best_strategy,
                 description="Get the best strategy found so far from iteration history."),
            Tool(name="get_iteration_history", func=get_iteration_history,
                 description="View all attempts, filtered by 'kept' or 'discarded'."),
            Tool(name="get_research_history", func=get_research_history,
                 description="Query past research iterations from ChromaDB memory. Args: JSON with goal and k."),
            Tool(name="generate_strategy_for_regime", func=generate_strategy_for_regime,
                 description="Generate a strategy tailored to the current market regime. "
                             "Args: JSON with regime (optional, auto-detected if omitted) and strategy_type_hint. "
                             "Automatically restricts to strategies recommended for that regime."),
            Tool(name="run_hyperopt", func=run_hyperopt,
                 description="Run Freqtrade hyperopt for automatic parameter optimization. "
                             "Args: JSON with strategy_id, epochs (default 50), loss function. "
                             "Use after generate_strategy to find optimal parameters automatically."),
            Tool(name="walk_forward_validate", func=walk_forward_validate,
                 description="Validate strategy robustness across multiple time windows. "
                             "Args: JSON with strategy_id and windows (default 3). "
                             "Use before declaring a strategy as kept to check it is not overfit."),
        ]