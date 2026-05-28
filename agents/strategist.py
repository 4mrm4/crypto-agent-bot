"""Strategist agent — creates, backtests, and iterates on trading strategies."""

import json
import logging
import tempfile
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
- sma_crossover: Simple moving average crossover (fast_ma, slow_ma params)
- macd_crossover: MACD line crossing signal line
- rsi_oversold: Buy when RSI exits oversold (<30), sell when overbought (>70)
- bollinger_bands: Buy when price touches lower band, sell at upper band
- combined_sma_rsi: SMA crossover with RSI filter
- custom: Provide raw indicator_code, entry_condition, exit_condition

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

IMPORTANT: Use ONLY plain ASCII text. No emoji, no Unicode symbols, no special characters."""

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

            Types: sma_crossover, macd_crossover, rsi_oversold, bollinger_bands, combined_sma_rsi, custom

            You can also set backtest config: timerange (e.g. "20250101-20251231"),
            pairs (list of exchange symbols), and timeframe (e.g. "1h", "15m", "4h").

            Examples:
              SMA:     {"strategy_type": "sma_crossover", "fast_ma": 10, "slow_ma": 30}
              MACD:    {"strategy_type": "macd_crossover", "timerange": "20250101-20251231", "timeframe": "15m"}
              RSI:     {"strategy_type": "rsi_oversold", "pairs": ["BTC/USDT", "ETH/USDT"]}
              Custom:  {"strategy_type": "custom", "indicator_code": "...", "entry_condition": "...", "exit_condition": "..."}

            Returns the strategy ID."""
            import json, uuid
            try:
                params = json.loads(params_json)
            except json.JSONDecodeError:
                params = {}

            strategy_type = params.pop("strategy_type", "sma_crossover")
            valid_types = {"sma_crossover", "macd_crossover", "rsi_oversold",
                           "bollinger_bands", "combined_sma_rsi", "custom"}
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
            return summary

        def set_backtest_config(config_json: str = "{}") -> str:
            """Set global backtest configuration that applies to all subsequent runs.
            Pass JSON with any of: timerange, pairs, timeframe, stoploss, trailing_stop.

            Examples:
              {"timerange": "20250101-20251231", "timeframe": "15m", "pairs": ["BTC/USDT", "ETH/USDT"]}
              {"timerange": "20260101-", "timeframe": "1h"}
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
            timerange = strat_params.pop("timerange", global_cfg.get("timerange", "20260101-"))
            pairs = strat_params.pop("pairs", global_cfg.get("pairs", None))
            strat_params.setdefault("timeframe", "1d")
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
            rec = IterationRecord(strat_params, result)
            rec.evaluate()
            self._iteration_history.append(rec)
            lines.append(f"  Verdict: {rec.verdict.upper()} -- {rec.reason}")

            return "\n".join(lines)

        def suggest_next_params(_: str = "") -> str:
            """Analyse iteration history and suggest what to try next.
            Works with any strategy type. Looks at past keep/discard records
            and recommends the most promising direction.
            Call this after run_backtest + interpret_metrics."""

            if not self._iteration_history:
                return ("No previous attempts found. Start with:\n"
                        "  generate_strategy with strategy_type='sma_crossover', fast_ma=10, slow_ma=30, stoploss=-0.05\n"
                        "Then backtest and iterate.")

            # Group best result by strategy type
            best_by_type = {}
            seen_types = set()
            for rec in self._iteration_history:
                st = rec.params.get("strategy_type", "sma_crossover")
                seen_types.add(st)
                sharpe = rec.metrics.get("sharpe_ratio", -999)
                if isinstance(sharpe, (int, float)):
                    prev = best_by_type.get(st, {}).get("sharpe", -999)
                    if sharpe > prev:
                        best_by_type[st] = {"params": rec.params, "sharpe": sharpe, "metrics": rec.metrics}

            # Analyse recent issues
            recent = self._iteration_history[-3:]
            issues_seen = []
            for rec in recent:
                if rec.verdict == "discarded":
                    issues_seen.append(rec.reason)
            common_issues = "; ".join(sorted(set(issues_seen))) if issues_seen else "none"

            lines = [
                f"Iteration history: {len(self._iteration_history)} attempts across types: {', '.join(sorted(seen_types))}",
                f"Recent issues: {common_issues}",
            ]

            for st, best in best_by_type.items():
                lines.append(f"Best {st}: Sharpe={best['sharpe']:.2f} params={best['params']}")

            # Generic metric-driven suggestions (type-agnostic)
            if "Sharpe" in common_issues:
                lines.append("Suggestion: Tighten stoploss, add a filter indicator, or try a different strategy type.")
            elif "Win rate" in common_issues:
                lines.append("Suggestion: Stricter entry conditions -- increase indicator thresholds or combine with a filter.")
            elif "Drawdown" in common_issues:
                lines.append("Suggestion: Reduce stoploss, add trailing stop, or lower position sizing.")
            elif "profit" in common_issues.lower():
                lines.append("Suggestion: Try the opposite signal direction, a different pair, or a different timeframe.")
            else:
                lines.append("Suggestion: Try adjusting parameters or switching to a different strategy type.")

            # For SMA-based strategies, suggest numerical ranges
            sma_types = [st for st in seen_types if st in ("sma_crossover", "combined_sma_rsi")]
            if sma_types and best_by_type:
                best = best_by_type.get(sma_types[0])
                if best:
                    bf = best["params"].get("fast_ma", 10)
                    bs = best["params"].get("slow_ma", 30)
                    tested_fast = set()
                    tested_slow = set()
                    for rec in self._iteration_history:
                        f = rec.params.get("fast_ma")
                        s = rec.params.get("slow_ma")
                        if f: tested_fast.add(f)
                        if s: tested_slow.add(s)

                    lines.append("")
                    lines.append("SMA next params to try:")
                    candidates = [(bf + 5, bs + 10), (bf, bs + 20), (bf + 10, bs + 30), (bf - 3, bs - 5)]
                    found = False
                    for cf, cs in candidates:
                        if cf not in tested_fast and cs not in tested_slow and cf < cs:
                            lines.append(f"  fast_ma={cf}, slow_ma={cs}")
                            found = True
                            break
                    if not found:
                        lines.append("  (nearby SMA combos exhausted -- try a different strategy type)")

            return "\n".join(lines)

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
            """Download historical market data for backtesting.
            Pass JSON: {"pairs": ["BTC/USDT"], "timeframe": "1h", "timerange": "20260101-"}
            Defaults: pairs=["BTC/USDT"], timeframe=config.TIMEFRAME, timerange="20260101-"
            Returns download status and row counts."""
            import json
            try:
                params = json.loads(params_json)
            except json.JSONDecodeError:
                params = {}
            from config import settings
            pairs = params.get("pairs", [settings.SYMBOL])
            timerange = params.get("timerange", "20260101-")
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
            records = {r.params.get("_sid", ""): r for r in self._iteration_history}
            matched = []
            for sid in ids:
                rec = records.get(sid)
                if rec:
                    matched.append((sid, rec))
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
                             "Types: sma_crossover, macd_crossover, rsi_oversold, bollinger_bands, combined_sma_rsi, custom."),
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
        ]