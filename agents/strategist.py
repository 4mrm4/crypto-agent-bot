"""Strategist agent — creates, backtests, and iterates on trading strategies."""

import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from langchain_core.tools import Tool

from agents.base import BaseAgent
from backtesting.engine import BacktestEngine

logger = logging.getLogger(__name__)

STRATEGIST_SYSTEM_PROMPT = """You are a quantitative trading strategist. Your job is to:
1. Create trading strategies by specifying parameters (fast_ma, slow_ma, stoploss)
2. Run backtests on those strategies
3. Interpret the metrics and suggest improvements

Use generate_sma_strategy to create a strategy with SMA crossover parameters.
Then use run_backtest to test it. Check the metrics — aim for:
- Sharpe ratio > 1.0
- Win rate > 50%
- Max drawdown < 5%
- Positive profit ratio

If the metrics are poor, adjust parameters and re-run. Report your findings.

IMPORTANT: Always use generate_sma_strategy FIRST to create a strategy before backtesting it.
IMPORTANT: Use ONLY plain ASCII text. No emoji, no Unicode symbols, no special characters."""


class StrategistAgent(BaseAgent):
    """Specialised agent that creates, backtests and optimises trading strategies."""

    def __init__(self, engine: Optional[BacktestEngine] = None):
        self._engine = engine or BacktestEngine()
        # Track strategies we've generated
        self._generated_strategies: Dict[str, Dict[str, Any]] = {}
        tools = self._build_tools()
        super().__init__(
            name="strategist",
            tools=tools,
            system_prompt=STRATEGIST_SYSTEM_PROMPT,
        )

    def _build_tools(self):
        def generate_sma_strategy(params_json: str = "{}") -> str:
            """Generate an SMA crossover strategy. Pass JSON with: fast_ma, slow_ma, stoploss.
            Example: '{"fast_ma": 10, "slow_ma": 30, "stoploss": -0.05}'
            Returns strategy identifier you can pass to run_backtest."""
            try:
                params = json.loads(params_json) if params_json.strip() else {}
            except json.JSONDecodeError:
                return f"Error: invalid JSON. Got: {params_json[:100]}"

            params.setdefault("fast_ma", 10)
            params.setdefault("slow_ma", 30)
            params.setdefault("stoploss", -0.05)

            strat_id = f"sma_f{params['fast_ma']}_s{params['slow_ma']}"
            self._generated_strategies[strat_id] = params

            return (
                f"Strategy '{strat_id}' created.\n"
                f"  fast_ma={params['fast_ma']}, slow_ma={params['slow_ma']}, "
                f"stoploss={params['stoploss']}\n"
                f"Use run_backtest with strategy_id='{strat_id}' to test it."
            )

        def run_backtest_fn(kwargs_json: str = "{}") -> str:
            """Run a backtest for a previously generated strategy.
            Pass JSON: {"strategy_id": "...", "timerange": "20260427-20260527", "pair": "BTC/USDT"}
            strategy_id comes from generate_sma_strategy's output."""
            try:
                kwargs = json.loads(kwargs_json) if kwargs_json.strip() else {}
            except json.JSONDecodeError:
                return f"Error: invalid JSON. Got: {kwargs_json[:100]}"

            strat_id = kwargs.get("strategy_id", "")
            if strat_id not in self._generated_strategies:
                available = list(self._generated_strategies.keys())
                return (
                    f"Error: strategy '{strat_id}' not found. "
                    f"Generate one first with generate_sma_strategy. Available: {available}"
                )

            params = self._generated_strategies[strat_id]
            timerange = kwargs.get("timerange", "20260427-20260527")
            pair = kwargs.get("pair", "BTC/USDT")

            try:
                result = self._engine.run_backtest(
                    strategy_params=params,
                    timerange=timerange,
                    pairs=[pair],
                )
            except Exception as exc:
                return f"Backtest error: {exc}"

            if "error" in result:
                return f"Backtest parsing error: {result['error']}"

            return (
                f"Backtest results for '{strat_id}' on {pair} ({timerange}):\n"
                f"  Total trades:  {result['total_trades']}\n"
                f"  Profit ratio:  {result['profit_ratio']:.4f}\n"
                f"  Win rate:      {result['win_rate']:.2%}\n"
                f"  Max drawdown:  {result['max_drawdown']:.4f}\n"
                f"  Sharpe ratio:  {result['sharpe_ratio']:.2f}\n"
            )

        def interpret_metrics_fn(metrics_json: str = "{}") -> str:
            """Interpret backtest metrics. Pass JSON with the metric values.
            Example: {"sharpe_ratio": 1.5, "win_rate": 0.55, "profit_ratio": 0.02}"""
            try:
                metrics = json.loads(metrics_json) if metrics_json.strip() else {}
            except json.JSONDecodeError:
                return f"Error: invalid JSON. Got: {metrics_json[:100]}"

            sharpe = metrics.get("sharpe_ratio", 0)
            win_rate = metrics.get("win_rate", 0)
            profit = metrics.get("profit_ratio", 0)
            drawdown = metrics.get("max_drawdown", 1)

            verdict_parts = []
            if sharpe > 1.0:
                verdict_parts.append("GOOD Sharpe (>1.0)")
            elif sharpe > 0.5:
                verdict_parts.append("OK Sharpe (0.5-1.0)")
            else:
                verdict_parts.append("POOR Sharpe (<0.5)")

            if win_rate > 0.5:
                verdict_parts.append("win rate >50%")
            else:
                verdict_parts.append(f"win rate {win_rate:.0%}")

            verdict = " / ".join(verdict_parts)
            return (
                f"Metrics interpretation:\n"
                f"  Sharpe: {sharpe:.2f} — {'good' if sharpe > 1 else 'needs improvement'}\n"
                f"  Win rate: {win_rate:.2%}\n"
                f"  Profit: {profit:.4f}\n"
                f"  Drawdown: {drawdown:.4f}\n"
                f"  Verdict: {verdict}\n"
                f"  Hint: {'Try adjusting fast_ma/slow_ma or increasing stoploss' if sharpe < 1 else 'Strategy is viable'}"
            )

        return [
            Tool(name="generate_sma_strategy", func=generate_sma_strategy,
                 description="Create an SMA crossover strategy. Args: JSON with fast_ma, slow_ma, stoploss. Returns strategy_id."),
            Tool(name="run_backtest", func=run_backtest_fn,
                 description="Backtest a strategy. Args: JSON with strategy_id, timerange, pair. Returns metrics."),
            Tool(name="interpret_metrics", func=interpret_metrics_fn,
                 description="Interpret backtest metrics and get improvement hints. Args: JSON with sharpe_ratio, win_rate, profit_ratio, max_drawdown."),
        ]