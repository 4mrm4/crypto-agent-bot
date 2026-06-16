"""Strategist agent — designs and iterates on trading strategy specifications."""

import json
import logging
import uuid
from typing import Any, Dict, List

from langchain_core.tools import Tool

from agents.base import BaseAgent
from agents.iteration_tracker import IterationRecord

logger = logging.getLogger(__name__)

STRATEGIST_SYSTEM_PROMPT = """You are a quantitative trading strategist. Your job is to:
1. Create a SINGLE trading strategy by specifying type and parameters
2. Use get_strategy_concepts to retrieve proven concept templates
3. Check past research via get_research_history to avoid repeating failures
4. Use suggest_next_params to decide what to tweak next
5. Generate and refine one strategy at a time

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
- vwap_deviation: VWAP deviation mean reversion (vwap_period, deviation_threshold)
- ema_ribbon: EMA ribbon alignment (ema_min, ema_max, ema_step)
- stoch_rsi: Stochastic RSI K/D crossover (stoch_rsi_period, oversold, overbought)
- adx_filter: ADX trend strength + DI direction filter (adx_period, adx_threshold)

REGIME GUIDANCE:
- strong_uptrend -> prefer sma_crossover, momentum, combined_sma_rsi, ema_ribbon, adx_filter
- ranging -> prefer bollinger_bands, rsi_oversold, mean_reversion, vwap_deviation, stoch_rsi
- volatile -> prefer breakout, volatility_squeeze
- weak_trend -> prefer macd_crossover, combined_sma_rsi

ADX AS MODIFIER:
ADX filter (adx_filter) can be composed with other types by passing
adx_period + adx_threshold alongside the base strategy params.
The BacktestEngine treats adx_filter as a standalone type in Phase 1;
future versions will support true composition (e.g. "sma_crossover + adx_filter").

WORK ITERATIVELY — one strategy at a time. Do NOT enumerate many variants at once.
Use this cycle:
  get_strategy_concepts -> generate_strategy -> suggest_next_params -> repeat

Target metrics:
- Sharpe ratio > 0.8
- Win rate > 40%
- Max drawdown < 15%
- At least 5 trades
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
- VWAP Deviation: price below VWAP threshold -> use vwap_deviation(vwap_period=20, deviation_threshold=15)
- EMA Ribbon: multi-EMA bullish alignment -> use ema_ribbon(ema_min=3, ema_max=24, ema_step=3)
- StochRSI: RSI-based stochastic cross -> use stoch_rsi(stoch_rsi_period=14, oversold=20, overbought=80)
- ADX Trend: trend strength + direction -> use adx_filter(adx_period=14, adx_threshold=25)

Map the research goal to the most suitable concept, then pick its
freqtrade_type and suggested_params as your starting point.

IMPORTANT: Use ONLY plain ASCII text. No emoji, no Unicode symbols, no special characters.

At the END of your response, ALWAYS output a line in EXACTLY this format:
next: backtest strategy_type=STRATEGY_TYPE params={"key": "value"}

Replace STRATEGY_TYPE and params with the actual strategy you generated.
Example:
next: backtest strategy_type=sma_crossover params={"fast_ma": 10, "slow_ma": 30}

Optional trailing stop params (add to params dict to enable):
  "trailing_stop": true, "trailing_stop_positive": 0.01, "trailing_stop_positive_offset": 0.02
  "trailing_only_offset_is_reached": false
With these, the stoploss tightens to 1% once price reaches 2% profit.

The 'next: ' prefix is how the system creates follow-up tasks.
If you do not output this line, the strategy will never be backtested.
"""

class StrategistAgent(BaseAgent):
    """Specialised agent that designs strategy specifications."""

    def __init__(self):
        # Track strategies we've generated
        self._generated_strategies: Dict[str, Dict[str, Any]] = {}
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
            volatility_squeeze, sentiment_driven, multi_timeframe,
            vwap_deviation, ema_ribbon, stoch_rsi, adx_filter

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
                           "multi_timeframe", "vwap_deviation",
                           "ema_ribbon", "stoch_rsi", "adx_filter"}
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

        def suggest_next_params(_: str = "") -> str:
            """Analyse past generated strategies and recommend next parameters to try.
            Looks at the last generated strategy and suggests logical tweaks
            based on strategy type defaults."""
            if not self._generated_strategies:
                return ("No strategies generated yet. Start with:\n"
                        "  generate_strategy with strategy_type='sma_crossover', fast_ma=10, slow_ma=30, stoploss=-0.05")

            last_key = list(self._generated_strategies.keys())[-1]
            last = self._generated_strategies[last_key]
            strategy_type = last.get("strategy_type", "sma_crossover")

            # Build generic suggestions by strategy type
            suggestions = {"strategy_type": strategy_type}
            if strategy_type in ("sma_crossover", "combined_sma_rsi"):
                current_fast = last.get("fast_ma", 10)
                current_slow = last.get("slow_ma", 30)
                suggestions["fast_ma"] = current_fast + 5
                suggestions["slow_ma"] = current_slow + 10
            elif strategy_type == "rsi_oversold":
                suggestions["rsi_buy_threshold"] = 25
                suggestions["rsi_sell_threshold"] = 75
            elif strategy_type == "bollinger_bands":
                suggestions["bb_period"] = last.get("bb_period", 20) + 5
            elif strategy_type == "macd_crossover":
                suggestions["macd_fast"] = 12
                suggestions["macd_slow"] = 26
                suggestions["macd_signal"] = 9

            return (
                f"Last strategy [{last_key}]: type={strategy_type}, "
                f"params={json.dumps(last, default=str)}\n\n"
                f"Suggested next params to try: {json.dumps(suggestions, indent=2)}\n"
                f"Note: generate_strategy with these params, then pass the strategy_id "
                f"to BacktesterAgent for backtesting."
            )

        def get_strategy_concepts(_: str = "") -> str:
            """Retrieve the proven strategy concept templates.
            Returns a curated list of battle-tested strategy concepts
            mapped to their freqtrade_type and suggested starting params.
            Use this when deciding what kind of strategy to create."""
            return ("STRATEGY CONCEPT LIBRARY:\n\n"
                    "- Golden Cross: SMA50/200 crossover -> sma_crossover(fast_ma=50, slow_ma=200)\n"
                    "- RSI Divergence: oversold bounce -> rsi_oversold(rsi_buy_threshold=30, rsi_sell_threshold=70)\n"
                    "- Volatility Breakout: N-day high + volume -> breakout(lookback=20)\n"
                    "- MACD Histogram Reversal: zero-cross -> macd_crossover(macd_fast=12, macd_slow=26, macd_signal=9)\n"
                    "- BB Squeeze: contraction then expansion -> volatility_squeeze(bb_period=20)\n"
                    "- Fear/Greed Contrarian: extreme fear buy -> sentiment_driven(rsi_period=14)\n"
                    "- Multi-TF Trend: short cross + long confirm -> multi_timeframe(fast_ma=20, slow_ma=50)\n"
                    "- Momentum + Volume: ROC + volume spike -> momentum(roc_period=10)\n"
                    "- Mean Reversion: BB + RSI oversold -> mean_reversion(bb_period=20, rsi_period=14)\n"
                    "- VWAP Deviation: price below VWAP threshold -> vwap_deviation(vwap_period=20, deviation_threshold=15)\n"
                    "- EMA Ribbon: multi-EMA bullish alignment -> ema_ribbon(ema_min=3, ema_max=24, ema_step=3)\n"
                    "- StochRSI: RSI stochastic crossover -> stoch_rsi(stoch_rsi_period=14, oversold=20, overbought=80)\n"
                    "- ADX Trend: trend strength + direction -> adx_filter(adx_period=14, adx_threshold=25)\n\n"
                    "REGIME GUIDANCE:\n"
                    "- strong_uptrend -> sma_crossover, momentum, combined_sma_rsi, ema_ribbon, adx_filter\n"
                    "- ranging -> bollinger_bands, rsi_oversold, mean_reversion, vwap_deviation, stoch_rsi\n"
                    "- volatile -> breakout, volatility_squeeze\n"
                    "- weak_trend -> macd_crossover, combined_sma_rsi")

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
            research = [r for r in results if r["metadata"].get("type") == "research_iteration"]
            if not research:
                return "No past research iterations found."
            lines = [f"Found {len(research)} past research iterations:"]
            for r in research:
                lines.append(f"  - [{r['metadata'].get('iteration', '?')}] {r['metadata'].get('verdict', '?')}: {r['text'][:150]}")
            return "\n".join(lines)

        return [
            Tool(name="generate_strategy", func=generate_strategy,
                 description="Create a strategy. Pass JSON with strategy_type and type-specific params. "
                             "Valid types: sma_crossover, macd_crossover, rsi_oversold, bollinger_bands, "
                             "combined_sma_rsi, momentum, breakout, mean_reversion, volatility_squeeze, "
                             "sentiment_driven, multi_timeframe, vwap_deviation, ema_ribbon, stoch_rsi, adx_filter."),
            Tool(name="suggest_next_params", func=suggest_next_params,
                 description="Analyse past generated strategies and recommend next parameters to try."),
            Tool(name="get_strategy_concepts", func=get_strategy_concepts,
                 description="Retrieve proven strategy concept templates mapped to types and suggested params."),
            Tool(name="get_research_history", func=get_research_history,
                 description="Query past research iterations from ChromaDB memory. Args: JSON with goal and k."),
        ]