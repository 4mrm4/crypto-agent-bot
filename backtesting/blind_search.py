"""
BlindParameterSearch — generates parameter batches BEFORE seeing any results.

The LLM's job is to define the search space, not to pick parameters reactively.
This prevents the manual curve-fitting that happens when the LLM sees results
and adjusts parameters in response.

Phases:
  1. generate_search_space — LLM defines N variants WITHOUT seeing results
  2. batch_backtest — run all variants, LLM sees nothing
  3. compute_aggregate_stats — LLM sees only distribution statistics
  4. get_direction_from_llm — LLM picks parameter region to explore
  5. select_best_for_wfv — pure quantitative selection, no LLM
"""

import logging
from typing import Any, Dict, List, Optional

from backtesting.data_split import DATA_SPLIT
from backtesting.engine import BacktestEngine

logger = logging.getLogger(__name__)


class BlindParameterSearch:
    """Generates parameter batches blind, then selects the best quantitatively.

    The LLM defines the search space before seeing any results.
    Individual backtest results are never shown to the LLM — only aggregate stats.
    """

    def __init__(self, engine: Optional[BacktestEngine] = None):
        self._engine = engine or BacktestEngine()

    def generate_search_space(
        self,
        strategy_type: str,
        regime: str,
        llm_agent,
        n_variants: int = 20,
    ) -> List[Dict[str, Any]]:
        """Prompt the LLM to define N parameter variants WITHOUT showing any results first.

        Args:
            strategy_type: Type of strategy (e.g. 'sma_crossover')
            regime: Current market regime
            llm_agent: StrategistAgent or any agent capable of generating strategies
            n_variants: Number of distinct parameter combinations to generate

        Returns:
            List of parameter dicts. LLM does not run backtests yet.
        """
        prompt = (
            f"You are defining a parameter search space for {strategy_type} "
            f"in the {regime} market regime.\n\n"
            f"Generate {n_variants} distinct parameter combinations covering:\n"
            f"- Conservative values (wide ranges, lower sensitivity)\n"
            f"- Aggressive values (tight ranges, high sensitivity)\n"
            f"- Middle ground\n\n"
            f"IMPORTANT: You will NOT see backtest results during this phase. "
            f"Do NOT reference previous backtest results. "
            f"Focus on parameter diversity across the full sensible range.\n\n"
            f"Output a JSON list of parameter dicts only, no explanation."
        )

        try:
            result = llm_agent.run(prompt)
            output = result.get("output", "")

            # Parse JSON from output
            import json, re
            # Find JSON array in output
            json_match = re.search(r"\[.*\]", output, re.DOTALL)
            if json_match:
                variants = json.loads(json_match.group(0))
            else:
                # Try parsing entire output as JSON
                variants = json.loads(output)

            if not isinstance(variants, list):
                logger.warning("LLM did not return a list, got %s", type(variants))
                return self._generate_default_variants(strategy_type, n_variants)

            logger.info(
                "LLM generated %d parameter variants for %s/%s",
                len(variants), strategy_type, regime,
            )
            return variants[:n_variants]

        except Exception as exc:
            logger.warning("Blind search space generation failed: %s", exc)
            return self._generate_default_variants(strategy_type, n_variants)

    def _generate_default_variants(
        self, strategy_type: str, n: int
    ) -> List[Dict[str, Any]]:
        """Fallback: generate diverse parameter variants deterministically."""
        from backtesting.strategy_templates import STRATEGY_REGISTRY
        registry = STRATEGY_REGISTRY.get(strategy_type, {})
        defaults = dict(registry.get("default_params", {}))

        variants = []
        for i in range(n):
            variant = dict(defaults)
            # Add slight random variation to numeric params
            for key, val in defaults.items():
                if isinstance(val, (int, float)) and key != "startup_candle_count":
                    spread = val * 0.5
                    variant[key] = val + (i - n // 2) * spread / (n // 2)
            variants.append(variant)
        return variants

    def batch_backtest(
        self,
        variants: List[Dict[str, Any]],
        strategy_type: str,
        pairs: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Run all variants against DATA_SPLIT.research_timerange().

        Returns raw results list. LLM does not see this.
        """
        results = []
        for i, params in enumerate(variants):
            try:
                result = self._engine.run_backtest(
                    strategy_params=params,
                    strategy_type=strategy_type,
                    timerange=DATA_SPLIT.research_timerange(),
                    pairs=pairs,
                )
                results.append({
                    "variant_index": i,
                    "params": params,
                    "sharpe_ratio": result.get("sharpe_ratio", 0),
                    "win_rate": result.get("win_rate", 0),
                    "max_drawdown": result.get("max_drawdown", 0),
                    "total_trades": result.get("total_trades", 0),
                    "profit_ratio": result.get("profit_ratio", 0),
                    "error": result.get("error", ""),
                })
            except Exception as exc:
                logger.warning("Batch variant %d failed: %s", i, exc)
                results.append({
                    "variant_index": i,
                    "params": params,
                    "error": str(exc),
                    "sharpe_ratio": 0, "win_rate": 0,
                    "max_drawdown": 0, "total_trades": 0,
                })

        logger.info(
            "Batch backtest complete: %d/%d variants succeeded",
            sum(1 for r in results if not r.get("error")),
            len(variants),
        )
        return results

    def compute_aggregate_stats(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compute statistics the LLM IS allowed to see.

        Includes:
        - Median Sharpe, Win Rate, Drawdown across all variants
        - Standard deviation (spread of results)
        - % variants that pass convergence criteria
        - Best-performing parameter REGION (not specific values)
        - Worst-performing parameter region

        Does NOT include: individual run metrics, specific winning params,
        the actual best result.
        """
        import numpy as np

        valid = [r for r in results if r.get("total_trades", 0) >= 5]
        if not valid:
            return {
                "median_sharpe": 0.0,
                "median_win_rate": 0.0,
                "std_sharpe": 0.0,
                "pct_passing": 0.0,
                "best_region": "none",
                "worst_region": "none",
                "n_valid": 0,
                "n_total": len(results),
            }

        sharpes = np.array([r.get("sharpe_ratio", 0) for r in valid])
        win_rates = np.array([r.get("win_rate", 0) for r in valid])
        drawdowns = np.array([abs(r.get("max_drawdown", 0)) for r in valid])

        # Determine parameter region from top/bottom performers
        sorted_by_sharpe = sorted(valid, key=lambda r: r.get("sharpe_ratio", 0), reverse=True)
        top_quarter = sorted_by_sharpe[:max(1, len(sorted_by_sharpe) // 4)]
        bottom_quarter = sorted_by_sharpe[-max(1, len(sorted_by_sharpe) // 4):]

        # "Region" = range of key parameters
        def _describe_region(variants: List[dict]) -> str:
            if not variants:
                return "unknown"
            key_params = [k for k in variants[0].get("params", {})
                         if isinstance(variants[0]["params"][k], (int, float))
                         and k != "startup_candle_count"]
            if not key_params:
                return "standard_range"
            parts = []
            for k in key_params[:3]:  # Limit to 3 keys
                vals = [v["params"].get(k, 0) for v in variants]
                parts.append(f"{k}=[{min(vals):.1f}..{max(vals):.1f}]")
            return ", ".join(parts)

        best_region = _describe_region(top_quarter)
        worst_region = _describe_region(bottom_quarter)

        # % passing convergence criteria
        passing = sum(
            1 for r in valid
            if r.get("sharpe_ratio", 0) >= 0.5
            and r.get("win_rate", 0) >= 0.30
            and abs(r.get("max_drawdown", 0)) <= 0.15
        )

        return {
            "median_sharpe": round(float(np.median(sharpes)), 3),
            "median_win_rate": round(float(np.median(win_rates)), 3),
            "median_drawdown": round(float(np.median(drawdowns)), 3),
            "std_sharpe": round(float(np.std(sharpes)), 3),
            "pct_passing": round(passing / len(valid) * 100, 1),
            "best_region": best_region,
            "worst_region": worst_region,
            "n_valid": len(valid),
            "n_total": len(results),
            "sharpe_range": [
                round(float(np.min(sharpes)), 2),
                round(float(np.max(sharpes)), 2),
            ],
        }

    def get_direction_from_llm(
        self,
        aggregate_stats: Dict[str, Any],
        strategy_type: str,
        llm_agent,
    ) -> Dict[str, Any]:
        """Show LLM only aggregate_stats and ask for directional guidance.

        The LLM sees distributions only — never individual results.
        Returns directional guidance, not specific parameter values.
        """
        prompt = (
            f"Given the following aggregate statistics from a blind batch backtest "
            f"of {strategy_type} strategies, what parameter region should we explore next?\n\n"
            f"Aggregate statistics:\n"
            f"- Median Sharpe: {aggregate_stats.get('median_sharpe', 0):.3f}\n"
            f"- Median Win Rate: {aggregate_stats.get('median_win_rate', 0):.1%}\n"
            f"- Median Drawdown: {aggregate_stats.get('median_drawdown', 0):.1%}\n"
            f"- Sharpe spread (std): {aggregate_stats.get('std_sharpe', 0):.3f}\n"
            f"- % passing minimum criteria: {aggregate_stats.get('pct_passing', 0):.1f}%\n"
            f"- Best parameter region: {aggregate_stats.get('best_region', 'unknown')}\n"
            f"- Worst parameter region: {aggregate_stats.get('worst_region', 'unknown')}\n"
            f"- Sharpe range: {aggregate_stats.get('sharpe_range', [0, 0])}\n\n"
            f"Do NOT suggest specific parameter values. "
            f"Describe the region of parameter space to explore next "
            f"(e.g. 'tighten RSI threshold range toward oversold', "
            f"'move toward faster moving averages')."
        )

        try:
            result = llm_agent.run(prompt)
            direction = result.get("output", "")
            return {"direction": direction[:500]}
        except Exception as exc:
            logger.warning("LLM direction failed: %s", exc)
            return {"direction": "continue with current parameter range"}

    def select_best_for_wfv(
        self, results: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Select the best variant for WFV WITHOUT LLM involvement.

        Pure quantitative selection: highest composite score.
        LLM never sees this selection was made.
        """
        valid = [r for r in results if not r.get("error") and r.get("total_trades", 0) >= 5]
        if not valid:
            return None

        def _composite_score(r: dict) -> float:
            sharpe = max(r.get("sharpe_ratio", 0), 0)
            wr = r.get("win_rate", 0)
            dd = abs(r.get("max_drawdown", 0))
            trades = r.get("total_trades", 0)

            score = (
                min(sharpe / 2.0, 1.0) * 0.35
                + wr * 0.20
                + max(0, 1 - dd / 0.15) * 0.25
                + min(trades / 30, 1.0) * 0.20
            )
            return score

        best = max(valid, key=_composite_score)
        logger.info(
            "Best variant selected quantitatively: "
            "Sharpe=%.2f WR=%.1f%% DD=%.1f%% Trades=%d",
            best.get("sharpe_ratio", 0),
            best.get("win_rate", 0) * 100,
            abs(best.get("max_drawdown", 0)) * 100,
            best.get("total_trades", 0),
        )
        return {
            "params": best["params"],
            "metrics": {
                "sharpe_ratio": best.get("sharpe_ratio", 0),
                "win_rate": best.get("win_rate", 0),
                "max_drawdown": best.get("max_drawdown", 0),
                "total_trades": best.get("total_trades", 0),
            },
        }
