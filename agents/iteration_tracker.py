"""Iteration tracker agent -- records, retrieves, and manages strategy results."""

import json
import logging
from typing import Any, Dict, List, Optional

from langchain_core.tools import Tool

from agents.base import BaseAgent
from orchestration.evaluation import evaluate_strategy_quality

logger = logging.getLogger(__name__)

ITERATION_TRACKER_PROMPT = """You are an iteration tracking specialist. Your job is to
record, retrieve, and manage trading strategy results. You don't design or
backtest strategies -- you keep the record.

Available tools:
- get_best_strategy: Get the current best strategy from iteration history
- get_iteration_history: View all past attempts with keep/discard verdicts
- store_strategy_result: Persist a strategy result to memory
- store_strategy_insight: Save a strategic observation

IMPORTANT: Use ONLY plain ASCII text. No emoji, no Unicode symbols."""

class IterationRecord:
    """Tracks one backtest attempt in the optimization loop."""

    def __init__(self, params: Dict[str, Any], metrics: Dict[str, Any]):
        self.params = dict(params)
        self.metrics = dict(metrics)
        self.verdict: str = "unknown"
        self.reason: str = ""

    def evaluate(self) -> "IterationRecord":
        """Assign keep/discard verdict via shared evaluation module."""
        self.verdict, self.reason = evaluate_strategy_quality(self.metrics)
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "params": self.params,
            "metrics": self.metrics,
            "verdict": self.verdict,
            "reason": self.reason,
        }


class IterationTrackerAgent(BaseAgent):
    """Specialised agent that records, retrieves, and manages strategy results."""

    def __init__(self):
        # Track iteration history (keep/discard records)
        self._iteration_history: List[IterationRecord] = []
        # Track the best strategy found
        self._best_strategy: Optional[Dict[str, Any]] = None
        self._best_params: Optional[Dict[str, Any]] = None
        # Current market context (set by Hermes before task dispatch)
        self._current_regime: str = ""
        self._current_sentiment: float = 0.0
        tools = self._build_tools()
        super().__init__(
            name="iteration_tracker",
            tools=tools,
            system_prompt=ITERATION_TRACKER_PROMPT,
        )

    # ------------------------------------------------------------------
    # Public helpers (not tools -- called by Hermes after strategist runs)
    # ------------------------------------------------------------------

    def add_record(self, record: IterationRecord):
        """Add an iteration record (called by Hermes after strategist run)."""
        self._iteration_history.append(record)
        self._update_best(record)

    def _update_best(self, record: IterationRecord):
        """Update best strategy if this record is better."""
        sharpe = record.metrics.get("sharpe_ratio", 0)
        current_best = self._best_params.get("sharpe_ratio", -999) if self._best_params else -999
        if sharpe > current_best:
            self._best_strategy = record.params
            self._best_params = {"sharpe_ratio": sharpe, **record.metrics}

    # ------------------------------------------------------------------
    # Tool builders
    # ------------------------------------------------------------------

    def _build_tools(self):
        # ------------------------------------------------------------------
        # Tool: get_best_strategy
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
        # Tool: get_iteration_history
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
                    lines.append(f"  [{tag}] type={st} [{param_summary}] -- {r.metrics['error']}")
                else:
                    lines.append(f"  [{tag}] type={st} [{param_summary}] "
                                 f"Sharpe={sharpe} WR={wr} trades={trades} -- {r.reason}")
            return "\n".join(lines)

        # ------------------------------------------------------------------
        # Tool: store_strategy_result
        # ------------------------------------------------------------------

        def store_strategy_result(params_json: str = "{}") -> str:
            """Persist a completed backtest result to ChromaDB memory.
            Pass JSON with: strategy_type, params, metrics, regime, sentiment, timerange.
            Returns confirmation message."""
            try:
                data = json.loads(params_json)
            except json.JSONDecodeError:
                return "Error: pass valid JSON with strategy_type, params, and metrics."

            strategy_type = data.get("strategy_type", "")
            params = data.get("params", {})
            metrics = data.get("metrics", {})
            regime = data.get("regime", self._current_regime)
            sentiment = data.get("sentiment_score", self._current_sentiment)
            timerange = data.get("timerange", "")

            if not strategy_type or not metrics:
                return "Error: 'strategy_type' and 'metrics' are required."

            try:
                from memory.vector_store import VectorStore
                vs = VectorStore()
                vs.store_strategy_result(
                    strategy_type=strategy_type,
                    params=params,
                    metrics=metrics,
                    regime=regime,
                    sentiment_score=sentiment,
                    timerange=timerange,
                )
                return f"Strategy result stored: type={strategy_type}, sharpe={metrics.get('sharpe_ratio', '?')}"
            except Exception as exc:
                logger.exception("Failed to store strategy result")
                return f"Error storing strategy result: {exc}"

        # ------------------------------------------------------------------
        # Tool: store_strategy_insight
        # ------------------------------------------------------------------

        def store_strategy_insight(params_json: str = "{}") -> str:
            """Save a strategic observation to ChromaDB memory.
            Pass JSON with: text (required), and optional metadata dict.
            Returns confirmation message."""
            try:
                data = json.loads(params_json)
            except json.JSONDecodeError:
                return "Error: pass valid JSON with 'text' field."

            text = data.get("text", "")
            if not text:
                return "Error: 'text' field is required."

            metadata = data.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}

            try:
                from memory.vector_store import VectorStore
                vs = VectorStore()
                vs.store_insight(text=text, metadata=metadata)
                return f"Insight stored (first 80 chars): {text[:80]}"
            except Exception as exc:
                logger.exception("Failed to store insight")
                return f"Error storing insight: {exc}"

        return [
            Tool(name="get_best_strategy", func=get_best_strategy,
                 description="Get the best strategy found so far from iteration history."),
            Tool(name="get_iteration_history", func=get_iteration_history,
                 description="View all attempts, filtered by 'kept' or 'discarded'."),
            Tool(name="store_strategy_result", func=store_strategy_result,
                 description="Persist a completed backtest result to ChromaDB memory. "
                             "Args: JSON with strategy_type, params, metrics, regime, sentiment_score, timerange."),
            Tool(name="store_strategy_insight", func=store_strategy_insight,
                 description="Save a strategic observation to ChromaDB memory. "
                             "Args: JSON with text and optional metadata dict."),
        ]
