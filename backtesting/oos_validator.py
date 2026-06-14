"""
OOSValidator — the ONLY module that touches holdout data.

RULES (must never be violated):
- Never called from AutonomousResearchLoop
- Never called from HermesOrchestrator
- Results never stored in ChromaDB (would contaminate future cycles)
- Results logged to a SEPARATE file: oos_results.jsonl
- Results never passed back to any LLM
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from backtesting.data_split import DATA_SPLIT
from backtesting.engine import BacktestEngine

logger = logging.getLogger(__name__)


@dataclass
class OOSResult:
    """Result of a single OOS validation run on holdout data."""
    strategy_id: str
    strategy_type: str
    research_sharpe: float
    oos_sharpe: float
    net_sharpe: float  # Sharpe after transaction cost drag
    research_win_rate: float
    oos_win_rate: float
    degradation_pct: float
    passed: bool
    recommendation: str  # "deploy" | "monitor_longer" | "reject"
    validated_at: str
    holdout_window: str
    oos_trades: int = 0
    oos_max_drawdown: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class OOSValidator:
    """Runs final validation on the holdout window.

    Only instantiate when the user explicitly requests OOS validation.
    Never called from autonomous pipelines.
    """

    OOS_RESULTS_PATH = Path("./workspace/oos_results.jsonl")

    def __init__(self, engine: Optional[BacktestEngine] = None):
        self._engine = engine or BacktestEngine()
        self.OOS_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        from config import settings
        from data.database import TradingDatabase
        self._db = TradingDatabase()

    def validate_strategy(
        self,
        strategy_type: str,
        strategy_params: Optional[Dict[str, Any]] = None,
        research_metrics: Optional[Dict[str, Any]] = None,
        strategy_id: str = "",
    ) -> OOSResult:
        """Run backtest on DATA_SPLIT.holdout_timerange() ONLY.

        Pass criteria (deliberately lower bar than research, but on unseen data):
          Sharpe >= 0.8
          Win Rate >= 0.42
          Max Drawdown <= 0.15
          Min Trades >= 10
        """
        strategy_params = strategy_params or {}
        research_metrics = research_metrics or {}

        # Run backtest on holdout data ONLY
        holdout_result = self._engine.run_backtest(
            strategy_params=strategy_params,
            strategy_type=strategy_type,
            timerange=DATA_SPLIT.holdout_timerange(),
        )

        oos_sharpe = float(holdout_result.get("sharpe_ratio", 0))
        oos_win_rate = float(holdout_result.get("win_rate", 0))
        oos_dd = abs(float(holdout_result.get("max_drawdown", 0)))
        oos_trades = int(holdout_result.get("total_trades", 0))

        research_sharpe = float(research_metrics.get("sharpe_ratio", 0))
        research_win_rate = float(research_metrics.get("win_rate", 0))

        # ── Net-of-costs metrics ──
        from backtesting.cost_model import TransactionCostModel
        cost_model = TransactionCostModel.from_settings()
        profit = float(holdout_result.get("profit_ratio", 0))
        avg_return = profit / max(oos_trades, 1)
        net_sharpe = cost_model.net_sharpe(oos_sharpe, avg_return)

        # Compute degradation
        degradation_pct = 0.0
        if research_sharpe > 0:
            degradation_pct = (research_sharpe - oos_sharpe) / research_sharpe
        elif research_sharpe <= 0:
            # Negative/flat research Sharpe means research data is unreliable;
            # mark inconclusive rather than silently passing with 0% degradation.
            degradation_pct = 1.0

        # Pass criteria on NET metrics (costs already baked in)
        passed = (
            net_sharpe >= 0.8
            and oos_win_rate >= 0.42
            and oos_dd <= 0.15
            and oos_trades >= 10
        )

        # Compute recommendation
        if passed and degradation_pct <= 0.30:
            recommendation = "deploy"
        elif passed and degradation_pct <= 0.50:
            recommendation = "monitor_longer"
        else:
            recommendation = "reject"

        # Reject if degradation > 50% regardless of absolute metrics
        if degradation_pct > 0.50:
            recommendation = "reject"
            passed = False

        result = OOSResult(
            strategy_id=strategy_id,
            strategy_type=strategy_type,
            research_sharpe=research_sharpe,
            oos_sharpe=oos_sharpe,
            net_sharpe=round(max(net_sharpe, 0), 2),
            research_win_rate=research_win_rate,
            oos_win_rate=oos_win_rate,
            degradation_pct=round(degradation_pct, 4),
            passed=passed,
            recommendation=recommendation,
            validated_at=datetime.utcnow().isoformat(),
            holdout_window=DATA_SPLIT.holdout_timerange(),
            oos_trades=oos_trades,
            oos_max_drawdown=round(oos_dd, 4),
        )

        # Log to separate file — NEVER ChromaDB
        self._log_result(result)

        logger.info(
            "OOS validation: %s | research Sharpe=%.2f -> OOS Sharpe=%.2f "
            "(degradation=%.1f%%) | recommendation=%s",
            strategy_type, research_sharpe, oos_sharpe,
            degradation_pct * 100, recommendation,
        )

        return result

    def batch_validate(
        self,
        strategies: List[Dict[str, Any]],
    ) -> List[OOSResult]:
        """Validate multiple strategies in batch.

        Each strategy dict must have:
          - strategy_type: str
          - strategy_params: dict (optional)
          - research_metrics: dict (optional)
          - strategy_id: str
        """
        results = []
        for s in strategies:
            try:
                result = self.validate_strategy(
                    strategy_type=s.get("strategy_type", "sma_crossover"),
                    strategy_params=s.get("strategy_params"),
                    research_metrics=s.get("research_metrics"),
                    strategy_id=s.get("strategy_id", ""),
                )
                results.append(result)
            except Exception as exc:
                logger.error("Batch OOS failed for %s: %s", s.get("strategy_id"), exc)
        return results

    @staticmethod
    def compute_degradation(research_metrics: dict, oos_metrics: dict) -> float:
        """Return degradation as a percentage.

        e.g. research Sharpe 1.5, OOS Sharpe 0.9 -> 40% degradation.
        Anything > 50% degradation -> reject regardless of absolute metrics.
        """
        research_sharpe = float(research_metrics.get("sharpe_ratio", 0))
        oos_sharpe = float(oos_metrics.get("sharpe_ratio", 0))
        if research_sharpe <= 0:
            return 0.0
        return (research_sharpe - oos_sharpe) / research_sharpe

    def _log_result(self, result: OOSResult) -> None:
        """Append to oos_results.jsonl + SQLite. NEVER writes to ChromaDB."""
        with open(self.OOS_RESULTS_PATH, "a") as f:
            f.write(json.dumps(result.to_dict()) + "\n")
        # Mirror to SQLite
        try:
            self._db.insert_oos_result({
                "strategy_id": result.strategy_id,
                "strategy_type": result.strategy_type,
                "sharpe": result.oos_sharpe,
                "net_sharpe": result.net_sharpe,
                "win_rate": result.oos_win_rate,
                "max_drawdown": result.oos_max_drawdown,
                "trade_count": result.oos_trades,
                "passed": result.passed,
                "recommendation": result.recommendation,
            })
        except Exception as exc:
            logger.warning("Failed to mirror OOS result to SQLite: %s", exc)

    def get_results(self) -> List[OOSResult]:
        """Read all OOS results from the log file (NOT ChromaDB)."""
        if not self.OOS_RESULTS_PATH.exists():
            return []
        results = []
        with open(self.OOS_RESULTS_PATH) as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    results.append(OOSResult(**data))
                except Exception:
                    pass
        return results

    def get_pending_validation(self) -> List[dict]:
        """Return strategies awaiting OOS validation from ChromaDB.

        This reads from ChromaDB only to FIND strategies, not to store results.
        """
        try:
            from memory.vector_store import VectorStore
            vs = VectorStore()
            # Find strategies tagged as 'pending_oos'
            results = vs.query_similar("pending_oos", k=50)
            pending = []
            for r in results:
                meta = r.get("metadata", {})
                if meta.get("status") == "pending_oos":
                    pending.append({
                        "strategy_type": meta.get("strategy_type", ""),
                        "strategy_id": meta.get("strategy_id", ""),
                        "params": {},
                    })
            return pending
        except Exception as exc:
            logger.warning("Could not query pending OOS strategies: %s", exc)
            return []
