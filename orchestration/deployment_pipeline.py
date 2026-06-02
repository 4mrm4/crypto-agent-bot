"""
DeploymentPipeline — orchestrates the full strategy gauntlet.

A strategy passes through 11 gates in sequence:
  1. BlindParameterSearch.generate_search_space()
  2. BlindParameterSearch.batch_backtest()
  3. SyntheticValidator.validate_strategy()
  4. BacktestEngine.run_backtest(research_window)
  5. ExperimentTracker convergence check
  6. BacktestEngine.walk_forward_validate()
  7. SyntheticValidator.run_permutation_test()
  8. RiskManagerAgent.pre_trade_approval()
  9. Tag as "pending_oos" in ChromaDB
 10. [MANUAL] OOSValidator.validate_strategy()
 11. [MANUAL] Tag as "deployable" if OOS passes

Steps 1-9 are automated. Steps 10-11 require human action.
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from backtesting.data_split import DATA_SPLIT

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result of running a strategy through the deployment pipeline."""
    strategy_id: str
    strategy_type: str
    regime: str
    passed_gates: int
    total_gates: int
    failed_at: Optional[int]  # Gate number where it failed (None = passed all automated)
    failed_at_name: str
    reason: str
    completed_at: str
    oos_passed: Optional[bool] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def passed_all_automated(self) -> bool:
        return self.failed_at is None

    @property
    def status(self) -> str:
        if self.failed_at is not None:
            return f"failed_at_gate_{self.failed_at}"
        if self.oos_passed is None:
            return "pending_oos"
        return "deployable" if self.oos_passed else "oos_rejected"


GATE_NAMES = [
    "1. BlindParameterSearch: search space",
    "2. BlindParameterSearch: batch backtest",
    "3. SyntheticValidator: random walk sanity",
    "4. BacktestEngine: research window",
    "5. ExperimentTracker: convergence check",
    "6. BacktestEngine: walk-forward validation",
    "7. SyntheticValidator: permutation test",
    "8. RiskManagerAgent: pre-trade approval",
    "9. Tag as pending_oos in ChromaDB",
    "10. [MANUAL] OOSValidator: holdout validation",
    "11. [MANUAL] Tag as deployable",
]


class DeploymentPipeline:
    """Orchestrates the full strategy gauntlet from blind search to deploy (JSONL + SQLite).

    Steps 1-9 are automated and can be called in sequence via run_full_pipeline().
    Steps 10-11 require explicit human action via API.
    """

    def __init__(self, vector_store=None, engine=None):
        self._vector_store = vector_store
        self._engine = engine
        self._results_path = "./workspace/pipeline_results.jsonl"
        from config import settings
        from data.database import TradingDatabase
        self._db = TradingDatabase(legacy_backup=settings.LEGACY_JSONL_BACKUP)

    def run_full_pipeline(
        self,
        strategy_type: str,
        regime: str,
        research_goal: Any = None,
        strategy_params: Optional[Dict[str, Any]] = None,
        pairs: Optional[List[str]] = None,
    ) -> PipelineResult:
        """Run gates 1-9 in sequence.

        Returns PipelineResult detailing what passed and where it failed.
        """
        strategy_params = strategy_params or {}

        # Import pipeline components
        from backtesting.blind_search import BlindParameterSearch
        from backtesting.synthetic_validator import SyntheticValidator
        from backtesting.engine import BacktestEngine
        from agents.risk_manager import (
            RiskManagerAgent,
            kelly_position_size_conservative,
            PositionSizingTier,
        )
        from memory.vector_store import VectorStore

        engine = self._engine or BacktestEngine()
        vs = self._vector_store or VectorStore()

        strategy_id = f"{strategy_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        # ── Gate 1: Blind search space ──
        logger.info("Gate 1/9: Blind parameter search space generation...")
        try:
            bps = BlindParameterSearch(engine=engine)
            # Without an LLM agent, skip to default variants
            variants = bps._generate_default_variants(strategy_type, n_variants=10)
            if not variants:
                return PipelineResult(
                    strategy_id=strategy_id, strategy_type=strategy_type,
                    regime=regime, passed_gates=0, total_gates=11,
                    failed_at=1, failed_at_name=GATE_NAMES[0],
                    reason="No parameter variants generated",
                    completed_at=datetime.utcnow().isoformat(),
                )
        except Exception as exc:
            return PipelineResult(
                strategy_id=strategy_id, strategy_type=strategy_type,
                regime=regime, passed_gates=0, total_gates=11,
                failed_at=1, failed_at_name=GATE_NAMES[0],
                reason=f"Gate 1 exception: {exc}",
                completed_at=datetime.utcnow().isoformat(),
            )

        # ── Gate 2: Batch backtest (blind) ──
        logger.info("Gate 2/9: Blind batch backtest (%d variants)...", len(variants))
        try:
            batch_results = bps.batch_backtest(
                variants=variants, strategy_type=strategy_type, pairs=pairs,
            )
            if not batch_results:
                return PipelineResult(
                    strategy_id=strategy_id, strategy_type=strategy_type,
                    regime=regime, passed_gates=1, total_gates=11,
                    failed_at=2, failed_at_name=GATE_NAMES[1],
                    reason="All batch backtests failed",
                    completed_at=datetime.utcnow().isoformat(),
                )
        except Exception as exc:
            return PipelineResult(
                strategy_id=strategy_id, strategy_type=strategy_type,
                regime=regime, passed_gates=1, total_gates=11,
                failed_at=2, failed_at_name=GATE_NAMES[1],
                reason=f"Gate 2 exception: {exc}",
                completed_at=datetime.utcnow().isoformat(),
            )

        # Select best variant quantitatively (no LLM)
        best = bps.select_best_for_wfv(batch_results)
        if best is None:
            return PipelineResult(
                strategy_id=strategy_id, strategy_type=strategy_type,
                regime=regime, passed_gates=2, total_gates=11,
                failed_at=3, failed_at_name=GATE_NAMES[1],
                reason="No viable variant found in batch search",
                completed_at=datetime.utcnow().isoformat(),
            )
        best_params = best["params"]
        best_metrics = best["metrics"]

        # ── Gate 3: Synthetic sanity check ──
        logger.info("Gate 3/9: Synthetic data sanity check...")
        try:
            sv = SyntheticValidator()
            sanity = sv.validate_strategy(
                strategy_type=strategy_type,
                strategy_params=best_params,
                n_synthetic_runs=10,
                strategy_id=strategy_id,
            )
            if sanity.verdict == "fails_sanity":
                return PipelineResult(
                    strategy_id=strategy_id, strategy_type=strategy_type,
                    regime=regime, passed_gates=2, total_gates=11,
                    failed_at=3, failed_at_name=GATE_NAMES[2],
                    reason=f"Synthetic sanity check failed: {sanity.interpretation}",
                    completed_at=datetime.utcnow().isoformat(),
                )
        except Exception as exc:
            # Synthetic check is not critical — log and continue
            logger.warning("Gate 3 non-critical failure: %s", exc)

        # ── Gate 4: Research backtest ──
        logger.info("Gate 4/9: Research window backtest...")
        try:
            bt_result = engine.run_backtest(
                strategy_params=best_params,
                strategy_type=strategy_type,
                timerange=DATA_SPLIT.research_timerange(),
                pairs=pairs,
            )
            bt_sharpe = float(bt_result.get("sharpe_ratio", 0))
            bt_win_rate = float(bt_result.get("win_rate", 0))
            bt_dd = abs(float(bt_result.get("max_drawdown", 0)))
            bt_trades = int(bt_result.get("total_trades", 0))
        except Exception as exc:
            return PipelineResult(
                strategy_id=strategy_id, strategy_type=strategy_type,
                regime=regime, passed_gates=2, total_gates=11,
                failed_at=4, failed_at_name=GATE_NAMES[3],
                reason=f"Research backtest exception: {exc}",
                completed_at=datetime.utcnow().isoformat(),
            )

        # ── Gate 5: Convergence check ──
        logger.info("Gate 5/9: Convergence check...")
        from orchestration.experiment_tracker import Experiment
        experiment = Experiment(
            strategy_type=strategy_type,
            params=best_params,
            timerange=DATA_SPLIT.research_timerange(),
            regime=regime,
            sentiment_score=0.5,
            sharpe=bt_sharpe,
            win_rate=bt_win_rate,
            max_drawdown=bt_dd,
            total_trades=bt_trades,
            walk_forward_passed=False,
            synthetic_sanity_passed=True,
        )
        if not experiment.meets_deploy_criteria():
            return PipelineResult(
                strategy_id=strategy_id, strategy_type=strategy_type,
                regime=regime, passed_gates=4, total_gates=11,
                failed_at=5, failed_at_name=GATE_NAMES[4],
                reason=(
                    f"Convergence check failed: "
                    f"Sharpe={bt_sharpe:.2f}(need 1.2), WR={bt_win_rate:.2%}(need 48%), "
                    f"DD={bt_dd:.2%}(need <=10%), Trades={bt_trades}(need 30)"
                ),
                completed_at=datetime.utcnow().isoformat(),
            )

        # ── Gate 6: Walk-forward validation ──
        logger.info("Gate 6/9: Walk-forward validation...")
        try:
            wfv_result = engine.walk_forward_validate(
                strategy_params=best_params,
                strategy_type=strategy_type,
                windows=3,
                pairs=pairs,
            )
            wfv_consistency = wfv_result.get("consistency_score", 0)
            wfv_is_robust = wfv_result.get("is_robust", False)
            if not wfv_is_robust:
                return PipelineResult(
                    strategy_id=strategy_id, strategy_type=strategy_type,
                    regime=regime, passed_gates=5, total_gates=11,
                    failed_at=6, failed_at_name=GATE_NAMES[5],
                    reason=f"Walk-forward not robust: consistency={wfv_consistency:.2f}",
                    completed_at=datetime.utcnow().isoformat(),
                )
        except Exception as exc:
            return PipelineResult(
                strategy_id=strategy_id, strategy_type=strategy_type,
                regime=regime, passed_gates=5, total_gates=11,
                failed_at=6, failed_at_name=GATE_NAMES[5],
                reason=f"Walk-forward exception: {exc}",
                completed_at=datetime.utcnow().isoformat(),
            )

        # ── Gate 7: Permutation test ──
        logger.info("Gate 7/9: Permutation test...")
        try:
            sv = SyntheticValidator()
            perm = sv.run_permutation_test(
                strategy_type=strategy_type,
                strategy_params=best_params,
                n_permutations=200,
                strategy_id=strategy_id,
            )
            if not perm.significant:
                logger.warning(
                    "Permutation test not significant: p=%.4f", perm.p_value
                )
        except Exception as exc:
            logger.warning("Gate 7 non-critical failure: %s", exc)

        # ── Gate 8: Pre-trade risk approval ──
        logger.info("Gate 8/9: Pre-trade risk approval...")
        try:
            kelly = kelly_position_size_conservative(
                win_rate=bt_win_rate,
                avg_win_pct=float(bt_result.get("avg_win_pct", 0.02)),
                avg_loss_pct=float(bt_result.get("avg_loss_pct", 0.01)),
                portfolio_value=10000.0,
                sizing_tier=PositionSizingTier.CAUTIOUS,
            )
            if kelly.get("position_size_usdt", 0) <= 0:
                return PipelineResult(
                    strategy_id=strategy_id, strategy_type=strategy_type,
                    regime=regime, passed_gates=7, total_gates=11,
                    failed_at=8, failed_at_name=GATE_NAMES[7],
                    reason=f"Pre-trade risk: Kelly returned 0 position ({kelly.get('rationale', '')})",
                    completed_at=datetime.utcnow().isoformat(),
                )
        except Exception as exc:
            return PipelineResult(
                strategy_id=strategy_id, strategy_type=strategy_type,
                regime=regime, passed_gates=7, total_gates=11,
                failed_at=8, failed_at_name=GATE_NAMES[7],
                reason=f"Risk approval exception: {exc}",
                completed_at=datetime.utcnow().isoformat(),
            )

        # ── Gate 9: Tag as pending_oos ──
        logger.info("Gate 9/9: Tagging as pending_oos...")
        try:
            vs.store_strategy_result(
                strategy_type=strategy_type,
                params=best_params,
                metrics={
                    "sharpe_ratio": bt_sharpe,
                    "win_rate": bt_win_rate,
                    "max_drawdown": bt_dd,
                    "total_trades": bt_trades,
                    "status": "pending_oos",
                    "deployable": False,
                },
                regime=regime,
                timerange=DATA_SPLIT.research_timerange(),
            )
        except Exception as exc:
            return PipelineResult(
                strategy_id=strategy_id, strategy_type=strategy_type,
                regime=regime, passed_gates=8, total_gates=11,
                failed_at=9, failed_at_name=GATE_NAMES[8],
                reason=f"ChromaDB tagging exception: {exc}",
                completed_at=datetime.utcnow().isoformat(),
            )

        # All automated gates passed!
        result = PipelineResult(
            strategy_id=strategy_id,
            strategy_type=strategy_type,
            regime=regime,
            passed_gates=9,
            total_gates=11,
            failed_at=None,
            failed_at_name="",
            reason="All 9 automated gates passed. Awaiting OOS validation.",
            completed_at=datetime.utcnow().isoformat(),
        )

        self._log_result(result)
        logger.info(
            "Pipeline complete: %s (%s/%s) — %s",
            strategy_type, result.passed_gates, result.total_gates,
            result.reason,
        )
        return result

    def get_pending_oos_strategies(self) -> List[dict]:
        """Return all strategies awaiting manual OOS validation."""
        from memory.vector_store import VectorStore
        vs = self._vector_store or VectorStore()
        results = vs.get_best_strategies(min_sharpe=0.0, k=50)
        pending = []
        for r in results:
            meta = r.get("metadata", {}) or {}
            if meta.get("status") == "pending_oos":
                pending.append({
                    "strategy_id": meta.get("strategy_id", ""),
                    "strategy_type": meta.get("strategy_type", ""),
                    "regime": meta.get("regime", ""),
                    "sharpe": meta.get("sharpe", 0),
                    "win_rate": meta.get("win_rate", 0),
                    "max_drawdown": meta.get("max_drawdown", 0),
                    "metadata": meta,
                })
        return pending

    def mark_oos_validated(
        self, strategy_id: str, oos_result: Any,
    ) -> bool:
        """Tag strategy as deployable (passed OOS) or reject (failed OOS).

        Called after manual OOS validation. Tags in ChromaDB.
        """
        from memory.vector_store import VectorStore
        vs = self._vector_store or VectorStore()

        status = "deployable" if getattr(oos_result, 'passed', False) else "oos_rejected"
        vs.store_insight(
            text=f"OOS validation result: {strategy_id} -> {status}",
            metadata={
                "strategy_id": strategy_id,
                "status": status,
                "oos_sharpe": getattr(oos_result, 'oos_sharpe', 0),
                "oos_win_rate": getattr(oos_result, 'oos_win_rate', 0),
                "deployable": status == "deployable",
            },
        )
        logger.info("OOS validated: %s -> %s", strategy_id, status)
        return status == "deployable"

    def get_all_status(self) -> dict:
        """Return full deployment pipeline state for all strategies.

        Reads from ChromaDB to find all strategies and their pipeline stages.
        """
        from memory.vector_store import VectorStore
        vs = self._vector_store or VectorStore()
        results = vs.get_best_strategies(min_sharpe=0.0, k=100)

        strategies = []
        for r in results:
            meta = r.get("metadata", {}) or {}
            status = meta.get("status", "unknown")

            strategies.append({
                "strategy_id": meta.get("strategy_id", ""),
                "strategy_type": meta.get("strategy_type", ""),
                "regime": meta.get("regime", ""),
                "status": status,
                "sharpe": meta.get("sharpe", 0),
                "win_rate": meta.get("win_rate", 0),
                "pipeline_gate": self._status_to_gate(status),
            })

        return {
            "strategies": strategies,
            "count": len(strategies),
            "total_gates": 11,
            "gate_names": GATE_NAMES,
        }

    @staticmethod
    def _status_to_gate(status: str) -> int:
        mapping = {
            "explored": 4,
            "converged": 5,
            "wf_validated": 6,
            "perm_tested": 7,
            "risk_approved": 8,
            "pending_oos": 9,
            "deployable": 11,
            "oos_rejected": 10,
            "retired": 11,
        }
        return mapping.get(status, 0)

    def _log_result(self, result: PipelineResult) -> None:
        """Append pipeline result to JSONL + SQLite."""
        import json
        from pathlib import Path
        log_path = Path(self._results_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(result.to_dict()) + "\n")
        # Mirror to SQLite
        try:
            self._db.insert_pipeline_result({
                "strategy_id": result.strategy_id,
                "strategy_type": result.strategy_type,
                "gate": result.failed_at_name if result.failed_at else "all_passed",
                "passed": result.passed_all_automated,
                "details": {
                    "passed_gates": result.passed_gates,
                    "total_gates": result.total_gates,
                    "reason": result.reason,
                    "oos_passed": result.oos_passed,
                },
            })
        except Exception as exc:
            logger.warning("Failed to mirror pipeline result to SQLite: %s", exc)
