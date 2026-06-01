"""
StrategyManager — continuously evaluates all deployed strategies.
Retires decaying ones. Triggers re-research for their regime.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DECAY_THRESHOLD = 0.80   # live_sharpe / backtest_sharpe < 0.8 → flag
RETIRE_THRESHOLD = 0.50  # live_sharpe / backtest_sharpe < 0.5 → auto-retire


class DecayReport:
    """Report of a single strategy's decay assessment."""
    def __init__(self, strategy_id: str, strategy_type: str, regime: str,
                 backtest_sharpe: float, live_sharpe: float,
                 decay_score: float, action: str, reason: str):
        self.strategy_id = strategy_id
        self.strategy_type = strategy_type
        self.regime = regime
        self.backtest_sharpe = backtest_sharpe
        self.live_sharpe = live_sharpe
        self.decay_score = decay_score
        self.action = action  # "healthy" | "decaying" | "retired"
        self.reason = reason

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


class StrategyManager:
    """
    Evaluates all deployed strategies for decay.
    Retires strategies that have degraded significantly.
    """

    def __init__(self, vector_store=None, experiment_tracker=None, event_bus=None):
        self._vector_store = vector_store
        self._experiment_tracker = experiment_tracker
        self._event_bus = event_bus
        self._deployed: List[Dict[str, Any]] = []

    def load_deployed(self):
        """Load all tagged 'deployable' strategies from ChromaDB."""
        if not self._vector_store:
            logger.warning("No vector_store — cannot load deployed strategies")
            return
        try:
            strategies = self._vector_store.get_best_strategies(min_sharpe=0.0, k=50)
            self._deployed = []
            for s in strategies:
                meta = s.get("metadata", {}) or {}
                if meta.get("deployable", False):
                    self._deployed.append({
                        "id": s.get("id", ""),
                        "strategy_type": meta.get("strategy_type", "unknown"),
                        "regime": meta.get("regime", "unknown"),
                        "backtest_sharpe": float(meta.get("sharpe", 0)),
                        "backtest_win_rate": float(meta.get("win_rate", 0)),
                        "max_drawdown": float(meta.get("max_drawdown", 0)),
                        "metadata": meta,
                    })
            logger.info("Loaded %d deployed strategies", len(self._deployed))
        except Exception as exc:
            logger.warning("Failed to load deployed strategies: %s", exc)

    async def evaluate_all_deployed(self) -> List[DecayReport]:
        """
        Evaluate every deployed strategy for performance decay.
        Returns list of DecayReport sorted by severity.
        """
        self.load_deployed()
        reports = []

        for strategy in self._deployed:
            report = await self._evaluate_single(strategy)
            if report:
                reports.append(report)

        # Sort by decay score (most decayed first)
        reports.sort(key=lambda r: r.decay_score)

        return reports

    async def _evaluate_single(self, strategy: dict) -> Optional[DecayReport]:
        """Evaluate a single strategy for decay."""
        strategy_type = strategy.get("strategy_type", "unknown")
        regime = strategy.get("regime", "unknown")
        strategy_id = strategy.get("id", "")
        backtest_sharpe = strategy.get("backtest_sharpe", 0)

        if backtest_sharpe <= 0:
            return None

        # Get live performance from AuditLog (if available)
        live_sharpe = await self._get_live_sharpe(strategy_type)
        if live_sharpe is None:
            live_sharpe = backtest_sharpe  # no data yet = assume healthy
            return None

        decay_score = live_sharpe / backtest_sharpe if backtest_sharpe > 0 else 1.0

        if decay_score >= DECAY_THRESHOLD:
            return DecayReport(
                strategy_id=strategy_id,
                strategy_type=strategy_type,
                regime=regime,
                backtest_sharpe=backtest_sharpe,
                live_sharpe=live_sharpe,
                decay_score=decay_score,
                action="healthy",
                reason=f"Healthy: live={live_sharpe:.2f} vs backtest={backtest_sharpe:.2f}",
            )

        if decay_score < RETIRE_THRESHOLD:
            # Auto-retire
            await self.retire_strategy(strategy_id, f"Decayed: live={live_sharpe:.2f} vs backtest={backtest_sharpe:.2f}")
            return DecayReport(
                strategy_id=strategy_id,
                strategy_type=strategy_type,
                regime=regime,
                backtest_sharpe=backtest_sharpe,
                live_sharpe=live_sharpe,
                decay_score=decay_score,
                action="retired",
                reason=f"Auto-retired: decay score={decay_score:.2%}",
            )

        # Flagged decay
        await self._emit("strategy_decay_detected", {
            "strategy_type": strategy_type,
            "regime": regime,
            "decay_pct": round((1 - decay_score) * 100, 1),
            "backtest_sharpe": round(backtest_sharpe, 2),
            "live_sharpe": round(live_sharpe, 2),
        })
        return DecayReport(
            strategy_id=strategy_id,
            strategy_type=strategy_type,
            regime=regime,
            backtest_sharpe=backtest_sharpe,
            live_sharpe=live_sharpe,
            decay_score=decay_score,
            action="decaying",
            reason=f"Decaying: live={live_sharpe:.2f} ({(1-decay_score)*100:.0f}% drop)",
        )

    async def _get_live_sharpe(self, strategy_type: str) -> Optional[float]:
        """Get live Sharpe ratio for a strategy from vector store metadata."""
        if not self._vector_store:
            return None
        try:
            results = self._vector_store.query_similar(
                f"live_performance_{strategy_type}", k=1
            )
            if results:
                meta = results[0].get("metadata", {})
                return float(meta.get("live_sharpe", 0)) if meta.get("live_sharpe") else None
        except Exception:
            pass
        return None

    async def retire_strategy(self, strategy_id: str, reason: str):
        """Tag a strategy as retired in ChromaDB."""
        if not self._vector_store:
            return
        try:
            self._vector_store.store_insight(
                text=f"RETIRED: {strategy_id} — {reason}",
                metadata={
                    "strategy_id": strategy_id,
                    "status": "retired",
                    "retired_at": datetime.utcnow().isoformat(),
                    "reason": reason,
                },
            )
            logger.warning("Strategy %s retired: %s", strategy_id, reason)
            await self._emit("strategy_retired", {
                "strategy_id": strategy_id,
                "reason": reason,
            })
        except Exception as exc:
            logger.warning("Failed to retire strategy %s: %s", strategy_id, exc)

    async def promote_strategy(self, strategy_id: str, metadata: dict):
        """Tag a strategy as deployable in ChromaDB."""
        if not self._vector_store:
            return
        try:
            meta = dict(metadata)
            meta["deployable"] = True
            meta["promoted_at"] = datetime.utcnow().isoformat()
            self._vector_store.store_insight(
                text=f"PROMOTED: {strategy_id} — approved for deployment",
                metadata=meta,
            )
            logger.info("Strategy %s promoted to deployable", strategy_id)
        except Exception as exc:
            logger.warning("Failed to promote strategy %s: %s", strategy_id, exc)

    async def _emit(self, event_type: str, payload: dict):
        if self._event_bus:
            try:
                await self._event_bus.publish(event_type, payload)
            except Exception:
                pass

    def get_deployed_count(self) -> int:
        return len(self._deployed)
