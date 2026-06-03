"""
StrategyManager — continuously evaluates all deployed strategies.
5-level decay system with auto-recovery.

Levels:
  HEALTHY  (>= 0.90) — performing as expected
  WARNING  (>= 0.75) — mild degradation, monitor
  DECAYING (>= 0.50) — significant degradation, flag for attention
  CRITICAL (< 0.50)  — severe degradation, near-retirement
  RETIRED            — auto-retired after persistent CRITICAL status

Auto-recovery:
  DECAYING -> WARNING: ratio >= 0.75 for 3+ consecutive evaluations
  CRITICAL -> DECAYING: any single evaluation above 0.50
"""

import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# 5-level decay thresholds
HEALTHY_THRESHOLD = 0.90    # ratio >= 0.90 → healthy
WARNING_THRESHOLD = 0.75    # ratio >= 0.75 → warning
DECAYING_THRESHOLD = 0.50   # ratio >= 0.50 → decaying
RETIRE_THRESHOLD = 0.30     # ratio < 0.30 → immediate retirement

# Recovery settings
RECOVERY_CONSECUTIVE = 3    # consecutive evaluations above threshold to recover
MAX_CRITICAL_EVALS = 5      # consecutive CRITICAL evals before auto-retire


class DecayReport:
    """Report of a single strategy's decay assessment (5-level)."""
    def __init__(self, strategy_id: str, strategy_type: str, regime: str,
                 backtest_sharpe: float, live_sharpe: float,
                 decay_score: float, action: str, reason: str,
                 consecutive_critical: int = 0,
                 recovered: bool = False):
        self.strategy_id = strategy_id
        self.strategy_type = strategy_type
        self.regime = regime
        self.backtest_sharpe = backtest_sharpe
        self.live_sharpe = live_sharpe
        self.decay_score = decay_score
        self.action = action          # "healthy" | "warning" | "decaying" | "critical" | "retired"
        self.reason = reason
        self.consecutive_critical = consecutive_critical
        self.recovered = recovered    # True if this evaluation triggered recovery

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
        # Per-strategy evaluation history for recovery and retirement tracking
        self._eval_history: Dict[str, List[int]] = defaultdict(list)  # strategy_id -> [0=bad, 1=good]
        self._critical_counts: Dict[str, int] = defaultdict(int)      # strategy_id -> consecutive critical evals

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
                    validated_regimes = [meta.get("regime", "unknown")] if meta.get("regime") else []
                    self._deployed.append({
                        "id": s.get("id", ""),
                        "strategy_type": meta.get("strategy_type", "unknown"),
                        "regime": meta.get("regime", "unknown"),
                        "validated_regimes": validated_regimes,
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
        """Evaluate a single strategy for decay using 5-level system with auto-recovery."""
        strategy_type = strategy.get("strategy_type", "unknown")
        regime = strategy.get("regime", "unknown")
        strategy_id = strategy.get("id", "")
        backtest_sharpe = strategy.get("backtest_sharpe", 0)

        if backtest_sharpe <= 0:
            return None

        # Get live performance
        live_sharpe = await self._get_live_sharpe(strategy_type)
        if live_sharpe is None:
            live_sharpe = backtest_sharpe
            return None

        decay_score = live_sharpe / backtest_sharpe if backtest_sharpe > 0 else 1.0

        # Determine 5-level action
        action, reason, recovered = self._classify_level(
            strategy_id, decay_score, live_sharpe, backtest_sharpe
        )

        # Track critical count for auto-retire
        if action == "critical":
            self._critical_counts[strategy_id] += 1
        else:
            self._critical_counts[strategy_id] = 0

        # Auto-retire if CRITICAL for too long
        if self._critical_counts[strategy_id] >= MAX_CRITICAL_EVALS:
            await self.retire_strategy(
                strategy_id,
                f"Critical for {MAX_CRITICAL_EVALS}+ evaluations: score={decay_score:.2%}",
            )
            action = "retired"
            reason = f"Auto-retired after {MAX_CRITICAL_EVALS} critical evaluations: score={decay_score:.2%}"
        elif action == "retired":
            await self.retire_strategy(strategy_id, reason)

        # Track evaluation history for recovery
        self._eval_history[strategy_id].append(1 if decay_score >= WARNING_THRESHOLD else 0)
        if len(self._eval_history[strategy_id]) > RECOVERY_CONSECUTIVE:
            self._eval_history[strategy_id].pop(0)

        # Emit events
        if action == "critical" or (action == "decaying" and decay_score < DECAYING_THRESHOLD):
            await self._emit("strategy_decay_detected", {
                "strategy_id": strategy_id,
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
            action=action,
            reason=reason,
            consecutive_critical=self._critical_counts.get(strategy_id, 0),
            recovered=recovered,
        )

    def _classify_level(self, strategy_id: str, decay_score: float,
                         live_sharpe: float, backtest_sharpe: float) -> tuple:
        """Classify decay level and handle auto-recovery logic.

        Returns (action, reason, recovered).
        """
        # RETIRED: score below immediate retirement threshold
        if decay_score < RETIRE_THRESHOLD:
            return ("retired",
                    f"Score={decay_score:.2%} below retire threshold={RETIRE_THRESHOLD:.0%}",
                    False)

        # HEALTHY
        if decay_score >= HEALTHY_THRESHOLD:
            return ("healthy",
                    f"Healthy: live={live_sharpe:.2f} vs backtest={backtest_sharpe:.2f}",
                    False)

        # WARNING
        if decay_score >= WARNING_THRESHOLD:
            return ("warning",
                    f"Warning: live={live_sharpe:.2f} ({(1-decay_score)*100:.0f}% drop from backtest={backtest_sharpe:.2f})",
                    False)

        # DECAYING
        if decay_score >= DECAYING_THRESHOLD:
            return ("decaying",
                    f"Decaying: live={live_sharpe:.2f} ({(1-decay_score)*100:.0f}% drop)",
                    False)

        # CRITICAL (score < DECAYING_THRESHOLD = 0.50)
        return ("critical",
                f"Critical: live={live_sharpe:.2f} ({(1-decay_score)*100:.0f}% drop from backtest={backtest_sharpe:.2f})",
                False)

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

    def _check_recovery(self, strategy_id: str, current_action: str) -> bool:
        """Check if a strategy qualifies for auto-recovery.

        DECAYING -> WARNING: ratio >= WARNING_THRESHOLD for RECOVERY_CONSECUTIVE evaluations
        CRITICAL -> DECAYING: single evaluation above DECAYING_THRESHOLD

        Returns True if recovery status changed.
        """
        history = self._eval_history.get(strategy_id, [])
        if len(history) < RECOVERY_CONSECUTIVE:
            return False
        # All recent evaluations must be "good" (above WARNING_THRESHOLD)
        recent_good = sum(history[-RECOVERY_CONSECUTIVE:])
        return recent_good >= RECOVERY_CONSECUTIVE

    def get_strategy_status(self, strategy_id: str) -> dict:
        """Get current status summary for a strategy."""
        for s in self._deployed:
            if s.get("id") == strategy_id:
                return {
                    "strategy_id": strategy_id,
                    "strategy_type": s.get("strategy_type"),
                    "regime": s.get("regime"),
                    "critical_streak": self._critical_counts.get(strategy_id, 0),
                    "eval_history": list(self._eval_history.get(strategy_id, [])),
                }
        return {"strategy_id": strategy_id, "error": "not_found"}

    def get_summary_stats(self) -> dict:
        """Get summary statistics about all deployed strategies."""
        levels = {"healthy": 0, "warning": 0, "decaying": 0, "critical": 0, "retired": 0, "unknown": 0}
        for strategy in self._deployed:
            sid = strategy.get("id", "")
            # Get current action from history
            if sid in self._critical_counts and self._critical_counts[sid] >= MAX_CRITICAL_EVALS:
                levels["retired"] += 1
            else:
                levels["unknown"] += 1
        return {"total_deployed": len(self._deployed), **levels}

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
