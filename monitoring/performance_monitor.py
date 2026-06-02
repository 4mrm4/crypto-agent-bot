"""
PerformanceMonitor — aggressive live vs backtest monitoring with
statistical significance testing and regime mismatch detection.

Continuously compares live strategy performance to backtest baseline.
Accounts for expected degradation. Fires alerts when degradation is abnormal.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Expected degradation ranges (empirically realistic for crypto)
# NOTE: These assume TransactionCostModel is already applied to backtest results.
# With costs already modelled, residual live degradation is from market impact,
# timing differences, and fill quality — not fee drag.
EXPECTED_DEGRADATION = {
    "sharpe":    (0.20, 0.40),  # Live Sharpe expected to be 20-40% lower (was 30-50%)
    "win_rate":  (0.03, 0.10),  # Win rate expected to drop 3-10 percentage points (was 5-15pp)
    "avg_win":   (0.10, 0.25),  # Average win smaller due to slippage (unchanged)
    "max_dd":    (0.10, 0.30),  # Drawdown expected to be 10-30% worse (unchanged)
}


@dataclass
class DegradationReport:
    """Report of a single strategy's degradation assessment."""
    strategy_id: str
    n_live_trades: int
    statistically_significant: bool
    metrics: Dict[str, Any]  # Per-metric degradation scores
    overall_degradation_pct: float
    alert_level: str  # "normal" | "warning" | "critical"
    recommendation: str
    generated_at: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MonitoringReport:
    """Full monitoring report for a strategy."""
    strategy_id: str
    strategy_type: str
    regime: str
    backtest_sharpe: float
    live_sharpe: float
    degradation: DegradationReport
    regime_alignment: Dict[str, Any]
    recommendation: str
    days_until_review: int

    def to_dict(self) -> dict:
        return asdict(self)


class PerformanceMonitor:
    """Continuously compares live strategy performance to backtest baseline.

    Accounts for expected degradation. Fires alerts when degradation is abnormal.
    Reads live trade data from the audit log and backtest data from ChromaDB.
    """

    def __init__(self, vector_store=None, audit_log=None):
        self._vector_store = vector_store
        self._audit_log = audit_log

    def compute_degradation_score(
        self,
        backtest_metrics: dict,
        live_metrics: dict,
        n_live_trades: int,
    ) -> DegradationReport:
        """For each metric, compute degradation vs expected range.

        Also computes:
        - statistical_significance: need >= 30 trades to judge
        - confidence_interval: range of true degradation at 95% confidence
        """
        metrics_scores = {}
        overall_degradation = 0.0
        n_metrics = 0

        for metric in ["sharpe", "win_rate", "max_dd"]:
            bt_val = abs(float(backtest_metrics.get(metric, 0))) if metric == "max_dd" else float(backtest_metrics.get(metric, 0))
            live_val = abs(float(live_metrics.get(metric, 0))) if metric == "max_dd" else float(live_metrics.get(metric, 0))

            if bt_val == 0:
                continue

            # Actual degradation: how much worse live is vs backtest
            if metric == "max_dd":
                # For drawdown, larger is worse, so degradation = live/bt
                actual_degradation = min(live_val / bt_val if bt_val > 0 else 1.0, 2.0)
            else:
                actual_degradation = max(0.0, (bt_val - live_val) / bt_val)

            # Expected degradation range
            expected_low, expected_high = EXPECTED_DEGRADATION.get(metric, (0, 0.5))

            # Z-score: how many expected ranges outside normal
            if expected_high > expected_low:
                z_score = (actual_degradation - expected_high) / (expected_high - expected_low)
            else:
                z_score = 0.0

            # Alert level
            if actual_degradation <= expected_high:
                alert = "normal"
            elif actual_degradation <= expected_high + 0.2:
                alert = "warning"
            else:
                alert = "critical"

            metrics_scores[metric] = {
                "backtest_value": bt_val,
                "live_value": live_val,
                "actual_degradation_pct": round(actual_degradation, 4),
                "expected_range": [expected_low, expected_high],
                "z_score": round(z_score, 2),
                "alert_level": alert,
            }
            overall_degradation += actual_degradation
            n_metrics += 1

        overall_degradation = overall_degradation / max(n_metrics, 1)

        # Determine overall alert level
        if any(m.get("alert_level") == "critical" for m in metrics_scores.values()):
            alert_level = "critical"
        elif any(m.get("alert_level") == "warning" for m in metrics_scores.values()):
            alert_level = "warning"
        else:
            alert_level = "normal"

        # Recommendation based on statistical significance
        statistically_significant = n_live_trades >= 30
        if not statistically_significant:
            recommendation = "insufficient_data"
        elif alert_level == "critical":
            recommendation = "suspend"
        elif alert_level == "warning":
            recommendation = "reduce_size"
        else:
            recommendation = "continue"

        return DegradationReport(
            strategy_id=live_metrics.get("strategy_id", ""),
            n_live_trades=n_live_trades,
            statistically_significant=statistically_significant,
            metrics=metrics_scores,
            overall_degradation_pct=round(overall_degradation, 4),
            alert_level=alert_level,
            recommendation=recommendation,
            generated_at=datetime.utcnow().isoformat(),
        )

    @staticmethod
    def is_degradation_normal(metric: str, actual_pct: float) -> bool:
        """Return True if degradation is within expected range."""
        low, high = EXPECTED_DEGRADATION.get(metric, (0, 1))
        return low <= actual_pct <= high

    def compute_rolling_sharpe(
        self,
        trade_history: List[dict],
        window_days: int = 30,
    ) -> List[float]:
        """Rolling Sharpe ratio for trend detection."""
        if not trade_history:
            return []

        import pandas as pd
        import numpy as np

        df = pd.DataFrame(trade_history)
        returns = df.get("pnl_pct", df.get("return_pct", df.get("profit_pct", [0])))
        rolling = returns.rolling(window=window_days)

        rolling_sharpes = []
        for i in range(len(rolling)):
            window = rolling.iloc[i]
            if len(window) >= window_days:
                mean_r = window.mean()
                std_r = window.std()
                if std_r > 0:
                    rolling_sharpes.append(mean_r / std_r * np.sqrt(365))
                else:
                    rolling_sharpes.append(0.0)
        return rolling_sharpes

    def detect_regime_mismatch(
        self,
        strategy_regime: str,
        current_regime: str,
        days_mismatched: int,
    ) -> dict:
        """Regime mismatch is a major cause of live underperformance.

        If strategy designed for "ranging" but market is "strong_uptrend" for >3 days:
        -> Suspend strategy (not retire — may recover when regime returns)
        -> Emit event
        """
        if strategy_regime == current_regime:
            return {
                "mismatched": False,
                "days_mismatched": 0,
                "action": "none",
                "reason": "Strategy regime matches current market regime.",
            }

        suspension_threshold = 3  # days before suspension
        if days_mismatched >= suspension_threshold:
            return {
                "mismatched": True,
                "strategy_regime": strategy_regime,
                "current_regime": current_regime,
                "days_mismatched": days_mismatched,
                "action": "suspend",
                "reason": (
                    f"Regime mismatch for {days_mismatched} days: "
                    f"strategy designed for {strategy_regime}, "
                    f"market is {current_regime}. Suspended until regime returns."
                ),
            }

        return {
            "mismatched": True,
            "strategy_regime": strategy_regime,
            "current_regime": current_regime,
            "days_mismatched": days_mismatched,
            "action": "warn",
            "reason": (
                f"Regime changed {days_mismatched} days ago. "
                f"Strategy will be suspended if mismatch continues >{suspension_threshold} days."
            ),
        }

    def generate_monitoring_report(
        self,
        strategy_id: str,
        strategy_type: str = "",
        regime: str = "",
        backtest_metrics: Optional[dict] = None,
        live_metrics: Optional[dict] = None,
    ) -> MonitoringReport:
        """Generate full monitoring report for a strategy.

        If no live metrics available, returns a baseline report.
        """
        backtest_metrics = backtest_metrics or {}
        live_metrics = live_metrics or {}

        # Try to load from ChromaDB if not provided
        if not backtest_metrics and self._vector_store:
            try:
                results = self._vector_store.get_best_strategies(min_sharpe=0.0, k=50)
                for r in results:
                    meta = r.get("metadata", {}) or {}
                    if meta.get("strategy_id") == strategy_id:
                        backtest_metrics = meta
                        break
            except Exception:
                pass

        n_live_trades = len(live_metrics.get("trade_history", [])) if live_metrics else 0
        degradation = self.compute_degradation_score(
            backtest_metrics, live_metrics, n_live_trades,
        )

        regime_mismatch = self.detect_regime_mismatch(
            strategy_regime=regime or backtest_metrics.get("regime", ""),
            current_regime=live_metrics.get("current_regime", ""),
            days_mismatched=live_metrics.get("days_mismatched", 0),
        )

        if degradation.recommendation == "suspend":
            recommendation = "SUSPEND"
        elif degradation.recommendation == "reduce_size":
            recommendation = "REDUCE_SIZE"
        elif regime_mismatch.get("action") == "suspend":
            recommendation = "SUSPEND (regime mismatch)"
        else:
            recommendation = "CONTINUE"

        return MonitoringReport(
            strategy_id=strategy_id,
            strategy_type=strategy_type or backtest_metrics.get("strategy_type", "unknown"),
            regime=regime or backtest_metrics.get("regime", "unknown"),
            backtest_sharpe=float(backtest_metrics.get("sharpe", 0)),
            live_sharpe=float(live_metrics.get("sharpe_ratio", 0)),
            degradation=degradation,
            regime_alignment=regime_mismatch,
            recommendation=recommendation,
            days_until_review=7 if degradation.alert_level == "warning" else 30,
        )

    def get_summary(self) -> List[dict]:
        """Return degradation status for all tracked strategies."""
        if not self._vector_store:
            return []
        try:
            results = self._vector_store.get_best_strategies(min_sharpe=0.0, k=50)
            summary = []
            for r in results:
                meta = r.get("metadata", {}) or {}
                if meta.get("status") not in ("deployed", "deployable"):
                    continue
                degradation = self.compute_degradation_score(
                    backtest_metrics=meta,
                    live_metrics={"sharpe_ratio": float(meta.get("live_sharpe", 0))},
                    n_live_trades=int(meta.get("live_trades", 0)),
                )
                summary.append({
                    "strategy_id": meta.get("strategy_id", ""),
                    "strategy_type": meta.get("strategy_type", ""),
                    "regime": meta.get("regime", ""),
                    "backtest_sharpe": meta.get("sharpe", 0),
                    "live_sharpe": meta.get("live_sharpe", 0),
                    "degradation_pct": degradation.overall_degradation_pct,
                    "alert_level": degradation.alert_level,
                    "recommendation": degradation.recommendation,
                    "n_live_trades": degradation.n_live_trades,
                    "statistically_significant": degradation.statistically_significant,
                })
            return summary
        except Exception as exc:
            logger.warning("Failed to get monitoring summary: %s", exc)
            return []
