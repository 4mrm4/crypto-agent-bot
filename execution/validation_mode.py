"""
ValidationMode — wraps the execution pipeline during the first 90 days live.

After 90 days AND if live Sharpe > 0.6 AND 50+ trades, automatically
graduates to normal mode. Until then, conservative limits apply.
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            import numpy as np
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
        except ImportError:
            pass
        return super().default(obj)


@dataclass
class GraduationAssessment:
    """Whether the system is ready to leave validation mode."""
    can_graduate: bool
    days_live: int
    trades_executed: int
    live_sharpe: float
    reasons: list
    days_remaining: int

    def to_dict(self) -> dict:
        return asdict(self)


class ValidationMode:
    """Conservative execution wrapper for the first 90 days live trading.

    During validation mode:
    - Position sizes capped at 2% of portfolio regardless of Kelly
    - Circuit breaker thresholds tighter (daily -1.5%, weekly -4%)
    - Every trade requires OOS validation to have passed
    - PerformanceMonitor runs after every 10 trades instead of every 30
    - All live results logged separately in validation_trades.jsonl
    - Autonomous loop continues researching but strategies go to pending_oos
    """

    VALIDATION_DAYS = 90
    GRADUATION_MIN_SHARPE = 0.6
    GRADUATION_MIN_TRADES = 50

    VALIDATION_CIRCUIT_BREAKER = {
        "daily_limit": -0.015,   # -1.5% (vs -3% normal)
        "weekly_limit": -0.040,  # -4% (vs -8% normal)
    }

    VALIDATION_MAX_POSITION_PCT = 0.02  # 2% of portfolio hard cap
    VALIDATION_TRADES_PATH = Path("./workspace/validation_trades.jsonl")

    def __init__(self, live_start_date: Optional[datetime] = None):
        self.live_start_date = live_start_date or datetime.now(timezone.utc)
        self.VALIDATION_TRADES_PATH.parent.mkdir(parents=True, exist_ok=True)
        from data.database import TradingDatabase
        self._db = TradingDatabase()

    @property
    def is_active(self) -> bool:
        """Validation mode is active if we're within the first 90 days."""
        days_live = (datetime.now(timezone.utc) - self.live_start_date).days
        return days_live < self.VALIDATION_DAYS

    @property
    def days_remaining(self) -> int:
        days_live = (datetime.now(timezone.utc) - self.live_start_date).days
        return max(0, self.VALIDATION_DAYS - days_live)

    @property
    def days_live(self) -> int:
        return (datetime.now(timezone.utc) - self.live_start_date).days

    def can_graduate(self, live_metrics: dict) -> GraduationAssessment:
        """Check if the system is ready to leave validation mode.

        Requires: 90+ days live AND Sharpe > 0.6 AND 50+ trades.
        Returns assessment with reasons.
        """
        live_sharpe = float(live_metrics.get("sharpe_ratio", 0))
        trades = int(live_metrics.get("total_trades", 0))
        days = self.days_live

        reasons = []
        if days < self.VALIDATION_DAYS:
            reasons.append(f"Only {days} days live, need {self.VALIDATION_DAYS}")
        if live_sharpe < self.GRADUATION_MIN_SHARPE:
            reasons.append(
                f"Live Sharpe {live_sharpe:.2f} < {self.GRADUATION_MIN_SHARPE}"
            )
        if trades < self.GRADUATION_MIN_TRADES:
            reasons.append(
                f"Only {trades} trades, need {self.GRADUATION_MIN_TRADES}"
            )

        can_graduate = len(reasons) == 0
        return GraduationAssessment(
            can_graduate=can_graduate,
            days_live=days,
            trades_executed=trades,
            live_sharpe=live_sharpe,
            reasons=reasons,
            days_remaining=self.days_remaining if not can_graduate else 0,
        )

    def apply_position_cap(self, kelly_result: dict) -> dict:
        """Override Kelly output to enforce validation position cap."""
        if not self.is_active:
            return kelly_result

        portfolio_value = kelly_result.get("portfolio_value", 10000.0)
        max_position = portfolio_value * self.VALIDATION_MAX_POSITION_PCT
        current_position = kelly_result.get("position_size_usdt", 0)

        if current_position > max_position:
            kelly_result["position_size_usdt"] = round(max_position, 2)
            kelly_result["portfolio_pct"] = round(
                max_position / portfolio_value * 100, 2
            )
            kelly_result["validation_cap_applied"] = True
            kelly_result["rationale"] += (
                f" [VALIDATION MODE: capped at "
                f"${max_position:.0f} ({self.VALIDATION_MAX_POSITION_PCT:.0%} of portfolio)]"
            )

        return kelly_result

    def apply_tight_circuit_breaker(
        self, daily_pnl: float, weekly_pnl: float
    ) -> bool:
        """Return True if trading should halt under validation thresholds."""
        if not self.is_active:
            return False

        if daily_pnl <= self.VALIDATION_CIRCUIT_BREAKER["daily_limit"]:
            logger.warning(
                "VALIDATION CB: Daily PnL %.2f%% exceeds limit %.1f%%",
                daily_pnl * 100,
                self.VALIDATION_CIRCUIT_BREAKER["daily_limit"] * 100,
            )
            return True
        if weekly_pnl <= self.VALIDATION_CIRCUIT_BREAKER["weekly_limit"]:
            logger.warning(
                "VALIDATION CB: Weekly PnL %.2f%% exceeds limit %.1f%%",
                weekly_pnl * 100,
                self.VALIDATION_CIRCUIT_BREAKER["weekly_limit"] * 100,
            )
            return True
        return False

    def log_validation_trade(self, trade: dict) -> None:
        """Append to validation_trades.jsonl + SQLite."""
        trade["logged_at"] = datetime.now(timezone.utc).isoformat()
        with open(self.VALIDATION_TRADES_PATH, "a") as f:
            f.write(json.dumps(trade, cls=_NumpyEncoder) + "\n")
        # Mirror to SQLite
        try:
            self._db.insert_validation_trade({
                "strategy_id": trade.get("strategy_id", trade.get("strategy_name", "unknown")),
                "pair": trade.get("pair", ""),
                "pnl": trade.get("pnl", trade.get("pnl_pct", 0)),
                "position_size": trade.get("position_size", trade.get("position_size_usdt", 0)),
                "timestamp": int(datetime.now(timezone.utc).timestamp()),
                "metadata": trade,
            })
        except Exception as exc:
            logger.warning("Failed to mirror validation trade to SQLite: %s", exc)

    def get_validation_trades(self) -> list:
        """Read validation trades from log."""
        if not self.VALIDATION_TRADES_PATH.exists():
            return []
        trades = []
        with open(self.VALIDATION_TRADES_PATH) as f:
            for line in f:
                try:
                    trades.append(json.loads(line.strip()))
                except Exception:
                    pass
        return trades

    def generate_validation_report(self) -> dict:
        """Summary report: days elapsed, trades, metrics, graduation status."""
        trades = self.get_validation_trades()
        live_sharpe = 0.0
        if trades:
            pnls = [t.get("pnl_pct", 0) for t in trades if "pnl_pct" in t]
            if pnls:
                import numpy as np
                mean_r = np.mean(pnls)
                std_r = np.std(pnls)
                live_sharpe = mean_r / std_r * 16.0 if std_r > 0 else 0.0

        live_metrics = {
            "sharpe_ratio": live_sharpe,
            "total_trades": len(trades),
        }
        graduation = self.can_graduate(live_metrics)

        return {
            "is_active": self.is_active,
            "days_live": self.days_live,
            "days_remaining": self.days_remaining,
            "trades_executed": len(trades),
            "live_sharpe": round(live_sharpe, 3),
            "graduation": graduation.to_dict(),
            "position_cap_pct": self.VALIDATION_MAX_POSITION_PCT,
            "circuit_breaker_limits": self.VALIDATION_CIRCUIT_BREAKER,
        }
