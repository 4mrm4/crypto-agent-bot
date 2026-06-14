"""AuditLog — structured, append-only JSONL log of every trade decision.

Records signal, risk_manager output, execution result, fill details, and PnL.
Never deleted. Used for strategy decay detection and performance analysis.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AuditEntry:
    """Single entry in the audit log."""
    signal_id: str
    pair: str
    side: str
    strategy_type: str
    strategy_name: str
    regime: str
    confidence: float
    position_size_usdt: float
    entry_price: float
    status: str                          # approved | executed | rejected | failed
    risk_verdict: str                    # from pre_trade_approval
    circuit_breaker_state: dict          # snapshot at time of trade
    correlation_result: dict             # from correlation check
    kelly_result: dict                   # from Kelly sizing
    exit_price: Optional[float] = None
    pnl_usdt: Optional[float] = None
    pnl_pct: Optional[float] = None
    slippage_pct: Optional[float] = None
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in asdict(self).items()}


class AuditLog:
    """Append-only JSONL audit log with SQLite mirror. One JSON object per line."""

    def __init__(self, save_path: str = "./workspace/audit_log.jsonl"):
        self._save_path = Path(save_path)
        self._save_path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: List[AuditEntry] = []
        self._load()
        # SQLite mirror
        from data.database import TradingDatabase
        self._db = TradingDatabase()
        logger.info("AuditLog loaded: %d entries", len(self._entries))

    def _load(self):
        if self._save_path.exists():
            with open(self._save_path) as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        self._entries.append(AuditEntry(**data))
                    except Exception:
                        pass

    def record(self, entry: AuditEntry):
        """Append an entry to the log (memory + disk + SQLite)."""
        self._entries.append(entry)
        with open(self._save_path, "a") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")
        # Mirror to SQLite
        try:
            self._db.insert_trade({
                "strategy_id": entry.strategy_name,
                "pair": entry.pair,
                "side": entry.side,
                "entry_price": entry.entry_price,
                "exit_price": entry.exit_price,
                "quantity": entry.position_size_usdt / entry.entry_price if entry.entry_price else 0,
                "pnl": entry.pnl_usdt,
                "pnl_pct": entry.pnl_pct,
                "fee_total": 0,
                "regime": entry.regime,
                "timestamp_open": int(entry.timestamp.timestamp()) if hasattr(entry.timestamp, 'timestamp') else 0,
                "metadata": {
                    "strategy_type": entry.strategy_type,
                    "confidence": entry.confidence,
                    "status": entry.status,
                    "risk_verdict": entry.risk_verdict,
                    "kelly_result": entry.kelly_result,
                },
            })
        except Exception as exc:
            logger.warning("Failed to mirror audit entry to SQLite: %s", exc)

    def query_by_strategy(self, strategy_name: str) -> List[AuditEntry]:
        """Return all entries for a given strategy name."""
        return [e for e in self._entries if e.strategy_name == strategy_name]

    def query_by_pair(self, pair: str) -> List[AuditEntry]:
        return [e for e in self._entries if e.pair == pair]

    def query_recent(self, limit: int = 100) -> List[AuditEntry]:
        return self._entries[-limit:]

    def compute_rolling_performance(self, strategy_name: str, days: int = 30) -> dict:
        """Compute rolling 30-day performance metrics for a strategy."""
        entries = self.query_by_strategy(strategy_name)
        # Filter to last N days
        cutoff = datetime.utcnow().timestamp() - days * 86400
        recent = [e for e in entries if hasattr(e, 'timestamp') and
                  isinstance(e.timestamp, datetime) and
                  e.timestamp.timestamp() > cutoff]

        if not recent:
            return {
                "strategy": strategy_name,
                "trade_count": 0,
                "total_pnl_pct": 0.0,
                "win_rate": 0.0,
                "sharpe": 0.0,
                "days": days,
                "vs_backtest_delta": 0.0,
            }

        filled = [e for e in recent if e.pnl_pct is not None]
        if not filled:
            return {
                "strategy": strategy_name,
                "trade_count": len(recent),
                "total_pnl_pct": 0.0,
                "win_rate": 0.0,
                "sharpe": 0.0,
                "days": days,
            }

        pnl_values = [e.pnl_pct for e in filled if e.pnl_pct is not None]
        wins = [p for p in pnl_values if p > 0]
        win_rate = len(wins) / len(pnl_values) if pnl_values else 0.0
        total_pnl = sum(pnl_values)
        sharpe = (float(sum(pnl_values)) / max(len(pnl_values), 1)) / max(
            (float(sum((p - sum(pnl_values)/len(pnl_values))**2 for p in pnl_values)) / len(pnl_values))**0.5 if len(pnl_values) > 1 else 0.01, 0.01
        ) if pnl_values else 0.0

        return {
            "strategy": strategy_name,
            "trade_count": len(filled),
            "total_trades": len(recent),
            "total_pnl_pct": round(total_pnl, 4),
            "win_rate": round(win_rate, 4),
            "sharpe": round(sharpe, 4),
            "avg_pnl": round(float(sum(pnl_values)) / len(pnl_values), 4) if pnl_values else 0.0,
            "days": days,
        }

    def get_all_strategies(self) -> List[str]:
        return list(set(e.strategy_name for e in self._entries))

    def count(self) -> int:
        return len(self._entries)

    def to_dict_list(self) -> List[dict]:
        return [e.to_dict() for e in self._entries]
