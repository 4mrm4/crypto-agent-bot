"""TradeSignal dataclass — structured trade proposal from strategy evaluation."""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional


@dataclass
class TradeSignal:
    """A complete trade signal with full provenance for the risk manager and executor."""
    pair: str
    side: str                            # "buy" | "sell"
    strategy_name: str
    strategy_type: str
    regime: str
    confidence: float                    # 0.0-1.0 from risk manager
    sharpe: float
    win_rate: float
    max_drawdown: float
    suggested_stoploss: float            # e.g. -0.04 for 4%
    suggested_take_profit: float         # e.g. 0.08 for 8%
    source_agent: str                    # "signal_scanner" | "strategist"
    price: float = 0.0                   # Current market price at signal time
    position_size_usdt: float = 0.0      # From Kelly sizing
    kelly_fraction: float = 0.0          # Fraction of Kelly used
    quality_score: float = 1.0           # ML model prediction, 0.0-1.0, default 1.0 (unfiltered)
    quality_multiplier: float = 1.0      # Applied to position size after Kelly sizing
    backtest_metrics: Optional[dict] = None  # Raw backtest metrics for ML scorer
    signal_id: str = ""                  # Auto-generated
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    executed_at: Optional[datetime] = None
    filled_price: Optional[float] = None
    status: str = "pending"              # pending | approved | executed | rejected | failed

    def to_dict(self) -> dict:
        return {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in asdict(self).items()}

    def to_freqtrade_config(self) -> dict:
        """Serialise to a Freqtrade-compatible config snippet."""
        return {
            "pair": self.pair,
            "side": self.side,
            "stake_amount": self.position_size_usdt,
            "stoploss": self.suggested_stoploss,
            "time_in_force": "GTC",
        }
