"""Transaction cost model for backtest fidelity.

Provides realistic fee and slippage assumptions so backtests reflect
execution costs rather than idealised P&L.
"""

import logging
from dataclasses import dataclass, asdict
from typing import Literal

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class TransactionCostModel:
    """Realistic transaction cost assumptions for backtest fidelity.

    Defaults reflect Binance spot tier-0 (30-day volume < 1M BTC):
      - maker_fee:   0.10%   (limit order, adds liquidity)
      - taker_fee:   0.075%  (market order, removes liquidity) — note: Binance spot
                             taker is typically higher than maker; this 0.075 is a
                             blended estimate for the bot's typical execution mix
      - slippage_pct: 0.05%  (fixed estimate — scales with order size / volume)
      - slippage_model: "fixed" (future: "volume_scaled")
    """
    maker_fee: float = 0.001       # 0.10%
    taker_fee: float = 0.00075     # 0.075%
    slippage_pct: float = 0.0005   # 0.05%
    slippage_model: Literal["fixed", "volume_scaled"] = "fixed"

    def total_cost_per_trade(self) -> float:
        """Combined cost for a round-trip trade (entry + exit).

        Assumes entry at taker rate, exit at maker rate (typical for signal-based bots
        that need immediate entry but can set limit exits).
        """
        return self.taker_fee + self.maker_fee + self.slippage_pct * 2

    def annual_cost_drag(self, trades_per_year: int) -> float:
        """Estimate of total cost drag as a fraction of notional per year."""
        return trades_per_year * self.total_cost_per_trade()

    def to_freqtrade_fee(self) -> float:
        """Return a single fee rate for Freqtrade's ``--fee`` flag.

        Freqtrade applies this as a flat per-trade fee (both entry and exit).
        We use the blended rate (max of maker/taker + slippage) as a conservative
        simplification.
        """
        return max(self.maker_fee, self.taker_fee) + self.slippage_pct

    @classmethod
    def from_settings(cls) -> "TransactionCostModel":
        """Construct from config/settings."""
        return cls(
            maker_fee=settings.MAKER_FEE,
            taker_fee=settings.TAKER_FEE,
            slippage_pct=settings.SLIPPAGE_PCT,
            slippage_model=settings.SLIPPAGE_MODEL,  # type: ignore
        )

    def net_sharpe(self, gross_sharpe: float, avg_return_per_trade: float) -> float:
        """Estimate net Sharpe after cost drag.

        This is a linear approximation: costs reduce returns directly.
        For a strategy with consistent returns, net Sharpe scales roughly as:
          net = gross * (1 - cost_per_trade / avg_return_per_trade)
        """
        cost_per_trade = self.total_cost_per_trade()
        if avg_return_per_trade <= 0 or cost_per_trade <= 0:
            return gross_sharpe
        return gross_sharpe * max(0, 1 - cost_per_trade / avg_return_per_trade)
