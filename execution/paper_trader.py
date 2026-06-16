"""PaperTrader — simulates live trading with a dummy balance using strategy signals."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from data.fetcher import MarketDataFetcher

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """A single simulated trade."""
    side: str  # "buy" or "sell"
    entry_time: datetime
    entry_price: float
    size: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    status: str = "open"  # open | closed


class PaperTrader:
    """Simulated trading environment using live price feeds."""

    def __init__(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1h",
        initial_balance: float = 10000.0,
        fetcher: Optional[MarketDataFetcher] = None,
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position: Optional[str] = None
        self.entry_price: Optional[float] = None
        self.position_size: float = 0.0
        self.trades: List[Trade] = []
        self.equity_curve: List[Dict[str, Any]] = []
        self._fetcher = fetcher or MarketDataFetcher()

    def run(self, signal_fn: Callable[[pd.DataFrame], str], max_candles: int = 100) -> Dict[str, Any]:
        logger.info("Paper trading %s | Balance: $%.2f", self.symbol, self.initial_balance)

        df = self._fetcher.fetch_ohlcv(self.symbol, self.timeframe, limit=max_candles)

        for i in range(1, len(df)):
            window = df.iloc[:i + 1]
            candle = df.iloc[i]
            signal = signal_fn(window)

            if signal == "buy" and self.position is None:
                self.position = "long"
                self.entry_price = candle["close"]
                self.position_size = self.balance * 0.95
                self.balance -= self.position_size
                self.trades.append(Trade(
                    side="buy", entry_time=candle.name,
                    entry_price=self.entry_price, size=self.position_size,
                ))

            elif signal == "sell" and self.position == "long":
                exit_price = candle["close"]
                proceeds = self.position_size * (exit_price / self.entry_price)
                pnl = proceeds - self.position_size
                self.balance += proceeds
                if self.trades and self.trades[-1].status == "open":
                    t = self.trades[-1]
                    t.exit_time = candle.name
                    t.exit_price = exit_price
                    t.pnl = pnl
                    t.pnl_pct = (exit_price / self.entry_price - 1)
                    t.status = "closed"
                self.position = None
                self.entry_price = None
                self.position_size = 0

            self.equity_curve.append({
                "timestamp": candle.name, "balance": self.balance, "signal": signal,
            })

        closed = [t for t in self.trades if t.status == "closed"]
        wins = [t for t in closed if t.pnl > 0]
        total_pnl = sum(t.pnl for t in closed)
        return {
            "symbol": self.symbol, "timeframe": self.timeframe,
            "initial_balance": self.initial_balance,
            "final_balance": self.balance + self.position_size,
            "total_pnl": total_pnl,
            "total_return_pct": (total_pnl / self.initial_balance) * 100,
            "num_trades": len(closed),
            "win_rate": round(len(wins) / len(closed), 3) if closed else 0,
            "trades": self.trades,
        }


def sma_crossover_signal(fast: int = 10, slow: int = 30) -> Callable[[pd.DataFrame], str]:
    """Return a signal function: 'buy' when fast SMA crosses above slow SMA."""
    def signal(df: pd.DataFrame) -> str:
        close = df["close"]
        fast_sma = close.rolling(fast).mean()
        slow_sma = close.rolling(slow).mean()
        if len(df) < slow + 2:
            return "hold"
        if fast_sma.iloc[-2] <= slow_sma.iloc[-2] and fast_sma.iloc[-1] > slow_sma.iloc[-1]:
            return "buy"
        elif fast_sma.iloc[-2] >= slow_sma.iloc[-2] and fast_sma.iloc[-1] < slow_sma.iloc[-1]:
            return "sell"
        return "hold"
    return signal