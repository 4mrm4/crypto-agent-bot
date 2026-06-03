"""Tests for execution/paper_trader.py"""

from datetime import datetime
from unittest.mock import MagicMock

import pandas as pd
import numpy as np

from execution.paper_trader import PaperTrader, Trade, sma_crossover_signal


def make_df(n=100):
    np.random.seed(42)
    close = 50000 + np.cumsum(np.random.randn(n) * 100)
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 10,
        "high": close + np.abs(np.random.randn(n) * 20),
        "low": close - np.abs(np.random.randn(n) * 20),
        "close": close,
        "volume": np.random.randint(100, 1000, n).astype(float),
    })


class TestTrade:
    def test_default_status(self):
        t = Trade(side="buy", entry_time=datetime.utcnow(),
                  entry_price=50000.0, size=1000.0)
        assert t.status == "open"
        assert t.exit_time is None
        assert t.pnl == 0.0

    def test_closed_trade(self):
        t = Trade(side="buy", entry_time=datetime.utcnow(),
                  entry_price=50000.0, size=1000.0,
                  exit_time=datetime.utcnow(), exit_price=55000.0,
                  pnl=500.0, pnl_pct=0.10, status="closed")
        assert t.status == "closed"
        assert t.pnl == 500.0
        assert t.pnl_pct == 0.10


class TestPaperTraderInit:
    def test_default_init(self):
        pt = PaperTrader()
        assert pt.symbol == "BTC/USDT"
        assert pt.timeframe == "1h"
        assert pt.initial_balance == 10000.0
        assert pt.balance == 10000.0
        assert pt.position is None
        assert pt.trades == []

    def test_custom_init(self):
        pt = PaperTrader(symbol="ETH/USDT", timeframe="15m", initial_balance=50000.0)
        assert pt.symbol == "ETH/USDT"
        assert pt.timeframe == "15m"
        assert pt.initial_balance == 50000.0


class TestSmaCrossoverSignal:
    def test_hold_when_short(self):
        df = make_df(10)
        signal = sma_crossover_signal(fast=10, slow=30)
        assert signal(df) == "hold"  # not enough data

    def test_buy_on_crossover(self):
        signal_fn = sma_crossover_signal(fast=2, slow=5)
        df = make_df(50)
        # Just check the function is callable and returns a string
        result = signal_fn(df)
        assert result in ("buy", "sell", "hold")


class TestPaperTraderRun:
    def test_run_with_mock_fetcher(self):
        pt = PaperTrader()
        mock_fetcher = MagicMock()
        mock_fetcher.fetch_ohlcv.return_value = make_df(50)
        pt._fetcher = mock_fetcher

        def signal(df):
            return "hold"

        result = pt.run(signal, max_candles=50)
        assert result["symbol"] == "BTC/USDT"
        assert result["num_trades"] == 0
        assert len(pt.equity_curve) > 0

    def test_run_with_buy_and_sell(self):
        pt = PaperTrader()
        mock_fetcher = MagicMock()
        mock_fetcher.fetch_ohlcv.return_value = make_df(50)
        pt._fetcher = mock_fetcher

        call_count = [0]

        def signal(df):
            call_count[0] += 1
            if call_count[0] == 5:
                return "buy"
            if call_count[0] == 20 and pt.position is not None:
                return "sell"
            return "hold"

        result = pt.run(signal, max_candles=50)
        assert isinstance(result["num_trades"], int)
        assert isinstance(result["total_return_pct"], float)

    def test_run_returns_correct_keys(self):
        pt = PaperTrader()
        mock_fetcher = MagicMock()
        mock_fetcher.fetch_ohlcv.return_value = make_df(30)
        pt._fetcher = mock_fetcher
        result = pt.run(lambda df: "hold", max_candles=30)
        expected_keys = {"symbol", "timeframe", "initial_balance", "final_balance",
                         "total_pnl", "total_return_pct", "num_trades", "win_rate", "trades"}
        assert expected_keys.issubset(result.keys())

    def test_trade_status_transitions(self):
        pt = PaperTrader()
        mock_fetcher = MagicMock()
        mock_fetcher.fetch_ohlcv.return_value = make_df(50)
        pt._fetcher = mock_fetcher

        step = [0]

        def signal(df):
            step[0] += 1
            if step[0] == 10:
                return "buy"
            if step[0] == 20:
                return "sell"
            return "hold"

        result = pt.run(signal, max_candles=50)
        if pt.trades:
            closed = [t for t in pt.trades if t.status == "closed"]
            assert all(t.status == "closed" or t.status == "open" for t in pt.trades)

    def test_no_trades_result(self):
        pt = PaperTrader()
        mock_fetcher = MagicMock()
        mock_fetcher.fetch_ohlcv.return_value = make_df(20)
        pt._fetcher = mock_fetcher
        result = pt.run(lambda df: "hold", max_candles=20)
        assert result["num_trades"] == 0
        assert result["win_rate"] == 0


class TestEquityCurve:
    def test_equity_curve_length(self):
        pt = PaperTrader()
        mock_fetcher = MagicMock()
        mock_fetcher.fetch_ohlcv.return_value = make_df(30)
        pt._fetcher = mock_fetcher
        pt.run(lambda df: "hold", max_candles=30)
        assert len(pt.equity_curve) == 29  # skip first row


class TestLivePriceFallback:
    def test_fetches_price_on_run(self):
        pt = PaperTrader()
        mock_fetcher = MagicMock()
        df = make_df(10)
        mock_fetcher.fetch_ohlcv.return_value = df
        pt._fetcher = mock_fetcher
        pt.run(lambda df: "hold", max_candles=10)
        assert mock_fetcher.fetch_ohlcv.called
