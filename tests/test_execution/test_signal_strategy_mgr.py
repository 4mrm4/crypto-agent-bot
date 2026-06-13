"""Tests for signal scanner, strategy manager, and exchange modules."""

from execution.signal_scanner import SignalScanner, SignalResult
from execution.trade_signal import TradeSignal
from execution.audit_log import AuditLog, AuditEntry
from orchestration.strategy_manager import StrategyManager, DecayReport


def test_signal_scanner_init():
    scanner = SignalScanner(pairs=["BTC/USDT"], scan_interval=999)
    assert scanner._pairs == ["BTC/USDT"]
    assert scanner._scan_interval == 999


def test_signal_result_dataclass():
    sr = SignalResult(signal="buy", confidence=0.75, indicators={"rsi": 30}, pair="BTC/USDT", strategy_type="rsi_oversold", regime="ranging")
    assert sr.signal == "buy"
    assert sr.confidence == 0.75
    assert sr.indicators["rsi"] == 30
    assert sr.timestamp is not None


def test_trade_signal_dataclass():
    ts = TradeSignal(pair="ETH/USDT", side="buy", strategy_name="test", strategy_type="sma_crossover",
                     regime="uptrend", confidence=0.8, sharpe=1.5, win_rate=0.6, max_drawdown=0.05,
                     suggested_stoploss=-0.04, suggested_take_profit=0.08, source_agent="test")
    assert ts.pair == "ETH/USDT"
    assert ts.side == "buy"
    assert ts.status == "pending"
    d = ts.to_dict()
    assert d["pair"] == "ETH/USDT"
    assert "created_at" in d


def test_audit_log_record_and_query(tmp_path):
    log = AuditLog(save_path=str(tmp_path / "test_audit.jsonl"))
    assert log.count() == 0
    entry = AuditEntry(signal_id="test1", pair="BTC/USDT", side="buy", strategy_type="sma",
                       strategy_name="test", regime="uptrend", confidence=0.8,
                       position_size_usdt=100, entry_price=50000, status="executed",
                       risk_verdict="approved", circuit_breaker_state={}, correlation_result={}, kelly_result={})
    log.record(entry)
    assert log.count() == 1
    found = log.query_by_strategy("test")
    assert len(found) == 1
    assert found[0].pair == "BTC/USDT"


def test_strategy_manager_init():
    mgr = StrategyManager()
    assert mgr.get_deployed_count() == 0


def test_decay_report_dataclass():
    dr = DecayReport(
        strategy_id="s1", strategy_type="sma_crossover", regime="uptrend",
        backtest_sharpe=1.5, live_sharpe=0.6, decay_score=0.4,
        action="retired", reason="Significant decay",
    )
    assert dr.action == "retired"
    assert dr.decay_score == 0.4
    d = dr.to_dict()
    assert d["strategy_type"] == "sma_crossover"
