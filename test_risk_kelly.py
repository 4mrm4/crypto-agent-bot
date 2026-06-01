"""Tests for Kelly Criterion position sizing and correlation gate."""

import json

from agents.risk_manager import RiskManagerAgent, CircuitBreakerState


def test_kelly_basic():
    """Basic Kelly: 60% win rate, 2:1 reward:risk."""
    result = _call_kelly(win_rate=0.60, avg_win=0.10, avg_loss=0.05, portfolio=10000)
    assert result["kelly_fraction"] > 0
    assert result["position_size_usdt"] > 0
    assert result["position_size_usdt"] <= 1000  # 10% cap


def test_kelly_negative_edge():
    """Kelly returns 0 when edge is negative."""
    result = _call_kelly(win_rate=0.30, avg_win=0.05, avg_loss=0.10, portfolio=10000)
    assert result["position_size_usdt"] == 0
    assert result["kelly_fraction"] == 0


def test_kelly_never_exceeds_max():
    """Kelly can never exceed 10% of portfolio."""
    result = _call_kelly(win_rate=0.90, avg_win=0.20, avg_loss=0.01, portfolio=10000)
    assert result["position_size_usdt"] <= 1000  # 10% of 10000


def test_kelly_quarter_fraction_default():
    """Default max_kelly_fraction = 0.25 produces smaller positions (below cap)."""
    full = _call_kelly(win_rate=0.55, avg_win=0.04, avg_loss=0.02, portfolio=10000, max_frac=1.0)
    quarter = _call_kelly(win_rate=0.55, avg_win=0.04, avg_loss=0.02, portfolio=10000, max_frac=0.25)
    assert quarter["position_size_usdt"] < full["position_size_usdt"]


def test_circuit_breaker_not_halted_by_default():
    state = CircuitBreakerState.status()
    assert state["halted"] is False


def test_circuit_breaker_halt_works():
    CircuitBreakerState.clear()
    CircuitBreakerState.halt("Test halt", duration_minutes=10)
    state = CircuitBreakerState.status()
    assert state["halted"] is True
    assert state["reason"] == "Test halt"
    CircuitBreakerState.clear()


def test_circuit_breaker_auto_resume():
    CircuitBreakerState.clear()
    CircuitBreakerState.halt("Short halt", duration_minutes=0)
    # Duration = 0 means resume_after is now, so is_halted should be False
    # (or very close — we accept either with a tiny tolerance)
    import time
    time.sleep(0.01)
    halted = CircuitBreakerState.is_halted()
    assert halted is False


def test_pre_trade_approval_rejects_when_circuit_breaker_halted():
    agent = RiskManagerAgent()
    CircuitBreakerState.halt("Test", duration_minutes=60)
    result = json.loads(agent.get_tool("pre_trade_approval").func(json.dumps({
        "circuit_breaker_result": {"trading_allowed": False, "reason": "Test halt"},
        "strategy_metrics": {"sharpe_ratio": 1.5},
    })))
    assert result["approved"] is False
    CircuitBreakerState.clear()
    assert result["confidence"] == 0.0


def test_risk_assessment_rejects_high_risk():
    result = json.loads(RiskManagerAgent().get_tool("assess_strategy_risk").func(json.dumps({
        "sharpe_ratio": 0.3,
        "win_rate": 0.25,
        "max_drawdown": 0.25,
        "total_trades": 5,
        "profit_factor": 0.8,
    })))
    assert result["verdict"] == "reject"
    assert result["concern_count"] >= 2


def _call_kelly(win_rate, avg_win, avg_loss, portfolio, max_frac=0.25):
    result = RiskManagerAgent().get_tool("kelly_position_size").func(json.dumps({
        "win_rate": win_rate,
        "avg_win_pct": avg_win,
        "avg_loss_pct": avg_loss,
        "portfolio_value": portfolio,
        "max_kelly_fraction": max_frac,
        "oos_degradation_pct": 0.0, "sizing_tier": "normal",
    }))
    return json.loads(result)
