"""Tests for Kelly Criterion position sizing and correlation gate."""

import json

from agents.risk_manager import RiskManagerAgent, CircuitBreakerState, PerAssetDrawdownTracker


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


def test_circuit_breaker_clamps_raw_percentage_pnl():
    """LLM passes daily_pnl_pct=-50 (-50%) — clamped to -0.50, matches 50% limit, no halt."""
    result = _call_circuit_breaker({"daily_pnl_pct": -50, "daily_limit": 0.50})
    assert result["trading_allowed"] is True, (
        f"Expected trading_allowed=True after clamping -50 to -0.50 (== 50% limit), got {result}"
    )


def test_circuit_breaker_clamps_then_halt_on_excess():
    """LLM passes daily_pnl_pct=-50, clamped to -0.50, still exceeds 3% limit — halts."""
    result = _call_circuit_breaker({"daily_pnl_pct": -50, "daily_limit": 0.03})
    assert result["trading_allowed"] is False, (
        f"Expected trading_allowed=False for -50 clamped to -0.50 vs 3% limit, got {result}"
    )


def test_circuit_breaker_zeroes_implausible_pnl():
    """LLM passes daily_pnl_pct=-0.95 (95% drawdown) — sanity guard zeroes it."""
    result = _call_circuit_breaker({"daily_pnl_pct": -0.95, "daily_limit": 0.03})
    assert result["trading_allowed"] is True, (
        f"Expected trading_allowed=True for zeroed implausible PnL, got {result}"
    )


def test_circuit_breaker_still_halted_on_real_drawdown():
    """A legitimate daily drawdown exceeding limit should still halt."""
    result = _call_circuit_breaker({"daily_pnl_pct": -0.05, "daily_limit": 0.03})
    assert result["trading_allowed"] is False, (
        f"Expected trading_allowed=False for -5% drawdown vs 3% limit, got {result}"
    )


def test_circuit_breaker_handles_none_pnl():
    """LLM passes daily_pnl_pct=None — guard prevents TypeError crash."""
    result = _call_circuit_breaker({"daily_pnl_pct": None, "daily_limit": 0.03})
    # Should not crash; None -> 0.0 should allow trading
    assert "trading_allowed" in result


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


# ── PerAssetDrawdownTracker tests ──

def test_drawdown_tracker_initial_state():
    """New tracker has no drawdown for unknown pairs."""
    tracker = PerAssetDrawdownTracker()
    assert tracker.current_drawdown("BTC/USDT") == 0.0
    assert tracker.drawdown_sizing_factor("BTC/USDT") == 1.0


def test_drawdown_tracker_peak_tracking():
    """Tracker records peak and computes drawdown correctly."""
    tracker = PerAssetDrawdownTracker()
    tracker.update("BTC/USDT", 100.0)
    assert tracker.current_drawdown("BTC/USDT") == 0.0
    assert tracker.drawdown_sizing_factor("BTC/USDT") == 1.0

    # Price drops 10%
    tracker.update("BTC/USDT", 90.0)
    assert round(tracker.current_drawdown("BTC/USDT"), 4) == 0.1
    # Sizing factor: 1 - (0.1 / 0.15) = 0.3333
    assert round(tracker.drawdown_sizing_factor("BTC/USDT"), 4) == round(1.0 - (0.1 / 0.15), 4)


def test_drawdown_tracker_new_peak_resets():
    """New high updates peak, drawdown resets."""
    tracker = PerAssetDrawdownTracker()
    tracker.update("BTC/USDT", 100.0)
    tracker.update("BTC/USDT", 80.0)
    assert round(tracker.current_drawdown("BTC/USDT"), 4) == 0.2

    # New peak above 100
    tracker.update("BTC/USDT", 110.0)
    assert tracker.current_drawdown("BTC/USDT") == 0.0


def test_drawdown_tracker_zero_at_threshold():
    """Sizing factor reaches 0 at max_drawdown_threshold (default 15%)."""
    tracker = PerAssetDrawdownTracker(max_drawdown_threshold=0.15)
    tracker.update("BTC/USDT", 100.0)
    # Drop exactly 15%
    tracker.update("BTC/USDT", 85.0)
    # dd = 0.15, factor = 1 - (0.15/0.15) = 0.0
    assert tracker.drawdown_sizing_factor("BTC/USDT") == 0.0


def test_drawdown_tracker_above_threshold():
    """Drawdown exceeding threshold also returns 0.0 factor."""
    tracker = PerAssetDrawdownTracker(max_drawdown_threshold=0.15)
    tracker.update("BTC/USDT", 100.0)
    tracker.update("BTC/USDT", 50.0)  # 50% drawdown >> 15% threshold
    assert tracker.drawdown_sizing_factor("BTC/USDT") == 0.0


def test_drawdown_normalised_size_returns_dict():
    """drawdown_normalised_size returns correct structure and adjusted size."""
    tracker = PerAssetDrawdownTracker()
    tracker.update("BTC/USDT", 100.0)
    tracker.update("BTC/USDT", 90.0)  # 10% drawdown

    result = tracker.drawdown_normalised_size(1000.0, "BTC/USDT")
    assert result["pair"] == "BTC/USDT"
    assert result["base_position_usdt"] == 1000.0
    assert result["current_drawdown"] == 0.1
    # factor = 1 - (0.1/0.15) = 0.3333...
    raw_factor = 1.0 - (0.1 / 0.15)
    assert result["drawdown_sizing_factor"] == round(raw_factor, 4)
    assert result["adjusted_position_usdt"] == round(1000.0 * raw_factor, 2)


def test_drawdown_normalised_size_no_drawdown():
    """No drawdown → adjusted size equals base size."""
    tracker = PerAssetDrawdownTracker()
    tracker.update("BTC/USDT", 100.0)
    result = tracker.drawdown_normalised_size(500.0, "BTC/USDT")
    assert result["adjusted_position_usdt"] == 500.0
    assert result["drawdown_sizing_factor"] == 1.0


def test_drawdown_normalised_size_zero_at_threshold():
    """At threshold drawdown, adjusted position is 0."""
    tracker = PerAssetDrawdownTracker(max_drawdown_threshold=0.10)
    tracker.update("BTC/USDT", 100.0)
    tracker.update("BTC/USDT", 90.0)  # exactly 10% = threshold
    result = tracker.drawdown_normalised_size(500.0, "BTC/USDT")
    assert result["adjusted_position_usdt"] == 0.0


def test_drawdown_tracker_reset_peak():
    """reset_peak sets peak to current value."""
    tracker = PerAssetDrawdownTracker()
    tracker.update("BTC/USDT", 100.0)
    tracker.update("BTC/USDT", 80.0)
    assert tracker.current_drawdown("BTC/USDT") > 0
    tracker.reset_peak("BTC/USDT")
    assert tracker.current_drawdown("BTC/USDT") == 0.0


def test_drawdown_tracker_reset_all():
    """reset_all clears all state."""
    tracker = PerAssetDrawdownTracker()
    tracker.update("BTC/USDT", 100.0)
    tracker.update("ETH/USDT", 200.0)
    tracker.update("BTC/USDT", 80.0)
    tracker.reset_all()
    assert tracker.current_drawdown("BTC/USDT") == 0.0
    assert tracker.current_drawdown("ETH/USDT") == 0.0


def test_risk_manager_accepts_drawdown_tracker():
    """RiskManagerAgent accepts a custom drawdown tracker."""
    tracker = PerAssetDrawdownTracker(max_drawdown_threshold=0.20)
    agent = RiskManagerAgent(drawdown_tracker=tracker)
    # The update_drawdown tool exists
    assert agent.get_tool("update_drawdown") is not None
    assert agent.get_tool("apply_drawdown_sizing") is not None


def test_apply_drawdown_sizing_tool():
    """apply_drawdown_sizing tool returns correct dict via LangChain."""
    tracker = PerAssetDrawdownTracker()
    tracker.update("ETH/USDT", 100.0)
    tracker.update("ETH/USDT", 85.0)  # 15% drawdown
    agent = RiskManagerAgent(drawdown_tracker=tracker)

    result = json.loads(agent.get_tool("apply_drawdown_sizing").func(json.dumps({
        "pair": "ETH/USDT",
        "base_position_usdt": 500.0,
    })))
    assert result["pair"] == "ETH/USDT"
    # At 15% drawdown with default 15% threshold, factor = 0
    assert result["adjusted_position_usdt"] == 0.0


def test_update_drawdown_tool():
    """update_drawdown tool updates tracker and returns correct state."""
    tracker = PerAssetDrawdownTracker()
    agent = RiskManagerAgent(drawdown_tracker=tracker)

    result = json.loads(agent.get_tool("update_drawdown").func(json.dumps({
        "positions": [
            {"pair": "BTC/USDT", "current_value_usdt": 150.0},
            {"pair": "ETH/USDT", "current_value_usdt": 100.0},
        ]
    })))
    assert result["updated"] is True
    assert "BTC/USDT" in result["drawdowns"]
    assert "ETH/USDT" in result["drawdowns"]
    # No drawdown on first update (peak = value)
    assert result["drawdowns"]["BTC/USDT"]["drawdown"] == 0.0
    assert result["drawdowns"]["BTC/USDT"]["sizing_factor"] == 1.0


def test_update_drawdown_then_apply_sizing():
    """Full pipeline: update drawdown → apply sizing with detected drawdown."""
    tracker = PerAssetDrawdownTracker()
    agent = RiskManagerAgent(drawdown_tracker=tracker)

    # Step 1: Set initial peak
    json.loads(agent.get_tool("update_drawdown").func(json.dumps({
        "positions": [{"pair": "BTC/USDT", "current_value_usdt": 100.0}]
    })))

    # Step 2: Value drops (simulate drawdown)
    json.loads(agent.get_tool("update_drawdown").func(json.dumps({
        "positions": [{"pair": "BTC/USDT", "current_value_usdt": 85.0}]
    })))

    # Step 3: Apply sizing — should be reduced
    result = json.loads(agent.get_tool("apply_drawdown_sizing").func(json.dumps({
        "pair": "BTC/USDT",
        "base_position_usdt": 1000.0,
    })))
    assert result["drawdown_pct"] == 0.15  # 15% drawdown
    assert result["adjusted_position_usdt"] == 0.0  # at threshold


def test_empty_update_drawdown():
    """Empty positions list returns no drawdowns without error."""
    agent = RiskManagerAgent()
    result = json.loads(agent.get_tool("update_drawdown").func(json.dumps({
        "positions": []
    })))
    assert result["updated"] is False
    assert result["drawdowns"] == {}


def test_drawdown_tracker_custom_threshold():
    """Custom max_drawdown_threshold changes the sizing decay slope."""
    tracker = PerAssetDrawdownTracker(max_drawdown_threshold=0.05)  # 5% threshold
    tracker.update("BTC/USDT", 100.0)
    tracker.update("BTC/USDT", 95.0)  # exactly 5%
    assert tracker.drawdown_sizing_factor("BTC/USDT") == 0.0

    tracker2 = PerAssetDrawdownTracker(max_drawdown_threshold=0.05)
    tracker2.update("BTC/USDT", 100.0)
    tracker2.update("BTC/USDT", 97.5)  # 2.5% drawdown
    # factor = 1 - (0.025/0.05) = 0.5
    assert tracker2.drawdown_sizing_factor("BTC/USDT") == 0.5


def test_drawdown_tracker_linear_decay():
    """Sizing factor decreases linearly with drawdown depth."""
    tracker = PerAssetDrawdownTracker(max_drawdown_threshold=0.20)
    tracker.update("BTC/USDT", 100.0)

    pairs = [
        (100.0, 1.0),   # 0%  dd → factor 1.0
        (95.0, 0.75),   # 5%  dd → factor 0.75
        (90.0, 0.5),    # 10% dd → factor 0.5
        (85.0, 0.25),   # 15% dd → factor 0.25
        (80.0, 0.0),    # 20% dd → factor 0.0
    ]

    for price, expected_factor in pairs:
        # Reset tracker for clean state each time
        t = PerAssetDrawdownTracker(max_drawdown_threshold=0.20)
        t.update("X", 100.0)
        t.update("X", price)
        dd = t.current_drawdown("X")
        factor = t.drawdown_sizing_factor("X")
        assert abs(factor - expected_factor) < 0.001, (
            f"dd={dd:.2%}, expected factor={expected_factor}, got {factor}"
        )


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


def _call_circuit_breaker(params: dict) -> dict:
    """Call the circuit_breaker_check tool with given params and return parsed result."""
    CircuitBreakerState.clear()
    result = RiskManagerAgent().get_tool("circuit_breaker_check").func(json.dumps(params))
    return json.loads(result)
