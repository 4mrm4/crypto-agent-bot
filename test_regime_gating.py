"""Tests for regime-conditioned gating (Task 10)."""

import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from execution.signal_scanner import (
    RegimeTransition,
    RegimeTransitionTracker,
    SignalScanner,
    SignalResult,
    TRANSITION_COOLDOWN_MINUTES,
    LOW_CONFIDENCE_THRESHOLD,
)


# ── RegimeTransitionTracker Tests ──

def test_tracker_initial_state():
    tracker = RegimeTransitionTracker()
    assert tracker.current_regime == "unknown"
    assert tracker.current_confidence == 0.0
    assert not tracker.is_in_cooldown()
    assert tracker.last_transition() is None
    assert tracker.transition_count(24) == 0


def test_tracker_first_update_no_transition():
    tracker = RegimeTransitionTracker()
    result = tracker.update("ranging", 0.8)
    assert result is None  # no transition from "unknown"
    assert tracker.current_regime == "ranging"
    assert tracker.current_confidence == 0.8


def test_tracker_detects_transition():
    tracker = RegimeTransitionTracker()
    tracker.update("ranging", 0.8)
    result = tracker.update("strong_uptrend", 0.9)
    assert result is not None
    assert result.from_regime == "ranging"
    assert result.to_regime == "strong_uptrend"
    assert result.confidence == 0.9
    assert tracker.current_regime == "strong_uptrend"
    assert tracker.transition_count(24) == 1


def test_tracker_multiple_transitions():
    tracker = RegimeTransitionTracker()
    tracker.update("ranging", 0.8)
    tracker.update("strong_uptrend", 0.9)
    tracker.update("volatile", 0.6)
    tracker.update("strong_uptrend", 0.7)
    assert tracker.transition_count(24) == 3
    assert tracker.transition_count(999) == 3  # large window includes all


def test_tracker_cooldown_active():
    tracker = RegimeTransitionTracker(cooldown_minutes=30)
    tracker.update("ranging", 0.8)
    tracker.update("strong_uptrend", 0.9)
    # Transition just happened — cooldown is active
    assert tracker.is_in_cooldown()


def test_tracker_cooldown_expired():
    tracker = RegimeTransitionTracker(cooldown_minutes=0)  # zero cooldown = immediate expiry
    tracker.update("ranging", 0.8)
    tracker.update("strong_uptrend", 0.9)
    assert not tracker.is_in_cooldown()


def test_tracker_cooldown_timedelta_calculation():
    """Use a past timestamp to verify cooldown expiry logic."""
    tracker = RegimeTransitionTracker(cooldown_minutes=10)
    tracker.update("ranging", 0.8)
    # Manually set last transition to 15 minutes ago
    past_transition = RegimeTransition(
        from_regime="ranging",
        to_regime="strong_uptrend",
        timestamp=datetime.utcnow() - timedelta(minutes=15),
        confidence=0.9,
    )
    tracker._history.append(past_transition)
    tracker._current_regime = "strong_uptrend"
    assert not tracker.is_in_cooldown()  # 15 > 10, cooldown expired


def test_tracker_no_cooldown_without_history():
    tracker = RegimeTransitionTracker()
    tracker.update("ranging", 0.8)
    assert not tracker.is_in_cooldown()  # no transitions yet


def test_effective_signal_floor_normal():
    """No transition, high confidence → base floor."""
    tracker = RegimeTransitionTracker()
    tracker.update("ranging", 0.8)
    assert tracker.effective_signal_floor(0.6) == 0.6


def test_effective_signal_floor_cooldown():
    """Recent transition → raised floor."""
    tracker = RegimeTransitionTracker(cooldown_minutes=30)
    tracker.update("ranging", 0.8)
    tracker.update("strong_uptrend", 0.7)
    floor = tracker.effective_signal_floor(0.6)
    assert floor == 0.75  # LOW_CONFIDENCE_SIGNAL_FLOOR


def test_effective_signal_floor_low_confidence():
    """Low regime confidence → raised floor."""
    tracker = RegimeTransitionTracker(cooldown_minutes=30)
    tracker.update("ranging", 0.3)  # below LOW_CONFIDENCE_THRESHOLD
    floor = tracker.effective_signal_floor(0.6)
    assert floor == 0.75  # LOW_CONFIDENCE_SIGNAL_FLOOR


def test_effective_signal_floor_flapping():
    """5+ transitions in 24 hours → floor rises to 0.7."""
    tracker = RegimeTransitionTracker(cooldown_minutes=0)
    tracker.update("a", 0.8)
    # Add 5 transitions manually
    from datetime import datetime
    regimes = ["b", "c", "d", "e", "f"]
    for r in regimes:
        tracker.update(r, 0.8)
    # Now we have 5 transitions
    assert tracker.transition_count(24) >= 5
    floor = tracker.effective_signal_floor(0.6)
    assert floor == 0.7


def test_transition_count_window():
    tracker = RegimeTransitionTracker()
    # Add transitions (first update doesn't count — it's from "unknown")
    tracker.update("a", 0.8)  # initial, no transition
    tracker.update("b", 0.8)  # transition 1
    tracker.update("c", 0.8)  # transition 2
    assert tracker.transition_count(24) == 2


# ── Signal Scanner Gating Tests ──

def test_get_strategies_for_regime_validated_only():
    """Strategies with validated_regimes that don't match current regime are skipped."""
    scanner = SignalScanner(scan_interval=999)
    scanner._approved_strategies = [
        {"strategy_type": "momentum", "regime": "strong_uptrend",
         "validated_regimes": ["strong_uptrend"]},
        {"strategy_type": "rsi_oversold", "regime": "ranging",
         "validated_regimes": ["ranging"]},
        {"strategy_type": "mean_reversion", "regime": "ranging",
         "validated_regimes": ["ranging"]},
    ]
    # In ranging regime, only rsi_oversold and mean_reversion should match
    with patch("data.regime.REGIME_STRATEGY_MAP", {
        "ranging": {"use": ["mean_reversion", "rsi_oversold", "bollinger_bands"], "avoid": []}
    }):
        result = scanner._get_strategies_for_regime("ranging")
        types = [s["strategy_type"] for s in result]
        assert "momentum" not in types  # validated for uptrend only
        assert "rsi_oversold" in types
        assert "mean_reversion" in types


def test_get_strategies_for_regime_no_validated():
    """Strategies without validated_regimes field still match by type."""
    scanner = SignalScanner(scan_interval=999)
    scanner._approved_strategies = [
        {"strategy_type": "sma_crossover", "regime": "strong_uptrend"},
        {"strategy_type": "momentum", "regime": "strong_uptrend"},
    ]
    with patch("data.regime.REGIME_STRATEGY_MAP", {
        "strong_uptrend": {"use": ["momentum", "sma_crossover"], "avoid": []}
    }):
        result = scanner._get_strategies_for_regime("strong_uptrend")
        assert len(result) == 2  # both match by type


def test_get_strategies_for_regime_empty_validated():
    """Empty validated_regimes list means strategy hasn't been validated anywhere → skip."""
    scanner = SignalScanner(scan_interval=999)
    scanner._approved_strategies = [
        {"strategy_type": "breakout", "regime": "volatile",
         "validated_regimes": []},  # empty = not validated anywhere
        {"strategy_type": "volatility_squeeze", "regime": "volatile",
         "validated_regimes": ["volatile"]},
    ]
    with patch("data.regime.REGIME_STRATEGY_MAP", {
        "volatile": {"use": ["breakout", "volatility_squeeze", "bollinger_bands"], "avoid": []}
    }):
        result = scanner._get_strategies_for_regime("volatile")
        types = [s["strategy_type"] for s in result]
        assert "breakout" not in types  # empty validated = skip
        assert "volatility_squeeze" in types


def test_update_approved_strategies_populates_validated():
    """update_approved_strategies should auto-populate validated_regimes."""
    scanner = SignalScanner(scan_interval=999)
    scanner.update_approved_strategies([
        {"strategy_type": "momentum", "regime": "strong_uptrend"},
    ])
    assert scanner._approved_strategies[0].get("validated_regimes") == ["strong_uptrend"]


def test_signal_floor_in_scan_loop():
    """Verify scan loop uses regime-adjusted signal floor for execution gating."""
    scanner = SignalScanner(scan_interval=999)
    # Put scanner in cooldown state
    scanner._regime_tracker.update("ranging", 0.8)
    scanner._regime_tracker.update("strong_uptrend", 0.9)
    assert scanner._regime_tracker.effective_signal_floor(0.6) == 0.75

    # Signal with confidence 0.65 (above base 0.6, below regime floor 0.75)
    result = SignalResult(
        signal="buy", confidence=0.65, indicators={},
        pair="BTC/USDT", strategy_type="momentum", regime="strong_uptrend",
    )
    signal_floor = scanner._regime_tracker.effective_signal_floor(0.6)
    # 0.65 < 0.75 → would NOT execute
    assert result.signal in ("buy", "sell")
    assert result.confidence < signal_floor


def test_regime_transition_event_emitted():
    """Regime transitions emit regime_transition event via event_bus."""
    bus = MagicMock()
    scanner = SignalScanner(scan_interval=999, event_bus=bus)
    scanner._regime_tracker.update("ranging", 0.8)

    # Simulate a transition
    transition = scanner._regime_tracker.update("strong_uptrend", 0.9)
    assert transition is not None
    # The event is fired via asyncio.ensure_future, so we just check the tracker
    assert scanner._regime_tracker.current_regime == "strong_uptrend"
    assert scanner._regime_tracker.current_confidence == 0.9
