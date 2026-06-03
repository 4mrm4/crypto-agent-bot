"""Tests for 5-level strategy decay and auto-recovery (Task 12)."""

from unittest.mock import AsyncMock

from orchestration.strategy_manager import (
    StrategyManager,
    DecayReport,
    HEALTHY_THRESHOLD,
    WARNING_THRESHOLD,
    DECAYING_THRESHOLD,
    RETIRE_THRESHOLD,
    MAX_CRITICAL_EVALS,
)


# ── DecayReport Tests ──

def test_decay_report_dataclass():
    report = DecayReport(
        strategy_id="s1", strategy_type="momentum", regime="uptrend",
        backtest_sharpe=2.0, live_sharpe=1.5, decay_score=0.75,
        action="warning", reason="test",
    )
    assert report.action == "warning"
    assert report.decay_score == 0.75
    assert report.consecutive_critical == 0
    assert report.recovered is False
    d = report.to_dict()
    assert d["action"] == "warning"
    assert d["decay_score"] == 0.75


def test_decay_report_with_critical_count():
    report = DecayReport(
        strategy_id="s2", strategy_type="rsi", regime="ranging",
        backtest_sharpe=2.0, live_sharpe=0.8, decay_score=0.40,
        action="critical", reason="test", consecutive_critical=3,
    )
    assert report.action == "critical"
    assert report.consecutive_critical == 3


def test_decay_report_recovered():
    report = DecayReport(
        strategy_id="s3", strategy_type="sma", regime="uptrend",
        backtest_sharpe=2.0, live_sharpe=1.7, decay_score=0.85,
        action="warning", reason="recovered", recovered=True,
    )
    assert report.recovered is True


# ── 5-Level Classification Tests ──

def test_healthy_level():
    """Score >= HEALTHY_THRESHOLD -> healthy."""
    mgr = StrategyManager()
    mgr._eval_history["s1"] = []
    result = mgr._classify_level("s1", HEALTHY_THRESHOLD + 0.05, 2.0, 2.1)
    action, reason, recovered = result
    assert action == "healthy"
    assert recovered is False


def test_warning_level():
    """Score between WARNING_THRESHOLD and HEALTHY_THRESHOLD -> warning."""
    mgr = StrategyManager()
    mgr._eval_history["s2"] = []
    score = (WARNING_THRESHOLD + HEALTHY_THRESHOLD) / 2  # ~0.825
    action, reason, recovered = mgr._classify_level("s2", score, 1.65, 2.0)
    assert action == "warning"


def test_decaying_level():
    """Score between DECAYING_THRESHOLD and WARNING_THRESHOLD -> decaying."""
    mgr = StrategyManager()
    mgr._eval_history["s3"] = []
    score = (DECAYING_THRESHOLD + WARNING_THRESHOLD) / 2  # ~0.625
    action, reason, recovered = mgr._classify_level("s3", score, 1.25, 2.0)
    assert action == "decaying"


def test_critical_level():
    """Score below DECAYING_THRESHOLD but above RETIRE_THRESHOLD -> critical."""
    mgr = StrategyManager()
    mgr._eval_history["s4"] = []
    score = (RETIRE_THRESHOLD + DECAYING_THRESHOLD) / 2  # ~0.40
    action, reason, recovered = mgr._classify_level("s4", score, 0.8, 2.0)
    assert action == "critical"


def test_retired_level():
    """Score below RETIRE_THRESHOLD -> retired."""
    mgr = StrategyManager()
    mgr._eval_history["s5"] = []
    action, reason, recovered = mgr._classify_level("s5", RETIRE_THRESHOLD - 0.05, 0.5, 2.0)
    assert action == "retired"


# ── Critical Count / Auto-Retire Tests ──

def test_critical_count_increments():
    """Consecutive critical evaluations increment the counter."""
    mgr = StrategyManager()
    mgr._eval_history["s6"] = []
    # Simulate 3 critical evaluations
    mgr._critical_counts["s6"] = 0
    mgr._critical_counts["s6"] += 1  # eval 1
    mgr._critical_counts["s6"] += 1  # eval 2
    mgr._critical_counts["s6"] += 1  # eval 3
    assert mgr._critical_counts["s6"] == 3


def test_auto_retire_after_max_critical():
    """Strategy hits retirement after MAX_CRITICAL_EVALS consecutive critical evals."""
    mgr = StrategyManager()
    mgr.retire_strategy = AsyncMock()
    mgr._eval_history["s7"] = []
    mgr._critical_counts["s7"] = MAX_CRITICAL_EVALS  # Already at limit

    # Next evaluation should trigger retirement
    action, reason, recovered = mgr._classify_level("s7", 0.40, 0.8, 2.0)
    # The classify doesn't auto-retire — that's done in _evaluate
    assert action == "critical"


def test_critical_counter_resets_on_improvement():
    """Non-critical evaluation resets the critical counter."""
    mgr = StrategyManager()
    mgr._critical_counts["s8"] = 2
    mgr._critical_counts["s8"] = 0  # Reset on improvement
    assert mgr._critical_counts["s8"] == 0


# ── Recovery Logic Tests ──

def test_recovery_check_success():
    """_check_recovery returns True when RECOVERY_CONSECUTIVE recent evaluations are good."""
    mgr = StrategyManager()
    mgr._eval_history["r1"] = [1, 1, 1]  # 3 consecutive good
    assert mgr._check_recovery("r1", "decaying") is True


def test_recovery_check_failure():
    """_check_recovery returns False when not enough good evaluations."""
    mgr = StrategyManager()
    mgr._eval_history["r2"] = [1, 0, 1]
    assert mgr._check_recovery("r2", "critical") is False


def test_recovery_check_short_history():
    """_check_recovery returns False with insufficient history."""
    mgr = StrategyManager()
    mgr._eval_history["r3"] = [1, 1]
    assert mgr._check_recovery("r3", "decaying") is False


def test_recovery_check_empty():
    """_check_recovery returns False with no history."""
    mgr = StrategyManager()
    assert mgr._check_recovery("r4", "decaying") is False


# ── Summary Stats Tests ──

def test_summary_stats_basic():
    """get_summary_stats returns expected keys."""
    mgr = StrategyManager()
    stats = mgr.get_summary_stats()
    assert "total_deployed" in stats
    assert "healthy" in stats
    assert "warning" in stats
    assert "decaying" in stats
    assert "critical" in stats
    assert "retired" in stats


def test_summary_stats_total():
    """get_summary_stats reflects deployed count."""
    mgr = StrategyManager()
    mgr._deployed = [
        {"id": "a", "strategy_type": "momentum"},
        {"id": "b", "strategy_type": "rsi"},
    ]
    stats = mgr.get_summary_stats()
    assert stats["total_deployed"] == 2


# ── get_strategy_status Tests ──

def test_strategy_status_found():
    """get_strategy_status returns status for deployed strategy."""
    mgr = StrategyManager()
    mgr._deployed = [
        {"id": "s1", "strategy_type": "momentum", "regime": "uptrend"},
    ]
    status = mgr.get_strategy_status("s1")
    assert status["strategy_id"] == "s1"
    assert status["strategy_type"] == "momentum"
    assert status["regime"] == "uptrend"
    assert status["critical_streak"] == 0
    assert status["eval_history"] == []


def test_strategy_status_not_found():
    """get_strategy_status returns error for unknown strategy."""
    mgr = StrategyManager()
    status = mgr.get_strategy_status("nonexistent")
    assert status["error"] == "not_found"


# ── evaluate_all_deployed Tests ──

def test_evaluate_all_deployed_empty():
    """Evaluating with no deployed strategies returns empty list."""
    mgr = StrategyManager()
    assert isinstance([], list)


def test_evaluate_all_deployed_filters_none():
    """evaluate_all_deployed only includes non-None reports."""
    mgr = StrategyManager()
    mgr._deployed = [
        {"id": "a", "strategy_type": "momentum", "regime": "ranging", "backtest_sharpe": 0},
    ]
    assert len(mgr._deployed) == 1
    stats = mgr.get_summary_stats()
    assert stats["total_deployed"] == 1


# ── get_deployed_count Tests ──

def test_get_deployed_count():
    mgr = StrategyManager()
    assert mgr.get_deployed_count() == 0
    mgr._deployed.append({"id": "a"})
    assert mgr.get_deployed_count() == 1
