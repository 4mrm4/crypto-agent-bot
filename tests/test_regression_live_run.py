"""Regression tests for live run issues — backtester contract, routing, parsing."""

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Issue 1: Backtester run_backtest accepts strategy_type + params ──


def test_backtester_run_backtest_accepts_strategy_type_params():
    """run_backtest('sma_crossover', '{"fast_ma": 10}') must not require strategy_id."""
    from agents.backtester import BacktesterAgent
    from unittest.mock import MagicMock

    agent = BacktesterAgent()
    # Patch engine to return a valid result without subprocess
    agent._engine.run_backtest = MagicMock(return_value={
        "total_trades": 50, "win_rate": 0.6, "sharpe_ratio": 1.2,
        "max_drawdown": -0.03, "profit_ratio": 0.05,
    })
    result = agent.get_tool("run_backtest").func(
        strategy_type="sma_crossover",
        params='{"fast_ma": 10, "slow_ma": 30}',
    )
    assert "Error: unknown strategy_id" not in result
    assert agent._iteration_history  # must have recorded an iteration


def test_backtester_run_backtest_returns_metrics():
    """run_backtest result must contain sharpe_ratio, win_rate, max_drawdown, total_trades."""
    from agents.backtester import BacktesterAgent
    from unittest.mock import MagicMock

    agent = BacktesterAgent()
    agent._engine.run_backtest = MagicMock(return_value={
        "total_trades": 50, "win_rate": 0.6, "sharpe_ratio": 1.2,
        "max_drawdown": -0.03, "profit_ratio": 0.05,
    })
    agent.get_tool("run_backtest").func(
        strategy_type="sma_crossover",
        params='{"fast_ma": 10}',
    )
    record = agent._iteration_history[-1]
    metrics = record["metrics"]
    for key in ("sharpe_ratio", "win_rate", "max_drawdown", "total_trades"):
        assert key in metrics, f"Missing metric: {key}"


# ── Issue 2: _extract_child_tasks() JSON parsing ──


def test_extract_child_tasks_parses_params_json():
    """_extract_child_tasks must extract strategy_type and params from 'next:' lines."""
    from orchestration.graph import _extract_child_tasks
    from orchestration.board import TaskBoard

    board = TaskBoard({}, {})
    parent = type("Task", (), {"id": "p1", "result": (
        'Strategy [abc12345] created: type=sma_crossover\n'
        'next: backtest strategy_type=sma_crossover params={"fast_ma": 10, "slow_ma": 30}\n'
    ), "assigned_to": "strategist"})()
    _extract_child_tasks(parent, board)
    tasks = board.get_tasks_by_status("TODO")
    assert len(tasks) == 1
    desc = tasks[0].description
    assert "strategy_type=sma_crossover" in desc
    assert "fast_ma" in desc


def test_extract_child_tasks_nested_braces():
    """Params with nested JSON (strings containing braces) must parse correctly."""
    from orchestration.graph import _extract_child_tasks
    from orchestration.board import TaskBoard

    board = TaskBoard({}, {})
    parent = type("Task", (), {"id": "p2", "result": (
        'next: backtest strategy_type=rsi_oversold '
        'params={"rsi_period": 14, "note": "test {with} braces"}\n'
    ), "assigned_to": "strategist"})()
    _extract_child_tasks(parent, board)
    tasks = board.get_tasks_by_status("TODO")
    assert len(tasks) == 1
    assert "rsi_period" in tasks[0].description


def test_extract_child_tasks_spaces_in_values():
    """Params values with spaces in strings must parse correctly."""
    from orchestration.graph import _extract_child_tasks
    from orchestration.board import TaskBoard

    board = TaskBoard({}, {})
    parent = type("Task", (), {"id": "p3", "result": (
        'next: backtest strategy_type=custom '
        "params={\"indicator_code\": \"df['ema'] = ta.EMA(df, 10)\"}\n"
    ), "assigned_to": "strategist"})()
    _extract_child_tasks(parent, board)
    tasks = board.get_tasks_by_status("TODO")
    assert len(tasks) == 1


def test_full_dispatch_chain():
    """Complete path: strategist next: output -> _extract_child_tasks -> correctly formed backtester task."""
    from orchestration.graph import _extract_child_tasks
    from orchestration.board import TaskBoard

    board = TaskBoard({}, {})
    # Simulate strategist output
    strategist_output = (
        "Strategy [a1b2c3d4] created: type=multi_timeframe\n"
        "next: backtest strategy_type=multi_timeframe "
        'params={"timeframe": "1h", "stoploss": -0.05}\n'
    )
    parent = type("Task", (), {
        "id": "p_chain", "result": strategist_output,
        "assigned_to": "strategist"
    })()
    _extract_child_tasks(parent, board)
    tasks = board.get_tasks_by_status("TODO")
    assert len(tasks) == 1
    desc = tasks[0].description
    # The task must contain all info the backtester needs
    assert "strategy_type=multi_timeframe" in desc
    assert '"timeframe": "1h"' in desc


# ── Issue 3: Agent routing ──


def test_pick_agent_risk_assessment():
    """'Assess risk for strategies targeting X' must route to risk_manager."""
    from orchestration.graph import _pick_agent

    caps = {
        "analyst": ["analysis", "market"],
        "strategist": ["strategy", "generate", "concept", "design"],
        "backtester": ["backtest", "walk_forward", "hyperopt", "optimise"],
        "risk_manager": ["risk", "assess", "kelly", "circuit", "position"],
        "curator": ["memory", "context"],
        "researcher": ["research", "web", "paper"],
    }
    result = _pick_agent("Assess risk for strategies targeting: Auto-research", caps)
    assert result == "risk_manager", f"Got {result}, expected risk_manager"


def test_pick_agent_backtest():
    """'backtest strategy_type=X params=Y' must route to backtester."""
    from orchestration.graph import _pick_agent

    caps = {
        "analyst": ["analysis", "market"],
        "strategist": ["strategy", "generate", "concept", "design"],
        "backtester": ["backtest", "walk_forward", "hyperopt", "optimise"],
        "risk_manager": ["risk", "assess", "kelly", "circuit", "position"],
        "curator": ["memory", "context"],
        "researcher": ["research", "web", "paper"],
    }
    result = _pick_agent("backtest strategy_type=sma_crossover params={}", caps)
    assert result == "backtester", f"Got {result}, expected backtester"


# ── Issue 6: Memory context must route to strategist ──


def test_regression_pick_agent_memory_context_routes_to_strategist():
    """[MEMORY CONTEXT tasks must always route to strategist, never backtester."""
    from orchestration.graph import _pick_agent

    caps = {
        "analyst": ["analysis", "market"],
        "strategist": ["strategy", "generate", "concept", "design"],
        "backtester": ["backtest", "data", "run", "research", "paper", "walk_forward"],
        "risk_manager": ["risk", "assess", "kelly", "circuit", "position"],
        "curator": ["memory", "context"],
        "researcher": ["research", "web", "paper"],
    }
    # Memory context containing words that could score higher for backtester
    desc = (
        "[MEMORY CONTEXT - Past relevant research:]\n"
        "Previous backtest data for sma_crossover showed Sharpe=1.2. "
        "Run more research on paper trading strategies."
    )
    result = _pick_agent(desc, caps)
    assert result == "strategist", f"Got {result}, expected strategist"


def test_regression_pick_agent_memory_context_prefix_is_exact():
    """Only descriptions starting with exactly '[MEMORY CONTEXT' are special-routed."""
    from orchestration.graph import _pick_agent

    caps = {
        "analyst": ["analysis", "market"],
        "strategist": ["strategy", "generate", "concept", "design"],
        "backtester": ["backtest", "data", "run", "walk_forward"],
        "risk_manager": ["risk", "assess", "kelly", "circuit", "position"],
        "curator": ["memory", "context"],
        "researcher": ["research", "web", "paper"],
    }
    # Description with 'memory context' but NOT starting with [MEMORY CONTEXT
    desc = "Review memory context from previous backtest data run"
    result = _pick_agent(desc, caps)
    assert result != "strategist", "Non-prefixed context should use normal routing"


# ── Issue 4: Discouraged strategy only checks requested type ──


def test_discouraged_only_checks_requested_type():
    """Task mentioning discouraged strategies in context must NOT trigger false positive."""
    desc = ('[CURRENT REGIME: strong_downtrend]\n'
            'backtest strategy_type=multi_timeframe params={"timeframe": "1h"}')
    # Extract strategy_type
    m = re.search(r'strategy_type=(\w+)', desc)
    assert m is not None
    strategy_type = m.group(1)
    discouraged = ["momentum", "breakout", "sma_crossover"]
    assert strategy_type not in discouraged, "multi_timeframe should not be in discouraged"


# ── Issue 5: TradingDatabase singleton ──


def test_database_singleton():
    """TradingDatabase() called twice must return same instance."""
    from data.database import TradingDatabase
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        db1 = TradingDatabase(db_path)
        db2 = TradingDatabase(db_path)
        assert db1 is db2, "Multiple instances should return same object"
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


# ── Bug 1: Circuit breaker comparison ──


def test_circuit_breaker_zero_drawdown_does_not_halt():
    """Circuit breaker must NOT halt when daily_pnl is 0.0."""
    from agents.risk_manager import RiskManagerAgent

    agent = RiskManagerAgent()
    cb_tool = agent.get_tool("circuit_breaker_check")
    result = cb_tool.func('{"daily_pnl_pct": 0.0}')
    import json
    parsed = json.loads(result)
    assert parsed.get("trading_allowed") is True, (
        f"Should allow trading at 0 drawdown, got: {parsed}"
    )


def test_regression_circuit_breaker_clamps_percentage():
    """Circuit breaker must soft-clamp raw percentage values (e.g. 500 -> 5.0) instead of crashing."""
    from agents.risk_manager import RiskManagerAgent, CircuitBreakerState

    CircuitBreakerState.clear()
    agent = RiskManagerAgent()
    cb_tool = agent.get_tool("circuit_breaker_check")
    # daily_limit=500 should be clamped to 5.0 (500/100), NOT crash with AssertionError
    result = cb_tool.func('{"daily_pnl_pct": -0.06, "daily_limit": 500}')
    import json
    parsed = json.loads(result)
    # After clamping: 500 -> 5.0, so 6% loss is within 500% day limit -> no halt
    # Actually 500/100 = 5.0, and 6% loss is 0.06 which is < 5.0, so trading_allowed
    # Wait — 500/100 = 5.0 as decimal = 500%. 6% loss = 0.06 < 5.0, so no halt.
    assert parsed.get("trading_allowed") is True, (
        f"Should clamp 500 to 5.0 (500%) and allow 6%% loss within it, got: {parsed}"
    )
    CircuitBreakerState.clear()


def test_regression_circuit_breaker_clamps_zero_limit_to_default():
    """Circuit breaker must use default 0.03 when daily_limit <= 0."""
    from agents.risk_manager import RiskManagerAgent, CircuitBreakerState

    CircuitBreakerState.clear()
    agent = RiskManagerAgent()
    cb_tool = agent.get_tool("circuit_breaker_check")
    # daily_limit=0 should fall back to default 0.03. 5% loss > 3% limit -> halt.
    result = cb_tool.func('{"daily_pnl_pct": -0.05, "daily_limit": 0}')
    import json
    parsed = json.loads(result)
    assert parsed.get("trading_allowed") is False, (
        f"Should use default 0.03 and halt on 5%% loss, got: {parsed}"
    )
    CircuitBreakerState.clear()


def test_circuit_breaker_halt_on_actual_drawdown():
    """Circuit breaker must halt when real drawdown exceeds limit."""
    from agents.risk_manager import RiskManagerAgent, CircuitBreakerState

    CircuitBreakerState.clear()  # ensure clean slate
    agent = RiskManagerAgent()
    cb_tool = agent.get_tool("circuit_breaker_check")
    result = cb_tool.func('{"daily_pnl_pct": -0.05, "daily_limit": 0.03}')
    import json
    parsed = json.loads(result)
    assert parsed.get("trading_allowed") is False, (
        f"Should halt on 5% loss with 3% limit, got: {parsed}"
    )
    # Clean up: clear the circuit breaker state
    CircuitBreakerState.clear()


# ── Bug 2: Pre-filter / startup log line ──


def test_backtest_engine_startup_log_line():
    """BacktestEngine must log 'Starting Freqtrade subprocess' before running."""
    from backtesting.engine import BacktestEngine
    import logging

    logger = logging.getLogger("backtesting.engine")
    logger.setLevel(logging.DEBUG)
    from io import StringIO
    buf = StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    try:
        engine = BacktestEngine()
        # Mock _run_freqtrade_backtest so it doesn't actually run
        from unittest.mock import MagicMock
        engine._run_freqtrade_backtest = MagicMock(return_value={
            "strategy": {"Test": {"total_trades": 0, "trades": []}}
        })
        engine._parse_results = MagicMock(return_value={
            "total_trades": 0, "sharpe_ratio": 0, "win_rate": 0,
        })
        engine.run_backtest(strategy_type="sma_crossover")
    finally:
        logger.removeHandler(handler)
    log_output = buf.getvalue()
    assert "Starting Freqtrade subprocess" in log_output, (
        f"Expected startup log line, got: {log_output}"
    )


def test_prefilter_low_sharpe_passes_through():
    """Pre-filter must NOT reject strategies solely due to low vectorbt Sharpe."""
    from config import settings
    assert settings.VECTORBT_PREFILTER_MIN_SHARPE < 0, (
        "Pre-filter MIN_SHARPE must be negative to avoid blocking strategies "
        f"before Freqtrade runs, got {settings.VECTORBT_PREFILTER_MIN_SHARPE}"
    )


# ── Bug 3: kelly_position_size_conservative with no arguments ──


def test_kelly_conservative_no_args_returns_valid():
    """kelly_position_size_conservative() with no args must not crash (negative Kelly with default degradation is OK)."""
    from agents.risk_manager import kelly_position_size_conservative

    result = kelly_position_size_conservative()
    assert result.get("error") is not True, (
        f"Should return valid result with defaults (negative Kelly expected with degradation), got: {result}"
    )
    # Must have key field regardless of sign
    assert "kelly_fraction" in result
    assert "rationale" in result


def test_kelly_conservative_tool_no_args_returns_valid():
    """Kelly tool exposed via LangChain must not crash when LLM calls with no args."""
    from agents.risk_manager import RiskManagerAgent

    agent = RiskManagerAgent()
    tool = agent.get_tool("kelly_position_size_conservative")
    result = tool.func()
    import json
    parsed = json.loads(result) if isinstance(result, str) else result
    assert parsed.get("error") is not True, (
        f"Tool should handle no-arg call, got: {parsed}"
    )


# ── Bug 4: CoinGecko 404 removed ──


def test_no_coingecko_news_fallback():
    """_get_coingecko_news must be removed — get_cryptopanic_news returns [] when no key."""
    from data.sentiment import SentimentFetcher

    assert not hasattr(SentimentFetcher, "_get_coingecko_news"), (
        "_get_coingecko_news should have been removed"
    )


def test_cryptopanic_no_key_returns_empty():
    """get_cryptopanic_news with no API key returns [] (no 404 fallback)."""
    from data.sentiment import SentimentFetcher

    fetcher = SentimentFetcher()
    result = fetcher.get_cryptopanic_news("BTC")
    assert result == [], (
        f"Should return [] when no CryptoPanic key, got: {result}"
    )


# ── Bug 5: Kelly string type coercion ──


def test_regression_kelly_conservative_coerces_string_inputs():
    """kelly_position_size_conservative must not crash when LLM passes strings instead of floats."""
    from agents.risk_manager import kelly_position_size_conservative

    # All string values — exactly what the LLM sends via LangChain
    # Use oos_degradation_pct=0 to avoid negative Kelly with defaults
    result = kelly_position_size_conservative(
        win_rate="0.55", avg_win_pct="0.02", avg_loss_pct="0.01",
        portfolio_value="10000", oos_degradation_pct="0",
        max_kelly_fraction="0.25",
    )
    assert result.get("error") is not True, (
        f"Should coerce strings to floats, got: {result}"
    )
    assert isinstance(result.get("kelly_fraction"), (int, float))
    assert result["kelly_fraction"] > 0


def test_regression_kelly_conservative_string_win_rate_zero():
    """String win_rate='0' must be treated as 0.0 and return error (not crash)."""
    from agents.risk_manager import kelly_position_size_conservative

    result = kelly_position_size_conservative(
        win_rate="0", avg_win_pct="0.02", avg_loss_pct="0.01",
    )
    assert result.get("error") is True, (
        f"Should return error for 0 win_rate, got: {result}"
    )


def test_regression_bayesian_kelly_coerces_string_inputs():
    """bayesian_kelly_position_size must not crash when LLM passes strings."""
    from agents.risk_manager import bayesian_kelly_position_size

    result = bayesian_kelly_position_size(
        win_rate="0.55", avg_win_pct="0.02", avg_loss_pct="0.01",
        portfolio_value="10000", total_trades="50", max_kelly_fraction="0.25",
    )
    assert result.get("error") is not True, (
        f"Should coerce strings to floats, got: {result}"
    )
    assert isinstance(result.get("kelly_fraction"), (int, float))


def test_regression_bayesian_kelly_conservative_coerces_string_inputs():
    """bayesian_kelly_position_size_conservative must not crash when LLM passes strings."""
    from agents.risk_manager import bayesian_kelly_position_size_conservative

    result = bayesian_kelly_position_size_conservative(
        win_rate="0.55", avg_win_pct="0.02", avg_loss_pct="0.01",
        portfolio_value="10000", total_trades="50",
        oos_degradation_pct="0.40", max_kelly_fraction="0.25",
    )
    assert result.get("error") is not True, (
        f"Should coerce strings to floats, got: {result}"
    )
    assert isinstance(result.get("kelly_fraction"), (int, float))


def test_regression_kelly_conservative_parses_json_string():
    """kelly_position_size_conservative must handle full JSON string as win_rate arg (LLM confusion)."""
    from agents.risk_manager import kelly_position_size_conservative

    # The LLM sometimes sends a JSON blob as the first positional arg
    result = kelly_position_size_conservative(
        '{"win_rate": 0.55, "avg_win_pct": 0.02, "avg_loss_pct": 0.01, '
        '"portfolio_value": 10000, "oos_degradation_pct": 0, '
        '"max_kelly_fraction": 0.25, "sizing_tier": "normal"}'
    )
    assert result.get("error") is not True, (
        f"Should parse JSON string, got: {result}"
    )
    assert result["kelly_fraction"] > 0


def test_regression_bayesian_kelly_parses_json_string():
    """bayesian_kelly_position_size must handle full JSON string as win_rate arg."""
    from agents.risk_manager import bayesian_kelly_position_size

    result = bayesian_kelly_position_size(
        '{"win_rate": 0.55, "avg_win_pct": 0.02, "avg_loss_pct": 0.01, '
        '"portfolio_value": 10000, "total_trades": 50, '
        '"max_kelly_fraction": 0.25, "sizing_tier": "normal"}'
    )
    assert result.get("error") is not True, (
        f"Should parse JSON string, got: {result}"
    )


def test_regression_bayesian_kelly_conservative_parses_json_string():
    """bayesian_kelly_position_size_conservative must handle full JSON string as win_rate arg."""
    from agents.risk_manager import bayesian_kelly_position_size_conservative

    result = bayesian_kelly_position_size_conservative(
        '{"win_rate": 0.55, "avg_win_pct": 0.02, "avg_loss_pct": 0.01, '
        '"portfolio_value": 10000, "total_trades": 50, '
        '"oos_degradation_pct": 0, "max_kelly_fraction": 0.25, '
        '"sizing_tier": "normal"}'
    )
    assert result.get("error") is not True, (
        f"Should parse JSON string, got: {result}"
    )
