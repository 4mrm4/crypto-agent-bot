"""Tests for orchestration/hermes.py — HermesOrchestrator event system, metric extraction."""

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from orchestration.hermes import HermesOrchestrator


# ── Fixtures ──

@pytest.fixture
def mock_agents():
    return {
        "analyst": MagicMock(),
        "strategist": MagicMock(),
        "backtester": MagicMock(),
        "curator": MagicMock(),
        "researcher": MagicMock(),
        "risk_manager": MagicMock(),
        "iteration_tracker": MagicMock(),
    }


@pytest.fixture
def orchestrator(mock_agents):
    return HermesOrchestrator(agents=mock_agents)


@pytest.fixture
def orchestrator_with_callback(mock_agents):
    callback = MagicMock()
    return HermesOrchestrator(agents=mock_agents, event_callback=callback)


# ── Test: Constructor ──

class TestHermesInit:
    def test_default_init(self, mock_agents):
        h = HermesOrchestrator(agents=mock_agents)
        assert h._event_callback is None
        assert h.board is not None
        assert h.agents == mock_agents

    def test_init_with_event_callback(self, mock_agents):
        cb = MagicMock()
        h = HermesOrchestrator(agents=mock_agents, event_callback=cb)
        assert h._event_callback is cb

    def test_init_with_circuit_breaker(self, mock_agents):
        cb_state = MagicMock()
        h = HermesOrchestrator(agents=mock_agents, circuit_breaker=cb_state)
        assert h._circuit_breaker is cb_state

    def test_init_default_circuit_breaker(self, mock_agents):
        h = HermesOrchestrator(agents=mock_agents)
        assert h._circuit_breaker is not None


# ── Test: Event callback ──

class TestEventCallback:
    def test_on_event_calls_callback(self, orchestrator_with_callback):
        cb = orchestrator_with_callback._event_callback
        orchestrator_with_callback.on_event("test_event", {"key": "value"})
        cb.assert_called_once_with("test_event", {"key": "value"})

    def test_on_event_no_callback_does_not_raise(self, orchestrator):
        orchestrator.on_event("test_event", {"key": "value"})  # should not raise

    def test_on_event_callback_exception_caught(self, orchestrator_with_callback):
        cb = orchestrator_with_callback._event_callback
        cb.side_effect = RuntimeError("callback failed")
        orchestrator_with_callback.on_event("test_event", {"key": "value"})  # should not raise

    def test_on_event_logs_callback_exception(self, mock_agents, caplog):
        def failing_cb(_, __):
            raise RuntimeError("callback failed")
        h = HermesOrchestrator(agents=mock_agents, event_callback=failing_cb)
        with caplog.at_level(logging.WARNING):
            h.on_event("test_event", {"key": "value"})
            assert "Event callback failed" in caplog.text


# ── Test: _emit ──

class TestEmit:
    def test_emit_passes_kwargs_as_dict(self, orchestrator_with_callback):
        cb = orchestrator_with_callback._event_callback
        orchestrator_with_callback._emit("hypothesis", hypothesis="test_hyp", iteration=1)
        cb.assert_called_once()
        args, kwargs = cb.call_args
        event_type, data = args
        assert event_type == "hypothesis"
        assert data == {"hypothesis": "test_hyp", "iteration": 1}

    def test_emit_with_empty_data(self, orchestrator_with_callback):
        cb = orchestrator_with_callback._event_callback
        orchestrator_with_callback._emit("iteration_start")
        cb.assert_called_once_with("iteration_start", {})


# ── Test: _extract_metrics (Level 1 — output dict keys) ──

class TestExtractMetrics:
    def test_returns_top_level_metrics(self, orchestrator):
        output = {
            "sharpe_ratio": 1.5,
            "win_rate": 0.6,
            "max_drawdown": -0.1,
            "profit_ratio": 2.0,
            "total_trades": 25,
        }
        metrics = orchestrator._extract_metrics(output)
        assert metrics["sharpe_ratio"] == 1.5
        assert metrics["win_rate"] == 0.6
        assert metrics["total_trades"] == 25

    def test_zero_trades_falls_to_level_2(self, orchestrator):
        """When total_trades is 0, _extract_metrics should check backtester history."""
        output = {"sharpe_ratio": 0, "total_trades": 0}
        backtester = orchestrator.agents["backtester"]
        backtester._iteration_history = [
            {"metrics": {"sharpe_ratio": 1.2, "win_rate": 0.55, "total_trades": 10}}
        ]
        metrics = orchestrator._extract_metrics(output)
        assert metrics["total_trades"] == 10
        assert metrics["sharpe_ratio"] == 1.2

    def test_no_backtester_history_returns_zeros(self, orchestrator):
        output = {"sharpe_ratio": 0, "total_trades": 0}
        orchestrator.agents["backtester"]._iteration_history = []
        metrics = orchestrator._extract_metrics(output)
        assert metrics["total_trades"] == 0

    def test_no_backtester_agent_returns_zeros(self, orchestrator):
        del orchestrator.agents["backtester"]
        output = {"sharpe_ratio": 0, "total_trades": 0}
        metrics = orchestrator._extract_metrics(output)
        assert metrics["total_trades"] == 0

    def test_backtester_no_history_attr(self, orchestrator):
        output = {"sharpe_ratio": 0, "total_trades": 0}
        del orchestrator.agents["backtester"]._iteration_history
        metrics = orchestrator._extract_metrics(output)
        assert metrics["total_trades"] == 0

    def test_backtester_history_filtered_by_trades(self, orchestrator):
        """Records with total_trades=0 should be skipped."""
        output = {"sharpe_ratio": 0, "total_trades": 0}
        backtester = orchestrator.agents["backtester"]
        backtester._iteration_history = [
            {"metrics": {"sharpe_ratio": 0, "total_trades": 0}},
            {"metrics": {"sharpe_ratio": 1.5, "total_trades": 15}},
        ]
        metrics = orchestrator._extract_metrics(output)
        assert metrics["total_trades"] == 15
        assert metrics["sharpe_ratio"] == 1.5

    def test_falls_to_level_3_experiments_jsonl(self, orchestrator, tmp_path):
        """When all else fails, read from experiments.jsonl."""
        output = {
            "sharpe_ratio": 0, "total_trades": 0,
            "goal": "Coverage gap: ... Researching multi_timeframe.",
        }
        orchestrator.agents["backtester"]._iteration_history = []
        exp_file = tmp_path / "experiments.jsonl"
        exp_file.write_text(
            json.dumps({"strategy_type": "multi_timeframe", "sharpe": 2.0,
                        "win_rate": 0.7, "max_drawdown": -0.05, "total_trades": 30})
        )
        with patch.object(Path, "exists", return_value=True), \
             patch("orchestration.hermes.Path.open", exp_file.open):
            with patch("orchestration.hermes.Path", return_value=exp_file):
                metrics = orchestrator._extract_metrics(output)
                assert metrics["total_trades"] == 30
                assert metrics["sharpe_ratio"] == 2.0

    def test_level_3_jsonl_missing_no_crash(self, orchestrator):
        """Missing experiments.jsonl returns empty dict, no crash."""
        output = {"sharpe_ratio": 0, "total_trades": 0, "goal": "Researching sma."}
        orchestrator.agents["backtester"]._iteration_history = []
        with patch("pathlib.Path.exists", return_value=False):
            metrics = orchestrator._extract_metrics(output)
            assert metrics["total_trades"] == 0


# ── Test: _extract_strategy_id ──

class TestExtractStrategyId:
    def test_extracts_from_board_done_tasks(self, orchestrator):
        orchestrator.board.add_task("test strategy", assigned_to="strategist")
        task = orchestrator.board.get_tasks_by_status("TODO")[0]
        task.status = "DONE"
        task.result = "Strategy [a1b2c3d4] performed well"
        strategy_id = orchestrator._extract_strategy_id({})
        assert strategy_id == "a1b2c3d4"


# ── Test: _build_summary ──

class TestBuildSummary:
    def test_returns_dict_with_goal(self, orchestrator):
        orchestrator.board.add_task("task 1")
        result = orchestrator._build_summary("test goal")
        assert result["goal"] == "test goal"
        assert "board_summary" in result
        assert "task_count" in result

    def test_task_count_reflects_board(self, orchestrator):
        orchestrator.board.add_task("task 1")
        orchestrator.board.add_task("task 2")
        result = orchestrator._build_summary("test")
        assert result["task_count"] == 2


# ── Test: Native emit in _generate_hypothesis ──

class TestGenerateHypothesisEmit:
    def test_emits_hypothesis_on_first_iteration(self, orchestrator_with_callback):
        """Simulates the first iteration of _generate_hypothesis by calling _emit directly."""
        h = orchestrator_with_callback
        h._emit("hypothesis", hypothesis="Test strategy with SMA crossover",
                iteration=1, max_iterations=5)
        cb = h._event_callback
        cb.assert_called_once()
        args = cb.call_args[0]
        assert args[0] == "hypothesis"
        assert args[1]["iteration"] == 1
        assert args[1]["max_iterations"] == 5
        assert "SMA" in args[1]["hypothesis"]


# ── Test: Native emit in _critique_iteration ──

class TestCritiqueIterationEmit:
    def test_emits_critique(self, orchestrator_with_callback):
        h = orchestrator_with_callback
        h._emit("critique", critique="Need better risk management",
                hypothesis="Test hypothesis")
        cb = h._event_callback
        cb.assert_called_once()
        args = cb.call_args[0]
        assert args[0] == "critique"
        assert "risk" in args[1]["critique"]


# ── Test: _run_research_goal emit points ──

class TestResearchGoalEmit:
    def test_emits_iteration_start(self, orchestrator_with_callback):
        h = orchestrator_with_callback
        h._emit("iteration_start", iteration=1, goal="test goal")
        cb = h._event_callback
        cb.assert_called_once_with("iteration_start", {"iteration": 1, "goal": "test goal"})

    def test_emits_task_done(self, orchestrator_with_callback):
        """Test task_done emission pattern from _run_research_goal."""
        h = orchestrator_with_callback
        h._emit("task_done", task_id="t1", agent="strategist",
                description="Generate strategy", result="SMA crossover strategy")
        cb = h._event_callback
        cb.assert_called_once()
        args = cb.call_args[0]
        assert args[0] == "task_done"

    def test_emits_iteration_result(self, orchestrator_with_callback):
        h = orchestrator_with_callback
        h._emit("iteration_result", iteration=1, task_count=3, metrics={"sharpe": 1.5})
        cb = h._event_callback
        cb.assert_called_once()
        args = cb.call_args[0]
        assert args[0] == "iteration_result"
        assert args[1]["iteration"] == 1
        assert args[1]["task_count"] == 3

    def test_emits_sentiment(self, orchestrator_with_callback):
        h = orchestrator_with_callback
        h._emit("sentiment", fear_greed={"value": 50}, bias="neutral")
        cb = h._event_callback
        cb.assert_called_once()
        assert cb.call_args[0][0] == "sentiment"
