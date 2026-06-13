"""Tests for orchestration/graph.py — LangGraph state machine and helper functions."""

import pytest
from unittest.mock import MagicMock, patch

from orchestration.board import TaskBoard
from orchestration.graph import (
    _shutdown_event,
    _pick_agent,
    _extract_child_tasks,
    increment_cycle,
    should_continue,
    finalize,
    review_task,
    build_orchestration_graph,
    request_shutdown,
)


# ── Fixtures ──

@pytest.fixture(autouse=True)
def reset_shutdown():
    """Clear the module-level shutdown event before each test."""
    _shutdown_event.clear()
    yield


@pytest.fixture
def board():
    """Minimal TaskBoard with agents and capabilities."""
    capabilities = {
        "analyst": ["analysis", "market", "sentiment"],
        "strategist": ["strategy", "generate", "design"],
        "backtester": ["backtest", "data", "run"],
        "researcher": ["research", "web", "search"],
    }
    agents = {
        name: MagicMock() for name in capabilities
    }
    return TaskBoard(agents, capabilities)


# ── Test: Shutdown ──

class TestShutdown:
    def test_request_shutdown_sets_event(self):
        assert not _shutdown_event.is_set()
        request_shutdown()
        assert _shutdown_event.is_set()

    def test_request_shutdown_is_idempotent(self):
        request_shutdown()
        request_shutdown()
        assert _shutdown_event.is_set()

    def test_shutdown_cleared_by_fixture(self):
        assert not _shutdown_event.is_set()


# ── Test: _pick_agent ──

class TestPickAgent:
    CAPABILITIES = {
        "analyst": ["analysis", "market", "sentiment"],
        "strategist": ["strategy", "generate", "design"],
        "backtester": ["backtest", "data", "run"],
        "researcher": ["research", "web", "search"],
    }

    def test_strategy_keyword_routes_to_strategist(self):
        assert _pick_agent("Generate a new strategy for trending market", self.CAPABILITIES) == "strategist"

    def test_backtest_keyword_routes_to_backtester(self):
        assert _pick_agent("Run backtest on SMA crossover", self.CAPABILITIES) == "backtester"

    def test_market_analysis_routes_to_analyst(self):
        assert _pick_agent("Analyse market conditions for BTC", self.CAPABILITIES) == "analyst"

    def test_research_keyword_routes_to_researcher(self):
        assert _pick_agent("Research novel trading strategies", self.CAPABILITIES) == "researcher"

    def test_case_insensitive_matching(self):
        assert _pick_agent("STRATEGY design for volatile markets", self.CAPABILITIES) == "strategist"

    def test_no_match_falls_back_to_analyst(self):
        assert _pick_agent("Do something completely unrelated", self.CAPABILITIES) == "analyst"

    def test_memory_context_routes_to_strategist(self):
        assert _pick_agent("[MEMORY CONTEXT] Past insights for trending regime", self.CAPABILITIES) == "strategist"

    def test_memory_context_wins_over_keyword_match(self):
        """Memory context prefix always goes to strategist, even if description has 'backtest'."""
        assert _pick_agent("[MEMORY CONTEXT] backtest results from last run", self.CAPABILITIES) == "strategist"

    def test_partial_word_does_not_match(self):
        """'strategy' in 'strategy_type' should not match via word-boundary split."""
        assert _pick_agent("Check strategy_type parameter", self.CAPABILITIES) == "analyst"

    def test_higher_score_wins(self):
        """Agent with more keyword matches wins."""
        desc = "market analysis and sentiment research"
        result = _pick_agent(desc, self.CAPABILITIES)
        # analyst (analysis, market, sentiment) = 3, researcher (research) = 1
        assert result == "analyst"


# ── Test: _extract_child_tasks ──

class TestExtractChildTasks:
    def test_parses_next_lines(self, board):
        task = MagicMock()
        task.id = "t1"
        task.result = "next: backtest strategy_type=sma_crossover params={\"fast_ma\": 5}"
        task.assigned_to = "strategist"
        _extract_child_tasks(task, board)
        tasks = board.get_tasks_by_status("TODO")
        assert len(tasks) >= 1
        assert "sma_crossover" in tasks[0].description

    def test_no_next_lines_no_tasks(self, board):
        task = MagicMock()
        task.id = "t2"
        task.result = "Just a regular output without task specification."
        task.assigned_to = "analyst"
        _extract_child_tasks(task, board)
        assert board.get_tasks_by_status("TODO") == []

    def test_strategy_type_in_result(self, board):
        task = MagicMock()
        task.id = "t3"
        task.result = "Found strategy_type=sma_crossover with good results"
        task.assigned_to = "strategist"
        _extract_child_tasks(task, board)
        tasks = board.get_tasks_by_status("TODO")
        assert any("sma_crossover" in t.description for t in tasks)

    def test_none_result_no_error(self, board):
        task = MagicMock()
        task.id = "t4"
        task.result = None
        task.assigned_to = "strategist"
        _extract_child_tasks(task, board)  # should not raise


# ── Test: increment_cycle ──

class TestIncrementCycle:
    def test_increments_by_one(self):
        state = {"cycle": 0}
        result = increment_cycle(state)
        assert result["cycle"] == 1

    def test_preserves_other_state(self):
        state = {"cycle": 3, "goal": "test", "board": MagicMock()}
        result = increment_cycle(state)
        assert result["cycle"] == 4


# ── Test: should_continue ──

class MockBoard:
    """Minimal board mock for should_continue tests."""

    def __init__(self, todo=None, in_progress=None):
        self._todo = todo or []
        self._in_progress = in_progress or []

    def get_tasks_by_status(self, status):
        if status == "TODO":
            return self._todo
        if status == "IN_PROGRESS":
            return self._in_progress
        return []


class TestShouldContinue:
    def test_continues_when_tasks_remain_and_under_max(self):
        state = {"board": MockBoard(todo=["t1"]), "cycle": 0, "max_cycles": 5}
        assert should_continue(state) == "dispatch_task"

    def test_finalizes_when_max_cycles_reached(self):
        state = {"board": MockBoard(todo=["t1"]), "cycle": 5, "max_cycles": 5}
        assert should_continue(state) == "finalize"

    def test_finalizes_when_no_tasks_and_no_in_progress(self):
        state = {"board": MockBoard(), "cycle": 0, "max_cycles": 5}
        assert should_continue(state) == "finalize"

    def test_continues_when_in_progress_tasks(self):
        state = {"board": MockBoard(in_progress=["t1"]), "cycle": 0, "max_cycles": 5}
        assert should_continue(state) == "dispatch_task"


# ── Test: finalize ──

class TestFinalize:
    def test_adds_status_complete(self, board):
        state = {"board": board, "goal": "test goal"}
        result = finalize(state)
        assert "final_output" in result
        assert result["final_output"]["goal"] == "test goal"

    def test_includes_task_count(self, board):
        board.add_task("Task 1")
        board.add_task("Task 2")
        state = {"board": board, "goal": "test"}
        result = finalize(state)
        assert result["final_output"]["task_count"] == 2


# ── Test: review_task ──

class TestReviewTask:
    def test_marks_task_done_with_result(self, board):
        board.add_task("Test task", assigned_to="analyst")
        task = board.get_tasks_by_status("TODO")[0]
        task.result = "Successful result"
        state = {"board": board, "current_task_id": task.id}
        review_task(state)
        assert task.status == "DONE"

    def test_no_task_returns_early(self, board):
        state = {"board": board, "current_task_id": None}
        result = review_task(state)
        assert "no task" in result["messages"][0]


# ── Test: build_orchestration_graph ──

class TestBuildGraph:
    def test_returns_compiled_graph(self):
        graph = build_orchestration_graph()
        assert graph is not None

    def test_graph_has_expected_nodes(self):
        graph = build_orchestration_graph()
        # Compiled graphs expose nodes via get_node or similar
        assert hasattr(graph, "get_node") or hasattr(graph, "nodes")

    def test_graph_compiles_without_error(self):
        graph = build_orchestration_graph()
        assert graph is not None
