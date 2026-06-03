"""Regression tests for the research loop — strategist generates, backtester backtests."""

from agents.strategist import STRATEGIST_SYSTEM_PROMPT
from agents.backtester import BACKTESTER_SYSTEM_PROMPT


class TestStrategistPromptFormat:
    """The strategist must output 'next: ' lines for the task extractor."""

    def test_prompt_contains_next_format(self):
        assert "next: backtest strategy_type=" in STRATEGIST_SYSTEM_PROMPT

    def test_prompt_contains_example(self):
        assert "sma_crossover" in STRATEGIST_SYSTEM_PROMPT
        assert "next: backtest strategy_type=sma_crossover" in STRATEGIST_SYSTEM_PROMPT

    def test_prompt_mentions_backtester(self):
        assert "backtest" in STRATEGIST_SYSTEM_PROMPT.lower()


class TestBacktesterPromptFormat:
    """The backtester prompt must mention it can run backtests."""

    def test_prompt_contains_run_backtest(self):
        assert "run_backtest" in BACKTESTER_SYSTEM_PROMPT


class TestTaskCreation:
    """Test that _extract_child_tasks creates tasks from 'next: ' lines."""

    def test_extract_creates_task(self):
        from orchestration.graph import _extract_child_tasks
        from orchestration.board import TaskBoard
        board = TaskBoard(agents={}, capabilities={})

        # Simulate a parent task whose result has a "next: " line
        class FakeTask:
            id = "parent123"
            result = (
                "Generated strategy: sma_crossover with fast_ma=10, slow_ma=30\n"
                "next: backtest strategy_type=sma_crossover params={\"fast_ma\": 10}\n"
                "next: record this result\n"
            )
            def __init__(self):
                self.description = "test"
                self.status = "TODO"
                self.assigned_to = None
                self.parent_id = None
                self.metadata = {}
                self.created_at = None

        task = FakeTask()
        _extract_child_tasks(task, board)

        todos = board.get_tasks_by_status("TODO")
        assert len(todos) >= 2, f"Expected at least 2 child tasks, got {len(todos)}"

        descriptions = [t.description for t in todos]
        backtest_tasks = [d for d in descriptions if "backtest" in d]
        assert len(backtest_tasks) >= 1, \
            f"No backtest task created. Tasks: {descriptions}"

    def test_extract_skips_without_next(self):
        from orchestration.graph import _extract_child_tasks
        from orchestration.board import TaskBoard
        board = TaskBoard(agents={}, capabilities={})

        class FakeTask:
            id = "parent456"
            result = "This result has no next: prefix in it."
            def __init__(self):
                self.description = "test"
                self.status = "TODO"
                self.assigned_to = None
                self.parent_id = None
                self.metadata = {}
                self.created_at = None

        task = FakeTask()
        before = len(board.get_tasks_by_status("TODO"))
        _extract_child_tasks(task, board)
        after = len(board.get_tasks_by_status("TODO"))
        assert after == before, "Should not create tasks without 'next: '"


class TestKeywordRouting:
    """Verify backtester keywords route correctly to backtester."""

    def test_backtest_routes_to_backtester(self):
        from orchestration.graph import _pick_agent
        capabilities = {
            "strategist": ["strategy", "generate", "concept", "design"],
            "backtester": ["backtest", "backtesting", "hyperopt", "run"],
            "analyst": ["analysis", "market"],
        }
        agent = _pick_agent("backtest strategy_type=sma_crossover params=...", capabilities)
        assert agent == "backtester", f"Expected backtester, got {agent}"

    def test_record_routes_to_iteration_tracker(self):
        from orchestration.graph import _pick_agent
        capabilities = {
            "strategist": ["strategy", "generate"],
            "iteration_tracker": ["record", "store", "iteration"],
            "analyst": ["analysis", "market"],
        }
        agent = _pick_agent("record this result", capabilities)
        assert agent == "iteration_tracker", f"Expected iteration_tracker, got {agent}"


class TestHermesTaskDescription:
    """Verify the research loop creates the right task description for strategist."""

    def test_strategist_not_tasked_with_backtest(self):
        from orchestration.hermes import HermesOrchestrator
        import inspect
        src = inspect.getsource(HermesOrchestrator._run_research_goal)
        # The strategist should only be asked to GENERATE, not backtest
        assert "backtest strategies for" not in src, \
            "Strategist should not be tasked with backtesting"
        assert "Generate strategies for" in src, \
            "Strategist should be tasked with generating only"
