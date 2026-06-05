"""Tests for the autonomous research loop."""

import json
from datetime import datetime

from orchestration.autonomous_loop import AutonomousResearchLoop, AutonomousLoopState
from orchestration.research import ResearchGoal


class FakeVectorStore:
    def __init__(self):
        self._data = []
    def get_best_strategies(self, regime="", min_sharpe=0.0, k=5):
        return []
    def query_similar(self, query, k=5):
        return []
    def store_insight(self, text, metadata=None, doc_id=None):
        self._data.append({"text": text, "metadata": metadata})


class FakeOrchestrator:
    def __init__(self):
        self.last_goal = None
    def run_research_loop(self, goal, max_iterations=3, max_cycles=4):
        self.last_goal = goal
        return {"converged": True, "total_iterations": 1, "board_summary": "done"}


def test_loop_state_defaults():
    state = AutonomousLoopState()
    assert state.is_running is False
    assert state.is_paused is False
    assert state.total_cycles == 0
    assert state.consecutive_failures == 0


def test_loop_init():
    loop = AutonomousResearchLoop(orchestrator=None, interval_minutes=30)
    assert loop.state.is_running is False
    assert loop._interval_seconds >= 300  # MIN_INTERVAL_SECONDS
    loop.shutdown()
    assert loop._shutdown is True


def test_coverage_gaps_returns_dict():
    vs = FakeVectorStore()
    loop = AutonomousResearchLoop(orchestrator=FakeOrchestrator(), vector_store=vs, interval_minutes=60)
    import asyncio
    gaps = asyncio.run(loop._compute_coverage_gaps())
    assert isinstance(gaps, dict)
    for regime in ["strong_uptrend", "weak_trend", "ranging", "volatile", "low_liquidity"]:
        assert regime in gaps


def test_generate_goal_when_no_coverage():
    vs = FakeVectorStore()
    orchestrator = FakeOrchestrator()
    loop = AutonomousResearchLoop(orchestrator=orchestrator, vector_store=vs, interval_minutes=60)
    # Set current regime to one with no recommended strategies, so
    # experiments.jsonl cross-reference returns 0.0 even with real data.
    loop.state.last_regime = "low_liquidity"
    goal = loop._generate_next_goal()
    # Should run synchronously without await
    import asyncio
    goal = asyncio.run(loop._generate_next_goal())
    assert goal is not None
    assert goal.triggered_by == "coverage_gap"
    assert goal.priority_score == 1.0
    assert goal.strategy_type_hint == "sma_crossover"  # fallback when recommended list is empty
    assert len(goal.motivation) > 10


def test_research_goal_defaults():
    goal = ResearchGoal(
        regime="volatile",
        strategy_type_hint="breakout",
        motivation="Testing breakout in volatile market",
        priority_score=0.8,
        triggered_by="exploration",
    )
    assert goal.regime == "volatile"
    assert goal.strategy_type_hint == "breakout"
    assert 0.0 <= goal.priority_score <= 1.0
    assert goal.created_at is not None


def test_loop_can_be_paused():
    loop = AutonomousResearchLoop(orchestrator=None)
    assert loop.state.is_paused is False
    loop.pause()
    assert loop.state.is_paused is True
    loop.resume()
    assert loop.state.is_paused is False
