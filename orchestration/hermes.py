"""Hermes-inspired multi-agent orchestrator with a Kanban task board."""

import logging
from typing import Any, Dict, List

from orchestration.board import TaskBoard
from orchestration.graph import build_orchestration_graph

logger = logging.getLogger(__name__)


class HermesOrchestrator:
    """Multi-agent orchestrator using a Kanban board and a LangGraph loop."""

    def __init__(self, agents: Dict[str, Any]):
        self._agent_capabilities: Dict[str, List[str]] = {
            "analyst": ["analysis", "market_research", "sentiment", "analyse", "market"],
            "strategist": ["strategy", "strategies", "backtest", "backtesting", "optimization", "optimise"],
            "risk_manager": ["risk", "assessment", "position_sizing"],
            "curator": ["memory", "context", "history"],
        }
        self.agents = agents
        self.board = TaskBoard(agents, self._agent_capabilities)
        self._graph = build_orchestration_graph()

    def run_research_goal(self, goal: str, max_cycles: int = 5) -> Dict[str, Any]:
        """Execute a full research lifecycle using the LangGraph state graph."""
        logger.info("=== Research goal: %s ===", goal)

        # Reset board for this run
        self.board = TaskBoard(self.agents, self._agent_capabilities)

        # Inject memory context if a curator agent is registered
        curator = self.agents.get("curator")
        if curator and hasattr(curator, "inject_context"):
            context = curator.inject_context(goal, k=3)
            if context:
                self.board.add_task(
                    description=f"Review past research context for: {goal}\nPast insights:\n{context}",
                    assigned_to="curator",
                )
                logger.info("Injected past insights from memory.")

        # Create initial tasks
        self.board.add_task(f"Analyse market conditions for: {goal}", assigned_to="analyst")
        self.board.add_task(
            f"Generate and backtest strategies for: {goal}", assigned_to="strategist"
        )
        self.board.add_task(
            f"Assess risk for strategies targeting: {goal}", assigned_to="risk_manager"
        )

        # Run the LangGraph
        initial_state = {
            "goal": goal,
            "board": self.board,
            "current_task_id": None,
            "current_agent_name": None,
            "cycle": 0,
            "max_cycles": max_cycles,
            "final_output": None,
            "messages": [],
            "curator_context": "",
        }

        final_state = self._graph.invoke(initial_state)

        output = final_state.get("final_output", self._build_summary(goal))

        # Store results in memory for future runs
        if curator:
            for tid, result_text in output.get("results", {}).items():
                if result_text:
                    curator.store_result(goal, str(result_text)[:500], {"task_id": tid})
            summary = output.get("board_summary", "")
            if summary:
                curator.store_result(goal, f"Board summary: {summary}", {"type": "summary"})
            logger.info("Stored goal results in memory.")

        logger.info("=== Research complete ===")
        return output

    def _build_summary(self, goal: str) -> Dict[str, Any]:
        """Assemble the final output dict from the board (fallback)."""
        done_tasks = self.board.get_tasks_by_status("DONE")
        return {
            "goal": goal,
            "board_summary": self.board.summary(),
            "task_count": len(self.board.tasks),
            "results": {t.id: t.result for t in done_tasks if t.result},
            "strategies": [
                str(t.result) for t in done_tasks
                if "strategy" in t.description.lower() and t.result
            ],
        }