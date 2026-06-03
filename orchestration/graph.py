"""LangGraph state graph for the Hermes multi-agent orchestration loop."""

import logging
import threading
from typing import Any, Dict, List, Literal, Optional, TypedDict

from langgraph.graph import END, StateGraph
import re

from orchestration.board import TaskBoard

logger = logging.getLogger(__name__)

_shutdown_event = threading.Event()


def request_shutdown():
    """Call this on SIGINT/KeyboardInterrupt to stop the graph cleanly."""
    _shutdown_event.set()


# ── State type ──

class OrchestratorState(TypedDict):
    """Shared state for the orchestration graph."""
    goal: str
    board: TaskBoard
    current_task_id: Optional[str]
    current_agent_name: Optional[str]
    cycle: int
    max_cycles: int
    final_output: Optional[Dict[str, Any]]
    messages: List[str]
    curator_context: str
    research_specs: Optional[List[Dict[str, Any]]]


# ── Graph node functions ──

def load_context(state: OrchestratorState) -> Dict[str, Any]:
    """Inject relevant memories from the curator as context for tasks."""
    board = state["board"]
    goal = state["goal"]

    # If a curator agent is registered, use inject_context
    if "curator" in board._agents:
        curator = board._agents["curator"]
        try:
            context = curator.inject_context(goal, k=3)
        except Exception as exc:
            logger.warning("Curator context injection failed: %s", exc)
            context = ""
    else:
        context = ""

    # Prepend memory context to the first task description if non-empty
    if context and board.get_tasks_by_status("TODO"):
        first_task = board.get_tasks_by_status("TODO")[0]
        first_task.description = (
            f"{context}\n\n---\n\n{first_task.description}"
        )
        logger.info("Injected %d chars of memory context into first task", len(context))

    return {"curator_context": context, "messages": [f"CONTEXT: loaded {len(context)} chars"]}


# ── Graph node functions ──

def dispatch_task(state: OrchestratorState) -> Dict[str, Any]:
    """Pick the next TODO task and assign an agent.

    Injects current market regime context into task descriptions
    so agents (especially the strategist) can make regime-aware decisions.
    """
    board = state["board"]
    goal = state["goal"]
    cycle = state["cycle"]
    max_cycles = state["max_cycles"]

    if cycle >= max_cycles:
        logger.info("Max cycles (%d) reached, finishing.", max_cycles)
        return {"current_task_id": None, "current_agent_name": None}

    todo_tasks = board.get_tasks_by_status("TODO")
    if not todo_tasks:
        logger.info("No more TODO tasks.")
        return {"current_task_id": None, "current_agent_name": None}

    task = todo_tasks[0]
    agent_name = _pick_agent(task.description, board._agent_capabilities)

    # ── Regime-aware routing ──
    # Inject current regime context into task description for regime-sensitive agents
    if agent_name in ("strategist", "analyst", "risk_manager"):
        try:
            from data.fetcher import MarketDataFetcher
            from data.regime import MarketRegimeDetector, REGIME_STRATEGY_MAP
            from config import settings

            fetcher = MarketDataFetcher()
            df = fetcher.fetch_ohlcv(settings.SYMBOL, settings.TIMEFRAME, limit=250)
            if df is not None and len(df) > 200:
                detector = MarketRegimeDetector()
                snapshot = detector.classify_regime_snapshot(df)
                regime_note = (
                    f"\n\n[CURRENT REGIME: {snapshot.regime} (confidence={snapshot.confidence:.0%})]"
                    f"\nADX={snapshot.adx:.1f}, ATR%={snapshot.atr_pct:.2%}, "
                    f"SMA200 distance={snapshot.sma200_distance:.2%}"
                    f"\nRecommended strategies: {', '.join(snapshot.recommended_strategies)}"
                    f"\nDiscouraged strategies: {', '.join(snapshot.discouraged_strategies)}"
                )
                task.description += regime_note
                logger.info(
                    "Injected regime context (%s, conf=%.0f%%) into %s task",
                    snapshot.regime, snapshot.confidence, agent_name,
                )

                # Apply penalty for discouraged strategies
                for discouraged in snapshot.discouraged_strategies:
                    if discouraged.lower() in task.description.lower():
                        logger.warning(
                            "Task uses discouraged strategy '%s' for regime '%s'",
                            discouraged, snapshot.regime,
                        )
                        task.description += (
                            f"\n[WARNING: '{discouraged}' is discouraged in {snapshot.regime} regime. "
                            f"Consider switching to: {', '.join(snapshot.recommended_strategies)}]"
                        )
        except Exception as exc:
            logger.debug("Regime injection skipped: %s", exc)

    task.status = "IN_PROGRESS"
    task.assigned_to = agent_name

    logger.info("[cycle %d] Dispatched '%s' to agent '%s'", cycle, task.description[:50], agent_name)
    return {
        "current_task_id": task.id,
        "current_agent_name": agent_name,
        "messages": [f"DISPATCH: {task.description[:80]} -> {agent_name}"],
    }


def execute_agent(state: OrchestratorState) -> Dict[str, Any]:
    """Run the assigned agent on the current task."""
    if _shutdown_event.is_set():
        logger.info("Shutdown requested — skipping agent execution")
        board = state["board"]
        task_id = state["current_task_id"]
        if task_id and task_id in board.tasks:
            board.tasks[task_id].status = "DONE"
            board.tasks[task_id].result = "Skipped: shutdown requested"
        return {"messages": ["EXECUTE: shutdown"]}
    board = state["board"]
    task_id = state["current_task_id"]
    agent_name = state["current_agent_name"]
    agents = board._agents  # injected reference

    if not task_id or not agent_name or agent_name not in agents:
        return {"messages": ["EXECUTE: no valid agent/task"]}

    task = board.tasks[task_id]
    agent = agents[agent_name]

    try:
        result = agent.run(task.description)
        output = result.get("output", "")
        task.result = output
        logger.info("[%s] completed: %.100s", agent_name, output[:100].replace("\n", " "))
    except Exception as exc:
        logger.exception("Agent %s failed: %s", agent_name, exc)
        task.result = f"Error: {exc}"

    return {"messages": [f"EXECUTE: {agent_name} finished task {task_id}"]}


def review_task(state: OrchestratorState) -> Dict[str, Any]:
    """Review task result — always pass in the basic version."""
    board = state["board"]
    task_id = state["current_task_id"]

    if not task_id:
        return {"messages": ["REVIEW: no task to review"]}

    task = board.tasks[task_id]

    # Simple evaluation: if the result is non-empty and has no "Error:", mark DONE
    if task.result and not str(task.result).startswith("Error:"):
        task.status = "DONE"
    else:
        # Mark as failed but still DONE to avoid infinite loops
        task.status = "DONE"
        logger.warning("Task %s finished with issues: %.80s", task_id, str(task.result)[:80])

    _extract_child_tasks(task, board)

    return {
        "messages": [f"REVIEW: task {task_id} -> {task.status}"],
    }


def should_continue(state: OrchestratorState) -> Literal["dispatch_task", "finalize"]:
    """Decide whether to loop or finalise."""
    board = state["board"]
    cycle = state["cycle"]

    todo = board.get_tasks_by_status("TODO")
    in_progress = board.get_tasks_by_status("IN_PROGRESS")

    if not todo and not in_progress:
        return "finalize"

    next_cycle = cycle + 1
    if next_cycle >= state["max_cycles"]:
        return "finalize"

    return "dispatch_task"

def increment_cycle(state: OrchestratorState) -> Dict[str, Any]:
    """Increment the cycle counter before looping back."""
    return {"cycle": state["cycle"] + 1}


def finalize(state: OrchestratorState) -> Dict[str, Any]:
    """Build the final summary output."""
    board = state["board"]
    goal = state["goal"]

    done_tasks = board.get_tasks_by_status("DONE")
    output = {
        "goal": goal,
        "board_summary": board.summary(),
        "task_count": len(board.tasks),
        "results": {t.id: t.result for t in done_tasks if t.result},
        "strategies": [
            str(t.result) for t in done_tasks
            if "strategy" in t.description.lower() and t.result
        ],
    }

    logger.info("=== Research complete: %s ===", goal)
    return {"final_output": output, "messages": ["FINALIZE: done"]}


# ── Helper functions ──

def _pick_agent(description: str, capabilities: Dict[str, List[str]]) -> str:
    """Score each agent by how many of their keywords appear in the description.
    Splits description into words so 'strategy' in 'strategy_type' is not a match."""
    words = set(description.lower().split())
    best_agent = "analyst"
    best_score = 0
    for agent_name, keywords in capabilities.items():
        score = sum(1 for kw in keywords if kw in words)
        if score > best_score:
            best_score = score
            best_agent = agent_name
    return best_agent



def _extract_child_tasks(parent: "Task", board: TaskBoard):
    """Extract follow-up tasks from agent output.
    Looks for explicit 'next: ' lines, then falls back to detecting
    strategy_type= references anywhere in the output."""
    result_text = str(parent.result) if parent.result else ""
    found_any = False
    for line in result_text.split("\n"):
        if line.strip().startswith("next: "):
            desc = line.strip().replace("next: ", "")
            board.add_task(description=desc, parent_id=parent.id, metadata={"auto": True})
            found_any = True
    if not found_any and "strategy_type=" in result_text:
        m = re.search(r'strategy_type[=:]\s*(\w+)', result_text)
        if m:
            board.add_task(description="backtest strategy_type=" + m.group(1),
                           parent_id=parent.id, metadata={"auto": True})
            found_any = True
    if not found_any and getattr(parent, "assigned_to", None) == "strategist" and parent.result:
        lowered = result_text.lower()
        # Normalize hyphens and spaces to underscores for matching
        normalized = lowered.replace("-", "_").replace(" ", "_")
        for kw in ["multi_timeframe", "sma_crossover", "macd_crossover", "rsi_oversold",
                    "bollinger_bands", "combined_sma_rsi", "momentum", "breakout",
                    "mean_reversion", "volatility_squeeze", "sentiment_driven"]:
            if kw in lowered or kw in normalized:
                board.add_task(description="backtest strategy_type=" + kw,
                               parent_id=parent.id, metadata={"auto": True})
                break




def build_orchestration_graph() -> StateGraph:
    """Build and compile the LangGraph state graph."""
    workflow = StateGraph(OrchestratorState)

    workflow.add_node("load_context", load_context)
    workflow.add_node("dispatch_task", dispatch_task)
    workflow.add_node("execute_agent", execute_agent)
    workflow.add_node("review_task", review_task)
    workflow.add_node("increment_cycle", increment_cycle)
    workflow.add_node("finalize", finalize)

    # Entry point — load memory context first
    workflow.set_entry_point("load_context")
    workflow.add_edge("load_context", "dispatch_task")

    # Main loop: dispatch -> execute -> review -> (increment_cycle -> dispatch | finalize)
    workflow.add_edge("dispatch_task", "execute_agent")
    workflow.add_edge("execute_agent", "review_task")
    workflow.add_conditional_edges(
        "review_task",
        should_continue,
        {"dispatch_task": "increment_cycle", "finalize": "finalize"},
    )
    workflow.add_edge("increment_cycle", "dispatch_task")
    workflow.add_edge("finalize", END)

    return workflow.compile()