"""LangGraph state graph for the Hermes multi-agent orchestration loop."""

import logging
from typing import Any, Dict, List, Literal, Optional, TypedDict

from langgraph.graph import END, StateGraph

from orchestration.board import TaskBoard

logger = logging.getLogger(__name__)


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
    """Pick the next TODO task and assign an agent."""
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
    task.status = "IN_PROGRESS"
    task.assigned_to = agent_name

    logger.info("[cycle %d] Dispatched '%s' to agent '%s'", cycle, task.description[:50], agent_name)
    return {
        "current_task_id": task.id,
        "current_agent_name": agent_name,
        "messages": [f"DISPATCH: {task.description} -> {agent_name}"],
    }


def execute_agent(state: OrchestratorState) -> Dict[str, Any]:
    """Run the assigned agent on the current task."""
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
    """Score each agent by how many of their keywords appear in the description."""
    desc_lower = description.lower()
    best_agent = "analyst"
    best_score = 0
    for agent_name, keywords in capabilities.items():
        score = sum(1 for kw in keywords if kw in desc_lower)
        if score > best_score:
            best_score = score
            best_agent = agent_name
    return best_agent


def _extract_child_tasks(parent: "Task", board: TaskBoard):
    """Extract follow-up tasks from agent output marked with 'next: '."""
    if parent.result and "next: " in str(parent.result):
        lines = str(parent.result).split("\n")
        for line in lines:
            if line.strip().startswith("next: "):
                desc = line.strip().replace("next: ", "")
                board.add_task(description=desc, parent_id=parent.id, metadata={"auto": True})


# ── Graph builder ──

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