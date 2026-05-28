"""FastAPI application for real-time Web UI backend."""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from api.event_bus import EventBus
from memory.vector_store import VectorStore
from orchestration.factory import make_orchestrator

logger = logging.getLogger(__name__)

app = FastAPI(title="Crypto Agent Bot")

# ── In-memory run state ──

_active_runs: Dict[str, Dict[str, Any]] = {}
_buses: Dict[str, EventBus] = {}

# ── Pydantic models ──


class RunRequest(BaseModel):
    goal: str
    max_cycles: int = 5
    max_iterations: int = 1


class RunResponse(BaseModel):
    run_id: str
    status: str


# ── REST endpoints ──


@app.get("/api/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.get("/")
async def serve_ui():
    """Serve the single-file React frontend."""
    from pathlib import Path
    ui_path = Path(__file__).parent.parent / "ui" / "index.html"
    if ui_path.exists():
        return HTMLResponse(ui_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>UI not found</h1>", status_code=404)


@app.post("/api/run")
async def start_run(req: RunRequest) -> RunResponse:
    """Start a research run in the background."""
    run_id = uuid.uuid4().hex[:8]
    bus = EventBus()
    _buses[run_id] = bus
    _active_runs[run_id] = {"status": "running", "goal": req.goal, "events": []}

    # Fire background task
    asyncio.create_task(_run_orchestration(run_id, req.goal, req.max_cycles, req.max_iterations, bus))

    return RunResponse(run_id=run_id, status="started")


@app.get("/api/history")
async def get_history(limit: int = 20):
    """Return recent ChromaDB records."""
    vs = VectorStore()
    if vs.count() == 0:
        return {"records": []}
    # Query broadly
    results = vs.query_similar("research strategy backtest", k=limit)
    records = []
    for r in results:
        records.append({
            "text": r["text"][:300],
            "metadata": r["metadata"],
        })
    return {"records": records}


@app.get("/api/strategies")
async def get_strategies():
    """Return strategist iteration records from ChromaDB."""
    vs = VectorStore()
    if vs.count() == 0:
        return {"strategies": []}
    results = vs.query_similar("kept_record discarded_record", k=50)
    strategies = []
    for r in results:
        meta = r["metadata"]
        if meta.get("type", "").endswith("_record"):
            strategies.append({
                "verdict": meta.get("type", "").replace("_record", ""),
                "reason": meta.get("reason", ""),
                "text": r["text"][:300],
            })
    return {"strategies": strategies}


# ── WebSocket ──


@app.websocket("/ws/run/{run_id}")
async def ws_run_events(websocket: WebSocket, run_id: str):
    await websocket.accept()
    # Wait up to 15 seconds for the bus to be created (POST handler may still be processing)
    bus = None
    for _ in range(30):
        bus = _buses.get(run_id)
        if bus:
            break
        await asyncio.sleep(0.5)
    if not bus:
        await websocket.send_json({"type": "error", "payload": {"message": "Run not found"}, "timestamp": datetime.utcnow().isoformat()})
        await websocket.close()
        return

    try:
        async for event in bus.subscribe():
            try:
                await websocket.send_json(event)
            except Exception:
                break
    except WebSocketDisconnect:
        pass


# ── Orchestration runner ──


async def _run_orchestration(run_id: str, goal: str, max_cycles: int, max_iterations: int, bus: EventBus):
    """Run orchestrator in a thread, publishing events to the bus."""
    from orchestration.hermes import HermesOrchestrator

    orchestrator = make_orchestrator()

    # Patch: inject event publishing into the orchestrator
    _patch_orchestrator(orchestrator, bus, run_id)

    await bus.publish("run_start", {"goal": goal, "max_cycles": max_cycles, "max_iterations": max_iterations})

    try:
        # Run in executor to avoid blocking the event loop
        loop = asyncio.get_running_loop()

        if max_iterations > 1:
            result = await loop.run_in_executor(
                None,
                lambda: orchestrator.run_research_loop(goal, max_cycles=max_cycles, max_iterations=max_iterations),
            )
        else:
            result = await loop.run_in_executor(
                None,
                lambda: orchestrator.run_research_goal(goal, max_cycles=max_cycles),
            )

        await bus.publish("run_complete", {
            "goal": goal,
            "board_summary": result.get("board_summary", ""),
            "total_iterations": result.get("total_iterations", 1),
            "converged": result.get("converged", False),
        })

        _active_runs[run_id] = {"status": "done", "goal": goal, "result": result}
    except Exception as exc:
        logger.exception("Run failed")
        await bus.publish("run_error", {"message": str(exc)})
        _active_runs[run_id] = {"status": "error", "goal": goal, "error": str(exc)}


def _patch_orchestrator(orchestrator: "HermesOrchestrator", bus: EventBus, run_id: str):
    """Patch orchestrator methods to emit events to the bus.

    Uses a callback function that schedules coroutines on the event loop.
    """
    orig_methods = {}

    def emit(event_type: str, payload: Dict[str, Any]):
        try:
            loop = asyncio.get_running_loop()
            asyncio.run_coroutine_threadsafe(
                bus.publish(event_type, payload), loop
            )
        except RuntimeError:
            pass  # No event loop available

    # Wrap _generate_hypothesis
    if hasattr(orchestrator, "_generate_hypothesis"):
        orig_gen = orchestrator._generate_hypothesis

        def patched_generate(goal, past_iterations, iter_num, max_iterations):
            hypothesis = orig_gen(goal, past_iterations, iter_num, max_iterations)
            emit("hypothesis", {
                "hypothesis": hypothesis,
                "iteration": iter_num,
                "max_iterations": max_iterations,
            })
            return hypothesis

        orchestrator._generate_hypothesis = patched_generate

    # Wrap _critique_iteration
    if hasattr(orchestrator, "_critique_iteration"):
        orig_crit = orchestrator._critique_iteration

        def patched_critique(output, goal, hypothesis):
            critique = orig_crit(output, goal, hypothesis)
            emit("critique", {
                "critique": critique,
                "hypothesis": hypothesis,
            })
            return critique

        orchestrator._critique_iteration = patched_critique

    # Wrap _run_research_goal to emit agent events
    if hasattr(orchestrator, "_run_research_goal"):
        orig_run = orchestrator._run_research_goal

        def patched_run(goal, max_cycles=5, hypothesis="", iteration=1):
            emit("iteration_start", {"iteration": iteration, "goal": goal[:200]})
            result = orig_run(goal, max_cycles=max_cycles, hypothesis=hypothesis, iteration=iteration)
            # Emit events for each board task
            done_tasks = orchestrator.board.get_tasks_by_status("DONE") if hasattr(orchestrator, "board") else []
            for task in done_tasks:
                if task.result:
                    emit("task_done", {
                        "task_id": task.id,
                        "agent": task.assigned_to,
                        "description": task.description[:200],
                        "result": str(task.result)[:500],
                    })
            emit("iteration_result", {
                "iteration": iteration,
                "task_count": len(done_tasks),
            })
            return result

        orchestrator._run_research_goal = patched_run