"""FastAPI application for real-time Web UI backend."""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from api.event_bus import EventBus
from data.api_health import APIHealthTracker
from memory.vector_store import VectorStore
from orchestration.factory import make_orchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-20s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Crypto Agent Bot")

# ── In-memory run state ──

_active_runs: Dict[str, Dict[str, Any]] = {}
_buses: Dict[str, EventBus] = {}
_startup_tasks: Dict[str, Dict[str, Any]] = {
    "market_data_stream": {"status": "not_started", "error": None},
    "autonomous_loop": {"status": "not_started", "error": None},
    "signal_scanner": {"status": "not_started", "error": None},
    "anomaly_detector": {"status": "not_started", "error": None},
}
_autonomous_loop_ref = None
_health_tracker = APIHealthTracker()

AUTONOMOUS_STATE_PATH = Path("./workspace/autonomous_state.json")


def _save_autonomous_state(enabled: bool, started_at: str = ""):
    AUTONOMOUS_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUTONOMOUS_STATE_PATH.write_text(json.dumps({"enabled": enabled, "started_at": started_at}, indent=2))


def _load_autonomous_state() -> dict:
    if AUTONOMOUS_STATE_PATH.exists():
        try:
            return json.loads(AUTONOMOUS_STATE_PATH.read_text())
        except Exception:
            pass
    return {"enabled": False, "started_at": ""}

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


@app.get("/api/data/health")
async def get_api_health():
    """Return health status for all external API sources."""
    all_health = _health_tracker.get_all_health()
    if not all_health:
        return {"message": "No external API sources tracked yet", "sources": {}}
    return {
        name: {
            "source": h.source,
            "last_success": h.last_success.isoformat() if h.last_success else None,
            "last_failure": h.last_failure.isoformat() if h.last_failure else None,
            "consecutive_failures": h.consecutive_failures,
            "is_healthy": h.is_healthy,
        }
        for name, h in all_health.items()
    }


def get_health_tracker() -> APIHealthTracker:
    """Return the global health tracker instance for use by integration modules."""
    return _health_tracker


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


# ── New Dashboard/Operation endpoints ──


@app.get("/api/autonomous/status")
async def autonomous_status():
    """Return current autonomous loop state, including persisted state."""
    global _autonomous_loop_ref
    persisted = _load_autonomous_state()
    # Check both the global ref (API-started) and app.state (CLI-started)
    loop = _autonomous_loop_ref or getattr(app.state, "autonomous_loop", None)
    if loop and loop.state.is_running:
        state = loop.get_state()
        state["persisted_enabled"] = persisted.get("enabled", False)
        return state
    return {"is_running": False, "is_paused": False, "persisted_enabled": persisted.get("enabled", False), "message": "Not started"}


@app.get("/api/autonomous/iterations")
async def autonomous_iterations():
    """Return iteration results from autonomous research cycles for UI charting."""
    global _autonomous_loop_ref
    loop = _autonomous_loop_ref or getattr(app.state, "autonomous_loop", None)
    if loop and hasattr(loop.state, "iteration_results"):
        results = loop.state.iteration_results
        # Compute summary stats
        discarded = sum(1 for r in results if r.get("verdict") == "discarded")
        kept = sum(1 for r in results if r.get("verdict") == "converged" or r.get("verdict") == "kept")
        best = max(
            (r.get("sharpe_ratio", 0) for r in results if isinstance(r.get("sharpe_ratio"), (int, float))),
            default=0,
        )
        return {
            "results": results,
            "discarded_count": discarded,
            "kept_count": kept,
            "best_sharpe": best,
        }
    return {"results": [], "discarded_count": 0, "kept_count": 0, "best_sharpe": 0}


@app.post("/api/autonomous/start")
async def autonomous_start():
    """Start the autonomous loop as a background task."""
    global _autonomous_loop_ref
    if _autonomous_loop_ref and _autonomous_loop_ref.state.is_running:
        return {"status": "already_running"}

    async def _build_and_start():
        global _autonomous_loop_ref
        try:
            logger.info("Building autonomous loop...")
            from orchestration.factory import make_orchestrator
            from api.event_bus import monkey_patch_hermes
            from memory.vector_store import VectorStore
            from orchestration.autonomous_loop import AutonomousResearchLoop
            from orchestration.experiment_tracker import ExperimentTracker
            from data.regime import MarketRegimeDetector
            from config import settings
            orchestrator = make_orchestrator()
            event_bus = getattr(app.state, "event_bus", None)
            if event_bus:
                monkey_patch_hermes(orchestrator, event_bus)
            vs = VectorStore(); et = ExperimentTracker(); rd = MarketRegimeDetector()
            loop = AutonomousResearchLoop(orchestrator=orchestrator, regime_detector=rd, experiment_tracker=et, vector_store=vs, interval_minutes=settings.AUTONOMOUS_INTERVAL_MINUTES, event_bus=getattr(app.state, "event_bus", None))
            _autonomous_loop_ref = loop
            _startup_tasks["autonomous_loop"] = {"status": "running", "error": None}
            asyncio.create_task(loop.run_forever())
            from datetime import datetime as dt
            _save_autonomous_state(enabled=True, started_at=dt.utcnow().isoformat())
            logger.info("Autonomous loop started via API")
        except Exception as exc:
            _startup_tasks["autonomous_loop"] = {"status": "failed", "error": str(exc)}
            logger.exception("Autonomous loop build failed: %s", exc)

    asyncio.create_task(_build_and_start())
    return {"status": "starting", "message": "Building loop in background (15-20s)"}


@app.post("/api/autonomous/stop")
async def autonomous_stop():
    """Stop the autonomous loop. Persists state for restart survival."""
    global _autonomous_loop_ref
    if _autonomous_loop_ref:
        _autonomous_loop_ref.shutdown()
        _autonomous_loop_ref = None
    _startup_tasks["autonomous_loop"] = {"status": "stopped", "error": None}
    _save_autonomous_state(enabled=True, started_at="")
    return {"status": "stopped"}


@app.post("/api/autonomous/pause")
async def autonomous_pause():
    """Pause the autonomous research loop."""
    loop = _autonomous_loop_ref or getattr(app.state, "autonomous_loop", None)
    if loop: loop.pause(); return {"status": "paused"}
    return JSONResponse({"error": "Not running"}, status_code=400)


@app.post("/api/autonomous/resume")
async def autonomous_resume():
    """Resume the autonomous research loop."""
    loop = _autonomous_loop_ref or getattr(app.state, "autonomous_loop", None)
    if loop: loop.resume(); return {"status": "resumed"}
    return JSONResponse({"error": "Not running"}, status_code=400)


@app.get("/api/circuit-breaker/status")
async def circuit_breaker_status():
    """Return current circuit breaker state."""
    from agents.risk_manager import CircuitBreakerState
    return CircuitBreakerState.status()


@app.post("/api/circuit-breaker/halt")
async def circuit_breaker_halt(reason: str = "Manual halt via API"):
    """Manually halt trading."""
    from agents.risk_manager import CircuitBreakerState
    duration = 60
    CircuitBreakerState.halt(reason, duration_minutes=duration)
    return {"status": "halted", "reason": reason, "resume_in_minutes": duration}


@app.post("/api/circuit-breaker/clear")
async def circuit_breaker_clear():
    """Manually clear the circuit breaker."""
    from agents.risk_manager import CircuitBreakerState
    CircuitBreakerState.clear()
    return {"status": "cleared"}


@app.get("/api/positions/open")
async def open_positions():
    """Return all open paper/live positions."""
    executor = getattr(app.state, "live_executor", None)
    if executor:
        return executor.get_open_positions()
    return []


@app.get("/api/positions/history")
async def position_history(limit: int = 100):
    """Return closed position history from audit log."""
    executor = getattr(app.state, "live_executor", None)
    if executor:
        audit = executor.get_audit_log()
        entries = audit.query_recent(limit=limit)
        return {"positions": [e.to_dict() for e in entries]}
    return {"positions": []}


@app.get("/api/signals/feed")
async def signal_feed(limit: int = 100):
    """Return last N signals from SignalScanner."""
    scanner = getattr(app.state, "signal_scanner", None)
    if scanner:
        history = scanner.get_signal_history(limit=limit)
        return {"signals": [{
            "pair": s.pair,
            "signal": s.signal,
            "confidence": s.confidence,
            "strategy_type": s.strategy_type,
            "regime": s.regime,
            "timestamp": s.timestamp.isoformat() if hasattr(s.timestamp, 'isoformat') else str(s.timestamp),
        } for s in history]}
    return {"signals": []}


@app.get("/api/strategies/deployable")
async def deployable_strategies():
    """Return deployable strategies from ChromaDB."""
    vs = VectorStore()
    results = vs.get_best_strategies(min_sharpe=0.0, k=50)
    deployable = []
    for r in results:
        meta = r.get("metadata", {}) or {}
        if meta.get("deployable", False) or meta.get("status") == "kept":
            deployable.append({
                "id": r.get("id", ""),
                "strategy_type": meta.get("strategy_type", "unknown"),
                "regime": meta.get("regime", "unknown"),
                "sharpe": meta.get("sharpe", 0),
                "win_rate": meta.get("win_rate", 0),
                "max_drawdown": meta.get("max_drawdown", 0),
                "metadata": meta,
            })
    return {"strategies": deployable}


@app.post("/api/strategies/{strategy_id}/retire")
async def retire_strategy(strategy_id: str, reason: str = "Manual retirement"):
    """Mark a strategy as retired in ChromaDB."""
    vs = VectorStore()
    vs.store_insight(
        text=f"RETIRED: {strategy_id} — {reason}",
        metadata={"strategy_id": strategy_id, "status": "retired", "reason": reason},
    )
    return {"status": "retired", "strategy_id": strategy_id}


@app.post("/api/validate/oos/{strategy_id}")
async def oos_validate_strategy(strategy_id: str):
    """Run OOS validation on holdout data for a strategy.

    Results are stored in oos_results.jsonl (NOT ChromaDB).
    This endpoint cannot be called from autonomous pipelines.
    """
    from backtesting.oos_validator import OOSValidator
    vs = VectorStore()

    # Find strategy by ID in ChromaDB
    results = vs.get_best_strategies(min_sharpe=0.0, k=50)
    strategy_meta = None
    for r in results:
        meta = r.get("metadata", {}) or {}
        if meta.get("strategy_id") == strategy_id or meta.get("id") == strategy_id:
            strategy_meta = meta
            break

    if not strategy_meta:
        return JSONResponse(
            {"error": f"Strategy {strategy_id} not found"}, status_code=404
        )

    validator = OOSValidator()
    result = validator.validate_strategy(
        strategy_type=strategy_meta.get("strategy_type", "sma_crossover"),
        strategy_params={},
        research_metrics={
            "sharpe_ratio": float(strategy_meta.get("sharpe", 0)),
            "win_rate": float(strategy_meta.get("win_rate", 0)),
        },
        strategy_id=strategy_id,
    )
    return {"result": result.to_dict()}


@app.get("/api/validate/oos/results")
async def oos_results():
    """Return all OOS validation results from oos_results.jsonl only.

    Never reads from ChromaDB — results are stored separately to prevent
    contamination of future research cycles.
    """
    from backtesting.oos_validator import OOSValidator
    validator = OOSValidator()
    results = validator.get_results()
    return {"results": [r.to_dict() for r in results], "count": len(results)}


@app.get("/api/monitoring/report/{strategy_id}")
async def monitoring_report(strategy_id: str):
    """Return full monitoring report for a strategy."""
    from monitoring.performance_monitor import PerformanceMonitor
    monitor = PerformanceMonitor()
    report = monitor.generate_monitoring_report(strategy_id)
    return {"report": report.to_dict() if hasattr(report, "to_dict") else str(report)}


@app.get("/api/monitoring/summary")
async def monitoring_summary():
    """Return degradation status for all strategies."""
    from monitoring.performance_monitor import PerformanceMonitor
    monitor = PerformanceMonitor()
    summary = monitor.get_summary()
    return {"summary": summary}


@app.get("/api/monitoring/oos/pending")
async def oos_pending():
    """Return strategies awaiting OOS validation."""
    from backtesting.oos_validator import OOSValidator
    validator = OOSValidator()
    pending = validator.get_pending_validation()
    return {"pending": pending, "count": len(pending)}


@app.get("/api/deployment/pipeline/status")
async def pipeline_status():
    """Return full deployment pipeline state for all strategies."""
    from orchestration.deployment_pipeline import DeploymentPipeline
    pipeline = DeploymentPipeline()
    status = pipeline.get_all_status()
    return status


@app.get("/api/risk/portfolio")
async def portfolio_risk():
    """Return portfolio risk metrics."""
    risk = {
        "total_open_positions": 0,
        "daily_pnl_pct": 0.0,
        "weekly_pnl_pct": 0.0,
        "max_drawdown": 0.0,
        "circuit_breaker": {"halted": False},
    }
    executor = getattr(app.state, "live_executor", None)
    if executor:
        risk["total_open_positions"] = len(executor.get_open_positions())
    from agents.risk_manager import CircuitBreakerState
    risk["circuit_breaker"] = CircuitBreakerState.status()
    return risk


@app.get("/api/startup/status")
async def startup_status():
    """Return status of all background tasks launched on startup."""
    return {"tasks": _startup_tasks}


@app.get("/api/regime/current")
async def current_regime():
    """Return current regime snapshot for all configured pairs."""
    from data.fetcher import MarketDataFetcher
    from data.regime import MarketRegimeDetector
    from config import settings
    try:
        fetcher = MarketDataFetcher()
        df = fetcher.fetch_ohlcv(settings.SYMBOL, "1h", limit=250)
        if df is not None and len(df) > 200:
            detector = MarketRegimeDetector()
            snapshot = detector.classify_regime_snapshot(df)
            return {
                "regime": snapshot.regime,
                "confidence": snapshot.confidence,
                "adx": snapshot.adx,
                "atr_pct": snapshot.atr_pct,
                "sma200_distance": snapshot.sma200_distance,
                "recommended_strategies": snapshot.recommended_strategies,
                "discouraged_strategies": snapshot.discouraged_strategies,
            }
    except Exception as exc:
        logger.warning("Regime detection failed: %s", exc)
    return {"regime": "unknown", "confidence": 0.0}


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


@app.websocket("/ws/autonomous")
async def ws_autonomous_events(websocket: WebSocket):
    """WebSocket for autonomous loop events (dashboard live updates)."""
    await websocket.accept()
    event_bus = getattr(app.state, "event_bus", None)
    if not event_bus:
        await websocket.send_json({"type": "error", "payload": {"message": "No autonomous event bus"}, "timestamp": datetime.utcnow().isoformat()})
        await websocket.close()
        return

    try:
        async for event in event_bus.subscribe():
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
    from orchestration.auto_research import run_auto_research

    orchestrator = make_orchestrator()

    # Reset and emit starting token count
    from agents.token_tracker import TokenTracker
    TokenTracker.get().reset()
    await bus.publish("token_usage", TokenTracker.get().get_usage())

    # Patch: inject event publishing into the orchestrator
    _patch_orchestrator(orchestrator, bus, run_id)

    logger.info("━" * 50)
    logger.info("[RUN %s] Starting goal: %s", run_id, goal)
    logger.info("[RUN %s] max_cycles=%s  max_iterations=%s", run_id, max_cycles, max_iterations)
    # Print goal plainly to terminal so it's visible above all async noise
    print(f"\n{'='*60}")
    print(f"  GOAL: {goal}")
    print(f"{'='*60}\n")
    await bus.publish("run_start", {"goal": goal, "max_cycles": max_cycles, "max_iterations": max_iterations})

    try:
        # Run in executor to avoid blocking the event loop
        loop = asyncio.get_running_loop()

        # ── 4B: Auto-research mode via goal prefix ──
        if goal.startswith("Auto-research:"):
            topic = goal[len("Auto-research:"):].strip()
            logger.info("[RUN %s] Auto-research mode: %s", run_id, topic)
            result = await loop.run_in_executor(
                None,
                lambda: run_auto_research(topic, event_bus=bus, loop=loop),
            )
        elif max_iterations > 1:
            logger.info("[RUN %s] AutoResearch loop enabled — will iterate up to %s times", run_id, max_iterations)
            result = await loop.run_in_executor(
                None,
                lambda: orchestrator.run_research_loop(goal, max_cycles=max_cycles, max_iterations=max_iterations),
            )
        else:
            logger.info("[RUN %s] Single-pass mode (use max_iterations > 1 for research loop)", run_id)
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

        # Emit sentiment if available in result (single-pass mode)
        if isinstance(result, dict):
            sentiment = result.get("sentiment")
            if sentiment:
                await bus.publish("sentiment", sentiment)

        logger.info("[RUN %s] ✅ Complete. Converged=%s | Iterations=%s", run_id, result.get("converged"), result.get("total_iterations"))

        _active_runs[run_id] = {"status": "done", "goal": goal, "result": result}
    except Exception as exc:
        logger.exception("[RUN %s] ❌ Failed: %s", run_id, exc)
        await bus.publish("run_error", {"message": str(exc)})
        _active_runs[run_id] = {"status": "error", "goal": goal, "error": str(exc)}


def _patch_orchestrator(orchestrator: "HermesOrchestrator", bus: EventBus, run_id: str):
    """Patch orchestrator methods to emit events to the bus.

    Captures the event loop at patch time (main thread) so emit() works
    from executor threads where asyncio.get_running_loop() would fail.
    """
    try:
        _loop = asyncio.get_running_loop()
    except RuntimeError:
        _loop = None

    def emit(event_type: str, payload: Dict[str, Any]):
        if _loop is None:
            return
        asyncio.run_coroutine_threadsafe(
            bus.publish(event_type, payload), _loop
        )

    def emit_tokens():
        """Emit current cumulative token usage."""
        try:
            from agents.token_tracker import TokenTracker
            emit("token_usage", TokenTracker.get().get_usage())
        except Exception:
            pass

    # Wrap _generate_hypothesis
    if hasattr(orchestrator, "_generate_hypothesis"):
        orig_gen = orchestrator._generate_hypothesis

        def patched_generate(goal, past_iterations, iter_num, max_iterations):
            hypothesis = orig_gen(goal, past_iterations, iter_num, max_iterations)
            logger.info("[RUN %s] Hypothesis iter %s/%s: %s", run_id, iter_num, max_iterations, hypothesis[:120])
            emit("hypothesis", {
                "hypothesis": hypothesis,
                "iteration": iter_num,
                "max_iterations": max_iterations,
            })
            emit_tokens()
            return hypothesis

        orchestrator._generate_hypothesis = patched_generate

    # Wrap _critique_iteration
    if hasattr(orchestrator, "_critique_iteration"):
        orig_crit = orchestrator._critique_iteration

        def patched_critique(output, goal, hypothesis):
            critique = orig_crit(output, goal, hypothesis)
            logger.info("[RUN %s] Critique for '%s': %s", run_id, hypothesis[:60], critique[:200])
            emit("critique", {
                "critique": critique,
                "hypothesis": hypothesis,
            })
            emit_tokens()
            return critique

        orchestrator._critique_iteration = patched_critique

    # Wrap _run_research_goal to emit agent events
    if hasattr(orchestrator, "_run_research_goal"):
        orig_run = orchestrator._run_research_goal

        def patched_run(goal, max_cycles=5, hypothesis="", iteration=1):
            logger.info("[RUN %s] ── Iteration %s starting ──", run_id, iteration)
            emit("iteration_start", {"iteration": iteration, "goal": goal[:200]})
            result = orig_run(goal, max_cycles=max_cycles, hypothesis=hypothesis, iteration=iteration)
            # Emit events for each board task
            done_tasks = orchestrator.board.get_tasks_by_status("DONE") if hasattr(orchestrator, "board") else []
            for task in done_tasks:
                if task.result:
                    logger.info("[RUN %s]   Task done [%s]: %s", run_id, task.assigned_to, task.description[:80])
                    emit("task_done", {
                        "task_id": task.id,
                        "agent": task.assigned_to,
                        "description": task.description[:200],
                        "result": str(task.result)[:500],
                    })
            # Extract and emit metrics
            metrics = orchestrator._extract_metrics(result) if hasattr(orchestrator, "_extract_metrics") else {}
            logger.info("[RUN %s] ── Iteration %s done — metrics: Sharpe=%.2f WR=%s DD=%s Trades=%s",
                        run_id, iteration,
                        metrics.get("sharpe_ratio", 0),
                        metrics.get("win_rate", "—"),
                        metrics.get("max_drawdown", "—"),
                        metrics.get("total_trades", 0))
            emit_tokens()
            emit("iteration_result", {
                "iteration": iteration,
                "task_count": len(done_tasks),
                "metrics": metrics,
            })
            # Emit sentiment if available in result
            if isinstance(result, dict):
                sentiment = result.get("sentiment")
                if sentiment:
                    emit("sentiment", sentiment)
            return result

        orchestrator._run_research_goal = patched_run


# ── Startup event: launch background tasks ──


@app.on_event("startup")
async def startup():
    """Launch background tasks on server startup."""
    logger.info("FastAPI server starting — launching background tasks")

    autonomous_loop = getattr(app.state, "autonomous_loop", None)
    event_bus = getattr(app.state, "event_bus", None)

    # 1. Market data stream
    try:
        from config import settings
        from data.stream import MarketDataStream
        pairs = [settings.SYMBOL]
        stream = MarketDataStream()
        await stream.connect(pairs)
        asyncio.create_task(stream.read_loop())
        app.state.market_data_stream = stream
        _startup_tasks["market_data_stream"] = {"status": "running", "error": None}
        logger.info("MarketDataStream started for %s", pairs)
    except Exception as exc:
        _startup_tasks["market_data_stream"] = {"status": "failed", "error": str(exc)}
        logger.warning("MarketDataStream startup skipped: %s", exc)

    # 2. Autonomous research loop
    persisted = _load_autonomous_state()
    if autonomous_loop and not autonomous_loop.state.is_running:
        asyncio.create_task(autonomous_loop.run_forever())
        _startup_tasks["autonomous_loop"] = {"status": "running", "error": None}
        logger.info("AutonomousResearchLoop started from app.state")
    elif persisted.get("enabled") and not _autonomous_loop_ref:
        _startup_tasks["autonomous_loop"] = {"status": "starting", "error": None}
        async def _rebuild_loop():
            try:
                from agents.analyst import AnalystAgent
                from agents.strategist import StrategistAgent
                from agents.risk_manager import RiskManagerAgent
                from agents.curator import CuratorAgent
                from agents.researcher import ResearcherAgent
                from backtesting.engine import BacktestEngine
                from memory.vector_store import VectorStore
                from orchestration.autonomous_loop import AutonomousResearchLoop
                from orchestration.experiment_tracker import ExperimentTracker
                from orchestration.factory import make_orchestrator
                from api.event_bus import monkey_patch_hermes
                from data.regime import MarketRegimeDetector
                from config import settings
                global _autonomous_loop_ref
                orchestrator = make_orchestrator()
                if event_bus:
                    monkey_patch_hermes(orchestrator, event_bus)
                vs = VectorStore(); et = ExperimentTracker(); rd = MarketRegimeDetector()
                loop = AutonomousResearchLoop(orchestrator=orchestrator, regime_detector=rd, experiment_tracker=et, vector_store=vs, interval_minutes=settings.AUTONOMOUS_INTERVAL_MINUTES, event_bus=event_bus)
                _autonomous_loop_ref = loop
                _startup_tasks["autonomous_loop"] = {"status": "running", "error": None}
                asyncio.create_task(loop.run_forever())
                logger.info("AutonomousResearchLoop restored from persisted state")
            except Exception as exc:
                _startup_tasks["autonomous_loop"] = {"status": "failed", "error": str(exc)}
                logger.exception("Auto-restart failed: %s", exc)
        asyncio.create_task(_rebuild_loop())
    else:
        _startup_tasks["autonomous_loop"] = {"status": "not_started", "error": "No loop configured"}

    try:
        scanner = getattr(app.state, "signal_scanner", None)
        if scanner:
            asyncio.create_task(scanner.scan_loop())
            _startup_tasks["signal_scanner"] = {"status": "running", "error": None}
            logger.info("SignalScanner started")
        else:
            _startup_tasks["signal_scanner"] = {"status": "not_started", "error": "No scanner configured"}
    except Exception as exc:
        _startup_tasks["signal_scanner"] = {"status": "failed", "error": str(exc)}
        logger.warning("SignalScanner startup skipped: %s", exc)

    # 4. Anomaly detector
    try:
        from monitoring.anomaly_detector import AnomalyDetector
        detector = AnomalyDetector(
            live_executor=getattr(app.state, "live_executor", None),
            signal_scanner=getattr(app.state, "signal_scanner", None),
            market_data_stream=getattr(app.state, "market_data_stream", None),
            event_bus=event_bus,
        )
        app.state.anomaly_detector = detector
        asyncio.create_task(detector.monitor_loop())
        _startup_tasks["anomaly_detector"] = {"status": "running", "error": None}
        logger.info("AnomalyDetector started")
    except Exception as exc:
        _startup_tasks["anomaly_detector"] = {"status": "failed", "error": str(exc)}
        logger.warning("AnomalyDetector startup skipped: %s", exc)