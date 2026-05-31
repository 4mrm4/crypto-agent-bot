"""Event bus for WebSocket streaming of orchestration events."""

import asyncio
import json
from datetime import datetime
from typing import Any, AsyncGenerator, Callable, Dict, Optional


class EventBus:
    """Simple asyncio.Queue-based event bus for streaming orchestration events."""

    def __init__(self, max_size: int = 500):
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_size)

    async def publish(self, event_type: str, payload: Dict[str, Any]):
        """Publish an event to all subscribers."""
        event = {
            "type": event_type,
            "payload": payload,
            "timestamp": datetime.utcnow().isoformat(),
        }
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            # Drop oldest event to make room
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(event)
            except asyncio.QueueEmpty:
                pass

    async def subscribe(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Async generator yielding events as they arrive."""
        while True:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=30.0)
                yield event
            except asyncio.TimeoutError:
                # Send heartbeat to keep connection alive
                yield {"type": "heartbeat", "payload": {}, "timestamp": datetime.utcnow().isoformat()}

    def make_callback(self) -> Callable[[str, Dict[str, Any]], None]:
        """Return a synchronous callback for use in orchestrator threads.

        Usage: pass this to HermesOrchestrator methods as event_callback.
        """
        loop: Optional[asyncio.AbstractEventLoop] = None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            pass

        def callback(event_type: str, payload: Dict[str, Any]):
            try:
                if loop and loop.is_running():
                    # Schedule on the event loop from another thread
                    asyncio.run_coroutine_threadsafe(
                        self.publish(event_type, payload), loop
                    )
                else:
                    # Fallback: synchronous add
                    event = {
                        "type": event_type,
                        "payload": payload,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                    try:
                        self._queue.put_nowait(event)
                    except asyncio.QueueFull:
                        try:
                            self._queue.get_nowait()
                            self._queue.put_nowait(event)
                        except asyncio.QueueEmpty:
                            pass
            except Exception:
                pass  # Never crash the orchestrator for a UI event

        return callback


def monkey_patch_hermes(orchestrator: Any, bus: EventBus, loop=None):
    """Patch HermesOrchestrator methods to emit events to the bus.

    Simplified version of server.py's _patch_orchestrator — suitable for
    auto_research mode where there's no run_id.
    """
    import logging
    logger = logging.getLogger("event_bus.monkey_patch")

    if loop is None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

    def emit(event_type: str, payload: Dict[str, Any]):
        if loop is None:
            return
        asyncio.run_coroutine_threadsafe(
            bus.publish(event_type, payload), loop
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
            emit("critique", {
                "critique": critique,
                "hypothesis": hypothesis,
            })
            emit_tokens()
            return critique

        orchestrator._critique_iteration = patched_critique

    # Wrap _run_research_goal to emit iteration events
    if hasattr(orchestrator, "_run_research_goal"):
        orig_run = orchestrator._run_research_goal

        def patched_run(goal, max_cycles=5, hypothesis="", iteration=1):
            emit("iteration_start", {"iteration": iteration, "goal": goal[:200]})
            result = orig_run(goal, max_cycles=max_cycles, hypothesis=hypothesis, iteration=iteration)
            metrics = orchestrator._extract_metrics(result) if hasattr(orchestrator, "_extract_metrics") else {}
            emit("iteration_result", {
                "iteration": iteration,
                "metrics": metrics,
            })
            emit_tokens()
            return result

        orchestrator._run_research_goal = patched_run

    logger.info("HermesOrchestrator patched for EventBus streaming.")