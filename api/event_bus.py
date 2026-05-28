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