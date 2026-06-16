"""Shared async-to-sync bridge — single canonical wrapper for calling async code from sync contexts.

Unifies the duplicated run_coroutine_threadsafe / asyncio.run fallback pattern
that was copied across regime.py and sentiment.py.
"""

import asyncio
from typing import Optional, TypeVar

T = TypeVar("T")


def run_async_in_sync(coro, timeout: float = 30) -> Optional[T]:
    """Run an async coroutine from a synchronous context.

    Tries asyncio.run_coroutine_threadsafe if an event loop is already running,
    otherwise falls back to asyncio.run().
    Returns the coroutine result, or None on any exception.
    """
    try:
        loop = asyncio.get_running_loop()
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result(timeout=timeout)
    except RuntimeError:
        # No running loop — use asyncio.run()
        return asyncio.run(coro)
