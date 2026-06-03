"""Token bucket rate limiter for external API calls."""
import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)


class RateLimiter:
    """Token bucket rate limiter.

    Uses a background thread to refill tokens at the configured rate.
    Thread-safe via a lock.
    """

    def __init__(self, rpm: int = 20):
        self._capacity = rpm
        self._tokens = float(rpm)
        self._refill_rate = rpm / 60.0  # tokens per second
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self):
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._capacity, self._tokens + elapsed * self._refill_rate)
        self._last_refill = now

    def acquire(self, block: bool = True, timeout: Optional[float] = None) -> bool:
        """Acquire a token. Returns True if acquired, False if rate limited.

        If block=True, waits up to `timeout` seconds for a token.
        If block=False, returns immediately with False if no token available.
        """
        deadline = time.monotonic() + timeout if timeout else None

        while True:
            with self._lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return True
                if not block:
                    return False

            # Blocking wait
            wait = 1.0 / self._refill_rate  # time for 1 token
            if deadline:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                wait = min(wait, remaining)
            time.sleep(wait)

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *args):
        pass
