"""Shared health tracker for all external API integrations."""
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Consecutive failures threshold to mark a source unhealthy
UNHEALTHY_THRESHOLD = 3


@dataclass
class APIHealth:
    """Health state for a single API source."""
    source: str
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    consecutive_failures: int = 0

    @property
    def is_healthy(self) -> bool:
        """True if consecutive failures below the unhealthy threshold."""
        return self.consecutive_failures < UNHEALTHY_THRESHOLD


class APIHealthTracker:
    """Tracks health of all external API sources.

    Stores state in-memory (simple dict). Can be extended to use
    StateBroker/Redis for persistence across restarts.
    """

    def __init__(self):
        self._state: Dict[str, APIHealth] = {}

    def _ensure(self, source: str) -> APIHealth:
        if source not in self._state:
            self._state[source] = APIHealth(source=source)
        return self._state[source]

    def record_success(self, source: str):
        """Record a successful API call — resets consecutive failures."""
        health = self._ensure(source)
        health.last_success = datetime.now(timezone.utc)
        health.consecutive_failures = 0
        logger.debug("APIHealth[%s]: success recorded", source)

    def record_failure(self, source: str, error: str):
        """Record an API failure — increments consecutive failures."""
        health = self._ensure(source)
        health.last_failure = datetime.now(timezone.utc)
        health.consecutive_failures += 1
        logger.warning(
            "APIHealth[%s]: failure #%d: %s",
            source, health.consecutive_failures, error,
        )

    def get_health(self, source: str) -> APIHealth:
        """Return current health for a source."""
        return self._ensure(source)

    def get_all_health(self) -> Dict[str, APIHealth]:
        """Return health for all tracked sources."""
        return dict(self._state)

    def get_degraded_sources(self) -> List[str]:
        """Return list of source names that are unhealthy."""
        return [
            name for name, h in self._state.items()
            if not h.is_healthy
        ]
