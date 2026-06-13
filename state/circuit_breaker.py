"""CircuitBreakerState — global halt switch for trading.

Plain instance class (not a singleton). One instance is created in main.py
and injected into every component that needs it: RiskManagerAgent,
LiveExecutor, AnomalyDetector, HermesOrchestrator.

Tests create a fresh instance per test case — no global state leakage.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


class CircuitBreakerState:
    """Global circuit breaker. Halted state is not shared across instances."""

    def __init__(self):
        self._halted: bool = False
        self._halt_reason: str = ""
        self._halted_at: Optional[datetime] = None
        self._resume_after: Optional[datetime] = None
        self._research_mode: bool = False  # Skip hard halt during research cycles (hallucinated PnL)

    # -- Research mode --

    @property
    def research_mode(self) -> bool:
        return self._research_mode

    @research_mode.setter
    def research_mode(self, value: bool):
        self._research_mode = value
        logger.debug("Circuit breaker research_mode set to %s", value)

    # -- Halt / resume --

    def halt(self, reason: str, duration_minutes: int = 60):
        self._halted = True
        self._halt_reason = reason
        self._halted_at = datetime.utcnow()
        self._resume_after = datetime.utcnow() + timedelta(minutes=duration_minutes)
        logger.warning("CIRCUIT BREAKER HALTED: %s (resume after %s)", reason, self._resume_after)

    def clear(self):
        self._halted = False
        self._halt_reason = ""
        self._halted_at = None
        self._resume_after = None
        logger.info("Circuit breaker cleared — trading resumed.")

    def is_halted(self) -> bool:
        if self._halted and self._resume_after and datetime.utcnow() > self._resume_after:
            self.clear()
        return self._halted

    def status(self) -> dict:
        return {
            "halted": self._halted,
            "reason": self._halt_reason,
            "halted_at": self._halted_at.isoformat() if self._halted_at else None,
            "resume_after": self._resume_after.isoformat() if self._resume_after else None,
            "research_mode": self._research_mode,
        }
