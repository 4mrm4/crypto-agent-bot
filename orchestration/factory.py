"""Orchestrator factory — shared between main.py and api/server.py."""

from agents.analyst import AnalystAgent
from agents.strategist import StrategistAgent
from agents.backtester import BacktesterAgent
from agents.iteration_tracker import IterationTrackerAgent
from agents.risk_manager import RiskManagerAgent
from agents.curator import CuratorAgent
from agents.researcher import ResearcherAgent
from orchestration.hermes import HermesOrchestrator
from state.circuit_breaker import CircuitBreakerState


def make_orchestrator():
    cb = CircuitBreakerState()
    return HermesOrchestrator(agents={
        "analyst": AnalystAgent(),
        "strategist": StrategistAgent(),
        "backtester": BacktesterAgent(),
        "iteration_tracker": IterationTrackerAgent(),
        "risk_manager": RiskManagerAgent(circuit_breaker=cb),
        "curator": CuratorAgent(),
        "researcher": ResearcherAgent(),
    }, circuit_breaker=cb)
