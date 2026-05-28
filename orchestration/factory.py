"""Orchestrator factory — shared between main.py and api/server.py."""


def make_orchestrator():
    from agents.analyst import AnalystAgent
    from agents.strategist import StrategistAgent
    from agents.risk_manager import RiskManagerAgent
    from agents.curator import CuratorAgent
    from agents.researcher import ResearcherAgent
    from orchestration.hermes import HermesOrchestrator
    return HermesOrchestrator(agents={
        "analyst": AnalystAgent(),
        "strategist": StrategistAgent(),
        "risk_manager": RiskManagerAgent(),
        "curator": CuratorAgent(),
        "researcher": ResearcherAgent(),
    })