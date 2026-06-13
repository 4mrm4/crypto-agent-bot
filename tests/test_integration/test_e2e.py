"""Phase 8 E2E — full pipeline: all agents, memory, orchestration."""

import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def main():
    sys.path.insert(0, ".")

    from agents.analyst import AnalystAgent
    from agents.strategist import StrategistAgent
    from agents.risk_manager import RiskManagerAgent
    from agents.curator import CuratorAgent
    from orchestration.hermes import HermesOrchestrator
    from workspace.vibe import VibeWorkspace

    print("=" * 60)
    print("Phase 8: End-to-End Integration")
    print("=" * 60)

    curator = CuratorAgent()
    orchestrator = HermesOrchestrator(agents={
        "analyst": AnalystAgent(),
        "strategist": StrategistAgent(),
        "risk_manager": RiskManagerAgent(),
        "curator": curator,
    })
    ws = VibeWorkspace(orchestrator=orchestrator)

    # 1. Research goal
    print("\n--- Goal ---")
    entry = ws.create_goal("Test SMA crossover for BTC/USDT", max_cycles=2)
    assert entry["status"] == "completed"
    print(f"Goal {entry['id']} complete. Tasks: {entry.get('result',{}).get('task_count',0)}")

    # 2. Memory populated
    mem_count = curator._vector_store.count()
    print(f"Memory: {mem_count} documents")
    assert mem_count > 0

    # 3. Retrieve memory
    mem = curator._vector_store.query_similar("BTC SMA crossover", k=2)
    print(f"Retrieved: {len(mem)} relevant insights")
    assert len(mem) > 0

    # 4. Risk assessment
    risk = RiskManagerAgent()
    rr = risk.run("Generate risk report: total_trades=12, sharpe=1.0, max_drawdown=0.04, win_rate=0.5")
    assert len(rr["output"]) > 0
    print(f"Risk report: {len(rr['output'])} chars")

    print("\n[OK] Phase 8 E2E test passed!")


if __name__ == "__main__":
    main()