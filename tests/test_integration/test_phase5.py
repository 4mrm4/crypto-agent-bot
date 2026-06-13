"""Phase 5 test — verify Workspace CLI works end-to-end."""

import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore


def main():
    sys.path.insert(0, ".")

    from agents.analyst import AnalystAgent
    from agents.strategist import StrategistAgent
    from orchestration.hermes import HermesOrchestrator
    from workspace.vibe import VibeWorkspace

    print("=" * 60)
    print("Phase 5: Research Workspace")
    print("=" * 60)

    analyst = AnalystAgent()
    strategist = StrategistAgent()
    orchestrator = HermesOrchestrator(agents={"analyst": analyst, "strategist": strategist})
    ws = VibeWorkspace(orchestrator=orchestrator)

    # Run a goal programmatically
    print("\n=== Creating goal ===")
    result = ws.create_goal("Test SMA crossover for BTC/USDT", max_cycles=3)
    assert result["status"] == "completed", f"Goal failed: {result.get('result')}"

    print("\n=== Listing goals ===")
    goals = ws.list_goals()
    assert len(goals) >= 1
    print(f"Goals in registry: {len(goals)}")

    print("\n=== Reviewing goal ===")
    reviewed = ws.review_goal(result["id"])
    assert reviewed is not None

    print("\n=== Accepting strategy ===")
    if result.get("strategies"):
        path = ws.accept_strategy(result["id"], 0)
        if path:
            print(f"Exported to: {path}")

    print("\n[OK] Phase 5 test passed!")


if __name__ == "__main__":
    main()