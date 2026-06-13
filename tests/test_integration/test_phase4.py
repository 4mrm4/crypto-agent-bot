"""Phase 4 test — verify multi-agent orchestration with Kanban/LangGraph."""

import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# Force UTF-8 for stdout so emoji doesn't crash Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore


def main():
    sys.path.insert(0, ".")

    from agents.analyst import AnalystAgent
    from agents.strategist import StrategistAgent
    from orchestration.hermes import HermesOrchestrator

    print("=" * 60)
    print("Phase 4: Multi-Agent Orchestration")
    print("=" * 60)

    # Create agents
    analyst = AnalystAgent()
    strategist = StrategistAgent()

    # Build orchestrator
    orchestrator = HermesOrchestrator(agents={
        "analyst": analyst,
        "strategist": strategist,
    })

    # Run a research goal
    print("\n--- Running research goal ---")
    goal = "Find a momentum strategy with Sharpe > 1 and max drawdown < 5% for BTC/USDT"
    result = orchestrator.run_research_goal(goal, max_cycles=4)

    print(f"\nGoal: {result['goal']}")
    print(f"Board: {result['board_summary']}")
    print(f"Tasks completed: {result['task_count']}")

    print("\nTask results:")
    for tid, output in result.get("results", {}).items():
        # Strip non-ASCII chars for Windows console
        safe = output[:400].encode("ascii", errors="replace").decode("ascii")
        print(f"\n  [{tid}] {safe}")

    print("\nStrategies found:")
    for s in result.get("strategies", []):
        safe = s[:300].encode("ascii", errors="replace").decode("ascii")
        print(f"  - {safe}")

    print("\n[OK] Phase 4 test passed!")


if __name__ == "__main__":
    main()