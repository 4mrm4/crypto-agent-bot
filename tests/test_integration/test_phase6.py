"""Phase 6 test — verify memory persists across consecutive goals."""

import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def main():
    sys.path.insert(0, ".")

    from agents.analyst import AnalystAgent
    from agents.strategist import StrategistAgent
    from agents.curator import CuratorAgent
    from orchestration.hermes import HermesOrchestrator
    from workspace.vibe import VibeWorkspace

    print("=" * 60)
    print("Phase 6: Memory and Continuous Learning")
    print("=" * 60)

    curator = CuratorAgent()
    orchestrator = HermesOrchestrator(agents={
        "analyst": AnalystAgent(),
        "strategist": StrategistAgent(),
        "curator": curator,
    })
    ws = VibeWorkspace(orchestrator=orchestrator)

    print("\n--- Goal 1: SMA crossover ---")
    r1 = ws.create_goal("Find SMA crossover for BTC/USDT", max_cycles=2)
    assert r1["status"] == "completed"
    print(f"Goal 1 done.")

    print("\n--- Goal 2: Similar strategy ---")
    r2 = ws.create_goal("Optimise SMA crossover for BTC", max_cycles=2)
    assert r2["status"] == "completed"
    print(f"Goal 2 done.")

    # Query memory
    print("\n--- Querying memory ---")
    results = curator._vector_store.query_similar("SMA crossover BTC", k=3)
    print(f"Retrieved {len(results)} insights")
    for r in results:
        print(f"  {r['text'][:120]}")
    assert len(results) > 0

    print(f"\nTotal: {curator._vector_store.count()} documents")
    print("[OK] Phase 6 test passed!")


if __name__ == "__main__":
    main()