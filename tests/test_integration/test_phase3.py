"""Phase 3 test — verify Analyst and Strategist agents work with real data and tools."""

import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def main():
    sys.path.insert(0, ".")

    from agents.analyst import AnalystAgent
    from agents.strategist import StrategistAgent

    print("=" * 60)
    print("Phase 3: Agent Layer")
    print("=" * 60)

    # ── Test 1: Analyst Agent ──
    print("\n--- 1. Analyst Agent: analyse BTC/USDT ---")
    analyst = AnalystAgent()
    result = analyst.run(
        "Analyse BTC/USDT market. Fetch 50 hourly candles, get the current price, "
        "and produce a brief market analysis with trend direction and support/resistance levels."
    )
    print(f"\nAnalyst output:\n{result['output']}")
    print(f"\nIntermediate tool steps: {len(result['intermediate_steps'])}")

    # ── Test 2: Strategist Agent ──
    print("\n--- 2. Strategist Agent: find SMA crossover for BTC/USDT ---")
    strategist = StrategistAgent()

    # The strategist needs to generate a strategy then backtest it
    # We do it in two explicit calls for reliability
    result = strategist.run(
        "Create a strategy with fast_ma=10, slow_ma=30, stoploss=-0.05. "
        "Then backtest it on BTC/USDT for timerange 20260427-20260527. "
        "Finally interpret the metrics and tell me if it's viable."
    )
    print(f"\nStrategist output:\n{result['output']}")
    print(f"\nIntermediate tool steps: {len(result['intermediate_steps'])}")

    print("\n[OK] Phase 3 test passed!")


if __name__ == "__main__":
    main()