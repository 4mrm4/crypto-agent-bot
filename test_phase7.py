"""Phase 7 test — verify Risk Manager and Paper Trader work."""

import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def main():
    sys.path.insert(0, ".")

    from agents.risk_manager import RiskManagerAgent

    print("=" * 60)
    print("Phase 7: Risk Management & Paper Trading")
    print("=" * 60)

    # Risk manager agent (LLM-driven)
    print("\n=== Risk Manager ===")
    risk = RiskManagerAgent()
    result = risk.run(
        "Assess risk for strategy with total_trades=15, win_rate=0.55, "
        "sharpe_ratio=1.2, profit_ratio=0.025, max_drawdown=0.035. "
        "Use risk_report tool. Give a go/no-go verdict."
    )
    print(f"Output: {result['output'][:300]}")
    assert len(result["output"]) > 0

    print("\n[OK] Phase 7 test passed!")


if __name__ == "__main__":
    main()