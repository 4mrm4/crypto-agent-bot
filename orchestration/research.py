"""Research iteration dataclass for the AutoResearch outer hypothesis loop."""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class ResearchIteration:
    """Tracks one full outer-loop iteration: hypothesis -> research -> critique."""
    hypothesis: str = ""
    strategy_id: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    verdict: str = "unknown"  # converged | discarded | max_iterations
    critique: str = ""
    iteration: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis": self.hypothesis,
            "strategy_id": self.strategy_id,
            "metrics": self.metrics,
            "verdict": self.verdict,
            "critique": self.critique,
            "iteration": self.iteration,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ResearchIteration":
        return ResearchIteration(
            hypothesis=data.get("hypothesis", ""),
            strategy_id=data.get("strategy_id", ""),
            metrics=data.get("metrics", {}),
            verdict=data.get("verdict", "unknown"),
            critique=data.get("critique", ""),
            iteration=data.get("iteration", 0),
        )


def check_convergence(metrics: Dict[str, Any], total_trades_min: int = 5) -> bool:
    """
    Check if research has converged to an acceptable strategy.
    Uses realistic targets based on available data window.

    Targets (relaxed for short data windows):
    - Sharpe >= 0.8 (was 1.5 — unrealistic on 30 days)
    - Win rate >= 40% (was 45%)
    - Max drawdown <= 15% (was 10%)
    - At least 5 trades (new — prevents convergence on 1-trade flukes)
    """
    sharpe = metrics.get("sharpe_ratio", 0)
    win_rate = metrics.get("win_rate", 0)
    drawdown = abs(metrics.get("max_drawdown", 0))
    total_trades = metrics.get("total_trades", 0)

    if total_trades < total_trades_min:
        return False
    if sharpe < 0.8:
        return False
    if win_rate < 0.40:
        return False
    if drawdown > 0.15:
        return False
    return True
