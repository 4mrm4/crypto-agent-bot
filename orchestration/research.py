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


def check_convergence(metrics: Dict[str, Any]) -> bool:
    """Check if metrics meet the convergence targets.

    Targets: Sharpe >= 1.5, win_rate >= 45%, max_drawdown <= 10%.
    """
    sharpe = metrics.get("sharpe_ratio", 0)
    win_rate = metrics.get("win_rate", 0)
    drawdown = abs(metrics.get("max_drawdown", 0))
    total_trades = metrics.get("total_trades", 0)

    if total_trades == 0:
        return False

    sharpe_ok = isinstance(sharpe, (int, float)) and sharpe >= 1.5
    wr_ok = isinstance(win_rate, (int, float)) and win_rate >= 0.45
    dd_ok = drawdown <= 0.10

    return sharpe_ok and wr_ok and dd_ok
