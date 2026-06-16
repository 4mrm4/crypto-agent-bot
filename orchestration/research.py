"""Research iteration dataclass for the AutoResearch outer hypothesis loop."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from orchestration.evaluation import check_convergence


@dataclass
class ResearchGoal:
    """Autonomously generated research goal — no human input needed."""
    regime: str                          # e.g. "ranging", "strong_uptrend"
    strategy_type_hint: str              # e.g. "mean_reversion", "momentum"
    motivation: str                      # Human-readable why this was chosen
    priority_score: float                # 0.0–1.0, higher = more urgent
    triggered_by: str                    # "decay" | "coverage_gap" | "regime_change" | "scheduled" | "exploration"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


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

