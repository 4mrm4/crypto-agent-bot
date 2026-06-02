"""Structured experiment tracker for strategy optimization."""
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class Experiment:
    strategy_type: str
    params: Dict[str, Any]
    timerange: str
    regime: str
    sentiment_score: float
    sharpe: float
    win_rate: float
    max_drawdown: float
    total_trades: int
    profit_factor: float = 1.0
    walk_forward_passed: bool = False
    monte_carlo_dd_95pct: float = 0.0
    synthetic_sanity_passed: bool = False  # Must pass random walk test
    verdict: str = "discarded"
    iteration: int = 0

    def score(self) -> float:
        """Composite score for ranking experiments.

        Uses tightened convergence criteria:
        - Sharpe >= 1.2 (weight 0.30)
        - Win rate >= 0.48 (weight 0.20)
        - Max drawdown <= 0.10 (weight 0.25)
        - Profit factor >= 1.5 (weight 0.15)
        - Min trades >= 30 (weight 0.10)
        - Walk-forward pass required (+0.15 bonus)
        """
        if self.total_trades < 30:
            return -999.0  # insufficient data
        if self.max_drawdown > 0.10:
            return -998.0  # excessive drawdown
        if self.sharpe < 0.5:
            return -997.0  # too low sharpe to consider

        return (
            min(self.sharpe / 1.2, 2.0) * 0.30 +
            min(self.win_rate / 0.48, 2.0) * 0.20 +
            max(0, 1 - self.max_drawdown / 0.10) * 0.25 +
            min(self.profit_factor / 1.5, 2.0) * 0.15 +
            min(self.total_trades / 30, 2.0) * 0.10 +
            (0.15 if self.walk_forward_passed else 0)
        )

    def meets_deploy_criteria(self) -> bool:
        """Check if this experiment meets the deployable strategy criteria."""
        return (
            self.sharpe >= 1.2
            and self.win_rate >= 0.48
            and self.max_drawdown <= 0.10
            and self.profit_factor >= 1.5
            and self.total_trades >= 30
            and self.walk_forward_passed
            and self.monte_carlo_dd_95pct <= 0.20
            and self.synthetic_sanity_passed
        )

    def to_dict(self) -> dict:
        return asdict(self)


class ExperimentTracker:
    """Tracks all backtest experiments this session with structured data (JSONL + SQLite)."""

    def __init__(self, save_path: str = "./workspace/experiments.jsonl"):
        self._experiments: List[Experiment] = []
        self._save_path = Path(save_path)
        self._save_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_existing()
        # SQLite mirror
        from config import settings
        from data.database import TradingDatabase
        self._db = TradingDatabase(legacy_backup=settings.LEGACY_JSONL_BACKUP)

    def _load_existing(self):
        if self._save_path.exists():
            with open(self._save_path) as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        self._experiments.append(Experiment(**data))
                    except Exception:
                        pass
            logger.info("Loaded %d past experiments", len(self._experiments))

    def record(self, experiment: Experiment):
        self._experiments.append(experiment)
        with open(self._save_path, "a") as f:
            f.write(json.dumps(experiment.to_dict()) + "\n")
        # Mirror to SQLite
        try:
            self._db.insert_experiment({
                "strategy_id": f"{experiment.strategy_type}_{experiment.iteration}",
                "strategy_type": experiment.strategy_type,
                "params": experiment.params,
                "metrics": {
                    "sharpe": experiment.sharpe,
                    "win_rate": experiment.win_rate,
                    "max_drawdown": experiment.max_drawdown,
                    "total_trades": experiment.total_trades,
                    "profit_factor": experiment.profit_factor,
                },
                "regime": experiment.regime,
                "status": "completed",
                "verdict": experiment.verdict,
            })
        except Exception as exc:
            logger.warning("Failed to mirror experiment to SQLite: %s", exc)

    def get_best(self, strategy_type: str = "", regime: str = "", k: int = 5) -> List[Experiment]:
        candidates = self._experiments
        if strategy_type:
            candidates = [e for e in candidates if e.strategy_type == strategy_type]
        if regime:
            candidates = [e for e in candidates if e.regime == regime]
        candidates = [e for e in candidates if e.total_trades >= 5]
        return sorted(candidates, key=lambda e: e.score(), reverse=True)[:k]

    def get_untested_params(
        self,
        strategy_type: str,
        param_grid: Dict[str, List[Any]],
    ) -> Optional[Dict[str, Any]]:
        """Return the first param combination from grid that hasn't been tested yet."""
        tested = set()
        for e in self._experiments:
            if e.strategy_type == strategy_type:
                tested.add(json.dumps(e.params, sort_keys=True))

        import itertools
        keys = list(param_grid.keys())
        for combo in itertools.product(*param_grid.values()):
            candidate = dict(zip(keys, combo))
            if json.dumps(candidate, sort_keys=True) not in tested:
                return candidate
        return None  # all combinations tested

    def suggest_next_params(
        self,
        strategy_type: str,
        current_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Data-driven parameter suggestion based on past experiments.
        Finds the best result so far and moves toward it, then explores
        nearby untested combinations.
        """
        past = self.get_best(strategy_type=strategy_type, k=10)
        if not past:
            return current_params

        best = past[0]
        next_params = dict(current_params)

        # Define valid ranges per param to prevent drift
        PARAM_RANGES = {
            "fast_ma": (3, 50),
            "slow_ma": (10, 200),
            "macd_fast": (5, 20),
            "macd_slow": (15, 50),
            "macd_signal": (5, 15),
            "rsi_period": (7, 25),
            "rsi_buy_threshold": (20, 40),
            "rsi_sell_threshold": (60, 85),
            "bb_period": (10, 40),
            "stoploss": (-0.20, -0.01),
        }

        # For each numeric param, move toward the best known value
        for key, val in best.params.items():
            if key not in next_params:
                continue
            if not isinstance(val, (int, float)):
                continue

            current = next_params[key]
            if not isinstance(current, (int, float)):
                continue

            # Move 30% toward best value
            step = (val - current) * 0.3
            new_val = current + step

            # Enforce type
            if isinstance(val, int):
                new_val = int(round(new_val))
                # Ensure fast_ma < slow_ma
                if key == "fast_ma" and "slow_ma" in next_params:
                    new_val = min(new_val, next_params["slow_ma"] - 3)
                if key == "slow_ma" and "fast_ma" in next_params:
                    new_val = max(new_val, next_params["fast_ma"] + 3)
            else:
                new_val = round(float(new_val), 4)

            # Clamp to valid range
            if key in PARAM_RANGES:
                lo, hi = PARAM_RANGES[key]
                new_val = max(lo, min(hi, new_val))

            next_params[key] = new_val

        return next_params

    def summary(self) -> str:
        if not self._experiments:
            return "No experiments recorded yet."
        kept = [e for e in self._experiments if e.verdict == "kept"]
        discarded = [e for e in self._experiments if e.verdict == "discarded"]
        best = self.get_best(k=1)
        lines = [
            f"Total experiments: {len(self._experiments)}",
            f"Kept: {len(kept)}, Discarded: {len(discarded)}",
        ]
        if best:
            b = best[0]
            lines.append(
                f"Best so far: {b.strategy_type} "
                f"Sharpe={b.sharpe:.2f} WR={b.win_rate:.0%} "
                f"Trades={b.total_trades} Regime={b.regime}"
            )
        return "\n".join(lines)
