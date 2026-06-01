"""
DataSplitConfig — single source of truth for all data windows.

Set once at import. Never modified by agents or the autonomous loop.
Frozen dataclass ensures runtime immutability.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple


@dataclass(frozen=True)
class DataSplitConfig:
    """Defines the hard research/holdout split for all backtesting.

    - research window: LLM CAN see results from this period
    - holdout window: LLM NEVER sees results from this period (reserved for OOS validation)
    - wfv_test_pct: percentage of research window reserved for walk-forward test folds
    """

    research_start: str      # "20170101" — earliest data the LLM can use
    research_end: str        # "20231231" — LLM never sees data after this
    holdout_start: str       # "20240101" — reserved for OOS validation only
    holdout_end: str         # "20260601" — end of OOS window
    wfv_test_pct: float      # 0.20 — 20% of research window used as WFV test set

    def validate(self) -> None:
        """Assert holdout_start > research_end. Raise if overlap detected."""
        r_end = datetime.strptime(self.research_end, "%Y%m%d")
        h_start = datetime.strptime(self.holdout_start, "%Y%m%d")
        assert h_start > r_end, (
            f"FATAL: Holdout start {self.holdout_start} overlaps research end {self.research_end}. "
            "This invalidates all research results. Fix DataSplitConfig immediately."
        )
        r_start = datetime.strptime(self.research_start, "%Y%m%d")
        assert r_start < r_end, (
            f"FATAL: Research start {self.research_start} is not before research end "
            f"{self.research_end}."
        )
        h_end = datetime.strptime(self.holdout_end, "%Y%m%d")
        assert h_start < h_end, (
            f"FATAL: Holdout start {self.holdout_start} is not before holdout end "
            f"{self.holdout_end}."
        )

    def research_timerange(self) -> str:
        """Freqtrade-compatible timerange string for research backtests."""
        return f"{self.research_start}-{self.research_end}"

    def holdout_timerange(self) -> str:
        """Freqtrade-compatible timerange string for OOS validation."""
        return f"{self.holdout_start}-{self.holdout_end}"

    def is_in_holdout(self, timerange: str) -> bool:
        """Check if a timerange string overlaps the holdout window."""
        # Extract all digit groups and look for dates >= holdout_start
        import re
        groups = re.findall(r"\d+", timerange)
        for g in groups:
            if len(g) >= 8:
                candidate = g[:8]
                if candidate >= self.holdout_start:
                    return True
        return False

    def wfv_splits(self, n_splits: int = 5) -> List[Tuple[str, str]]:
        """Return N walk-forward train/test splits within the research window only.

        Each split: (train_timerange, test_timerange).
        Test periods never overlap. Total test coverage = wfv_test_pct of research window.
        All splits stay strictly within the research window.
        """
        r_start = datetime.strptime(self.research_start, "%Y%m%d")
        r_end = datetime.strptime(self.research_end, "%Y%m%d")
        total_days = (r_end - r_start).days

        test_days = int(total_days * self.wfv_test_pct)
        test_window_days = test_days // n_splits

        if test_window_days < 7:
            # Not enough data for meaningful splits — reduce n_splits
            n_splits = max(2, min(10, test_days // 14))
            test_window_days = test_days // n_splits

        train_window_days = (total_days - test_days) // n_splits

        splits: List[Tuple[str, str]] = []
        for i in range(n_splits):
            train_start = r_start
            train_end = min(
                r_start + timedelta(days=train_window_days),
                r_end - timedelta(days=test_window_days),  # reserve room for test
            )
            test_start = min(train_end, r_end)
            test_end = min(test_start + timedelta(days=test_window_days), r_end)

            # Guard against empty windows
            if (train_end - train_start).days < 7 or (test_end - test_start).days < 3:
                break

            splits.append((
                f"{train_start.strftime('%Y%m%d')}-{train_end.strftime('%Y%m%d')}",
                f"{test_start.strftime('%Y%m%d')}-{test_end.strftime('%Y%m%d')}",
            ))

            # Next train window starts where this test ended (expanding window)
            r_start = test_end

        return splits


# ── Singleton — import this everywhere ──
from datetime import timedelta  # noqa: E402 — needed for wfv_splits

DATA_SPLIT = DataSplitConfig(
    research_start="20170101",
    research_end="20231231",
    holdout_start="20240101",
    holdout_end="20260601",
    wfv_test_pct=0.20,
)
DATA_SPLIT.validate()
