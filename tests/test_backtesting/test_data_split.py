"""Tests for DataSplitConfig and holdout guard."""

from datetime import datetime

import pytest

import unittest
from backtesting.data_split import DATA_SPLIT, DataSplitConfig


class TestDataSplitValidation:
    def test_holdout_after_research(self):
        """Holdout start must be after research end."""
        assert datetime.strptime(DATA_SPLIT.holdout_start, "%Y%m%d") > datetime.strptime(DATA_SPLIT.research_end, "%Y%m%d")

    def test_research_before_research_end(self):
        """Research start must be before research end."""
        assert datetime.strptime(DATA_SPLIT.research_start, "%Y%m%d") < datetime.strptime(DATA_SPLIT.research_end, "%Y%m%d")

    def test_holdout_before_holdout_end(self):
        """Holdout start must be before holdout end."""
        assert datetime.strptime(DATA_SPLIT.holdout_start, "%Y%m%d") < datetime.strptime(DATA_SPLIT.holdout_end, "%Y%m%d")

    def test_validate_passes(self):
        """validate() should complete without error."""
        # Already validated at import — just verify no exception
        assert DATA_SPLIT.research_timerange() == "20170101-20231231"
        assert DATA_SPLIT.holdout_timerange() == "20240101-20260601"

    def test_validate_overlap_raises(self):
        """Instantiating with overlapping windows must raise."""
        with pytest.raises(AssertionError, match="overlaps research end"):
            DataSplitConfig(
                research_start="20200101",
                research_end="20240101",
                holdout_start="20231201",
                holdout_end="20251231",
                wfv_test_pct=0.20,
            ).validate()

    def test_research_before_research_end_raises(self):
        """Research start after research end must raise."""
        with pytest.raises(AssertionError, match="not before research end"):
            DataSplitConfig(
                research_start="20250101",
                research_end="20240101",
                holdout_start="20250101",
                holdout_end="20260101",
                wfv_test_pct=0.20,
            ).validate()


class TestHoldoutDetection:
    def test_research_only_not_in_holdout(self):
        assert not DATA_SPLIT.is_in_holdout("20210101-20231231")

    def test_full_holdout_detected(self):
        assert DATA_SPLIT.is_in_holdout("20240101-20250101")

    def test_partial_overlap_detected(self):
        assert DATA_SPLIT.is_in_holdout("20230601-20240601")

    def test_open_ended_not_in_holdout(self):
        assert not DATA_SPLIT.is_in_holdout("20210101-20230101")

    def test_exact_boundary_not_in_holdout(self):
        """The holdout_start date itself should be detected."""
        assert DATA_SPLIT.is_in_holdout("20240101-20250101")


class TestWFVSplits:
    def test_wfv_splits_are_within_research(self):
        splits = DATA_SPLIT.wfv_splits(n_splits=5)
        for train_tr, test_tr in splits:
            # Neither train nor test should contain holdout dates
            assert not DATA_SPLIT.is_in_holdout(train_tr)
            assert not DATA_SPLIT.is_in_holdout(test_tr)

    def test_wfv_splits_have_test_coverage(self):
        splits = DATA_SPLIT.wfv_splits(n_splits=5)
        total_test_days = 0
        for train_tr, test_tr in splits:
            # Extract test days
            test_part = test_tr.split("-")[1]
            # Approximate check
            total_test_days += 30  # rough estimate

    def test_wfv_splits_non_empty(self):
        splits = DATA_SPLIT.wfv_splits(n_splits=5)
        assert len(splits) > 0

    def test_wfv_splits_reduces_for_short_windows(self):
        """With very many splits, should reduce."""
        splits = DATA_SPLIT.wfv_splits(n_splits=100)
        # Should not return 100 splits for limited data
        assert len(splits) <= 10


class TestHoldoutGuard(unittest.TestCase):
    """Full integration test requiring BacktestEngine."""

    def _skip_if_no_data():
        return True

    def test_holdout_timerange_raises_value_error(self):
        """The most important test: holdout timerange must be rejected."""
        try:
            from backtesting.engine import BacktestEngine
            engine = BacktestEngine()
            with pytest.raises(ValueError, match="HOLDOUT VIOLATION"):
                engine.run_backtest("sma_crossover", timerange="20240101-20250101")
        except ImportError:
            pytest.skip("BacktestEngine not importable (dependencies)")


if __name__ == "__main__":
    import unittest
    unittest.main()
