"""
CPCVValidator — Combinatorial Purged Cross-Validation for strategy robustness.

Replaces Gate 6 (walk-forward validation) in the deployment pipeline.
Tests a strategy across all C(n_folds, k_test) combinatorial train/test
splits, purging overlapping label spans and applying an embargo period.

Core components:
  - CPCVSplitter: generates combinatorial fold indices with purge/embargo
  - CPCVResult: dataclass for the distribution of per-path metrics
  - CPCVValidator: orchestrates strategy evaluation across all paths

Thresholds are deliberately modest — CPCV is a noise filter, not a quality
gate. Strict quality enforcement is Gate 10 (OOS validation).
"""

import itertools
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Default CPCV parameters ──
DEFAULT_N_FOLDS = 6
DEFAULT_K_TEST = 2
DEFAULT_EMBARGO_DAYS = 5
MIN_TRADES_PER_PATH = 5
MAX_NAN_RATIO = 0.30

# ── Pass/fail thresholds (noise filter, not quality gate) ──
MIN_MEDIAN_SHARPE = 0.0
MIN_LOWER_QUARTILE = -0.5
MIN_PCT_POSITIVE = 0.50


# ── CPCV Result ──

@dataclass
class CPCVResult:
    """Distribution of strategy performance across all CPCV paths."""
    sharpe_distribution: List[float] = field(default_factory=list)
    avg_sharpe: float = 0.0
    sharpe_std: float = 0.0
    sharpe_lower_quartile: float = 0.0
    sharpe_upper_quartile: float = 0.0
    median_sharpe: float = 0.0
    pct_positive_sharpe: float = 0.0
    n_paths: int = 0
    n_paths_valid: int = 0
    n_folds: int = DEFAULT_N_FOLDS
    passed: bool = False
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "avg_sharpe": round(self.avg_sharpe, 4),
            "sharpe_std": round(self.sharpe_std, 4),
            "sharpe_lower_quartile": round(self.sharpe_lower_quartile, 4),
            "sharpe_upper_quartile": round(self.sharpe_upper_quartile, 4),
            "median_sharpe": round(self.median_sharpe, 4),
            "pct_positive_sharpe": round(self.pct_positive_sharpe, 4),
            "n_paths": self.n_paths,
            "n_paths_valid": self.n_paths_valid,
            "n_folds": self.n_folds,
            "passed": self.passed,
            "reason": self.reason,
        }


# ── CPCV Splitter ──

class CPCVSplitter:
    """Generates Combinatorial Purged Cross-Validation train/test index splits.

    Given N total folds and k test folds per path, generates all
    C(n_folds, k_test) combinatorial combinations. Each combination
    yields a (train_idx, test_idx) pair.

    Purging removes training samples whose (hypothetical) label spans
    would overlap with the test window. Embargo removes an additional
    buffer of samples after each test fold boundary.

    Args:
        n_folds: Total number of folds to partition the data into.
        k_test: Number of test folds per combinatorial path.
        embargo_days: Number of samples to embargo after each test fold.
            Interpreted as bar count (not calendar days).
    """

    def __init__(self, n_folds: int = DEFAULT_N_FOLDS,
                 k_test: int = DEFAULT_K_TEST,
                 embargo_days: int = DEFAULT_EMBARGO_DAYS):
        if n_folds < 3:
            raise ValueError(f"n_folds must be >= 3, got {n_folds}")
        if k_test < 1 or k_test >= n_folds:
            raise ValueError(f"k_test must be 1 <= k_test < n_folds, got {k_test}")
        if embargo_days < 0:
            raise ValueError(f"embargo_days must be >= 0, got {embargo_days}")

        self.n_folds = n_folds
        self.k_test = k_test
        self.embargo = embargo_days

    @property
    def n_train_folds(self) -> int:
        return self.n_folds - self.k_test

    @property
    def n_paths(self) -> int:
        """Number of combinatorial paths: C(n_folds, k_test)."""
        from math import comb
        return comb(self.n_folds, self.k_test)

    def _get_fold_ranges(self, n_samples: int) -> List[Tuple[int, int]]:
        """Partition [0, n_samples) into n_folds contiguous index ranges.

        Returns list of (start, end) tuples, one per fold.
        """
        fold_size = n_samples // self.n_folds
        remainder = n_samples % self.n_folds
        folds = []
        start = 0
        for i in range(self.n_folds):
            # Distribute remainder across first folds
            size = fold_size + (1 if i < remainder else 0)
            end = start + size
            folds.append((start, end))
            start = end
        return folds

    def split(self, n_samples: int):
        """Generate (train_idx, test_idx) for each combinatorial path.

        Args:
            n_samples: Total number of samples (rows) in the dataset.

        Yields:
            (train_idx, test_idx): numpy arrays of indices for each path.
        """
        fold_ranges = self._get_fold_ranges(n_samples)
        all_folds = list(range(self.n_folds))

        # All C(n_folds, k_test) combinations of test folds
        for test_folds in itertools.combinations(all_folds, self.k_test):
            test_folds = set(test_folds)
            train_folds = [f for f in all_folds if f not in test_folds]

            # Test indices
            test_idx = []
            for f in sorted(test_folds):
                s, e = fold_ranges[f]
                test_idx.extend(range(s, e))

            # Train indices with purging and embargo
            train_idx = []
            for f in train_folds:
                s, e = fold_ranges[f]
                train_idx.extend(range(s, e))

            # Purge: remove train samples whose fold is adjacent to or overlapping test
            for tf in test_folds:
                test_start = fold_ranges[tf][0]
                test_end = fold_ranges[tf][1]

                # Purge train samples that are too close to test window
                # (within the same label span — roughly one fold width before test)
                purge_before = max(0, test_start - (test_end - test_start))
                # Embargo: remove train samples after test fold
                embargo_end = min(n_samples, test_end + self.embargo)

                train_idx = [i for i in train_idx
                             if i < purge_before or i >= test_end]
                # Apply embargo (remove samples after test window)
                train_idx = [i for i in train_idx
                             if i < test_end or i >= embargo_end]

            yield np.array(train_idx, dtype=np.int64), np.array(test_idx, dtype=np.int64)


# ── CPCV Validator ──

class CPCVValidator:
    """Runs Combinatorial Purged Cross-Validation on a strategy.

    Uses SignalFactory to generate signals per path and FastMetrics
    to compute per-path performance. Aggregates results into a CPCVResult.

    Paths with fewer than MIN_TRADES_PER_PATH trades produce NaN Sharpe
    and are excluded from the distribution. If more than MAX_NAN_RATIO
    of paths are NaN, validation fails.
    """

    def __init__(self, splitter: Optional[CPCVSplitter] = None):
        self._splitter = splitter or CPCVSplitter()

    def validate(
        self,
        df: pd.DataFrame,
        strategy_type: str,
        params: Optional[dict] = None,
        n_folds: Optional[int] = None,
        k_test: Optional[int] = None,
    ) -> CPCVResult:
        """Run CPCV on a strategy over the given OHLCV data.

        Args:
            df: OHLCV DataFrame.
            strategy_type: Strategy type key (must be in SignalFactory.REGISTRY).
            params: Strategy parameters (dict of param_name -> value).
            n_folds: Override default fold count.
            k_test: Override default test folds per path.

        Returns:
            CPCVResult with the distribution of per-path metrics.
        """
        n_folds = n_folds or self._splitter.n_folds
        k_test = k_test or self._splitter.k_test
        embargo = self._splitter.embargo

        splitter = CPCVSplitter(n_folds=n_folds, k_test=k_test, embargo_days=embargo)
        n_samples = len(df)

        if n_samples < n_folds * 20:
            return CPCVResult(
                n_paths=0, n_folds=n_folds, passed=False,
                reason=f"Insufficient samples: {n_samples} for {n_folds} folds (need {n_folds * 20})",
            )

        from backtesting.signal_factory import FastMetrics, SignalFactory

        sharpe_values: List[float] = []
        path_trades: List[int] = []
        n_total = 0

        for train_idx, test_idx in splitter.split(n_samples):
            n_total += 1

            if len(test_idx) < MIN_TRADES_PER_PATH:
                # Not enough test samples to evaluate
                continue

            try:
                # Generate signals on test portion using params
                test_df = df.iloc[test_idx].copy()
                signals = SignalFactory.generate(test_df, strategy_type, params or {})

                # Compute metrics
                metrics = FastMetrics.compute(test_df, signals)
                trades = metrics.get("total_trades", 0)
                sharpe = metrics.get("sharpe_ratio", 0.0)

                path_trades.append(trades)

                if trades >= MIN_TRADES_PER_PATH and sharpe != 0.0:
                    sharpe_values.append(sharpe)
                # else: NaN path, excluded
            except Exception as exc:
                logger.debug("CPCV path failed: %s", exc)
                path_trades.append(0)
                continue

        # Check NaN ratio
        if n_total == 0:
            return CPCVResult(
                n_paths=0, n_folds=n_folds, passed=False,
                reason="No valid CPCV paths generated",
            )

        n_valid = len(sharpe_values)
        nan_ratio = 1.0 - (n_valid / n_total) if n_total > 0 else 1.0

        if nan_ratio > MAX_NAN_RATIO:
            return CPCVResult(
                n_paths=n_total, n_paths_valid=n_valid, n_folds=n_folds,
                passed=False,
                reason=f"Insufficient trades across CPCV paths: {n_valid}/{n_total} valid "
                       f"({nan_ratio:.0%} NaN, threshold {MAX_NAN_RATIO:.0%})",
            )

        if n_valid == 0:
            return CPCVResult(
                n_paths=n_total, n_paths_valid=0, n_folds=n_folds,
                passed=False,
                reason="No valid paths with sufficient trades",
            )

        # Aggregate statistics
        arr = np.array(sharpe_values)
        avg_sharpe = float(np.mean(arr))
        sharpe_std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        median_sharpe = float(np.median(arr))
        lower_quartile = float(np.percentile(arr, 25))
        upper_quartile = float(np.percentile(arr, 75))
        pct_positive = float(np.mean(arr > 0))

        # Pass/fail decision
        passed = (
            median_sharpe >= MIN_MEDIAN_SHARPE
            and lower_quartile >= MIN_LOWER_QUARTILE
            and pct_positive >= MIN_PCT_POSITIVE
        )

        reason = ""
        if passed:
            reason = (
                f"CPCV passed: median Sharpe={median_sharpe:.2f}, "
                f"Q1={lower_quartile:.2f}, {pct_positive:.0%} paths positive"
                f" ({n_valid}/{n_total} valid paths)"
            )
        else:
            failures = []
            if median_sharpe < MIN_MEDIAN_SHARPE:
                failures.append(f"median Sharpe {median_sharpe:.2f} < {MIN_MEDIAN_SHARPE}")
            if lower_quartile < MIN_LOWER_QUARTILE:
                failures.append(f"lower quartile {lower_quartile:.2f} < {MIN_LOWER_QUARTILE}")
            if pct_positive < MIN_PCT_POSITIVE:
                failures.append(f"only {pct_positive:.0%} paths positive < {MIN_PCT_POSITIVE:.0%}")
            reason = f"CPCV failed: {', '.join(failures)} ({n_valid}/{n_total} valid paths)"

        return CPCVResult(
            sharpe_distribution=sharpe_values,
            avg_sharpe=avg_sharpe,
            sharpe_std=sharpe_std,
            sharpe_lower_quartile=lower_quartile,
            sharpe_upper_quartile=upper_quartile,
            median_sharpe=median_sharpe,
            pct_positive_sharpe=pct_positive,
            n_paths=n_total,
            n_paths_valid=n_valid,
            n_folds=n_folds,
            passed=passed,
            reason=reason,
        )
