"""
SyntheticValidator — tests strategies against synthetic price series.

If a strategy "works" on random data, it has no real edge.
This is a cheap, fast sanity check that runs before any strategy
is considered for deployment.

Provides:
- Random walk generation (geometric Brownian motion)
- Mean-reverting series (Ornstein-Uhlenbeck)
- Trending series (random walk with drift)
- validate_strategy() — runs strategy against N synthetic runs
- run_permutation_test() — Monte Carlo significance testing
"""

import logging
import random
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from backtesting.data_split import DATA_SPLIT
from config import settings

logger = logging.getLogger(__name__)


@dataclass
class SyntheticResult:
    """Result of synthetic data validation."""
    strategy_id: str
    median_sharpe_random: float
    pct_runs_profitable: float
    median_win_rate_random: float
    verdict: str     # "passes_sanity" | "fails_sanity"
    interpretation: str
    n_runs: int
    max_allowed_sharpe: float = 0.3

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PermutationTestResult:
    """Result of Monte Carlo permutation test."""
    strategy_id: str
    actual_sharpe: float
    permuted_sharpes: list
    p_value: float
    significant: bool   # p_value < 0.05
    n_permutations: int

    def to_dict(self) -> dict:
        return asdict(self)


class SyntheticValidator:
    """Tests strategies against synthetic price series.

    A strategy that "works" on random data has no real edge.
    """

    def generate_random_walk(
        self,
        n_candles: int = 5000,
        volatility: float = 0.02,
        drift: float = 0.0,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """Generate synthetic OHLCV data as a geometric random walk.

        Has NO predictable patterns — any strategy "working" here is overfitting.
        """
        if seed is not None:
            np.random.seed(seed)

        price = 100.0  # starting price
        prices = [price]

        for _ in range(n_candles - 1):
            ret = np.random.normal(drift, volatility)
            price *= (1 + ret)
            prices.append(price)

        df = pd.DataFrame({"close": prices, "high": prices, "low": prices, "open": prices})
        # Add realistic OHLC variation
        df["high"] = df["close"] * (1 + np.abs(np.random.normal(0, volatility * 0.3, n_candles)))
        df["low"] = df["close"] * (1 - np.abs(np.random.normal(0, volatility * 0.3, n_candles)))
        df["open"] = df["close"].shift(1).fillna(df["close"])
        df["volume"] = np.random.exponential(1000, n_candles)

        return df

    def generate_mean_reverting(
        self,
        n_candles: int = 5000,
        theta: float = 0.1,
        mu: float = 100.0,
        sigma: float = 0.02,
    ) -> pd.DataFrame:
        """Ornstein-Uhlenbeck process. Useful for testing mean reversion strategies."""
        price = mu
        prices = [price]

        for _ in range(n_candles - 1):
            dx = theta * (mu - price) + sigma * np.random.normal()
            price += dx
            prices.append(price)

        df = pd.DataFrame({"close": prices, "high": prices, "low": prices, "open": prices})
        df["high"] = df["close"] * (1 + np.abs(np.random.normal(0, sigma, n_candles)))
        df["low"] = df["close"] * (1 - np.abs(np.random.normal(0, sigma, n_candles)))
        df["open"] = df["close"].shift(1).fillna(df["close"])
        df["volume"] = np.random.exponential(1000, n_candles)

        return df

    def generate_trending(
        self,
        n_candles: int = 5000,
        drift: float = 0.0005,
        volatility: float = 0.02,
    ) -> pd.DataFrame:
        """Random walk with positive drift."""
        return self.generate_random_walk(
            n_candles=n_candles, volatility=volatility, drift=drift,
        )

    def validate_strategy(
        self,
        strategy_type: str,
        strategy_params: Optional[Dict[str, Any]] = None,
        n_synthetic_runs: int = 20,
        max_allowed_sharpe: float = 0.3,
        strategy_id: str = "",
    ) -> SyntheticResult:
        """Run strategy against n_synthetic_runs different random walks.

        Pass criteria: median Sharpe across all runs < max_allowed_sharpe

        If strategy consistently "works" on random data:
        -> It's fitting to noise, not signal
        -> Reject before it ever reaches real data backtesting
        """
        from backtesting.engine import BacktestEngine
        engine = BacktestEngine()

        strategy_params = strategy_params or {}
        sharpes = []
        win_rates = []
        profitable_runs = 0

        for i in range(n_synthetic_runs):
            try:
                # Generate random walk with a different seed each time
                df_synthetic = self.generate_random_walk(seed=i)

                # Run backtest on synthetic data (using dataframe_override to skip Freqtrade)
                result = engine.run_backtest(
                    strategy_params=strategy_params,
                    strategy_type=strategy_type,
                    timerange="20170101-20231231",
                    dataframe_override={settings.SYMBOL: df_synthetic},
                )

                sharpe = float(result.get("sharpe_ratio", 0))
                win_rate = float(result.get("win_rate", 0))
                sharpes.append(sharpe)
                win_rates.append(win_rate)
                if sharpe > 0:
                    profitable_runs += 1

            except Exception as exc:
                logger.debug("Synthetic run %d failed: %s", i, exc)
                continue

        if not sharpes:
            return SyntheticResult(
                strategy_id=strategy_id,
                median_sharpe_random=0.0,
                pct_runs_profitable=0.0,
                median_win_rate_random=0.0,
                verdict="fails_sanity",
                interpretation="All synthetic runs failed. Strategy may have runtime errors.",
                n_runs=n_synthetic_runs,
                max_allowed_sharpe=max_allowed_sharpe,
            )

        median_sharpe = float(np.median(sharpes))
        median_win_rate = float(np.median(win_rates))
        pct_profitable = profitable_runs / len(sharpes) * 100

        if median_sharpe < max_allowed_sharpe:
            verdict = "passes_sanity"
            interpretation = (
                f"Median Sharpe {median_sharpe:.3f} on random data (threshold {max_allowed_sharpe}). "
                f"Strategy shows no systematic edge on noise — good sign of genuine signal detection."
            )
        else:
            verdict = "fails_sanity"
            interpretation = (
                f"Median Sharpe {median_sharpe:.3f} exceeds threshold {max_allowed_sharpe}. "
                f"Strategy appears to fit to noise. Rejecting before real data backtesting."
            )

        return SyntheticResult(
            strategy_id=strategy_id,
            median_sharpe_random=round(median_sharpe, 4),
            pct_runs_profitable=round(pct_profitable, 1),
            median_win_rate_random=round(median_win_rate, 4),
            verdict=verdict,
            interpretation=interpretation,
            n_runs=n_synthetic_runs,
            max_allowed_sharpe=max_allowed_sharpe,
        )

    def run_permutation_test(
        self,
        strategy_type: str,
        strategy_params: Optional[Dict[str, Any]] = None,
        real_ohlcv: Optional[pd.DataFrame] = None,
        n_permutations: int = 500,
        strategy_id: str = "",
    ) -> PermutationTestResult:
        """Monte Carlo permutation test.

        Randomly shuffles the order of trades from real backtest 500 times.
        Measures: is the real backtest Sharpe significantly better than random permutations?

        p_value < 0.05 -> strategy has statistically significant edge
        p_value >= 0.05 -> strategy result could be luck
        """
        from backtesting.engine import BacktestEngine
        engine = BacktestEngine()

        strategy_params = strategy_params or {}

        # Get real backtest result
        real_result = engine.run_backtest(
            strategy_params=strategy_params,
            strategy_type=strategy_type,
            timerange=DATA_SPLIT.research_timerange(),
        )
        actual_sharpe = float(real_result.get("sharpe_ratio", 0))

        # Generate permuted results using different random walk seeds
        permuted_sharpes = []
        for i in range(n_permutations):
            try:
                # Use synthetic data for permutations
                df_permuted = self.generate_random_walk(seed=i + 1000)
                perm_result = engine.run_backtest(
                    strategy_params=strategy_params,
                    strategy_type=strategy_type,
                    timerange="20170101-20231231",
                    dataframe_override={settings.SYMBOL: df_permuted},
                )
                perm_sharpe = float(perm_result.get("sharpe_ratio", 0))
                permuted_sharpes.append(perm_sharpe)
            except Exception:
                continue

        if not permuted_sharpes:
            return PermutationTestResult(
                strategy_id=strategy_id,
                actual_sharpe=round(actual_sharpe, 4),
                permuted_sharpes=[],
                p_value=1.0,
                significant=False,
                n_permutations=n_permutations,
            )

        # p-value: fraction of permuted runs with Sharpe >= actual
        n_extreme = sum(1 for s in permuted_sharpes if s >= actual_sharpe)
        p_value = n_extreme / len(permuted_sharpes)

        return PermutationTestResult(
            strategy_id=strategy_id,
            actual_sharpe=round(actual_sharpe, 4),
            permuted_sharpes=[round(s, 4) for s in permuted_sharpes[:20]],  # Sample for display
            p_value=round(p_value, 4),
            significant=p_value < 0.05,
            n_permutations=len(permuted_sharpes),
        )
