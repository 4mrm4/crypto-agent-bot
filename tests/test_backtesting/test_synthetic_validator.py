import pytest
"""Tests for SyntheticValidator."""

from backtesting.synthetic_validator import SyntheticValidator


class TestSyntheticGeneration:
    def test_random_walk_shape(self):
        sv = SyntheticValidator()
        df = sv.generate_random_walk(n_candles=100, seed=42)
        assert len(df) == 100
        assert "close" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "volume" in df.columns

    def test_random_walk_different_seeds(self):
        sv = SyntheticValidator()
        df1 = sv.generate_random_walk(n_candles=100, seed=1)
        df2 = sv.generate_random_walk(n_candles=100, seed=2)
        # Different seeds should produce different series
        assert df1["close"].iloc[-1] != df2["close"].iloc[-1]

    def test_random_walk_no_drift(self):
        sv = SyntheticValidator()
        df = sv.generate_random_walk(n_candles=1000, drift=0.0, seed=42)
        # Close should end near starting price (100) with no drift
        assert 50 < df["close"].iloc[-1] < 200

    def test_mean_reverting_shape(self):
        sv = SyntheticValidator()
        df = sv.generate_mean_reverting(n_candles=100)
        assert len(df) == 100
        assert "close" in df.columns

    def test_trending_shape(self):
        sv = SyntheticValidator()
        df = sv.generate_trending(n_candles=100)
        assert len(df) == 100

    def test_random_walk_high_volatility(self):
        sv = SyntheticValidator()
        df = sv.generate_random_walk(volatility=0.05, n_candles=100, seed=1)
        # High vol should still produce valid data
        assert len(df) == 100


class TestSyntheticValidation:
    def test_validate_strategy_no_runs(self):
        """When all runs fail, should return fails_sanity."""
        sv = SyntheticValidator()
        result = sv.validate_strategy(
            strategy_type="nonexistent",
            n_synthetic_runs=3,
        )
        assert result.verdict in ("passes_sanity", "fails_sanity")

    def test_validate_strategy_returns_result(self):
        sv = SyntheticValidator()
        result = sv.validate_strategy(
            strategy_type="sma_crossover",
            n_synthetic_runs=3,
            strategy_id="test_id",
        )
        assert result.strategy_id == "test_id"
        assert result.n_runs > 0
        assert result.verdict in ("passes_sanity", "fails_sanity")

    def test_permutation_test_returns_result(self):
        sv = SyntheticValidator()
        result = sv.run_permutation_test(
            strategy_type="sma_crossover",
            n_permutations=50,
            strategy_id="test_perm",
        )
        assert result.strategy_id == "test_perm"
        assert result.n_permutations > 0
        assert isinstance(result.significant, bool)
