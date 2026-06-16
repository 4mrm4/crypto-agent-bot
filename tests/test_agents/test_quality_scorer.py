"""Tests for TradeQualityScorer — ML trade quality filter."""

import json
import os
import pickle
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

MODEL_PATH = Path("./workspace/quality_scorer.pkl")

import numpy as np
import pandas as pd
import pytest

from execution.quality_scorer import (
    BLOCK_THRESHOLD,
    HIGH_QUALITY_THRESHOLD,
    KNOWN_REGIMES,
    KNOWN_STRATEGY_TYPES,
    MIN_TRAINING_SAMPLES,
    TradeQualityScorer,
    QualityPrediction,
)
from execution.trade_signal import TradeSignal
from execution.live_executor import LiveExecutor


# ── Helpers ──

def _make_signal(strategy_type="momentum", regime="strong_uptrend",
                 sharpe=1.5, win_rate=0.55, max_drawdown=0.08,
                 confidence=0.7, **kwargs) -> TradeSignal:
    """Create a TradeSignal with sensible defaults."""
    return TradeSignal(
        pair="BTC/USDT",
        side="buy",
        strategy_name="test_strat",
        strategy_type=strategy_type,
        regime=regime,
        confidence=confidence,
        sharpe=sharpe,
        win_rate=win_rate,
        max_drawdown=max_drawdown,
        suggested_stoploss=-0.03,
        suggested_take_profit=0.06,
        source_agent="test",
        **kwargs,
    )


def _make_mock_experiment(**overrides) -> dict:
    """Build an experiment row matching database.py row format."""
    exp = {
        "id": "exp_test",
        "strategy_id": "strat_test",
        "strategy_type": "momentum",
        "params": "{}",
        "metrics": json.dumps({
            "sharpe": 1.5,
            "win_rate": 0.55,
            "max_drawdown": 0.08,
            "profit_factor": 1.8,
            "total_trades": 50,
        }),
        "regime": "strong_uptrend",
        "created_at": 1000000,
        "status": "completed",
        "verdict": "deployed",
    }
    exp.update(overrides)
    return exp


def _generate_training_rows(n: int, seed: int = 42) -> list:
    """Generate N synthetic experiment rows for training."""
    rng = np.random.RandomState(seed)
    rows = []
    for i in range(n):
        st = KNOWN_STRATEGY_TYPES[i % len(KNOWN_STRATEGY_TYPES)]
        reg = KNOWN_REGIMES[i % len(KNOWN_REGIMES)]
        sharpe = rng.uniform(0.5, 3.0)
        win_rate = rng.uniform(0.3, 0.8)
        dd = rng.uniform(0.02, 0.25)
        pf = rng.uniform(0.5, 4.0)
        tt = int(rng.uniform(10, 200))
        # Label: good if sharpe > 1.2 AND win_rate > 0.45 AND dd < 0.18
        label = 1 if (sharpe > 1.2 and win_rate > 0.45 and dd < 0.18) else 0
        verdict = "deployed" if label else "discarded"
        rows.append(_make_mock_experiment(
            id=f"exp_{i}",
            strategy_type=st,
            regime=reg,
            metrics=json.dumps({
                "sharpe": round(sharpe, 4),
                "win_rate": round(win_rate, 4),
                "max_drawdown": round(dd, 4),
                "profit_factor": round(pf, 4),
                "total_trades": tt,
            }),
            verdict=verdict,
        ))
    return rows


# ── Tests ──

class TestColdStart:
    """Scorer with no training data returns passthrough values."""

    def test_cold_start_returns_one(self):
        """Untrained scorer returns quality=1.0, multiplier=1.0, not blocked."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            model_path = Path(f.name)
        scorer = TradeQualityScorer(model_path=model_path)
        # No train() called — stays in cold start
        signal = _make_signal()
        pred = scorer.predict_quality(signal)
        assert pred.quality_score == 1.0
        assert pred.quality_multiplier == 1.0
        assert pred.blocked is False
        assert pred.model_trained is False
        assert "cold_start" in pred.reason

    def test_cold_start_accepts_signal_attributes(self):
        """Predict reads strategy_type/regime from signal duck-typing."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            model_path = Path(f.name)
        scorer = TradeQualityScorer(model_path=model_path)
        # Should not crash with minimal attributes
        pred = scorer.predict_quality(
            MagicMock(strategy_type="momentum", regime="uptrend",
                      sharpe=1.0, win_rate=0.5, max_drawdown=0.1)
        )
        assert pred.quality_score == 1.0  # cold start
        assert pred.blocked is False

    def test_cold_start_with_dict_signal(self):
        """Predict handles dict-like signal objects."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            model_path = Path(f.name)
        scorer = TradeQualityScorer(model_path=model_path)
        signal_dict = {
            "strategy_type": "momentum",
            "regime": "uptrend",
            "sharpe": 1.0,
            "win_rate": 0.5,
            "max_drawdown": 0.1,
        }
        pred = scorer.predict_quality(signal_dict)
        assert pred.quality_score == 1.0
        assert pred.blocked is False


class TestTraining:
    """Training the scorer from synthetic data."""

    def test_training_with_minimal_data(self):
        """Train on 30+ synthetic rows, verify predict returns meaningful values."""
        rows = _generate_training_rows(MIN_TRAINING_SAMPLES + 5)

        with patch("data.database.TradingDatabase") as mock_db_cls:
            mock_db = MagicMock()
            mock_db.query_experiments_with_verdict.return_value = rows
            mock_db_cls.return_value = mock_db

            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
                model_path = Path(f.name)
            scorer = TradeQualityScorer(model_path=model_path)
            trained = scorer.train()
            assert trained is True, "Should train with >= MIN_TRAINING_SAMPLES"
            assert scorer.is_trained()
            assert scorer._model is not None
            assert scorer._n_samples >= MIN_TRAINING_SAMPLES
            assert scorer._trained_at is not None

            # Predict on a known-good signal
            signal = _make_signal(strategy_type="momentum", regime="strong_uptrend")
            pred = scorer.predict_quality(signal)
            assert 0.0 <= pred.quality_score <= 1.0
            assert pred.model_trained is True
            assert pred.n_training_samples >= MIN_TRAINING_SAMPLES

    def test_insufficient_data_returns_false(self):
        """Train with < 30 samples returns False, stays cold."""
        rows = _generate_training_rows(10)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            model_path = Path(f.name)

        with patch("data.database.TradingDatabase") as mock_db_cls:
            mock_db = MagicMock()
            mock_db.query_experiments_with_verdict.return_value = rows
            mock_db_cls.return_value = mock_db

            scorer = TradeQualityScorer(model_path=model_path)
            trained = scorer.train()
            assert trained is False
            assert not scorer.is_trained()
            # Still cold-start
            pred = scorer.predict_quality(_make_signal())
            assert pred.quality_score == 1.0

    def test_force_training_override(self):
        """force=True overrides MIN_TRAINING_SAMPLES check."""
        rows = _generate_training_rows(5)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            model_path = Path(f.name)

        with patch("data.database.TradingDatabase") as mock_db_cls:
            mock_db = MagicMock()
            mock_db.query_experiments_with_verdict.return_value = rows
            mock_db_cls.return_value = mock_db

            scorer = TradeQualityScorer(model_path=model_path)
            trained = scorer.train(force=True)
            assert trained is True

    def test_training_empty_db(self):
        """Empty database leaves scorer in cold start."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            model_path = Path(f.name)
        with patch("data.database.TradingDatabase") as mock_db_cls:
            mock_db = MagicMock()
            mock_db.query_experiments_with_verdict.return_value = []
            mock_db_cls.return_value = mock_db

            scorer = TradeQualityScorer(model_path=model_path)
            trained = scorer.train()
            assert trained is False
            pred = scorer.predict_quality(_make_signal())
            assert pred.quality_score == 1.0


class TestFeatureEncoding:
    """Feature vector shape and content correctness."""

    def test_feature_encoding_shape(self):
        """Feature vector has correct number of elements."""
        scorer = TradeQualityScorer()
        n_st = len(KNOWN_STRATEGY_TYPES)  # 11
        n_reg = len(KNOWN_REGIMES)        # 7
        n_numeric = 6
        expected_size = n_st + n_reg + (n_st * n_reg) + n_numeric

        signal = _make_signal(strategy_type="momentum", regime="strong_uptrend")
        vec = scorer._build_feature_vector(signal, {})
        assert len(vec) == expected_size, (
            f"Expected {expected_size}, got {len(vec)}"
        )

    def test_one_hot_strategy_type(self):
        """Correct strategy_type column is 1, rest are 0."""
        scorer = TradeQualityScorer()
        feature_vec = scorer._build_feature_vector(
            _make_signal(strategy_type="bollinger_bands", regime="ranging"), {}
        )
        n_st = len(KNOWN_STRATEGY_TYPES)
        bollinger_idx = KNOWN_STRATEGY_TYPES.index("bollinger_bands")
        for i in range(n_st):
            if i == bollinger_idx:
                assert feature_vec[i] == 1.0, f"Expected 1 at strategy index {i}"
            else:
                assert feature_vec[i] == 0.0, f"Expected 0 at strategy index {i}"

    def test_one_hot_regime(self):
        """Correct regime column is 1, rest are 0."""
        scorer = TradeQualityScorer()
        n_st = len(KNOWN_STRATEGY_TYPES)
        feature_vec = scorer._build_feature_vector(
            _make_signal(strategy_type="momentum", regime="volatile"), {}
        )
        n_reg = len(KNOWN_REGIMES)
        volatile_idx = KNOWN_REGIMES.index("volatile")
        regime_start = n_st
        for i in range(n_reg):
            idx = regime_start + i
            if i == volatile_idx:
                assert feature_vec[idx] == 1.0, f"Expected 1 at regime index {i}"
            else:
                assert feature_vec[idx] == 0.0, f"Expected 0 at regime index {i}"

    def test_interaction_one_hot(self):
        """Strategy x regime interaction: only matching pair is 1."""
        scorer = TradeQualityScorer()
        n_st = len(KNOWN_STRATEGY_TYPES)
        n_reg = len(KNOWN_REGIMES)
        st = "macd_crossover"
        reg = "downtrend"
        feature_vec = scorer._build_feature_vector(
            _make_signal(strategy_type=st, regime=reg), {}
        )
        interaction_start = n_st + n_reg
        st_idx = KNOWN_STRATEGY_TYPES.index(st)
        reg_idx = KNOWN_REGIMES.index(reg)
        expected_interaction_idx = st_idx * n_reg + reg_idx

        for i in range(n_st * n_reg):
            idx = interaction_start + i
            if i == expected_interaction_idx:
                assert feature_vec[idx] == 1.0, f"Expected 1 at interaction idx {i}"
            else:
                assert feature_vec[idx] == 0.0, f"Expected 0 at interaction idx {i}"

    def test_numeric_features_negate_drawdown(self):
        """max_drawdown is negated in the feature vector."""
        scorer = TradeQualityScorer()
        signal = _make_signal(sharpe=2.0, win_rate=0.6, max_drawdown=0.05)
        vec = scorer._build_feature_vector(signal, {})
        # Numeric features start after one-hot + interaction
        n_st = len(KNOWN_STRATEGY_TYPES)
        n_reg = len(KNOWN_REGIMES)
        numeric_start = n_st + n_reg + (n_st * n_reg)
        # numeric order: sharpe, win_rate, max_drawdown_neg, profit_factor, log_total_trades, oos_passed
        assert vec[numeric_start] == 2.0     # sharpe
        assert vec[numeric_start + 1] == 0.6  # win_rate
        assert vec[numeric_start + 2] == -0.05  # negated max_drawdown
        assert vec[numeric_start + 3] == 1.0    # profit_factor (default)
        assert vec[numeric_start + 4] == 0.0    # log1p(0) = 0
        assert vec[numeric_start + 5] == 0.0    # oos_passed

    def test_unknown_strategy_regime(self):
        """Unknown strategy/regime values map to all zeros in one-hot blocks."""
        scorer = TradeQualityScorer()
        n_st = len(KNOWN_STRATEGY_TYPES)
        n_reg = len(KNOWN_REGIMES)
        # These are not in KNOWN_STRATEGY_TYPES / KNOWN_REGIMES
        vec = scorer._build_feature_vector(
            _make_signal(strategy_type="unknown_strat_xyz", regime="unknown_regime_xyz"), {}
        )
        # All one-hot entries should be 0
        onehot_len = n_st + n_reg + (n_st * n_reg)
        for i in range(onehot_len):
            assert vec[i] == 0.0, f"Expected 0 at one-hot index {i}"
        # Numeric features remain intact
        assert vec[onehot_len] == 1.5  # sharpe from _make_signal default


class TestQualityMultiplier:
    """Score → multiplier mapping."""

    def test_high_quality_full_multiplier(self):
        """Score >= 0.70 → multiplier 1.0."""
        assert TradeQualityScorer._score_to_multiplier(0.70) == 1.0
        assert TradeQualityScorer._score_to_multiplier(0.85) == 1.0
        assert TradeQualityScorer._score_to_multiplier(0.99) == 1.0
        assert TradeQualityScorer._score_to_multiplier(1.0) == 1.0

    def test_low_quality_blocked(self):
        """Score < 0.40 → multiplier 0.0 (block)."""
        assert TradeQualityScorer._score_to_multiplier(0.39) == 0.0
        assert TradeQualityScorer._score_to_multiplier(0.20) == 0.0
        assert TradeQualityScorer._score_to_multiplier(0.0) == 0.0
        assert TradeQualityScorer._score_to_multiplier(BLOCK_THRESHOLD - 0.001) == 0.0

    def test_mid_quality_proportional(self):
        """Score between 0.40 and 0.70 → multiplier = score."""
        assert TradeQualityScorer._score_to_multiplier(0.40) == 0.40
        assert TradeQualityScorer._score_to_multiplier(0.55) == 0.55
        assert TradeQualityScorer._score_to_multiplier(0.69) == 0.69

    def test_boundary_values(self):
        """Boundary at thresholds maps correctly."""
        assert TradeQualityScorer._score_to_multiplier(BLOCK_THRESHOLD) == 0.40  # >= 0.40, < 0.70
        assert TradeQualityScorer._score_to_multiplier(HIGH_QUALITY_THRESHOLD) == 1.0  # >= 0.70


class TestModelPersistence:
    """Save/load model round-trip."""

    def test_model_persistence(self, tmp_path):
        """Save then load model and verify consistent predictions."""
        rows = _generate_training_rows(MIN_TRAINING_SAMPLES)

        with patch("data.database.TradingDatabase") as mock_db_cls:
            mock_db = MagicMock()
            mock_db.query_experiments_with_verdict.return_value = rows
            mock_db_cls.return_value = mock_db
            mock_db_cls.return_value = mock_db

            model_path = tmp_path / "test_scorer.pkl"
            scorer = TradeQualityScorer(model_path=model_path)
            scorer.train()
            assert model_path.exists(), "Model file should exist after training"

            # Predict before reload
            signal = _make_signal(strategy_type="momentum", regime="strong_uptrend")
            pred_before = scorer.predict_quality(signal)

        # Load into a new scorer
        scorer2 = TradeQualityScorer(model_path=model_path)
        assert scorer2.is_trained(), "Should auto-load from model file"
        assert scorer2._n_samples == scorer._n_samples

        pred_after = scorer2.predict_quality(signal)
        assert pred_after.quality_score == pred_before.quality_score
        assert pred_after.quality_multiplier == pred_before.quality_multiplier
        assert pred_after.model_trained is True

    def test_model_persistence_no_file(self):
        """Scorer with non-existent model file starts in cold start."""
        scorer = TradeQualityScorer(model_path=Path("/nonexistent/path.pkl"))
        assert not scorer.is_trained()
        pred = scorer.predict_quality(_make_signal())
        assert pred.quality_score == 1.0

    def test_model_persistence_corrupted_file(self, tmp_path):
        """Corrupted pickle file falls back to cold start."""
        bad_path = tmp_path / "bad.pkl"
        bad_path.write_text("this is not a pickle")
        scorer = TradeQualityScorer(model_path=bad_path)
        assert not scorer.is_trained()
        pred = scorer.predict_quality(_make_signal())
        assert pred.quality_score == 1.0


class TestRetrainTrigger:
    """Auto-retrain every N trades."""

    def test_retrain_on_trade_count(self):
        """After RETRAIN_INTERVAL_TRADES calls to record_trade_executed, retrain triggers."""
        rows = _generate_training_rows(MIN_TRAINING_SAMPLES + 5)

        with patch("data.database.TradingDatabase") as mock_db_cls:
            mock_db = MagicMock()
            mock_db.query_experiments_with_verdict.return_value = rows
            mock_db_cls.return_value = mock_db

            scorer = TradeQualityScorer()
            scorer.train()
            original_train_count = scorer._n_samples

            # Mock train to track calls
            original_train = scorer.train
            train_call_count = [0]

            def tracking_train(force=False):
                train_call_count[0] += 1
                return original_train(force=force)

            scorer.train = tracking_train

            # Execute trades up to retrain threshold
            for i in range(25):
                scorer.record_trade_executed()
                if train_call_count[0] > 0:
                    break

            assert train_call_count[0] > 0, "Retrain should have been triggered"

    def test_no_retrain_below_threshold(self):
        """record_trade_executed < 25 should not trigger retrain."""
        scorer = TradeQualityScorer()
        scorer._trades_since_retrain = 0
        original_train = scorer.train
        train_called = [False]

        def noop_train(force=False):
            train_called[0] = True
            return False

        scorer.train = noop_train
        for _ in range(24):
            scorer.record_trade_executed()
        assert train_called[0] is False, "Train should not be called before threshold"


class TestIntegrationWithLiveExecutor:
    """Quality scorer integrated into LiveExecutor pipeline."""

    @pytest.mark.asyncio
    async def test_ml_filter_blocks_low_quality(self):
        """LiveExecutor blocks signal when quality < BLOCK_THRESHOLD."""
        mock_scorer = MagicMock(spec=TradeQualityScorer)
        mock_scorer.predict_quality.return_value = QualityPrediction(
            quality_score=0.25,
            quality_multiplier=0.0,
            n_training_samples=100,
            model_trained=True,
            blocked=True,
            reason="blocked",
        )

        executor = LiveExecutor(
            exchange_id="binance",
            paper_mode=True,
            quality_scorer=mock_scorer,
        )

        signal = _make_signal(position_size_usdt=100.0)
        result = await executor.execute_signal(signal)

        assert result.success is False
        assert "ML quality filter" in result.error
        # Signal should be updated
        assert signal.quality_score == 0.25
        assert signal.quality_multiplier == 0.0

    @pytest.mark.asyncio
    async def test_ml_filter_applies_multiplier(self):
        """LiveExecutor applies quality multiplier to position size."""
        mock_scorer = MagicMock(spec=TradeQualityScorer)
        mock_scorer.predict_quality.return_value = QualityPrediction(
            quality_score=0.55,
            quality_multiplier=0.55,
            n_training_samples=100,
            model_trained=True,
            blocked=False,
            reason="allowed",
        )

        executor = LiveExecutor(
            exchange_id="binance",
            paper_mode=True,
            quality_scorer=mock_scorer,
        )

        signal = _make_signal(position_size_usdt=200.0)
        result = await executor.execute_signal(signal)

        # Position size should be reduced by multiplier
        assert signal.position_size_usdt == 110.0  # 200 * 0.55 = 110.0

    @pytest.mark.asyncio
    async def test_ml_filter_high_quality_full_size(self):
        """High quality signal keeps full position size."""
        mock_scorer = MagicMock(spec=TradeQualityScorer)
        mock_scorer.predict_quality.return_value = QualityPrediction(
            quality_score=0.85,
            quality_multiplier=1.0,
            n_training_samples=100,
            model_trained=True,
            blocked=False,
            reason="allowed",
        )

        executor = LiveExecutor(
            exchange_id="binance",
            paper_mode=True,
            quality_scorer=mock_scorer,
        )

        signal = _make_signal(position_size_usdt=200.0)
        result = await executor.execute_signal(signal)

        assert signal.position_size_usdt == 200.0  # unchanged

    @pytest.mark.asyncio
    async def test_record_trade_on_success(self):
        """record_trade_executed is called after successful execution."""
        mock_scorer = MagicMock(spec=TradeQualityScorer)
        mock_scorer.predict_quality.return_value = QualityPrediction(
            quality_score=0.85,
            quality_multiplier=1.0,
            n_training_samples=100,
            model_trained=True,
            blocked=False,
            reason="allowed",
        )

        mock_fetcher = MagicMock()
        mock_fetcher.fetch_ohlcv.return_value = pd.DataFrame({
            "open": [50000.0]*10, "high": [50100.0]*10, "low": [49900.0]*10,
            "close": [50050.0]*10, "volume": [100.0]*10,
        })

        executor = LiveExecutor(
            exchange_id="binance",
            paper_mode=True,
            quality_scorer=mock_scorer,
            fetcher=mock_fetcher,
        )

        signal = _make_signal(position_size_usdt=50.0)
        result = await executor.execute_signal(signal)

        # Paper trade succeeds, so record_trade_executed should be called
        mock_scorer.record_trade_executed.assert_called_once()


class TestQualityPredictionDataclass:
    """QualityPrediction dataclass behavior."""

    def test_to_dict(self):
        pred = QualityPrediction(
            quality_score=0.5,
            quality_multiplier=0.5,
            n_training_samples=100,
            model_trained=True,
            blocked=False,
            reason="allowed",
        )
        d = pred.to_dict()
        assert d["quality_score"] == 0.5
        assert d["quality_multiplier"] == 0.5
        assert d["blocked"] is False
        assert d["model_trained"] is True
        assert d["reason"] == "allowed"

    def test_blocked_prediction(self):
        pred = QualityPrediction(
            quality_score=0.3,
            quality_multiplier=0.0,
            n_training_samples=100,
            model_trained=True,
            blocked=True,
            reason="blocked",
        )
        assert pred.blocked is True
        assert pred.quality_multiplier == 0.0


class TestSignalScannerMetrics:
    """Verify signal_scanner no longer uses hardcoded placeholder metrics."""

    def test_scanner_no_placeholder_constants(self):
        """Check that signal_scanner does not contain hardcoded placeholder values."""
        import execution.signal_scanner as scanner_module
        source = open(scanner_module.__file__).read()
        # These were the old placeholder values
        assert "sharpe=1.0" not in source
        assert "win_rate=0.5" not in source
        assert "max_drawdown=0.05" not in source
