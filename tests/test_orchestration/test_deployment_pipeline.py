"""Tests for DeploymentPipeline — run_full_pipeline, OOS operations, logging."""

from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from orchestration.deployment_pipeline import PipelineResult, GATE_NAMES


# ── Test: run_full_pipeline gate failures ──

class TestRunFullPipeline:
    """Mock each gate to fail at a specific point and verify the pipeline stops correctly."""

    @pytest.fixture
    def pipeline(self):
        with patch("data.database.TradingDatabase"):
            from orchestration.deployment_pipeline import DeploymentPipeline
            yield DeploymentPipeline(vector_store=MagicMock(), engine=None)

    def test_gate_1_fails_no_variants(self, pipeline):
        """Gate 1: BlindParameterSearch returns no variants."""
        with patch("backtesting.blind_search.BlindParameterSearch") as mock_bps:
            instance = mock_bps.return_value
            instance._generate_default_variants.return_value = []

            result = pipeline.run_full_pipeline(
                strategy_type="sma_crossover", regime="ranging",
            )

        assert result.failed_at == 1
        assert result.passed_gates == 0
        assert "No parameter variants" in result.reason

    def test_gate_1_exception(self, pipeline):
        """Gate 1: BlindParameterSearch raises exception."""
        with patch("backtesting.blind_search.BlindParameterSearch") as mock_bps:
            instance = mock_bps.return_value
            instance._generate_default_variants.side_effect = ValueError("search failed")

            result = pipeline.run_full_pipeline(
                strategy_type="sma_crossover", regime="ranging",
            )

        assert result.failed_at == 1
        assert "exception" in result.reason.lower()

    def test_gate_2_fails_batch_empty(self, pipeline):
        """Gate 2: batch_backtest returns empty results."""
        with patch("backtesting.blind_search.BlindParameterSearch") as mock_bps:
            instance = mock_bps.return_value
            instance._generate_default_variants.return_value = [{"params": {"fast_ma": 5}}]
            instance.batch_backtest.return_value = []

            result = pipeline.run_full_pipeline(
                strategy_type="sma_crossover", regime="ranging",
            )

        assert result.failed_at == 2
        assert result.passed_gates == 1

    def test_gate_3_select_best_none(self, pipeline):
        """After gate 2, select_best_for_wfv returns None."""
        with patch("backtesting.blind_search.BlindParameterSearch") as mock_bps:
            instance = mock_bps.return_value
            instance._generate_default_variants.return_value = [{"params": {"fast_ma": 5}}]
            instance.batch_backtest.return_value = [{"sharpe": 1.0}]
            instance.select_best_for_wfv.return_value = None

            result = pipeline.run_full_pipeline(
                strategy_type="sma_crossover", regime="ranging",
            )

        assert result.failed_at == 3
        assert result.passed_gates == 2

    def test_gate_5_convergence_fails(self, pipeline):
        """Gate 5: Experiment meets_deploy_criteria returns False."""
        with patch("backtesting.blind_search.BlindParameterSearch") as mock_bps, \
             patch("backtesting.synthetic_validator.SyntheticValidator") as mock_sv, \
             patch("backtesting.engine.BacktestEngine") as mock_be, \
             patch("backtesting.data_split.DATA_SPLIT") as mock_ds:

            bps_instance = mock_bps.return_value
            bps_instance._generate_default_variants.return_value = [{"params": {"fast_ma": 5}}]
            bps_instance.batch_backtest.return_value = [{"sharpe": 1.0}]
            bps_instance.select_best_for_wfv.return_value = {"params": {"fast_ma": 5}, "metrics": {}}

            sv_instance = mock_sv.return_value
            sv_instance.validate_strategy.return_value = MagicMock(verdict="passes_sanity")

            be_instance = mock_be.return_value
            be_instance.run_backtest.return_value = {
                "sharpe_ratio": 0.5, "win_rate": 0.3, "max_drawdown": -0.2, "total_trades": 10,
            }

            result = pipeline.run_full_pipeline(
                strategy_type="sma_crossover", regime="ranging",
            )

        assert result.failed_at == 5
        assert result.passed_gates == 4
        assert "Convergence check" in result.reason

    def test_gate_8_kelly_zeros(self, pipeline):
        """Gate 8: Kelly returns zero position size."""
        with patch("backtesting.blind_search.BlindParameterSearch") as mock_bps, \
             patch("backtesting.synthetic_validator.SyntheticValidator") as mock_sv, \
             patch("backtesting.engine.BacktestEngine") as mock_be, \
             patch("backtesting.data_split.DATA_SPLIT") as mock_ds, \
             patch("orchestration.experiment_tracker.Experiment.meets_deploy_criteria", return_value=True), \
             patch("backtesting.cpcv_validator.CPCVValidator") as mock_cpcv, \
             patch("agents.risk_manager.kelly_position_size_conservative") as mock_kelly:

            bps_instance = mock_bps.return_value
            bps_instance._generate_default_variants.return_value = [{"params": {"fast_ma": 5}}]
            bps_instance.batch_backtest.return_value = [{"sharpe": 1.0}]
            bps_instance.select_best_for_wfv.return_value = {"params": {"fast_ma": 5}, "metrics": {}}

            sv_instance = mock_sv.return_value
            sv_instance.validate_strategy.return_value = MagicMock(verdict="passes_sanity")

            be_instance = mock_be.return_value
            be_instance.run_backtest.return_value = {
                "sharpe_ratio": 1.5, "win_rate": 0.55, "max_drawdown": -0.08, "total_trades": 50,
                "avg_win_pct": 0.02, "avg_loss_pct": 0.01, "profit_ratio": 1.5,
            }

            mock_kelly.return_value = {"position_size_usdt": 0, "rationale": "too risky"}

            cpcv_instance = mock_cpcv.return_value
            cpcv_instance.validate.return_value = MagicMock(passed=True)

            result = pipeline.run_full_pipeline(
                strategy_type="sma_crossover", regime="ranging",
            )

        assert result.failed_at == 8
        assert result.passed_gates == 7

    def test_all_gates_pass(self, pipeline):
        """All 9 automated gates pass."""
        with patch("backtesting.blind_search.BlindParameterSearch") as mock_bps, \
             patch("backtesting.synthetic_validator.SyntheticValidator") as mock_sv, \
             patch("backtesting.engine.BacktestEngine") as mock_be, \
             patch("backtesting.data_split.DATA_SPLIT") as mock_ds, \
             patch("orchestration.experiment_tracker.Experiment.meets_deploy_criteria", return_value=True), \
             patch("agents.risk_manager.kelly_position_size_conservative") as mock_kelly, \
             patch("backtesting.cpcv_validator.CPCVValidator") as mock_cpcv:

            bps_instance = mock_bps.return_value
            bps_instance._generate_default_variants.return_value = [{"params": {"fast_ma": 5}}]
            bps_instance.batch_backtest.return_value = [{"sharpe": 1.0}]
            bps_instance.select_best_for_wfv.return_value = {"params": {"fast_ma": 5}, "metrics": {}}

            sv_instance = mock_sv.return_value
            sv_instance.validate_strategy.return_value = MagicMock(verdict="passes_sanity")
            sv_instance.run_permutation_test.return_value = MagicMock(significant=True)

            be_instance = mock_be.return_value
            be_instance.run_backtest.return_value = {
                "sharpe_ratio": 1.5, "win_rate": 0.55, "max_drawdown": -0.08, "total_trades": 50,
                "avg_win_pct": 0.02, "avg_loss_pct": 0.01, "profit_ratio": 1.5,
            }

            mock_kelly.return_value = {"position_size_usdt": 100.0, "rationale": "ok"}

            cpcv_instance = mock_cpcv.return_value
            cpcv_instance.validate.return_value = MagicMock(passed=True)

            result = pipeline.run_full_pipeline(
                strategy_type="sma_crossover", regime="ranging",
            )

        assert result.failed_at is None
        assert result.passed_gates == 9
        assert result.passed_all_automated


# ── Test: OOS operations ──

class TestOOSOperations:
    @pytest.fixture
    def pipeline(self):
        with patch("data.database.TradingDatabase"):
            from orchestration.deployment_pipeline import DeploymentPipeline
            yield DeploymentPipeline(vector_store=MagicMock(), engine=None)

    def test_get_pending_oos_strategies_returns_matching(self, pipeline):
        vs = pipeline._vector_store
        vs.get_best_strategies.return_value = [
            {"metadata": {"status": "pending_oos", "strategy_id": "s1", "sharpe": 1.2}},
            {"metadata": {"status": "deployable", "strategy_id": "s2", "sharpe": 1.5}},
            {"metadata": {"status": "pending_oos", "strategy_id": "s3", "sharpe": 0.9}},
        ]
        pending = pipeline.get_pending_oos_strategies()
        assert len(pending) == 2
        ids = [p["strategy_id"] for p in pending]
        assert "s1" in ids
        assert "s3" in ids
        assert "s2" not in ids

    def test_mark_oos_validated_deployable(self, pipeline):
        vs = pipeline._vector_store
        oos_result = MagicMock(passed=True, oos_sharpe=1.8, oos_win_rate=0.6)
        result = pipeline.mark_oos_validated("s1", oos_result)
        assert result is True
        vs.store_insight.assert_called_once()

    def test_mark_oos_validated_rejected(self, pipeline):
        vs = pipeline._vector_store
        oos_result = MagicMock(passed=False, oos_sharpe=0.3, oos_win_rate=0.4)
        result = pipeline.mark_oos_validated("s1", oos_result)
        assert result is False
        vs.store_insight.assert_called_once()

    def test_get_all_status_returns_summary(self, pipeline):
        vs = pipeline._vector_store
        vs.get_best_strategies.return_value = [
            {"metadata": {"status": "pending_oos", "strategy_id": "s1"}},
        ]
        summary = pipeline.get_all_status()
        assert summary["count"] == 1
        assert summary["total_gates"] == 11
        assert len(summary["strategies"]) == 1


# ── Test: _log_result ──

class TestLogResult:
    @pytest.fixture
    def pipeline(self):
        with patch("data.database.TradingDatabase"):
            from orchestration.deployment_pipeline import DeploymentPipeline
            yield DeploymentPipeline(vector_store=MagicMock(), engine=None)

    def test_log_result_writes_to_jsonl(self, pipeline, tmp_path):
        pipeline._results_path = str(tmp_path / "results.jsonl")
        result = PipelineResult(
            strategy_id="test_sma", strategy_type="sma_crossover",
            regime="ranging", passed_gates=9, total_gates=11,
            failed_at=None, failed_at_name="",
            reason="All passed", completed_at="2025-01-01T00:00:00",
        )
        pipeline._log_result(result)
        log_file = tmp_path / "results.jsonl"
        assert log_file.exists()
        content = log_file.read_text()
        assert "test_sma" in content
        assert "All passed" in content
