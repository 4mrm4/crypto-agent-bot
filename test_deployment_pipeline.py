"""Tests for DeploymentPipeline."""

from orchestration.deployment_pipeline import PipelineResult, GATE_NAMES


class TestPipelineResult:
    def test_passed_all_automated(self):
        result = PipelineResult(
            strategy_id="test", strategy_type="sma_crossover",
            regime="ranging", passed_gates=9, total_gates=11,
            failed_at=None, failed_at_name="",
            reason="All automated passed",
            completed_at="2025-01-01T00:00:00",
        )
        assert result.passed_all_automated
        assert result.status == "pending_oos"

    def test_failed_at_gate(self):
        result = PipelineResult(
            strategy_id="test", strategy_type="sma_crossover",
            regime="ranging", passed_gates=4, total_gates=11,
            failed_at=5, failed_at_name=GATE_NAMES[4],
            reason="Convergence check failed",
            completed_at="2025-01-01T00:00:00",
        )
        assert not result.passed_all_automated
        assert result.status == "failed_at_gate_5"

    def test_oos_passed(self):
        result = PipelineResult(
            strategy_id="test", strategy_type="sma_crossover",
            regime="ranging", passed_gates=11, total_gates=11,
            failed_at=None, failed_at_name="",
            reason="Deployable", completed_at="2025-01-01T00:00:00",
            oos_passed=True,
        )
        assert result.status == "deployable"

    def test_oos_rejected(self):
        result = PipelineResult(
            strategy_id="test", strategy_type="sma_crossover",
            regime="ranging", passed_gates=10, total_gates=11,
            failed_at=None, failed_at_name="",
            reason="OOS rejected", completed_at="2025-01-01T00:00:00",
            oos_passed=False,
        )
        assert result.status == "oos_rejected"

    def test_to_dict(self):
        result = PipelineResult(
            strategy_id="test", strategy_type="sma_crossover",
            regime="ranging", passed_gates=9, total_gates=11,
            failed_at=None, failed_at_name="",
            reason="Test", completed_at="2025-01-01T00:00:00",
        )
        d = result.to_dict()
        assert d["strategy_id"] == "test"
        assert result.passed_all_automated


class TestGateNames:
    def test_all_gates_named(self):
        assert len(GATE_NAMES) == 11
        assert "BlindParameterSearch" in GATE_NAMES[0]
        assert "SyntheticValidator" in GATE_NAMES[2]
        assert "BacktestEngine" in GATE_NAMES[3]
        assert "OOSValidator" in GATE_NAMES[9]
        assert "deployable" in GATE_NAMES[10]

    def test_gate_names_are_unique(self):
        assert len(GATE_NAMES) == len(set(GATE_NAMES))


class TestStatusToGate:
    def test_explored_maps_to_gate_4(self):
        from orchestration.deployment_pipeline import DeploymentPipeline
        assert DeploymentPipeline._status_to_gate("explored") == 4

    def test_pending_oos_maps_to_gate_9(self):
        from orchestration.deployment_pipeline import DeploymentPipeline
        assert DeploymentPipeline._status_to_gate("pending_oos") == 9

    def test_deployable_maps_to_gate_11(self):
        from orchestration.deployment_pipeline import DeploymentPipeline
        assert DeploymentPipeline._status_to_gate("deployable") == 11

    def test_unknown_maps_to_0(self):
        from orchestration.deployment_pipeline import DeploymentPipeline
        assert DeploymentPipeline._status_to_gate("unknown") == 0
