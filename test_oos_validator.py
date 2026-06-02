"""Tests for OOSValidator.

Critical: OOSValidator must never write to ChromaDB.
"""

import json
import os
from pathlib import Path

from backtesting.oos_validator import OOSValidator, OOSResult


class TestOOSValidator:
    def test_oos_result_dataclass(self):
        result = OOSResult(
            strategy_id="test_1",
            strategy_type="sma_crossover",
            research_sharpe=1.5,
            oos_sharpe=1.0,
            net_sharpe=0.85,
            research_win_rate=0.5,
            oos_win_rate=0.45,
            degradation_pct=0.33,
            passed=True,
            recommendation="deploy",
            validated_at="2025-01-01T00:00:00",
            holdout_window="20240101-20260101",
            oos_trades=25,
            oos_max_drawdown=0.08,
        )
        assert result.passed
        assert result.recommendation == "deploy"

    def test_oos_result_reject_high_degradation(self):
        result = OOSResult(
            strategy_id="test_2",
            strategy_type="sma_crossover",
            research_sharpe=2.0,
            oos_sharpe=0.5,
            net_sharpe=0.4,
            research_win_rate=0.6,
            oos_win_rate=0.3,
            degradation_pct=0.75,
            passed=False,
            recommendation="reject",
            validated_at="2025-01-01T00:00:00",
            holdout_window="20240101-20260101",
        )
        assert not result.passed
        assert result.recommendation == "reject"

    def test_oos_result_to_dict(self):
        result = OOSResult(
            strategy_id="test_3",
            strategy_type="sma_crossover",
            research_sharpe=1.0,
            oos_sharpe=0.8,
            net_sharpe=0.7,
            research_win_rate=0.5,
            oos_win_rate=0.4,
            degradation_pct=0.2,
            passed=True,
            recommendation="monitor_longer",
            validated_at="2025-01-01T00:00:00",
            holdout_window="20240101-20260101",
        )
        d = result.to_dict()
        assert d["strategy_id"] == "test_3"
        assert d["passed"] is True

    def test_compute_degradation(self):
        research = {"sharpe_ratio": 1.5}
        oos = {"sharpe_ratio": 0.9}
        deg = OOSValidator.compute_degradation(research, oos)
        assert abs(deg - 0.4) < 0.01  # 40% degradation

    def test_compute_degradation_no_research(self):
        deg = OOSValidator.compute_degradation({"sharpe_ratio": 0}, {"sharpe_ratio": 0})
        assert deg == 0.0

    def test_log_result_writes_to_separate_file(self, tmp_path):
        validator = OOSValidator()
        # Override path to temp
        validator.OOS_RESULTS_PATH = tmp_path / "oos_results.jsonl"

        result = OOSResult(
            strategy_id="test_log", strategy_type="sma_crossover",
            research_sharpe=1.0, oos_sharpe=0.8, net_sharpe=0.7,
            research_win_rate=0.5, oos_win_rate=0.4,
            degradation_pct=0.2, passed=True, recommendation="deploy",
            validated_at="2025-01-01T00:00:00",
            holdout_window="20240101-20260101",
        )
        validator._log_result(result)
        assert validator.OOS_RESULTS_PATH.exists()
        lines = validator.OOS_RESULTS_PATH.read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["strategy_id"] == "test_log"

    def test_get_results_from_file(self, tmp_path):
        validator = OOSValidator()
        validator.OOS_RESULTS_PATH = tmp_path / "oos_results.jsonl"

        # Write a test entry
        with open(validator.OOS_RESULTS_PATH, "w") as f:
            f.write(json.dumps({
                "strategy_id": "get_test", "strategy_type": "sma_crossover",
                "research_sharpe": 1.0, "oos_sharpe": 0.8, "net_sharpe": 0.7,
                "research_win_rate": 0.5, "oos_win_rate": 0.4,
                "degradation_pct": 0.2, "passed": True, "recommendation": "deploy",
                "validated_at": "2025-01-01T00:00:00",
                "holdout_window": "20240101-20260101",
            }) + "\n")

        results = validator.get_results()
        assert len(results) == 1
        assert results[0].strategy_id == "get_test"

    def test_get_results_no_file(self):
        validator = OOSValidator()
        validator.OOS_RESULTS_PATH = Path("/tmp/nonexistent_oos_results.jsonl")
        results = validator.get_results()
        assert results == []
