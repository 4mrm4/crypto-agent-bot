"""Tests for PerformanceMonitor."""

from monitoring.performance_monitor import PerformanceMonitor, EXPECTED_DEGRADATION


class TestDegradationScoring:
    def test_expected_degradation_ranges_exist(self):
        assert "sharpe" in EXPECTED_DEGRADATION
        assert "win_rate" in EXPECTED_DEGRADATION
        assert "max_dd" in EXPECTED_DEGRADATION

    def test_is_degradation_normal_within(self):
        assert PerformanceMonitor.is_degradation_normal("sharpe", 0.4)

    def test_is_degradation_normal_outside(self):
        assert not PerformanceMonitor.is_degradation_normal("sharpe", 0.8)

    def test_compute_degradation_no_trades(self):
        pm = PerformanceMonitor()
        report = pm.compute_degradation_score(
            {"sharpe": 1.5, "win_rate": 0.5},
            {"sharpe": 0.0, "win_rate": 0.0},
            n_live_trades=0,
        )
        assert not report.statistically_significant
        assert report.recommendation == "insufficient_data"

    def test_compute_degradation_sufficient_trades(self):
        pm = PerformanceMonitor()
        report = pm.compute_degradation_score(
            {"sharpe": 1.5, "win_rate": 0.5, "max_dd": 0.05},
            {"sharpe": 1.2, "win_rate": 0.45, "max_dd": 0.08},
            n_live_trades=30,
        )
        assert report.statistically_significant

    def test_compute_degradation_critical(self):
        pm = PerformanceMonitor()
        report = pm.compute_degradation_score(
            {"sharpe": 2.0, "win_rate": 0.6, "max_dd": 0.05},
            {"sharpe": 0.2, "win_rate": 0.2, "max_dd": 0.3},
            n_live_trades=30,
        )
        # At least one metric should be critical
        assert report.alert_level in ("critical", "warning")

    def test_degradation_report_has_expected_fields(self):
        pm = PerformanceMonitor()
        report = pm.compute_degradation_score(
            {"sharpe": 1.0, "win_rate": 0.5, "max_dd": 0.1},
            {"sharpe": 0.8, "win_rate": 0.45, "max_dd": 0.12},
            n_live_trades=50,
        )
        d = report.to_dict()
        assert "strategy_id" in d
        assert "n_live_trades" in d
        assert "statistically_significant" in d
        assert "metrics" in d
        assert "overall_degradation_pct" in d
        assert "alert_level" in d
        assert "recommendation" in d


class TestRegimeMismatch:
    def test_no_mismatch(self):
        pm = PerformanceMonitor()
        result = pm.detect_regime_mismatch("ranging", "ranging", 0)
        assert not result["mismatched"]

    def test_mismatch_warn_recent(self):
        pm = PerformanceMonitor()
        result = pm.detect_regime_mismatch("ranging", "uptrend", 1)
        assert result["mismatched"]
        assert result["action"] == "warn"

    def test_mismatch_suspend_after_threshold(self):
        pm = PerformanceMonitor()
        result = pm.detect_regime_mismatch("ranging", "uptrend", 5)
        assert result["mismatched"]
        assert result["action"] == "suspend"

    def test_mismatch_reason_includes_details(self):
        pm = PerformanceMonitor()
        result = pm.detect_regime_mismatch("ranging", "strong_uptrend", 5)
        assert "ranging" in result["reason"]
        assert "strong_uptrend" in result["reason"]
