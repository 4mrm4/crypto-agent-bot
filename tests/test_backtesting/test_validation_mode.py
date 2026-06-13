"""Tests for ValidationMode."""

from datetime import datetime, timedelta

from execution.validation_mode import ValidationMode, GraduationAssessment


class TestValidationMode:
    def test_validation_active_initially(self):
        vm = ValidationMode(live_start_date=datetime.utcnow())
        assert vm.is_active
        assert vm.days_remaining > 0

    def test_validation_inactive_after_90_days(self):
        vm = ValidationMode(live_start_date=datetime.utcnow() - timedelta(days=91))
        assert not vm.is_active
        assert vm.days_remaining == 0

    def test_position_cap_applied_in_validation(self):
        vm = ValidationMode(live_start_date=datetime.utcnow())
        kelly_result = {
            "position_size_usdt": 500.0,
            "portfolio_value": 10000.0,
            "portfolio_pct": 5.0,
            "rationale": "test",
        }
        capped = vm.apply_position_cap(kelly_result)
        assert capped["position_size_usdt"] <= 200.0  # 2% of 10000

    def test_position_cap_not_applied_outside_validation(self):
        vm = ValidationMode(live_start_date=datetime.utcnow() - timedelta(days=91))
        kelly_result = {
            "position_size_usdt": 500.0,
            "portfolio_value": 10000.0,
            "portfolio_pct": 5.0,
            "rationale": "test",
        }
        result = vm.apply_position_cap(kelly_result)
        assert result["position_size_usdt"] == 500.0  # unchanged

    def test_tight_circuit_breaker_triggers(self):
        vm = ValidationMode(live_start_date=datetime.utcnow())
        assert vm.apply_tight_circuit_breaker(-0.02, -0.01)  # daily -2% exceeds -1.5%

    def test_tight_circuit_breaker_weekly_triggers(self):
        vm = ValidationMode(live_start_date=datetime.utcnow())
        assert vm.apply_tight_circuit_breaker(-0.01, -0.05)  # weekly -5% exceeds -4%

    def test_tight_circuit_breaker_no_trigger(self):
        vm = ValidationMode(live_start_date=datetime.utcnow())
        assert not vm.apply_tight_circuit_breaker(-0.01, -0.02)  # within limits

    def test_tight_circuit_breaker_inactive(self):
        vm = ValidationMode(live_start_date=datetime.utcnow() - timedelta(days=91))
        assert not vm.apply_tight_circuit_breaker(-0.05, -0.10)  # not active

    def test_graduation_not_met_insufficient_days(self):
        vm = ValidationMode(live_start_date=datetime.utcnow())
        assessment = vm.can_graduate({"sharpe_ratio": 0.8, "total_trades": 100})
        assert not assessment.can_graduate
        assert "days live" in " ".join(assessment.reasons).lower()

    def test_graduation_not_met_low_sharpe(self):
        vm = ValidationMode(live_start_date=datetime.utcnow() - timedelta(days=100))
        assessment = vm.can_graduate({"sharpe_ratio": 0.3, "total_trades": 100})
        assert not assessment.can_graduate
        assert "sharpe" in " ".join(assessment.reasons).lower()

    def test_graduation_not_met_few_trades(self):
        vm = ValidationMode(live_start_date=datetime.utcnow() - timedelta(days=100))
        assessment = vm.can_graduate({"sharpe_ratio": 0.8, "total_trades": 5})
        assert not assessment.can_graduate
        assert "trades" in " ".join(assessment.reasons).lower()

    def test_graduation_met(self):
        vm = ValidationMode(live_start_date=datetime.utcnow() - timedelta(days=100))
        assessment = vm.can_graduate({"sharpe_ratio": 0.8, "total_trades": 60})
        assert assessment.can_graduate
        assert len(assessment.reasons) == 0

    def test_validation_report_structure(self):
        vm = ValidationMode(live_start_date=datetime.utcnow())
        report = vm.generate_validation_report()
        assert "is_active" in report
        assert "days_live" in report
        assert "days_remaining" in report
        assert "graduation" in report
        assert "position_cap_pct" in report
