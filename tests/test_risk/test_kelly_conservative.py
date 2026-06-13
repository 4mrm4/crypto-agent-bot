"""Tests for conservative Kelly position sizing."""

from agents.risk_manager import (
    kelly_position_size_conservative,
    PositionSizingTier,
    BACKTEST_OPTIMISM_FACTOR,
)


class TestConservativeKelly:
    def test_backtest_optimism_factor_exists(self):
        assert 0.0 < BACKTEST_OPTIMISM_FACTOR <= 1.0

    def test_kelly_conservative_positive_edge(self):
        result = kelly_position_size_conservative(
            win_rate=0.6,
            avg_win_pct=0.05,
            avg_loss_pct=0.03,
            portfolio_value=10000.0,
            oos_degradation_pct=0.0,  # No degradation to show positive edge
            sizing_tier=PositionSizingTier.NORMAL,
        )
        assert result["position_size_usdt"] > 0
        assert result["portfolio_pct"] > 0
        assert "haircut_applied" in result

        # With 40% degradation, the edge should be reduced or negative
        conservative = kelly_position_size_conservative(
            win_rate=0.6,
            avg_win_pct=0.05,
            avg_loss_pct=0.03,
            portfolio_value=10000.0,
            oos_degradation_pct=0.40,
            sizing_tier=PositionSizingTier.NORMAL,
        )
        assert conservative["position_size_usdt"] < result["position_size_usdt"]

    def test_kelly_conservative_negative_edge(self):
        result = kelly_position_size_conservative(
            win_rate=0.3,
            avg_win_pct=0.02,
            avg_loss_pct=0.05,
            portfolio_value=10000.0,
            sizing_tier=PositionSizingTier.NORMAL,
        )
        assert result["position_size_usdt"] == 0

    def test_kelly_conservative_degradation_reduces_size(self):
        no_degradation = kelly_position_size_conservative(
            win_rate=0.6, avg_win_pct=0.05, avg_loss_pct=0.03,
            portfolio_value=10000.0, oos_degradation_pct=0.0,
        )
        high_degradation = kelly_position_size_conservative(
            win_rate=0.6, avg_win_pct=0.05, avg_loss_pct=0.03,
            portfolio_value=10000.0, oos_degradation_pct=0.8,
        )
        assert high_degradation["position_size_usdt"] < no_degradation["position_size_usdt"]

    def test_validation_tier_caps_position(self):
        result = kelly_position_size_conservative(
            win_rate=0.7, avg_win_pct=0.08, avg_loss_pct=0.02,
            portfolio_value=10000.0, sizing_tier=PositionSizingTier.VALIDATION,
            oos_degradation_pct=0.0,
        )
        assert result["portfolio_pct"] <= 2.0  # 2% cap for validation

    def test_cautious_tier_caps_position(self):
        result = kelly_position_size_conservative(
            win_rate=0.7, avg_win_pct=0.08, avg_loss_pct=0.02,
            portfolio_value=10000.0, sizing_tier=PositionSizingTier.CAUTIOUS,
            oos_degradation_pct=0.0,
        )
        assert result["portfolio_pct"] <= 5.0  # 5% cap for cautious

    def test_normal_tier_caps_position(self):
        result = kelly_position_size_conservative(
            win_rate=0.7, avg_win_pct=0.08, avg_loss_pct=0.02,
            portfolio_value=10000.0, sizing_tier=PositionSizingTier.NORMAL,
            oos_degradation_pct=0.0,
        )
        assert result["portfolio_pct"] <= 10.0  # 10% cap for normal

    def test_kelly_invalid_inputs(self):
        result = kelly_position_size_conservative(
            win_rate=0.0, avg_win_pct=0.0, avg_loss_pct=0.0,
            portfolio_value=10000.0,
        )
        assert result.get("error") is True


class TestPositionSizingTier:
    def test_validation_tier_under_90_days(self):
        assert PositionSizingTier.from_live_days(50) == PositionSizingTier.VALIDATION

    def test_cautious_tier_90_180_days(self):
        assert PositionSizingTier.from_live_days(120) == PositionSizingTier.CAUTIOUS

    def test_cautious_tier_low_sharpe(self):
        assert PositionSizingTier.from_live_days(200, live_sharpe=0.3) == PositionSizingTier.CAUTIOUS

    def test_normal_tier_after_180_days_good_sharpe(self):
        assert PositionSizingTier.from_live_days(200, live_sharpe=0.8) == PositionSizingTier.NORMAL

    def test_validation_max_position_pct(self):
        assert PositionSizingTier.VALIDATION.max_position_pct() == 0.02

    def test_cautious_max_position_pct(self):
        assert PositionSizingTier.CAUTIOUS.max_position_pct() == 0.05

    def test_normal_max_position_pct(self):
        assert PositionSizingTier.NORMAL.max_position_pct() == 0.10
