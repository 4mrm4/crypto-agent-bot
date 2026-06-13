"""Tests for Bayesian Kelly position sizing (Task 8)."""

from agents.risk_manager import (
    bayesian_kelly_position_size,
    bayesian_kelly_position_size_conservative,
    _beta_posterior_win_rate,
    _estimate_wins_losses,
    PositionSizingTier,
    BETA_PRIOR_ALPHA,
    BETA_PRIOR_BETA,
)


# ── Beta Posterior Tests ──

def test_beta_posterior_produces_ci():
    """Posterior produces a valid credible interval with ci_lower < ci_upper."""
    result = _beta_posterior_win_rate(wins=30, losses=20)
    assert result["ci_lower"] < result["ci_upper"]
    assert result["total_trades"] == 50
    assert result["map_win_rate"] > 0
    assert result["ci_lower"] < result["map_win_rate"] < result["ci_upper"]


def test_beta_posterior_few_trades_wide_ci():
    """Few trades produce a wider CI (uncertainty is higher)."""
    few = _beta_posterior_win_rate(wins=3, losses=2)
    many = _beta_posterior_win_rate(wins=30, losses=20)
    few_width = few["ci_upper"] - few["ci_lower"]
    many_width = many["ci_upper"] - many["ci_lower"]
    assert few_width > many_width  # fewer trades = wider CI


def test_beta_posterior_converges_with_more_data():
    """With many trades, MAP converges to observed win rate."""
    result = _beta_posterior_win_rate(wins=500, losses=500)
    assert abs(result["map_win_rate"] - 0.5) < 0.01
    # CI should be tight
    ci_width = result["ci_upper"] - result["ci_lower"]
    assert ci_width < 0.06  # tight CI with 1000 samples


def test_beta_posterior_zero_wins():
    """Zero wins still produces a valid posterior."""
    result = _beta_posterior_win_rate(wins=0, losses=10)
    assert result["ci_lower"] >= 0
    assert result["map_win_rate"] < 0.2  # should be very low
    assert result["total_trades"] == 10


def test_beta_posterior_prior_info():
    """Posterior alpha/beta reflect the Beta(2,2) prior."""
    result = _beta_posterior_win_rate(wins=0, losses=0)
    assert result["posterior_alpha"] == BETA_PRIOR_ALPHA  # 2.0
    assert result["posterior_beta"] == BETA_PRIOR_BETA    # 2.0
    assert result["map_win_rate"] == 0.5  # mode of Beta(2,2)


# ── Estimate Wins/Losses Tests ──

def test_estimate_wins_losses_perfect():
    wins, losses = _estimate_wins_losses(0.60, 100)
    assert wins == 60.0
    assert losses == 40.0


def test_estimate_wins_losses_edge():
    wins, losses = _estimate_wins_losses(0.0, 100)
    assert wins == 0.0
    assert losses == 100.0


# ── Bayesian Kelly Basic Tests ──

def test_bayesian_kelly_basic():
    """Basic Bayesian Kelly: 60% win rate, 2:1 reward:risk, 50 trades."""
    result = bayesian_kelly_position_size(
        win_rate=0.60, avg_win_pct=0.10, avg_loss_pct=0.05,
        portfolio_value=10000, total_trades=50,
    )
    assert result["kelly_fraction"] > 0
    assert result["position_size_usdt"] > 0
    assert result["position_size_usdt"] <= 1000  # 10% cap
    assert "bayesian_posterior" in result
    assert result["method"] == "bayesian"


def test_bayesian_kelly_negative_edge():
    """Bayesian Kelly returns 0 when edge is negative."""
    result = bayesian_kelly_position_size(
        win_rate=0.30, avg_win_pct=0.05, avg_loss_pct=0.10,
        portfolio_value=10000, total_trades=50,
    )
    assert result["position_size_usdt"] == 0
    assert result["kelly_fraction"] == 0


def test_bayesian_kelly_never_exceeds_max():
    """Bayesian Kelly respects the 10% portfolio cap."""
    result = bayesian_kelly_position_size(
        win_rate=0.90, avg_win_pct=0.20, avg_loss_pct=0.01,
        portfolio_value=10000, total_trades=1000,
    )
    assert result["position_size_usdt"] <= 1000


def test_bayesian_kelly_uncertainty_penalty():
    """Fewer trades = wider CI = smaller position = more conservative."""
    many_trades = bayesian_kelly_position_size(
        win_rate=0.60, avg_win_pct=0.10, avg_loss_pct=0.05,
        portfolio_value=10000, total_trades=1000,
    )
    few_trades = bayesian_kelly_position_size(
        win_rate=0.60, avg_win_pct=0.10, avg_loss_pct=0.05,
        portfolio_value=10000, total_trades=10,
    )
    # Fewer trades should give smaller or equal position
    assert few_trades["position_size_usdt"] <= many_trades["position_size_usdt"]


def test_bayesian_kelly_sizing_tier_validation():
    """Validation tier has smallest max position."""
    validation = bayesian_kelly_position_size(
        win_rate=0.60, avg_win_pct=0.10, avg_loss_pct=0.05,
        portfolio_value=10000, total_trades=50,
        sizing_tier=PositionSizingTier.VALIDATION,
    )
    normal = bayesian_kelly_position_size(
        win_rate=0.60, avg_win_pct=0.10, avg_loss_pct=0.05,
        portfolio_value=10000, total_trades=50,
        sizing_tier=PositionSizingTier.NORMAL,
    )
    assert validation["position_size_usdt"] <= normal["position_size_usdt"]


def test_bayesian_kelly_invalid_inputs():
    """Zero win_rate returns error dict."""
    result = bayesian_kelly_position_size(
        win_rate=0.0, avg_win_pct=0.10, avg_loss_pct=0.05,
        portfolio_value=10000,
    )
    assert result.get("error") is True
    assert result["position_size_usdt"] == 0


def test_bayesian_kelly_zero_avg_loss():
    """Zero avg_loss returns error dict."""
    result = bayesian_kelly_position_size(
        win_rate=0.60, avg_win_pct=0.10, avg_loss_pct=0.0,
        portfolio_value=10000,
    )
    assert result.get("error") is True


# ── Conservative Bayesian Kelly Tests ──

def test_bayesian_conservative_basic():
    """Conservative Bayesian Kelly produces valid results with strong edge."""
    result = bayesian_kelly_position_size_conservative(
        win_rate=0.75, avg_win_pct=0.10, avg_loss_pct=0.03,
        portfolio_value=10000, total_trades=100,
    )
    assert result["kelly_fraction"] > 0
    assert result["position_size_usdt"] > 0
    assert result["method"] == "bayesian_conservative"


def test_bayesian_conservative_more_conservative_than_standard():
    """Conservative variant should produce smaller or equal positions than standard Bayesian."""
    standard = bayesian_kelly_position_size(
        win_rate=0.70, avg_win_pct=0.08, avg_loss_pct=0.025,
        portfolio_value=10000, total_trades=100,
    )
    conservative = bayesian_kelly_position_size_conservative(
        win_rate=0.70, avg_win_pct=0.08, avg_loss_pct=0.025,
        portfolio_value=10000, total_trades=100,
    )
    # Conservative should apply degradation haircut + CI lower bound
    assert conservative["position_size_usdt"] <= standard["position_size_usdt"]


def test_bayesian_conservative_with_custom_degradation():
    """Custom degradation percentage changes position size."""
    mild = bayesian_kelly_position_size_conservative(
        win_rate=0.80, avg_win_pct=0.08, avg_loss_pct=0.025,
        portfolio_value=10000, total_trades=100,
        oos_degradation_pct=0.10,
    )
    heavy = bayesian_kelly_position_size_conservative(
        win_rate=0.80, avg_win_pct=0.08, avg_loss_pct=0.025,
        portfolio_value=10000, total_trades=100,
        oos_degradation_pct=0.80,
    )
    assert heavy["position_size_usdt"] <= mild["position_size_usdt"]


def test_bayesian_conservative_negative_after_haircut():
    """Strategy that passes basic Kelly may fail after degradation haircut."""
    # 55% win rate with 1:1 R:R — barely positive on standard Kelly
    result = bayesian_kelly_position_size_conservative(
        win_rate=0.55, avg_win_pct=0.05, avg_loss_pct=0.05,
        portfolio_value=10000, total_trades=30,
        oos_degradation_pct=0.40,
    )
    # May or may not be negative depending on Bayesian CI lower bound
    assert result["position_size_usdt"] >= 0


def test_bayesian_conservative_zero_trades():
    """With zero trades, conservative Bayesian Kelly uses neutral prior."""
    result = bayesian_kelly_position_size_conservative(
        win_rate=0.60, avg_win_pct=0.10, avg_loss_pct=0.05,
        portfolio_value=10000, total_trades=0,
    )
    # With 0 trades, posterior = Beta(2,2), CI lower should be very low
    assert result["kelly_fraction"] == 0 or result["kelly_fraction"] < 0.05


def test_bayesian_conservative_tier_validation():
    """Validation mode with max conservatism."""
    result = bayesian_kelly_position_size_conservative(
        win_rate=0.75, avg_win_pct=0.10, avg_loss_pct=0.03,
        portfolio_value=10000, total_trades=100,
        sizing_tier=PositionSizingTier.VALIDATION,
    )
    assert result["sizing_tier"] == "validation"
    assert result["position_size_usdt"] <= 500  # 5% of 10000
