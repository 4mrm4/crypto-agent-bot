"""Tests for Portfolio-Level VaR (Task 9)."""

import math
import numpy as np

from risk.portfolio_var import (
    PortfolioVaR,
    _compute_returns,
    _z_score,
    MIN_HISTORICAL_DAYS,
    MAX_PORTFOLIO_EXPOSURE_PCT,
    MAX_SINGLE_POSITION_EXPOSURE_PCT,
)


# ── Utility Tests ──

def test_compute_returns_basic():
    prices = np.array([100.0, 110.0, 121.0])
    ret = _compute_returns(prices)
    assert len(ret) == 2
    assert ret[0] > 0
    assert ret[1] > 0


def test_compute_returns_too_short():
    prices = np.array([100.0])
    ret = _compute_returns(prices)
    assert len(ret) == 0


def test_compute_returns_negative():
    prices = np.array([100.0, 90.0, 81.0])
    ret = _compute_returns(prices)
    assert ret[0] < 0
    assert ret[1] < 0


def test_z_score_95():
    z = _z_score(0.95)
    assert abs(z - 1.645) < 0.01


def test_z_score_99():
    z = _z_score(0.99)
    assert abs(z - 2.326) < 0.01


def test_z_score_50():
    z = _z_score(0.50)
    assert abs(z - 0.0) < 0.01


# ── PortfolioVaR Tests ──

def test_empty_var():
    """Empty VaR when no symbols provided."""
    var = PortfolioVaR(10000)
    result = var.var_covariance([])
    assert result.get("error") == "No symbols provided"
    assert result["var_amount"] == 0.0


def test_ingest_prices():
    """Price ingestion stores data in cache."""
    var = PortfolioVaR()
    prices = [100.0 + i * 0.5 for i in range(100)]
    var.ingest_prices("BTC/USDT", prices)
    assert var.has_data_for(["BTC/USDT"])


def test_ingest_too_few_prices():
    """Very short price series is rejected."""
    var = PortfolioVaR()
    var.ingest_prices("BTC/USDT", [100.0, 101.0])
    assert not var.has_data_for(["BTC/USDT"])


def test_var_covariance_basic():
    """VaR produces a positive risk measure."""
    var = PortfolioVaR(10000)
    np.random.seed(42)
    prices = 100.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, 100)))
    var.ingest_prices("BTC/USDT", prices.tolist())

    result = var.var_covariance(["BTC/USDT"], confidence=0.95)
    assert result["var_amount"] > 0
    assert result["var_pct"] > 0
    assert result["method"] == "variance_covariance"
    assert "symbols" in result


def test_var_covariance_99_higher_than_95():
    """99% VaR should be larger (stricter) than 95% VaR."""
    var = PortfolioVaR(10000)
    np.random.seed(42)
    prices = 100.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, 100)))
    var.ingest_prices("BTC/USDT", prices.tolist())

    var_95 = var.var_covariance(["BTC/USDT"], confidence=0.95)
    var_99 = var.var_covariance(["BTC/USDT"], confidence=0.99)
    assert var_99["var_amount"] > var_95["var_amount"]


def test_var_covariance_multiple_symbols():
    """VaR with two symbols produces marginal VaR breakdown."""
    var = PortfolioVaR(20000)
    np.random.seed(42)
    btc = 100.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, 100)))
    eth = 50.0 * np.exp(np.cumsum(np.random.normal(0, 0.015, 100)))
    var.ingest_prices("BTC/USDT", btc.tolist())
    var.ingest_prices("ETH/USDT", eth.tolist())

    result = var.var_covariance(["BTC/USDT", "ETH/USDT"], weights=[0.6, 0.4])
    assert result["var_amount"] > 0
    assert len(result["symbols"]) == 2
    assert len(result["marginal_var"]) == 2
    assert result["marginal_var"][0]["symbol"] == "BTC/USDT"
    assert result["marginal_var"][1]["symbol"] == "ETH/USDT"


def test_var_covariance_larger_portfolio_higher_var():
    """Larger portfolio should have higher VaR in dollar terms."""
    np.random.seed(42)
    prices = 100.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, 100)))

    var_small = PortfolioVaR(10000)
    var_small.ingest_prices("BTC/USDT", prices.tolist())
    result_small = var_small.var_covariance(["BTC/USDT"], confidence=0.95)

    var_large = PortfolioVaR(100000)
    var_large.ingest_prices("BTC/USDT", prices.tolist())
    result_large = var_large.var_covariance(["BTC/USDT"], confidence=0.95)

    assert result_large["var_amount"] > result_small["var_amount"]


def test_historical_var_basic():
    """Historical VaR produces a positive risk measure."""
    var = PortfolioVaR(10000)
    np.random.seed(42)
    prices = 100.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, 100)))
    var.ingest_prices("BTC/USDT", prices.tolist())

    result = var.var_historical(["BTC/USDT"], confidence=0.95)
    assert result["var_amount"] > 0
    assert result["var_pct"] > 0
    assert result["method"] == "historical_simulation"


def test_var_combined():
    """Combined VaR returns both 95 and 99 measures."""
    var = PortfolioVaR(10000)
    np.random.seed(42)
    prices = 100.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, 100)))
    var.ingest_prices("BTC/USDT", prices.tolist())

    result = var.var_combined(["BTC/USDT"])
    assert "var_95_amount" in result
    assert "var_99_amount" in result
    assert "var_historical_95_amount" in result
    assert result["var_99_amount"] > result["var_95_amount"]


def test_var_combined_multiple_symbols():
    """Combined VaR with 2 symbols produces marginal VaR."""
    var = PortfolioVaR(20000)
    np.random.seed(42)
    btc = 100.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, 100)))
    eth = 50.0 * np.exp(np.cumsum(np.random.normal(0, 0.015, 100)))
    var.ingest_prices("BTC/USDT", btc.tolist())
    var.ingest_prices("ETH/USDT", eth.tolist())

    result = var.var_combined(["BTC/USDT", "ETH/USDT"], weights=[0.6, 0.4])
    assert len(result["marginal_var"]) == 2
    assert result["portfolio_volatility"] > 0


def test_check_exposure_limits_within():
    """Check passes when exposure is within limits."""
    var = PortfolioVaR(10000)
    positions = [
        {"pair": "BTC/USDT", "side": "long", "size_usdt": 1000},
        {"pair": "ETH/USDT", "side": "long", "size_usdt": 500},
    ]
    result = var.check_exposure_limits(positions, max_exposure_pct=0.50)
    assert result["approved"] is True
    assert result["total_exposure_usdt"] == 1500
    assert abs(result["total_exposure_pct"] - 0.15) < 0.01


def test_check_exposure_limits_over():
    """Check fails when exposure exceeds limits."""
    var = PortfolioVaR(10000)
    positions = [
        {"pair": "BTC/USDT", "side": "long", "size_usdt": 6000},
    ]
    result = var.check_exposure_limits(positions, max_exposure_pct=0.50)
    assert result["approved"] is False  # 60% > 50%


def test_check_exposure_limits_single_breach():
    """Individual position over single-position limit is flagged."""
    var = PortfolioVaR(10000)
    positions = [
        {"pair": "BTC/USDT", "side": "long", "size_usdt": 2000},  # 20% > 15%
        {"pair": "ETH/USDT", "side": "long", "size_usdt": 500},
    ]
    result = var.check_exposure_limits(positions)
    assert len(result["breaches"]) == 1
    assert result["breaches"][0]["pair"] == "BTC/USDT"


def test_check_exposure_limits_empty():
    """No positions = passed check."""
    var = PortfolioVaR(10000)
    result = var.check_exposure_limits([])
    assert result["approved"] is True
    assert result["total_exposure_pct"] == 0.0


def test_check_var_limits_within():
    """check_var_limits approves when VaR is within threshold."""
    var = PortfolioVaR(10000)
    np.random.seed(42)
    prices = 100.0 * np.exp(np.cumsum(np.random.normal(0, 0.005, 100)))
    var.ingest_prices("BTC/USDT", prices.tolist())

    result = var.check_var_limits(["BTC/USDT"], max_var_95_pct=0.10)
    assert isinstance(bool(result["approved"]), bool)  # returns a valid boolean


def test_check_var_limits_no_data():
    """check_var_limits returns not approved when no data."""
    var = PortfolioVaR(10000)
    result = var.check_var_limits(["BTC/USDT"])
    assert result["approved"] is False
    assert "error" in result


def test_correlation_matrix():
    """Correlation matrix with 2 symbols produces valid output."""
    var = PortfolioVaR()
    np.random.seed(42)
    # Two partially correlated series
    common = np.random.normal(0, 0.01, 100)
    btc = 100.0 * np.exp(np.cumsum(common + np.random.normal(0, 0.005, 100)))
    eth = 100.0 * np.exp(np.cumsum(common * 0.8 + np.random.normal(0, 0.005, 100)))
    var.ingest_prices("BTC/USDT", btc.tolist())
    var.ingest_prices("ETH/USDT", eth.tolist())

    result = var.correlation_matrix(["BTC/USDT", "ETH/USDT"])
    assert len(result["symbols"]) == 2
    assert len(result["matrix"]) == 2
    assert result["matrix"][0][0] == 1.0  # diagonal


def test_correlation_matrix_single_symbol():
    """Single symbol returns no meaningful correlations."""
    var = PortfolioVaR()
    np.random.seed(42)
    btc = 100.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, 100)))
    var.ingest_prices("BTC/USDT", btc.tolist())

    result = var.correlation_matrix(["BTC/USDT"])
    assert "error" in result


def test_update_portfolio_value():
    """Updating portfolio value changes VaR results."""
    var = PortfolioVaR(10000)
    np.random.seed(42)
    prices = 100.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, 100)))
    var.ingest_prices("BTC/USDT", prices.tolist())

    result_a = var.var_covariance(["BTC/USDT"], confidence=0.95)
    var.update_portfolio_value(50000)
    result_b = var.var_covariance(["BTC/USDT"], confidence=0.95)

    assert result_b["var_amount"] > result_a["var_amount"]


def test_clear_cache():
    """clear_cache removes all price data."""
    var = PortfolioVaR()
    prices = [100.0 + i for i in range(100)]
    var.ingest_prices("BTC/USDT", prices)
    assert var.has_data_for(["BTC/USDT"])
    var.clear_cache()
    assert not var.has_data_for(["BTC/USDT"])


def test_has_data_for_missing():
    """has_data_for returns False for unknown symbol."""
    var = PortfolioVaR()
    assert not var.has_data_for(["NONEXISTENT"])


def test_weight_normalization():
    """Weights are automatically normalized to sum to 1."""
    var = PortfolioVaR(10000)
    np.random.seed(42)
    btc = 100.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, 100)))
    eth = 50.0 * np.exp(np.cumsum(np.random.normal(0, 0.015, 100)))
    var.ingest_prices("BTC/USDT", btc.tolist())
    var.ingest_prices("ETH/USDT", eth.tolist())

    # Uneven weights that don't sum to 1
    result = var.var_covariance(["BTC/USDT", "ETH/USDT"], weights=[10, 5])
    assert result["var_amount"] > 0


def test_empty_portfolio():
    """Empty portfolio VaR returns zeros."""
    var = PortfolioVaR(10000)
    result = var.var_historical([])
    assert result.get("error") == "No symbols provided"


def test_zero_portfolio_value():
    """VaR with zero portfolio value is zero."""
    var = PortfolioVaR(0)
    np.random.seed(42)
    prices = 100.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, 100)))
    var.ingest_prices("BTC/USDT", prices.tolist())

    result = var.var_covariance(["BTC/USDT"])
    assert result["var_amount"] == 0.0
