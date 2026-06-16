"""
PortfolioVaR — portfolio-level Value at Risk with covariance-based and
historical simulation methods.

Supports:
  - 95% and 99% VaR (variance-covariance method)
  - Historical simulation VaR (empirical percentile)
  - Marginal VaR per position (contribution to total risk)
  - Position correlation matrix
  - Exposure limit checks against configurable thresholds
"""

import logging
import math
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default VaR parameters
DEFAULT_CONFIDENCE_95 = 0.95
DEFAULT_CONFIDENCE_99 = 0.99
DEFAULT_HOLDING_PERIOD = 1  # days
MIN_HISTORICAL_DAYS = 30   # minimum price history for VaR calculation
MAX_PORTFOLIO_EXPOSURE_PCT = 0.50  # total portfolio at risk cap
MAX_SINGLE_POSITION_EXPOSURE_PCT = 0.15  # single position cap


def _compute_returns(prices: np.ndarray) -> np.ndarray:
    """Compute log returns from a price series."""
    if len(prices) < 2:
        return np.array([])
    return np.diff(np.log(prices))


def _z_score(confidence: float) -> float:
    """Approximate z-score for a given confidence level.

    Uses the rational approximation for the normal distribution quantile.
    """
    from scipy.stats import norm
    return norm.ppf(confidence)


class PortfolioVaR:
    """Portfolio-level Value at Risk calculator.

    Uses variance-covariance (parametric) method as primary and
    historical simulation as secondary/verification method.
    """

    def __init__(self, portfolio_value: float = 10000.0):
        self._portfolio_value = portfolio_value
        # Cache for price data: symbol -> (date, prices_array)
        self._price_cache: Dict[str, np.ndarray] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        self._cache_ttl = timedelta(minutes=5)

    def update_portfolio_value(self, value: float):
        """Update the current portfolio value."""
        self._portfolio_value = value

    def ingest_prices(self, symbol: str, prices: List[float]):
        """Feed a price series for a symbol into the cache."""
        arr = np.array(prices, dtype=np.float64)
        if len(arr) >= MIN_HISTORICAL_DAYS:
            self._price_cache[symbol] = arr
            self._cache_timestamps[symbol] = datetime.now(timezone.utc)

    def clear_cache(self):
        """Clear all cached price data."""
        self._price_cache.clear()
        self._cache_timestamps.clear()

    def has_data_for(self, symbols: List[str]) -> bool:
        """Check if price data is available for all given symbols."""
        return all(s in self._price_cache and len(self._price_cache[s]) >= MIN_HISTORICAL_DAYS
                   for s in symbols)

    def _get_returns(self, symbol: str) -> np.ndarray:
        """Get log returns for a symbol from cache."""
        prices = self._price_cache.get(symbol)
        if prices is None or len(prices) < MIN_HISTORICAL_DAYS:
            return np.array([])
        return _compute_returns(prices)

    # ── Variance-Covariance VaR ──

    def var_covariance(
        self,
        symbols: List[str],
        weights: Optional[List[float]] = None,
        confidence: float = DEFAULT_CONFIDENCE_95,
        holding_period: int = DEFAULT_HOLDING_PERIOD,
    ) -> dict:
        """Compute parametric VaR using variance-covariance matrix.

        Args:
            symbols: List of position symbols (e.g. ["BTC/USDT", "ETH/USDT"])
            weights: Portfolio weights (equal-weighted if None)
            confidence: Confidence level (0.95 or 0.99)
            holding_period: Holding period in days

        Returns:
            dict with var_amount, var_pct, portfolio_value, confidence, method details
        """
        if not symbols:
            return self._empty_var_result("No symbols provided")

        n = len(symbols)
        if weights is None:
            weights = [1.0 / n] * n
        weights = np.array(weights, dtype=np.float64)
        weights = weights / weights.sum()  # normalize

        # Build return matrix
        returns_list = []
        valid_symbols = []
        valid_weights = []

        for i, sym in enumerate(symbols):
            ret = self._get_returns(sym)
            if len(ret) >= MIN_HISTORICAL_DAYS:
                returns_list.append(ret)
                valid_symbols.append(sym)
                valid_weights.append(weights[i])
            else:
                logger.warning("Insufficient data for %s (%d obs)", sym, len(ret))

        if len(returns_list) < 1:
            return self._empty_var_result("No symbols with sufficient data")

        # Align to shortest series
        min_len = min(len(r) for r in returns_list)
        aligned = np.array([r[-min_len:] for r in returns_list])  # shape: (n_symbols, n_obs)

        valid_weights = np.array(valid_weights, dtype=np.float64)
        valid_weights = valid_weights / valid_weights.sum()

        # Covariance matrix (annualized)
        cov_matrix = np.cov(aligned) * 252  # daily -> annual

        if len(valid_symbols) == 1:
            # Single symbol: variance is a scalar
            portfolio_variance = float(cov_matrix) if cov_matrix.ndim == 0 else float(cov_matrix[0, 0])
        else:
            portfolio_variance = valid_weights.T @ cov_matrix @ valid_weights
        portfolio_vol = math.sqrt(portfolio_variance)

        # Z-score for confidence level
        z = _z_score(confidence)

        # VaR at holding_period
        var_pct = z * portfolio_vol * math.sqrt(holding_period / 252)
        var_amount = self._portfolio_value * var_pct

        # Marginal VaR — contribution of each position
        marginal_var = []
        portfolio_std = portfolio_vol
        if portfolio_std > 0:
            for i, sym in enumerate(valid_symbols):
                if len(valid_symbols) == 1:
                    # Single position: marginal VaR = total VaR
                    mc_var = z * portfolio_std * math.sqrt(holding_period / 252)
                else:
                    cov_i = cov_matrix[i, :] @ valid_weights
                    mc_var = z * cov_i / portfolio_std * math.sqrt(holding_period / 252)
                marginal_var.append({
                    "symbol": sym,
                    "weight": round(float(valid_weights[i]), 4),
                    "marginal_var_pct": round(float(mc_var), 6),
                    "marginal_var_amount": round(float(mc_var * self._portfolio_value), 2),
                })

        return {
            "method": "variance_covariance",
            "var_amount": round(abs(var_amount), 2),
            "var_pct": round(abs(var_pct), 4),
            "portfolio_value": self._portfolio_value,
            "confidence": confidence,
            "holding_period_days": holding_period,
            "portfolio_volatility": round(float(portfolio_vol), 4),
            "symbols": valid_symbols,
            "marginal_var": marginal_var,
            "n_observations": min_len,
        }

    # ── Historical Simulation VaR ──

    def var_historical(
        self,
        symbols: List[str],
        weights: Optional[List[float]] = None,
        confidence: float = DEFAULT_CONFIDENCE_95,
        holding_period: int = DEFAULT_HOLDING_PERIOD,
    ) -> dict:
        """Compute VaR using historical simulation (empirical percentile).

        Args:
            Same as var_covariance.

        Returns:
            dict with var_amount, var_pct, method details
        """
        if not symbols:
            return self._empty_var_result("No symbols provided")

        n = len(symbols)
        if weights is None:
            weights = [1.0 / n] * n
        weights = np.array(weights, dtype=np.float64)
        weights = weights / weights.sum()

        # Build weighted portfolio returns
        returns_list = []
        valid_symbols = []
        valid_weights = []

        for i, sym in enumerate(symbols):
            ret = self._get_returns(sym)
            if len(ret) >= MIN_HISTORICAL_DAYS:
                returns_list.append(ret)
                valid_symbols.append(sym)
                valid_weights.append(weights[i])

        if len(returns_list) < 1:
            return self._empty_var_result("No symbols with sufficient data")

        min_len = min(len(r) for r in returns_list)
        aligned = np.array([r[-min_len:] for r in returns_list])

        valid_weights = np.array(valid_weights, dtype=np.float64)
        valid_weights = valid_weights / valid_weights.sum()

        # Weighted sum across positions for each time step
        portfolio_returns = valid_weights @ aligned  # shape: (n_obs,)

        # Scale to holding period
        scaled_returns = portfolio_returns * math.sqrt(holding_period)

        # Empirical percentile
        var_pct = float(np.percentile(scaled_returns, (1 - confidence) * 100))
        var_amount = self._portfolio_value * abs(var_pct)

        return {
            "method": "historical_simulation",
            "var_amount": round(abs(var_amount), 2),
            "var_pct": round(abs(var_pct), 4),
            "portfolio_value": self._portfolio_value,
            "confidence": confidence,
            "holding_period_days": holding_period,
            "symbols": valid_symbols,
            "n_observations": min_len,
        }

    # ── Combined VaR ──

    def var_combined(
        self,
        symbols: List[str],
        weights: Optional[List[float]] = None,
    ) -> dict:
        """Compute both 95% and 99% VaR using variance-covariance method.

        Returns consolidated risk report.
        """
        var_95 = self.var_covariance(symbols, weights, confidence=0.95)
        var_99 = self.var_covariance(symbols, weights, confidence=0.99)
        var_hist = self.var_historical(symbols, weights, confidence=0.95)

        if var_95.get("error") or var_99.get("error"):
            return self._empty_var_result("Insufficient data for combined VaR")

        return {
            "portfolio_value": self._portfolio_value,
            "symbols": var_95.get("symbols", []),
            "var_95_amount": var_95["var_amount"],
            "var_95_pct": var_95["var_pct"],
            "var_99_amount": var_99["var_amount"],
            "var_99_pct": var_99["var_pct"],
            "var_historical_95_amount": var_hist.get("var_amount", 0),
            "var_historical_95_pct": var_hist.get("var_pct", 0),
            "marginal_var": var_95.get("marginal_var", []),
            "portfolio_volatility": var_95.get("portfolio_volatility", 0),
            "n_observations": var_95.get("n_observations", 0),
        }

    # ── Exposure Limit Checks ──

    def check_exposure_limits(
        self, positions: List[dict], max_exposure_pct: float = MAX_PORTFOLIO_EXPOSURE_PCT
    ) -> dict:
        """Check if current portfolio exposure is within limits.

        Args:
            positions: List of position dicts with keys: pair, side, size_usdt
            max_exposure_pct: Maximum total exposure as fraction of portfolio

        Returns:
            dict with approved, total_exposure_pct, individual_checks
        """
        total_exposure = sum(abs(p.get("size_usdt", 0)) for p in positions)
        exposure_pct = total_exposure / self._portfolio_value if self._portfolio_value > 0 else 0

        individual_checks = []
        for p in positions:
            pair = p.get("pair", "unknown")
            size = abs(p.get("size_usdt", 0))
            pos_pct = size / self._portfolio_value if self._portfolio_value > 0 else 0
            within_limit = pos_pct <= MAX_SINGLE_POSITION_EXPOSURE_PCT
            individual_checks.append({
                "pair": pair,
                "size_usdt": round(size, 2),
                "exposure_pct": round(pos_pct, 4),
                "limit_pct": MAX_SINGLE_POSITION_EXPOSURE_PCT,
                "within_limit": within_limit,
            })

        overall_approved = exposure_pct <= max_exposure_pct

        return {
            "approved": bool(overall_approved),
            "total_exposure_usdt": round(total_exposure, 2),
            "total_exposure_pct": round(exposure_pct, 4),
            "max_exposure_pct": max_exposure_pct,
            "portfolio_value": self._portfolio_value,
            "individual_checks": individual_checks,
            "breaches": [c for c in individual_checks if not c["within_limit"]],
        }

    def check_var_limits(self, symbols: List[str], weights: Optional[List[float]] = None,
                         max_var_95_pct: float = 0.05) -> dict:
        """Check if portfolio VaR is within acceptable limits.

        Args:
            symbols: Position symbols
            weights: Position weights
            max_var_95_pct: Maximum acceptable 95% VaR as portfolio fraction

        Returns:
            dict with approved flag and VaR details
        """
        var_result = self.var_combined(symbols, weights)
        if var_result.get("error"):
            return {"approved": False, "error": var_result.get("error")}

        var_95_pct = var_result.get("var_95_pct", 1.0)
        approved = bool(var_95_pct <= max_var_95_pct)

        return {
            "approved": bool(approved),
            "var_95_pct": var_95_pct,
            "var_95_amount": var_result.get("var_95_amount", 0),
            "max_var_95_pct": max_var_95_pct,
            "portfolio_value": self._portfolio_value,
            "symbols": symbols,
        }

    # ── Correlation Matrix ──

    def correlation_matrix(self, symbols: List[str]) -> dict:
        """Compute pairwise return correlation matrix for given symbols.

        Returns:
            dict with matrix (list of lists), symbols, warnings for high correlations
        """
        returns_dict = {}
        for sym in symbols:
            ret = self._get_returns(sym)
            if len(ret) >= MIN_HISTORICAL_DAYS:
                returns_dict[sym] = ret[-252:]  # 1 year of daily returns

        if len(returns_dict) < 2:
            return {"symbols": symbols, "matrix": [], "high_correlations": [],
                    "error": "Need at least 2 symbols with data"}

        # Align
        min_len = min(len(r) for r in returns_dict.values())
        aligned = {sym: r[-min_len:] for sym, r in returns_dict.items()}

        sym_list = list(aligned.keys())
        df = pd.DataFrame(aligned)
        corr_matrix = df.corr().values

        # Find high correlations (> 0.7)
        high_corrs = []
        for i in range(len(sym_list)):
            for j in range(i + 1, len(sym_list)):
                corr_val = corr_matrix[i][j]
                if abs(corr_val) > 0.7:
                    high_corrs.append({
                        "pair1": sym_list[i],
                        "pair2": sym_list[j],
                        "correlation": round(float(corr_val), 4),
                    })

        return {
            "symbols": sym_list,
            "matrix": [[round(float(v), 4) for v in row] for row in corr_matrix],
            "high_correlations": high_corrs,
        }

    def _empty_var_result(self, reason: str) -> dict:
        """Return an empty VaR result with error."""
        return {
            "error": reason,
            "var_amount": 0.0,
            "var_pct": 0.0,
            "portfolio_value": self._portfolio_value,
            "method": "none",
        }
