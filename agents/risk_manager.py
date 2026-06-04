"""RiskManagerAgent — full portfolio risk management with Kelly sizing,
correlation checks, circuit breaker, and pre-trade approval.

This is a complete rewrite of the original risk manager with quantitative
tools that block bad trades rather than just producing text assessments.

Bayesian Kelly (v8+ upgrade):
- Models win rate as a Beta posterior: Beta(α_prior + wins, β_prior + losses)
- Uses lower bound of 90% credible interval as conservative win rate input
- Automatically shrinks position size when few trades are observed
- Converges to observed rate as trade count grows
"""

import json
import logging
import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from langchain_core.tools import Tool

from agents.base import BaseAgent
from data.fetcher import MarketDataFetcher
from data.regime import RegimeSnapshot

from enum import Enum

logger = logging.getLogger(__name__)

RISK_MANAGER_PROMPT = """You are a quantitative risk management specialist.
Your job is to assess trade proposals using statistical tools and block
any trade that doesn't meet risk thresholds.

Available tools:
- kelly_position_size: Compute optimal position size using fractional Kelly
- kelly_position_size_conservative: Compute position size with pessimism-adjusted Kelly (degradation haircut applied)
- bayesian_kelly: Compute position size using Bayesian Kelly with Beta posterior (recommended)
- bayesian_kelly_conservative: Bayesian Kelly + degradation haircut for maximum conservatism (recommended for new strategies)
- check_position_correlation: Prevent over-concentration across positions
- circuit_breaker_check: Verify global trading is allowed
- assess_strategy_risk: Get full risk report for a strategy
- pre_trade_approval: Final go/no-go gate before any trade

Always run ALL tools before approving a trade. One veto = no trade.
Be conservative. Capital preservation is priority #1.
Prefer bayesian_kelly_conservative over other sizing methods — it accounts for
both backtest optimism and win-rate uncertainty from limited trade samples.

IMPORTANT: Use ONLY plain ASCII text. No emoji, no Unicode symbols.
"""


# ── Shared circuit breaker state (module-level singleton) ──

class CircuitBreakerState:
    """Global circuit breaker. Halted state is shared across all instances."""
    _halted: bool = False
    _halt_reason: str = ""
    _halted_at: Optional[datetime] = None
    _resume_after: Optional[datetime] = None

    @classmethod
    def halt(cls, reason: str, duration_minutes: int = 60):
        cls._halted = True
        cls._halt_reason = reason
        cls._halted_at = datetime.utcnow()
        cls._resume_after = datetime.utcnow() + timedelta(minutes=duration_minutes)
        logger.warning("CIRCUIT BREAKER HALTED: %s (resume after %s)", reason, cls._resume_after)

    @classmethod
    def clear(cls):
        cls._halted = False
        cls._halt_reason = ""
        cls._halted_at = None
        cls._resume_after = None
        logger.info("Circuit breaker cleared — trading resumed.")

    @classmethod
    def is_halted(cls) -> bool:
        if cls._halted and cls._resume_after and datetime.utcnow() > cls._resume_after:
            cls.clear()
        return cls._halted

    @classmethod
    def status(cls) -> dict:
        return {
            "halted": cls._halted,
            "reason": cls._halt_reason,
            "halted_at": cls._halted_at.isoformat() if cls._halted_at else None,
            "resume_after": cls._resume_after.isoformat() if cls._resume_after else None,
        }


def _normalize(symbol: str) -> str:
    s = symbol.strip().upper()
    if "/" not in s:
        s = s + "/USDT"
    return s


# ── Conservative position sizing ──

from config import settings

BACKTEST_OPTIMISM_FACTOR = settings.BACKTEST_OPTIMISM_FACTOR  # Live metrics = X% of backtest


class PositionSizingTier(Enum):
    """Progressive position sizing tiers based on live trading maturity."""
    VALIDATION = "validation"      # First 90 days live: max 2% portfolio per trade
    CAUTIOUS = "cautious"          # 90-180 days: max 5% portfolio per trade
    NORMAL = "normal"              # 180+ days with good live track record: max 10%

    @classmethod
    def from_live_days(cls, days_live: int, live_sharpe: float = 0.0) -> "PositionSizingTier":
        if days_live < 90:
            return cls.VALIDATION
        if days_live < 180 or live_sharpe < 0.6:
            return cls.CAUTIOUS
        return cls.NORMAL

    def max_position_pct(self) -> float:
        if self == PositionSizingTier.VALIDATION:
            return 0.02   # 2%
        if self == PositionSizingTier.CAUTIOUS:
            return 0.05   # 5%
        return 0.10       # 10%


def kelly_position_size_conservative(
    win_rate: float = 0.5,
    avg_win_pct: float = 0.02,
    avg_loss_pct: float = 0.01,
    portfolio_value: float = 10000.0,
    oos_degradation_pct: float = 0.40,
    max_kelly_fraction: float = 0.25,
    sizing_tier: PositionSizingTier = PositionSizingTier.CAUTIOUS,
) -> dict:
    """Kelly position sizing with pessimism-adjusted inputs.

    Applies a degradation haircut BEFORE Kelly calculation to account for
    the known optimism in backtest metrics.

    Args:
        win_rate: Historical win rate as decimal (0.0-1.0), e.g. 0.55 = 55%.
        avg_win_pct: Average winning trade return as decimal, e.g. 0.02 = 2%.
        avg_loss_pct: Average losing trade loss as decimal, e.g. 0.01 = 1%.
        portfolio_value: Current portfolio value in USDT, e.g. 10000.0.
        oos_degradation_pct: Out-of-sample degradation haircut (default 0.40 = 40%).
        max_kelly_fraction: Fraction of full Kelly to use (default 0.25 = quarter Kelly).
        sizing_tier: PositionSizingTier enum (validation/cautious/normal).

    Example usage:
        kelly_position_size_conservative(
            win_rate=0.55, avg_win_pct=0.02, avg_loss_pct=0.01, portfolio_value=10000.0
        )

    Adjusted inputs:
    - adj_win_rate = win_rate * (1 - oos_degradation_pct)
    - adj_avg_win = avg_win * (1 - oos_degradation_pct)

    Additional caps:
    - PositionSizingTier.VALIDATION: max 2% of portfolio
    - PositionSizingTier.CAUTIOUS: max 5%
    - PositionSizingTier.NORMAL: max 10% (existing cap)
    """
    # Type coercion: LLM may pass strings via LangChain tools.
    # Also handle the case where the LLM passes a full JSON object as the first
    # positional arg (because the tool description says "Args: JSON with ...").
    if isinstance(win_rate, str) and win_rate.strip().startswith("{"):
        import json as _json
        try:
            _parsed = _json.loads(win_rate)
            win_rate = float(_parsed.get("win_rate", 0.5))
            avg_win_pct = float(_parsed.get("avg_win_pct", 0.02))
            avg_loss_pct = float(_parsed.get("avg_loss_pct", 0.01))
            portfolio_value = float(_parsed.get("portfolio_value", 10000.0))
            oos_degradation_pct = float(_parsed.get("oos_degradation_pct", 0.40))
            max_kelly_fraction = float(_parsed.get("max_kelly_fraction", 0.25))
            tier_name = _parsed.get("sizing_tier", "cautious")
            sizing_tier = PositionSizingTier(tier_name) if tier_name in ("validation", "cautious", "normal") else PositionSizingTier.CAUTIOUS
        except (_json.JSONDecodeError, ValueError):
            pass
    else:
        win_rate = float(win_rate)
        avg_win_pct = float(avg_win_pct)
        avg_loss_pct = float(avg_loss_pct)
        portfolio_value = float(portfolio_value)
        oos_degradation_pct = float(oos_degradation_pct)
        max_kelly_fraction = float(max_kelly_fraction)
    if sizing_tier is None:
        sizing_tier = PositionSizingTier.CAUTIOUS

    if win_rate <= 0 or avg_loss_pct <= 0:
        return {
            "kelly_fraction": 0.0,
            "position_size_usdt": 0.0,
            "portfolio_pct": 0.0,
            "rationale": "Invalid inputs: win_rate or avg_loss must be positive",
            "haircut_applied": "none",
            "error": True,
        }

    # Apply degradation haircut
    degradation_factor = max(0.0, 1.0 - oos_degradation_pct)
    adj_win_rate = win_rate * degradation_factor
    adj_avg_win = avg_win_pct * degradation_factor
    adj_avg_loss = avg_loss_pct

    # Full Kelly: f* = (p * b - q) / b
    b = adj_avg_win / adj_avg_loss if adj_avg_loss > 0 else 0
    p = adj_win_rate
    q = 1.0 - p
    raw_kelly = (p * b - q) / b if b > 0 else 0

    # Fractional Kelly
    kelly_used = raw_kelly * max_kelly_fraction

    # Apply tier-based portfolio cap
    max_pct = sizing_tier.max_position_pct()
    max_position = portfolio_value * max_pct
    kelly_position = portfolio_value * kelly_used
    final_position = min(max(kelly_position, 0.0), max_position)

    # Negative edge
    if raw_kelly <= 0:
        return {
            "kelly_fraction": 0.0,
            "position_size_usdt": 0.0,
            "portfolio_pct": 0.0,
            "full_kelly": round(raw_kelly, 4),
            "rationale": "Negative adjusted Kelly: no position recommended.",
            "haircut_applied": f"degradation={oos_degradation_pct:.0%}, tier={sizing_tier.value}",
        }

    return {
        "kelly_fraction": round(kelly_used, 4),
        "full_kelly": round(raw_kelly, 4),
        "position_size_usdt": round(final_position, 2),
        "portfolio_pct": round(final_position / portfolio_value * 100, 2),
        "max_position_usdt": round(max_position, 2),
        "sizing_tier": sizing_tier.value,
        "haircut_applied": (
            f"degradation={oos_degradation_pct:.0%}->adjusted WR={adj_win_rate:.2f}, "
            f"AW={adj_avg_win:.2%}, tier={sizing_tier.value} cap={max_pct:.0%}"
        ),
        "rationale": (
            f"Raw Kelly={raw_kelly:.2%}, fraction={max_kelly_fraction:.0%}, "
            f"used={kelly_used:.2%}. Degradation haircut={oos_degradation_pct:.0%}. "
            f"Tier={sizing_tier.value} cap={max_pct:.0%}. "
            f"Position=${final_position:.0f} ({final_position/portfolio_value*100:.1f}% of portfolio)."
        ),
    }


# ── Bayesian Kelly Position Sizing ──

BETA_PRIOR_ALPHA = 2.0  # Weakly informative prior
BETA_PRIOR_BETA = 2.0
BAYESIAN_CI_LOWER = 0.05  # 90% CI lower bound
BAYESIAN_CI_UPPER = 0.95  # 90% CI upper bound


def _beta_posterior_win_rate(wins: float, losses: float) -> dict:
    from scipy.stats import beta as beta_dist
    alpha_post = BETA_PRIOR_ALPHA + wins
    beta_post = BETA_PRIOR_BETA + losses
    if alpha_post > 1 and beta_post > 1:
        map_win_rate = (alpha_post - 1) / (alpha_post + beta_post - 2)
    else:
        map_win_rate = alpha_post / (alpha_post + beta_post)
    ci_lower = beta_dist.ppf(BAYESIAN_CI_LOWER, alpha_post, beta_post)
    ci_upper = beta_dist.ppf(BAYESIAN_CI_UPPER, alpha_post, beta_post)
    return {
        "map_win_rate": round(map_win_rate, 4),
        "ci_lower": round(float(ci_lower), 4),
        "ci_upper": round(float(ci_upper), 4),
        "total_trades": int(wins + losses),
        "posterior_alpha": round(alpha_post, 2),
        "posterior_beta": round(beta_post, 2),
    }


def _estimate_wins_losses(win_rate: float, total_trades: int) -> tuple:
    wins = win_rate * total_trades
    losses = total_trades - wins
    return wins, losses


def bayesian_kelly_position_size(
    win_rate: float = 0.5,
    avg_win_pct: float = 0.02,
    avg_loss_pct: float = 0.01,
    portfolio_value: float = 10000.0,
    total_trades: int = 50,
    max_kelly_fraction: float = 0.25,
    sizing_tier: "PositionSizingTier" = None,
) -> dict:
    """Bayesian Kelly position sizing using Beta posterior for win rate.

    Args:
        win_rate: Historical win rate as decimal (0.0-1.0), e.g. 0.55.
        avg_win_pct: Average winning trade return as decimal, e.g. 0.02.
        avg_loss_pct: Average losing trade loss as decimal, e.g. 0.01.
        portfolio_value: Current portfolio value in USDT, e.g. 10000.0.
        total_trades: Number of trades for Beta posterior (default 50).
        max_kelly_fraction: Fraction of full Kelly (default 0.25).
        sizing_tier: PositionSizingTier enum.
    """
    # Type coercion: LLM may pass strings via LangChain tools.
    # Also handle JSON-object string passed as first positional arg.
    if isinstance(win_rate, str) and win_rate.strip().startswith("{"):
        import json as _json
        try:
            _parsed = _json.loads(win_rate)
            win_rate = float(_parsed.get("win_rate", 0.5))
            avg_win_pct = float(_parsed.get("avg_win_pct", 0.02))
            avg_loss_pct = float(_parsed.get("avg_loss_pct", 0.01))
            portfolio_value = float(_parsed.get("portfolio_value", 10000.0))
            total_trades = int(_parsed.get("total_trades", 50))
            max_kelly_fraction = float(_parsed.get("max_kelly_fraction", 0.25))
            tier_name = _parsed.get("sizing_tier", "cautious")
            sizing_tier = PositionSizingTier(tier_name) if tier_name in ("validation", "cautious", "normal") else PositionSizingTier.CAUTIOUS
        except (_json.JSONDecodeError, ValueError):
            pass
    else:
        win_rate = float(win_rate)
        avg_win_pct = float(avg_win_pct)
        avg_loss_pct = float(avg_loss_pct)
        portfolio_value = float(portfolio_value)
        total_trades = int(total_trades)
        max_kelly_fraction = float(max_kelly_fraction)
    if sizing_tier is None:
        sizing_tier = PositionSizingTier.CAUTIOUS
    if win_rate <= 0 or avg_loss_pct <= 0:
        return {"kelly_fraction": 0.0, "position_size_usdt": 0.0, "portfolio_pct": 0.0,
                "rationale": "Invalid inputs: win_rate or avg_loss must be positive", "error": True}
    if sizing_tier is None:
        sizing_tier = PositionSizingTier.CAUTIOUS
    wins, losses = _estimate_wins_losses(win_rate, total_trades)
    posterior = _beta_posterior_win_rate(wins, losses)
    bayesian_win_rate = posterior["ci_lower"]
    b = avg_win_pct / avg_loss_pct if avg_loss_pct > 0 else 0
    p = bayesian_win_rate
    q = 1.0 - p
    raw_kelly = (p * b - q) / b if b > 0 else 0
    kelly_used = raw_kelly * max_kelly_fraction
    max_pct = sizing_tier.max_position_pct()
    max_position = portfolio_value * max_pct
    kelly_position = portfolio_value * kelly_used
    final_position = min(max(kelly_position, 0.0), max_position)
    if raw_kelly <= 0:
        return {"kelly_fraction": 0.0, "position_size_usdt": 0.0, "portfolio_pct": 0.0,
                "full_kelly": round(raw_kelly, 4), "bayesian_posterior": posterior,
                "rationale": "Negative Bayesian Kelly: no position recommended.",
                "method": "bayesian"}
    return {
        "kelly_fraction": round(kelly_used, 4),
        "full_kelly": round(raw_kelly, 4),
        "position_size_usdt": round(final_position, 2),
        "portfolio_pct": round(final_position / portfolio_value * 100, 2),
        "max_position_usdt": round(max_position, 2),
        "sizing_tier": sizing_tier.value,
        "bayesian_posterior": posterior,
        "bayesian_used_wr": round(bayesian_win_rate, 4),
        "observed_wr": round(win_rate, 4),
        "method": "bayesian",
        "rationale": "Bayesian Kelly used.",
    }


def bayesian_kelly_position_size_conservative(
    win_rate: float = 0.5,
    avg_win_pct: float = 0.02,
    avg_loss_pct: float = 0.01,
    portfolio_value: float = 10000.0,
    total_trades: int = 50,
    oos_degradation_pct: float = 0.40,
    max_kelly_fraction: float = 0.25,
    sizing_tier: "PositionSizingTier" = None,
) -> dict:
    # Type coercion: LLM may pass strings via LangChain tools.
    # Also handle JSON-object string passed as first positional arg.
    if isinstance(win_rate, str) and win_rate.strip().startswith("{"):
        import json as _json
        try:
            _parsed = _json.loads(win_rate)
            win_rate = float(_parsed.get("win_rate", 0.5))
            avg_win_pct = float(_parsed.get("avg_win_pct", 0.02))
            avg_loss_pct = float(_parsed.get("avg_loss_pct", 0.01))
            portfolio_value = float(_parsed.get("portfolio_value", 10000.0))
            total_trades = int(_parsed.get("total_trades", 50))
            oos_degradation_pct = float(_parsed.get("oos_degradation_pct", 0.40))
            max_kelly_fraction = float(_parsed.get("max_kelly_fraction", 0.25))
            tier_name = _parsed.get("sizing_tier", "cautious")
            sizing_tier = PositionSizingTier(tier_name) if tier_name in ("validation", "cautious", "normal") else PositionSizingTier.CAUTIOUS
        except (_json.JSONDecodeError, ValueError):
            pass
    else:
        win_rate = float(win_rate)
        avg_win_pct = float(avg_win_pct)
        avg_loss_pct = float(avg_loss_pct)
        portfolio_value = float(portfolio_value)
        total_trades = int(total_trades)
        oos_degradation_pct = float(oos_degradation_pct)
        max_kelly_fraction = float(max_kelly_fraction)
    if win_rate <= 0 or avg_loss_pct <= 0:
        return {"kelly_fraction": 0.0, "position_size_usdt": 0.0, "portfolio_pct": 0.0,
                "rationale": "Invalid inputs: win_rate or avg_loss must be positive", "error": True}
    if sizing_tier is None:
        sizing_tier = PositionSizingTier.CAUTIOUS
    degradation_factor = max(0.0, 1.0 - oos_degradation_pct)
    adj_win_rate = win_rate * degradation_factor
    adj_avg_win = avg_win_pct * degradation_factor
    adj_avg_loss = avg_loss_pct
    wins, losses = _estimate_wins_losses(adj_win_rate, total_trades)
    posterior = _beta_posterior_win_rate(wins, losses)
    bayesian_win_rate = posterior["ci_lower"]
    b = adj_avg_win / adj_avg_loss if adj_avg_loss > 0 else 0
    p = bayesian_win_rate
    q = 1.0 - p
    raw_kelly = (p * b - q) / b if b > 0 else 0
    kelly_used = raw_kelly * max_kelly_fraction
    max_pct = sizing_tier.max_position_pct()
    max_position = portfolio_value * max_pct
    kelly_position = portfolio_value * kelly_used
    final_position = min(max(kelly_position, 0.0), max_position)
    if raw_kelly <= 0:
        return {"kelly_fraction": 0.0, "position_size_usdt": 0.0, "portfolio_pct": 0.0,
                "full_kelly": round(raw_kelly, 4), "bayesian_posterior": posterior,
                "haircut_applied": "degradation={:.0%}, tier={}".format(oos_degradation_pct, sizing_tier.value),
                "rationale": "Negative conservative Bayesian Kelly: no position.",
                "method": "bayesian_conservative"}
    return {
        "kelly_fraction": round(kelly_used, 4),
        "full_kelly": round(raw_kelly, 4),
        "position_size_usdt": round(final_position, 2),
        "portfolio_pct": round(final_position / portfolio_value * 100, 2),
        "max_position_usdt": round(max_position, 2),
        "sizing_tier": sizing_tier.value,
        "bayesian_posterior": posterior,
        "bayesian_used_wr": round(bayesian_win_rate, 4),
        "degraded_wr": round(adj_win_rate, 4),
        "observed_wr": round(win_rate, 4),
        "oos_degradation_pct": oos_degradation_pct,
        "method": "bayesian_conservative",
        "haircut_applied": "degradation={:.0%}, tier={}, posterior Beta({:.1f},{:.1f})".format(
            oos_degradation_pct, sizing_tier.value, posterior["posterior_alpha"], posterior["posterior_beta"]),
        "rationale": "Conservative Bayesian Kelly used.",
    }






class RiskManagerAgent(BaseAgent):
    """Evaluates trading strategies for risk and produces quantitative go/no-go decisions."""

    def __init__(self, fetcher: Optional[MarketDataFetcher] = None):
        self._fetcher = fetcher or MarketDataFetcher()
        tools = self._build_tools()
        super().__init__(name="risk_manager", tools=tools, system_prompt=RISK_MANAGER_PROMPT)

    def _build_tools(self):
        # ── Tool 1: Kelly Criterion position sizing ──

        def kelly_position_size(params_json: str = "{}") -> str:
            """Compute optimal position size using fractional Kelly Criterion.

            Args: JSON with:
              - win_rate: float (0.0-1.0, from backtest)
              - avg_win_pct: float (e.g. 0.05 for 5% average win)
              - avg_loss_pct: float (e.g. 0.03 for 3% average loss)
              - portfolio_value: float (current portfolio USDT value)
              - max_kelly_fraction: float (default 0.25 = quarter Kelly)

            Returns JSON with kelly_fraction, position_size_usdt, rationale.
            Never returns more than 10% of portfolio per trade.
            """
            try:
                params = json.loads(params_json) if isinstance(params_json, str) else params_json
            except json.JSONDecodeError:
                return "Error: invalid JSON"

            win_rate = float(params.get("win_rate", 0.5))
            avg_win = float(params.get("avg_win_pct", 0.05))
            avg_loss = float(params.get("avg_loss_pct", 0.03))
            portfolio_value = float(params.get("portfolio_value", 10000.0))
            max_kelly_frac = float(params.get("max_kelly_fraction", 0.25))
            oos_degradation = float(params.get("oos_degradation_pct", 0.40))
            tier_name = params.get("sizing_tier", "cautious")

            if avg_loss <= 0:
                return json.dumps({
                    "kelly_fraction": 0.0,
                    "position_size_usdt": 0.0,
                    "rationale": "Avg loss must be positive",
                    "error": True,
                })

            # Apply degradation haircut to backtest metrics
            degradation_factor = max(0.0, 1.0 - oos_degradation)
            adj_win_rate = win_rate * degradation_factor
            adj_avg_win = avg_win * degradation_factor
            adj_avg_loss = avg_loss

            # Full Kelly: f* = (p * b - q) / b, where b = avg_win/avg_loss
            b = adj_avg_win / adj_avg_loss
            p = adj_win_rate
            q = 1.0 - p
            kelly_full = (p * b - q) / b

            # Apply fractional Kelly (default quarter)
            kelly_used = kelly_full * max_kelly_frac

            # Tier-based cap
            try:
                tier = PositionSizingTier(tier_name)
            except ValueError:
                tier = PositionSizingTier.CAUTIOUS
            max_pct = tier.max_position_pct()
            max_position = portfolio_value * max_pct
            kelly_position = portfolio_value * kelly_used
            final_position = min(max(kelly_position, 0.0), max_position)

            # Handle negative Kelly (edge flipped)
            if kelly_full <= 0:
                return json.dumps({
                    "kelly_fraction": 0.0,
                    "position_size_usdt": 0.0,
                    "full_kelly": float(kelly_full),
                    "oos_degradation_pct": oos_degradation,
                    "sizing_tier": tier.value,
                    "rationale": "Negative Kelly after degradation adjustment: no position recommended.",
                })

            return json.dumps({
                "kelly_fraction": round(kelly_used, 4),
                "full_kelly": float(kelly_full),
                "position_size_usdt": round(final_position, 2),
                "max_position_usdt": round(max_position, 2),
                "portfolio_pct": round(final_position / portfolio_value * 100, 2),
                "oos_degradation_pct": oos_degradation,
                "sizing_tier": tier.value,
                "haircut_applied": f"WR={win_rate:.2f}->{adj_win_rate:.2f}, AW={avg_win:.2%}->{adj_avg_win:.2%}",
                "rationale": (
                    f"Kelly full={kelly_full:.2%}, fraction={max_kelly_frac:.0%}, "
                    f"used={kelly_used:.2%}. Degradation haircut={oos_degradation:.0%}. "
                    f"Tier={tier.value} cap={max_pct:.0%}. "
                    f"Position=${final_position:.0f} "
                    f"({final_position/portfolio_value*100:.1f}% of portfolio)."
                ),
            })

        # ── Tool 2: Correlation gate ──

        def check_position_correlation(params_json: str = "{}") -> str:
            """Check correlation between proposed pair and current open positions.

            Args: JSON with:
              - proposed_pair: str (e.g. 'BTC/USDT')
              - open_positions: list of dicts with 'pair' field
              - max_correlation: float (default 0.7)

            Returns JSON with approved bool, max_correlation value, correlated_with.
            Uses 30-day rolling return correlation from daily close prices.
            """
            try:
                params = json.loads(params_json) if isinstance(params_json, str) else params_json
            except json.JSONDecodeError:
                return json.dumps({"approved": True, "error": "invalid JSON", "max_correlation": 0.0})

            proposed = _normalize(params.get("proposed_pair", "BTC/USDT"))
            open_positions = params.get("open_positions", [])
            max_corr = float(params.get("max_correlation", 0.7))

            if not open_positions:
                return json.dumps({
                    "approved": True,
                    "max_correlation": 0.0,
                    "correlated_with": None,
                    "rationale": "No open positions — no correlation risk.",
                })

            # Fetch 30 days of daily returns for proposed pair
            try:
                proposed_df = self._fetcher.fetch_ohlcv(proposed, "1d", limit=30)
                if proposed_df is None or len(proposed_df) < 10:
                    return json.dumps({
                        "approved": True,
                        "max_correlation": 0.0,
                        "correlated_with": None,
                        "rationale": f"Insufficient data for {proposed} — approving by default.",
                    })
                proposed_returns = proposed_df["close"].astype(float).pct_change().dropna()
            except Exception as exc:
                return json.dumps({
                    "approved": True,
                    "max_correlation": 0.0,
                    "error": str(exc),
                    "rationale": f"Could not fetch data for {proposed} — approving by default.",
                })

            max_corr_found = 0.0
            worst_pair = None

            for pos in open_positions:
                pos_pair = _normalize(pos.get("pair", ""))
                if not pos_pair or pos_pair == proposed:
                    continue
                try:
                    pos_df = self._fetcher.fetch_ohlcv(pos_pair, "1d", limit=30)
                    if pos_df is None or len(pos_df) < 10:
                        continue
                    pos_returns = pos_df["close"].astype(float).pct_change().dropna()
                    # Align lengths
                    min_len = min(len(proposed_returns), len(pos_returns))
                    if min_len < 5:
                        continue
                    corr = proposed_returns.iloc[-min_len:].corr(pos_returns.iloc[-min_len:])
                    if abs(corr) > max_corr_found:
                        max_corr_found = abs(corr)
                        worst_pair = pos_pair
                except Exception:
                    continue

            approved = max_corr_found <= max_corr

            return json.dumps({
                "approved": approved,
                "max_correlation": round(max_corr_found, 4),
                "correlated_with": worst_pair,
                "threshold": max_corr,
                "rationale": (
                    f"Approved: max correlation {max_corr_found:.2f} <= {max_corr:.2f} threshold"
                    if approved else
                    f"BLOCKED: {proposed} correlated {max_corr_found:.2f} with {worst_pair} "
                    f"(threshold {max_corr:.2f})"
                ),
            })

        # ── Tool 3: Circuit breaker check ──

        def circuit_breaker_check(params_json: str = "{}") -> str:
            """Check if trading is allowed by the circuit breaker.

            Args: JSON with optional:
              - daily_pnl_pct: float (default 0.0)
              - weekly_pnl_pct: float (default 0.0)
              - daily_limit: float (default -0.03 = -3%)
              - weekly_limit: float (default -0.08 = -8%)

            Returns JSON with trading_allowed, reason, resume_after.
            """
            try:
                params = json.loads(params_json) if isinstance(params_json, str) else params_json
            except json.JSONDecodeError:
                params = {}

            daily_pnl = float(params.get("daily_pnl_pct", 0.0))
            weekly_pnl = float(params.get("weekly_pnl_pct", 0.0))
            daily_limit_raw = float(params.get("daily_limit", -0.03))
            weekly_limit_raw = float(params.get("weekly_limit", -0.08))

            # Soft clamp: never raise AssertionError. If value > 1.0, treat as
            # percentage (divide by 100). If <= 0, use default.
            def _clamp_limit(raw: float, default: float) -> float:
                if raw <= 0:
                    logger.warning(
                        "Circuit breaker limit %f <= 0, using default %f",
                        raw, abs(default),
                    )
                    return abs(default)
                if raw > 1.0:
                    logger.warning(
                        "Circuit breaker limit %f > 1.0, dividing by 100 "
                        "(treated as percentage)",
                        raw,
                    )
                    return raw / 100.0
                return raw

            daily_limit = _clamp_limit(daily_limit_raw, 0.03)
            weekly_limit = _clamp_limit(weekly_limit_raw, 0.08)

            # Check if already halted
            if CircuitBreakerState.is_halted():
                status = CircuitBreakerState.status()
                return json.dumps({
                    "trading_allowed": False,
                    "reason": status["reason"],
                    "resume_after": status["resume_after"],
                    "triggered_by": "circuit_breaker",
                })

            # Check daily drawdown — only halt on actual negative P&L exceeding limit
            if daily_pnl < 0 and abs(daily_pnl) > daily_limit:
                CircuitBreakerState.halt(
                    f"Daily drawdown {daily_pnl:.2%} exceeds limit {daily_limit:.2%}",
                    duration_minutes=60,
                )
                return json.dumps({
                    "trading_allowed": False,
                    "reason": f"Daily drawdown {abs(daily_pnl):.2%} > {daily_limit:.2%} limit",
                    "triggered_by": "daily_drawdown",
                    "resume_after": CircuitBreakerState.status()["resume_after"],
                })

            # Check weekly drawdown
            if weekly_pnl < 0 and abs(weekly_pnl) > weekly_limit:
                CircuitBreakerState.halt(
                    f"Weekly drawdown {weekly_pnl:.2%} exceeds limit {weekly_limit:.2%}",
                    duration_minutes=360,  # 6 hours for weekly limit breach
                )
                return json.dumps({
                    "trading_allowed": False,
                    "reason": f"Weekly drawdown {abs(weekly_pnl):.2%} > {weekly_limit:.2%} limit",
                    "triggered_by": "weekly_drawdown",
                    "resume_after": CircuitBreakerState.status()["resume_after"],
                })

            return json.dumps({
                "trading_allowed": True,
                "reason": "All checks passed",
                "daily_pnl_pct": round(daily_pnl, 4),
                "weekly_pnl_pct": round(weekly_pnl, 4),
            })

        # ── Tool 4: Strategy risk assessment ──

        def assess_strategy_risk(params_json: str = "{}") -> str:
            """Assess a strategy's risk profile from backtest metrics.

            Args: JSON with:
              - sharpe_ratio: float
              - win_rate: float (0.0-1.0)
              - max_drawdown: float (e.g. 0.05 for 5%)
              - total_trades: int
              - profit_factor: float (total_profit / total_loss)

            Returns JSON with risk_score, verdict, concerns list.
            """
            try:
                params = json.loads(params_json) if isinstance(params_json, str) else params_json
            except json.JSONDecodeError:
                return json.dumps({"verdict": "error", "concerns": ["Invalid JSON"]})

            sharpe = float(params.get("sharpe_ratio", 0))
            win_rate = float(params.get("win_rate", 0))
            drawdown = abs(float(params.get("max_drawdown", 0)))
            trades = int(params.get("total_trades", 0))
            profit_factor = float(params.get("profit_factor", 1.0))

            concerns = []
            risk_score = 0.0  # 0 = safe, 1 = extremely risky

            if trades < 30:
                concerns.append(f"Low sample size ({trades} trades)")
                risk_score += 0.2
            if sharpe < 0.8:
                concerns.append(f"Low Sharpe ({sharpe:.2f})")
                risk_score += 0.25
            elif sharpe < 1.2:
                concerns.append(f"Marginal Sharpe ({sharpe:.2f})")
                risk_score += 0.1
            if win_rate < 0.40:
                concerns.append(f"Low win rate ({win_rate:.0%})")
                risk_score += 0.15
            if drawdown > 0.10:
                concerns.append(f"High drawdown ({drawdown:.2%})")
                risk_score += 0.2
            elif drawdown > 0.05:
                concerns.append(f"Elevated drawdown ({drawdown:.2%})")
                risk_score += 0.1
            if profit_factor < 1.2:
                concerns.append(f"Low profit factor ({profit_factor:.2f})")
                risk_score += 0.15

            # Verdict
            if risk_score >= 0.5 or not concerns:
                verdict = "reject"
            elif risk_score >= 0.3:
                verdict = "cautious"
            else:
                verdict = "accept"

            risk_level = ["low", "medium", "high", "extreme"][min(int(risk_score * 4), 3)]

            return json.dumps({
                "risk_score": round(risk_score, 3),
                "risk_level": risk_level,
                "verdict": verdict,
                "concern_count": len(concerns),
                "concerns": concerns,
                "max_drawdown": round(drawdown, 4),
                "profit_factor": round(profit_factor, 2),
            })

        # ── Tool 5: Pre-trade approval (final gate) ──

        def pre_trade_approval(params_json: str = "{}") -> str:
            """Final go/no-go gate before any trade. Runs all checks.

            Args: JSON with:
              - strategy_metrics: dict (sharpe_ratio, win_rate, max_drawdown, total_trades)
              - kelly_result: dict (from kelly_position_size)
              - correlation_result: dict (from check_position_correlation)
              - circuit_breaker_result: dict (from circuit_breaker_check)
              - regime: str (optional, current market regime)

            Returns JSON with approved bool, position_size_usdt, confidence, reasons list.
            """
            try:
                params = json.loads(params_json) if isinstance(params_json, str) else params_json
            except json.JSONDecodeError:
                return json.dumps({"approved": False, "error": "Invalid JSON"})

            reasons = []

            # 1. Circuit breaker check
            cb = params.get("circuit_breaker_result", {})
            if isinstance(cb, str):
                try:
                    cb = json.loads(cb)
                except json.JSONDecodeError:
                    cb = {}
            if not cb.get("trading_allowed", True):
                return json.dumps({
                    "approved": False,
                    "position_size_usdt": 0.0,
                    "confidence": 0.0,
                    "reasons": [f"Circuit breaker: {cb.get('reason', 'halted')}"],
                })

            # 2. Correlation check
            corr = params.get("correlation_result", {})
            if isinstance(corr, str):
                try:
                    corr = json.loads(corr)
                except json.JSONDecodeError:
                    corr = {}
            if not corr.get("approved", True):
                return json.dumps({
                    "approved": False,
                    "position_size_usdt": 0.0,
                    "confidence": 0.0,
                    "reasons": [f"Correlation: {corr.get('rationale', 'blocked')}"],
                })

            # 3. Strategy risk assessment
            metrics = params.get("strategy_metrics", {})
            if isinstance(metrics, str):
                try:
                    metrics = json.loads(metrics)
                except json.JSONDecodeError:
                    metrics = {}
            risk_result = json.loads(assess_strategy_risk(metrics)) if metrics else {}
            if risk_result.get("verdict") == "reject":
                return json.dumps({
                    "approved": False,
                    "position_size_usdt": 0.0,
                    "confidence": 0.0,
                    "reasons": risk_result.get("concerns", ["Strategy risk too high"]),
                })

            # 4. Kelly sizing
            kelly = params.get("kelly_result", {})
            if isinstance(kelly, str):
                try:
                    kelly = json.loads(kelly)
                except json.JSONDecodeError:
                    kelly = {}
            position_size = float(kelly.get("position_size_usdt", 0.0))
            kelly_frac = float(kelly.get("kelly_fraction", 0.0))

            if position_size <= 0:
                return json.dumps({
                    "approved": False,
                    "position_size_usdt": 0.0,
                    "confidence": 0.0,
                    "reasons": [kelly.get("rationale", "Kelly sizing returned zero position")],
                })

            # 5. Compute confidence score
            sharpe = float(metrics.get("sharpe_ratio", 0))
            win_rate = float(metrics.get("win_rate", 0))
            confidence = min(sharpe / 2.0, 1.0) * 0.5 + win_rate * 0.5

            reasons.append(f"Kelly position: ${position_size:.0f}")
            reasons.append(f"Confidence score: {confidence:.1%}")
            regime = params.get("regime", "unknown")
            reasons.append(f"Regime: {regime}")

            if risk_result.get("verdict") == "cautious":
                reasons.append("CAUTIOUS: strategy has moderate risk flags")
                confidence *= 0.8

            return json.dumps({
                "approved": True,
                "position_size_usdt": round(position_size, 2),
                "kelly_fraction": round(kelly_frac, 4),
                "confidence": round(confidence, 4),
                "reasons": reasons,
            })


        def bayesian_kelly_wrapper(params_json: str = "{}") -> str:
            """Compute position size using Bayesian Kelly with Beta posterior."""
            try:
                params = json.loads(params_json) if isinstance(params_json, str) else params_json
            except json.JSONDecodeError:
                return json.dumps({"error": "invalid JSON"})
            try:
                tier = PositionSizingTier(params.get("sizing_tier", "cautious"))
            except ValueError:
                tier = PositionSizingTier.CAUTIOUS
            result = bayesian_kelly_position_size(
                win_rate=float(params.get("win_rate", 0.5)),
                avg_win_pct=float(params.get("avg_win_pct", 0.05)),
                avg_loss_pct=float(params.get("avg_loss_pct", 0.03)),
                portfolio_value=float(params.get("portfolio_value", 10000.0)),
                total_trades=int(params.get("total_trades", 50)),
                max_kelly_fraction=float(params.get("max_kelly_fraction", 0.25)),
                sizing_tier=tier,
            )
            return json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in result.items()})

        def bayesian_kelly_conservative_wrapper(params_json: str = "{}") -> str:
            """Compute position size with Bayesian Kelly + degradation haircut."""
            try:
                params = json.loads(params_json) if isinstance(params_json, str) else params_json
            except json.JSONDecodeError:
                return json.dumps({"error": "invalid JSON"})
            try:
                tier = PositionSizingTier(params.get("sizing_tier", "cautious"))
            except ValueError:
                tier = PositionSizingTier.CAUTIOUS
            result = bayesian_kelly_position_size_conservative(
                win_rate=float(params.get("win_rate", 0.5)),
                avg_win_pct=float(params.get("avg_win_pct", 0.05)),
                avg_loss_pct=float(params.get("avg_loss_pct", 0.03)),
                portfolio_value=float(params.get("portfolio_value", 10000.0)),
                total_trades=int(params.get("total_trades", 50)),
                oos_degradation_pct=float(params.get("oos_degradation_pct", 0.40)),
                max_kelly_fraction=float(params.get("max_kelly_fraction", 0.25)),
                sizing_tier=tier,
            )
            return json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in result.items()})

        return [
            Tool(name="kelly_position_size", func=kelly_position_size,
                 description="Compute optimal position size using fractional Kelly Criterion. "
                             "Args: JSON with win_rate, avg_win_pct, avg_loss_pct, portfolio_value, max_kelly_fraction"),
            Tool(name="kelly_position_size_conservative", func=kelly_position_size_conservative,
                 description="Compute position size with pessimism-adjusted Kelly. Applies degradation haircut "
                             "to backtest metrics before calculation. Args: JSON with win_rate, avg_win_pct, "
                             "avg_loss_pct, portfolio_value, oos_degradation_pct (default 0.40), "
                             "max_kelly_fraction (default 0.25), sizing_tier ('validation'|'cautious'|'normal')"),
            Tool(name="bayesian_kelly", func=bayesian_kelly_wrapper,
                 description="(RECOMMENDED) Compute position size using Bayesian Kelly with Beta posterior "
                             "for win-rate uncertainty. Uses lower bound of 90% credible interval as "
                             "conservative win rate. Args: JSON with win_rate, avg_win_pct, avg_loss_pct, "
                             "portfolio_value, total_trades (default 50), max_kelly_fraction (default 0.25), "
                             "sizing_tier ('validation'|'cautious'|'normal')"),
            Tool(name="bayesian_kelly_conservative", func=bayesian_kelly_conservative_wrapper,
                 description="(RECOMMENDED) Most conservative sizing: Bayesian Kelly + degradation haircut "
                             "applied to backtest metrics before Beta posterior calculation. "
                             "Args: JSON with win_rate, avg_win_pct, avg_loss_pct, portfolio_value, "
                             "total_trades (default 50), oos_degradation_pct (default 0.40), "
                             "max_kelly_fraction (default 0.25), sizing_tier"),
            Tool(name="check_position_correlation", func=check_position_correlation,
                 description="Check if a proposed pair is too correlated with open positions. "
                             "Args: JSON with proposed_pair, open_positions list, max_correlation threshold"),
            Tool(name="circuit_breaker_check", func=circuit_breaker_check,
                 description="Check if trading is allowed by the global circuit breaker. "
                             "Args: JSON with daily_pnl_pct, weekly_pnl_pct, daily_limit, weekly_limit"),
            Tool(name="assess_strategy_risk", func=assess_strategy_risk,
                 description="Assess a strategy's risk profile from backtest metrics. "
                             "Args: JSON with sharpe_ratio, win_rate, max_drawdown, total_trades, profit_factor"),
            Tool(name="pre_trade_approval", func=pre_trade_approval,
                 description="FINAL GATE: run all checks before approving a trade. "
                             "Args: JSON with strategy_metrics, kelly_result, correlation_result, circuit_breaker_result, regime"),
        ]
