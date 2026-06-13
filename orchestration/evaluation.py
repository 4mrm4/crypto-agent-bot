"""Shared strategy quality evaluation — single source of truth for keep/discard verdicts.

All agent-level and orchestration-level evaluation converges here so that
threshold changes propagate everywhere they should, and nowhere they shouldn't.

The deployment gate in ``orchestration/experiment_tracker.py`` deliberately
uses **stricter** thresholds (Sharpe >= 1.2, WR >= 48 %, DD <= 10 %, plus
walk-forward, Monte Carlo, and synthetic sanity checks).  That module is the
right place for production-readiness decisions; this module handles the
inner-loop "is this worth pursuing ?" question.
"""

# ---------------------------------------------------------------------------
# Shared thresholds — convergence and keep / discard share these values.
# Changing them here affects both agent-level evaluation and the outer-loop
# convergence gate, which is the desired behaviour.
# ---------------------------------------------------------------------------
CONVERGENCE_SHARPE = 0.8
CONVERGENCE_WIN_RATE = 0.40
CONVERGENCE_MAX_DRAWDOWN = 0.15
CONVERGENCE_MIN_TRADES = 5


def evaluate_strategy_quality(metrics: dict) -> tuple:
    """Return ``(verdict, reason)`` based on shared convergence thresholds.

    Parameters
    ----------
    metrics :
        Must contain *sharpe_ratio*, *win_rate*, *max_drawdown*,
        *total_trades*, and either *profit_ratio* or *total_profit*.

    Returns
    -------
    tuple[str, str]
        ``("kept", "All targets met")`` or ``("discarded", "<reason>")``
        where *reason* lists every threshold that was missed.
    """
    issues = []
    sharpe = metrics.get("sharpe_ratio", 0)
    win_rate = metrics.get("win_rate", 0)
    drawdown = abs(metrics.get("max_drawdown", 0))
    total_trades = metrics.get("total_trades", 0)
    profit = metrics.get("profit_ratio", metrics.get("total_profit", 0))

    if total_trades < CONVERGENCE_MIN_TRADES:
        issues.append(
            f"Trades {total_trades} < {CONVERGENCE_MIN_TRADES}"
        )
    if sharpe < CONVERGENCE_SHARPE:
        issues.append(f"Sharpe {sharpe:.2f} < {CONVERGENCE_SHARPE}")
    if win_rate < CONVERGENCE_WIN_RATE:
        issues.append(
            f"Win rate {win_rate:.0%} < {CONVERGENCE_WIN_RATE}"
        )
    if drawdown > CONVERGENCE_MAX_DRAWDOWN:
        issues.append(
            f"Drawdown {drawdown:.2%} > {CONVERGENCE_MAX_DRAWDOWN}"
        )
    if profit <= 0:
        issues.append(f"Non-positive profit ({profit})")

    if issues:
        return "discarded", "; ".join(issues)
    return "kept", "All targets met"


def check_convergence(
    metrics: dict,
    total_trades_min: int = CONVERGENCE_MIN_TRADES,
) -> bool:
    """Return ``True`` when *metrics* meet the convergence criteria.

    This is a convenience wrapper around the same thresholds used by
    :func:`evaluate_strategy_quality` — kept as a separate function so
    callers that only need a boolean (HermesOrchestrator, outer loop)
    don't have to unpack a tuple they don't use.
    """
    sharpe = metrics.get("sharpe_ratio", 0)
    win_rate = metrics.get("win_rate", 0)
    drawdown = abs(metrics.get("max_drawdown", 0))
    total_trades = metrics.get("total_trades", 0)

    if total_trades < total_trades_min:
        return False
    if sharpe < CONVERGENCE_SHARPE:
        return False
    if win_rate < CONVERGENCE_WIN_RATE:
        return False
    if drawdown > CONVERGENCE_MAX_DRAWDOWN:
        return False
    return True
