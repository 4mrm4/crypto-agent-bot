# Fix: Backtester Missing, UI Not Showing Results, Circuit Breaker False Halt

## Bug 1 — Backtester agent missing from autonomous mode

**Root cause:** `main.py:_run_autonomous()` creates an agents dict without `"backtester": BacktesterAgent()`. The LangGraph dispatches backtest tasks to an agent that doesn't exist → returns immediately with no-op → `task.result = None` → metrics fall through to stale experiments.jsonl.

**Fix:** Add `"backtester": BacktesterAgent()` to the agents dict in `main.py:148-154`. Import `BacktesterAgent` at the top of the function.

## Bug 2 — UI doesn't show autonomous research results

**Root cause:** The Research tab listens for WebSocket events at `/ws/run/{run_id}`, but autonomous loop research cycles emit events through the EventBus without a connected WebSocket client. The Dashboard tab polls REST endpoints that don't include iteration history.

**Fix:** 
1. Add iteration history tracking to the autonomous loop state (accumulate iteration results across cycles)
2. Add `/api/autonomous/iterations` REST endpoint returning accumulated chart data
3. Wire the autonomous_state to include iteration metrics (current sharpe, best sharpe, discarded count)
4. Update UI Dashboard poll to also fetch `/api/autonomous/iterations` and display the Sharpe chart, best value, and discarded strategies

## Bug 3 — Circuit breaker false halt on daily drawdown

**Root cause:** `daily_pnl` is -50% (decimal -0.50), `daily_limit` is 3% (decimal 0.03). The check `daily_pnl < daily_limit` with `daily_pnl < 0` is True → halts. Already has a guard for `daily_pnl < 0` but the comparison is still incorrect because it compares a negative decimal against a positive decimal limit.

**Fix:** In `risk_manager.py`, the circuit breaker check should use `abs(daily_pnl) > abs(limit)` when `daily_pnl < 0`, ensuring we compare magnitude against the limit regardless of sign.

## Implementation Order

1. Bug 3 (circuit breaker) — simplest fix, isolated to one file
2. Bug 1 (backtester missing) — one-liner in main.py
3. Bug 2 (UI results) — multi-file change: autonomous loop state, API endpoint, UI component
