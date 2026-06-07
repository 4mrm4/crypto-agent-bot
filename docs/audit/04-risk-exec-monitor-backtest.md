# Audit: Risk, Execution, Monitoring & Backtesting

Generated 2026-06-06.

---

## RISK SUBSYSTEM

### `risk/portfolio_var.py`

**Code Smells**
- Duplicate `_build_returns_matrix` in `var_covariance` and `var_historical` (lines 112-141, 210-236) — textbook copy-paste
- Hardcoded 252 trading days in multiple places — should be module-level constant
- `_empty_var_result` is partial — missing keys like `var_95_amount`, `var_95_pct`
- `_price_cache` and `_cache_timestamps` have no thread locks — race in async context
- MIN_HISTORICAL_DAYS=30 but `correlation_matrix` truncates to 252 regardless of data available

**Missing Tests:** `check_var_limits`, `correlation_matrix`, `update_portfolio_value`

---

## EXECUTION SUBSYSTEM

### `execution/live_executor.py`

**CRITICAL BUGS**
- **OPENAI_API_KEY used as exchange API key (lines 106-107):** `"apiKey": settings.OPENAI_API_KEY` sent to exchange — security + functional bug
- **Paper execution fill_price is wrong (line 238):** `fill_price=result.get("final_balance", 0)` — final_balance is cash+position, not a price
- **New PaperTrader created for every signal (lines 219-224):** Paper trading state resets every signal, `max_candles=5`
- **TWAP uses market orders (lines 302-306):** Market orders on large TWAP slices cause significant slippage — should use limit orders
- **TWAP price aggregation uses wrong field (line 308):** `order.get("price", 0)` — for market orders, `price` may be None

**Design Issues**
- **No circuit breaker check in `_execute_live` (lines 249-283):** Only checked at `execute_signal` level
- **`exchange` property lazy-init not thread-safe (lines 100-112):** Multiple coroutines = multiple instances
- **Stoploss monitoring uses global setting for all positions (line 358):** Ignores per-strategy stoploss

**Missing Tests:** Zero test coverage — no `test_live_executor.py`

### `execution/paper_trader.py`

**Bugs**
- **Only handles "long" positions (lines 63, 73):** Sell signal with no position silently ignored
- **Hardcoded 0.95 position size (line 66):** Ignores Kelly sizing

**Code Smells**
- `sma_crossover_signal` function is dead code (lines 107-120)

### `execution/audit_log.py`

**Bugs**
- `_load` silently swallows ALL exceptions (lines 63-67): Corrupted lines silently dropped

**Code Smells**
- Duplicate SQLite mirroring pattern in 4 files
- Inconsistent field name mapping: `strategy_name` vs `strategy_id`
- Sharpe calculation on single line is unreadable (lines 144-146)

### `execution/validation_mode.py`

**Code Smells**
- `days_live` computed in 3 separate properties instead of once
- `apply_position_cap` mutates input dict by adding keys — side effect
- Hardcoded Sharpe scaling factor `16.0` with no explanation

### `execution/trade_signal.py`

**Design Issues**
- `kelly_fraction` field never serialized in `to_freqtrade_config` — position sizing info lost
- `signal_id` defaults to `""` and not included in config — cannot correlate orders back to provenance

---

## MONITORING SUBSYSTEM

### `monitoring/anomaly_detector.py`

**Code Smells**
- `_check_negative_kelly` is a no-op (body just `pass`) (lines 166-168)
- Hardcoded time thresholds throughout (600s, 14400s, 60s, etc.) — should be named constants
- `_check_price_source` has tight coupling to specific APIs

### `monitoring/performance_monitor.py`

**Bugs**
- **Rolling Sharpe uses trade-count window, not time window (line 183):** `returns.rolling(window=window_days)` — window is number of rows, not days. With irregular trades, "30-day" window is actually "last 30 trades"

**Design Issues**
- Degradation Z-score is not a real z-score (lines 107-110): Normalizes by range, not standard deviation
- `detect_regime_mismatch` parameters come from `live_metrics` which is always empty — never fires in practice
- `get_summary` reads `live_sharpe` from ChromaDB metadata — may be stale

### `monitoring/telegram_alerter.py`

**CRITICAL BUGS**
- **Callback handler never registered (lines 117-130):** Inline keyboard buttons (Approve/Reject) are sent but no callback query handler is registered. User taps do nothing. Human-in-the-loop approvals are non-functional.
- **No polling started (lines 124-128):** Comment says "unless specifically needed" but without polling, bot can't receive responses

**Design Issues**
- Module-level constants from `os.getenv()` at import time — not from `config.Settings`
- Inline keyboard uses pair as callback_data — same pair with new signal causes Telegram crash

---

## BACKTESTING SUBSYSTEM

### `backtesting/engine.py`

**Bugs**
- **Race condition in strategy file cleanup (lines 202-206):** Concurrent `run_backtest` calls delete each other's files
- **`_parse_results` crashes on empty zip (lines 503-505):** `[0]` on empty list = IndexError
- **Subprocess timeout orphans temp files (line 468):** TimeoutExpired leaves strategy/config files on disk

**Design Issues**
- Freqtrade executable path is Windows-specific (lines 126-127): Breaks cross-platform
- `_build_config` reads config.json from disk (lines 424-426): Fails if file missing
- `TransactionCostModel.from_settings()` called twice (lines 433, 463): Redundant

### `backtesting/synthetic_validator.py`

**CRITICAL BUGS**
- **Synthetic data generated but NEVER passed to backtest (lines 155-165):** `df_synthetic = self.generate_random_walk(seed=i)` but engine is called with hardcoded timerange `"20170101-20231231"`. Validator is a complete no-op — runs on real data.
- **Same bug in `run_permutation_test` (lines 249-257):** Permutation test also runs on real data
- **BacktestEngine has no mechanism to accept pre-loaded DataFrames (lines 160-164):** Always runs Freqtrade subprocess on file-based data

### `backtesting/cpcv_validator.py`

**Bugs**
- **Purge/embargo logic is incorrect for multi-fold (lines 165-181):** Later folds can undo earlier removals. First purge step is redundant with second. Label span assumption may be wrong for asymmetric labels.

### `backtesting/blind_search.py`

**Bugs**
- **`_generate_default_variants` can produce non-integer values for TA-Lib integer params (lines 103-109):** `fast_ma` periods need integers, spread formula can produce floats
- **`generate_search_space` requires LLM but pipeline bypasses it via private method (line 41-43)**

### `backtesting/oos_validator.py`

**Bugs**
- **`compute_degradation` static method not used consistently (lines 185-195):** `validate_strategy` computes degradation inline — could diverge
- **Pass criteria docstring vs code inconsistency (lines 72-75 vs 109-113):** Uses `net_sharpe` for pass but raw Sharpe for degradation — strategy passes on net metrics but rejected on raw degradation

### `backtesting/signal_factory.py`

**Design Issues**
- **`_build_signal` uses Python for-loop over all rows (lines 38-47):** Slow for 5000+ candles — should be vectorized
- **Missing exit condition coverage:** Some strategies produce multiple entries before exit, creating incorrect trade sequences

### `backtesting/cost_model.py`

**Code Smells**
- Inconsistent fee comments (lines 21-22): 0.075% called "blended estimate" but lower than maker fee — values inverted relative to market convention

### `backtesting/timerange_utils.py`

**Design Issues**
- 11-branch function handles all LLM-invented formats (lines 34-68): Fragile, silently returns default on unrecognized formats
- Truncation without warning: `[:8]` can produce wrong dates

### `backtesting/setup_data.py`

**Bugs**
- **Data directory hardcoded to "binance" but exchange is Kraken (line 32):** File check always fails, triggers unnecessary re-downloads
- **File matching patterns fragile:** May match trade data files, not OHLCV
- **Freqtrade executable detection only checks Windows paths (lines 79-84)**
- **`subprocess.run` with user-provided pairs/timeframes (lines 86-96):** Command injection risk
- **DEFAULT_START computed at module load time (line 14):** Date fixed at import, not at call time

---

## CROSS-CUTTING

### Files with Zero Test Coverage
- `execution/live_executor.py`
- `execution/signal_scanner.py`
- `execution/audit_log.py`
- `execution/trade_signal.py`
- `execution/price_feed.py`
- `backtesting/engine.py` (subprocess methods)
- `backtesting/cost_model.py`
- `backtesting/timerange_utils.py`
- `backtesting/strategy_templates.py`
- `backtesting/setup_data.py`
- `orchestration/deployment_pipeline.py` (run_full_pipeline)
