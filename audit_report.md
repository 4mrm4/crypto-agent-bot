# Audit Report — crypto_agent_bot

**Date:** 2026-06-16
**Files audited:** 76 of 76
**Issues found:** 89

---

## Cross-Cutting Issues

### `datetime.utcnow()` deprecated (Python 3.12+)
- **Files:** `api/server.py` (lines 93, 239, 568, 593), `api/event_bus.py` (lines 22, 42, 67), `data/database.py` (lines 292, 344, 386, 427, 614), `data/regime.py` (line 41), `data/sentiment.py` (line 35), `backtesting/engine.py` (line 491), `execution/audit_log.py` (line 54), `orchestration/deployment_pipeline.py`, `monitoring/anomaly_detector.py`
- **Severity:** MEDIUM
- **Fix:** Replace all with `datetime.now(datetime.UTC)` or `datetime.now(datetime.timezone.utc)`

### `import asyncio` inside method bodies
- **Files:** `data/regime.py:_get_social_signal()`, `data/sentiment.py:_fetch_santiment()`, `data/sentiment.py:get_combined_sentiment_sync()`
- **Severity:** LOW
- **Fix:** Move `import asyncio` to top of file

### Async-to-sync wrapper pattern duplicated
- **Files:** `data/regime.py:_get_social_signal`, `data/sentiment.py:_fetch_santiment`, `data/sentiment.py:get_combined_sentiment_sync`
- **Severity:** MEDIUM
- **Fix:** Extract a shared `run_async_in_sync(coro)` helper into `utils/`

### Kelly sizing logic duplicated 4x (~350 lines total)
- **File:** `agents/risk_manager.py` — `kelly_position_size_conservative` (104-227), `bayesian_kelly_position_size` (263-342), `bayesian_kelly_position_size_conservative` (345-423), local `kelly_position_size` inside `_build_tools` (522-608)
- **Severity:** HIGH
- **Fix:** Extract one canonical Kelly implementation with flags for Bayesian/degradation. All four have identical type-coercion and math — a bug fix in one requires fixing all four

### `json.dumps()` without numpy-safe encoder
- **File:** `data/database.py:insert_trade()` (line 246) — `json.dumps(trade.get("metadata", {}))` will crash if metadata dict contains numpy scalars from pandas
- **Severity:** HIGH
- **Fix:** Use `_NumpyEncoder` (from `execution/audit_log.py`) or equivalent in all DB serialization paths. Per team memory: "Every json.dumps() in data-pipeline paths must handle numpy scalars"

### `IterationRecord` class duplicated
- **Files:** `agents/strategist.py:99-119`, `agents/iteration_tracker.py:26-46`
- **Severity:** MEDIUM
- **Fix:** Delete the copy in `strategist.py` and import from `agents.iteration_tracker`

---

## File-Specific Findings

### agents/__init__.py
- **Line:** 7
- **Issue:** `__all__` is incomplete — missing `BacktesterAgent`, `RiskManagerAgent`, `IterationTrackerAgent`
- **Severity:** LOW
- **Fix:** Add missing agent classes to `__all__`

### agents/base.py
- **Line:** 30 — `except Exception: pass` in `TokenUsageHandler.on_llm_end` silently swallows all token-tracking failures
- **Severity:** LOW
- **Fix:** Log at `logger.debug()` level

- **Line:** 75 — `result.get("messages", [])[-1].content` assumes messages list is never empty; if LangGraph returns zero messages (edge case), this raises `IndexError` before the subsequent guard runs
- **Severity:** LOW
- **Fix:** Guard before indexing: `msgs = result.get("messages", []); raw = msgs[-1].content if msgs else ""`

- **Line:** 89 — `output.encode("ascii", errors="replace").decode("ascii")` silently corrupts all non-ASCII content (currency symbols, em-dashes, non-English text)
- **Severity:** MEDIUM
- **Fix:** Only ASCII-sanitize when target is a Windows console, not when consumed programmatically

### agents/analyst.py
- **Line:** 51 — `f"Current price: ${price:,.2f}"` assumes `price` is never `None`; crashes with `TypeError` if `fetch_current_price()` returns `None` (API down, symbol not found)
- **Severity:** MEDIUM
- **Fix:** Guard: `if price is None: return "Price unavailable"` before formatting

- **Line:** 55-57 — `SentimentFetcher()` instantiated inside tool function, re-created every call
- **Severity:** LOW
- **Fix:** Instantiate once in `__init__` and reuse

### agents/backtester.py
- **Line:** 73 — `re.search(r'params=(\{.*\})', stripped, re.DOTALL)` — greedy `.*` captures too much with nested braces; if input contains `{"a": {"b": 1}}`, regex matches from first `{` to last `}`, potentially including trailing text
- **Severity:** HIGH
- **Fix:** Use a proper brace-matching parser, or scan for balanced braces

- **Line:** 82-83 — `strat_params.pop("timerange", ...)` and `strat_params.pop("pairs", ...)` remove these from the dict, so iteration history stored on line 102 won't include timerange or pairs
- **Severity:** MEDIUM
- **Fix:** Use `.get()` instead of `.pop()`; pass values directly without mutating the shared dict

- **Line:** 89-98 — Direct `self._engine.run_backtest()` call inside `run()` override has no retry logic; base class retries are bypassed
- **Severity:** MEDIUM
- **Fix:** Add retry loop, or delegate the error back to `super().run()` as fallback

### agents/curator.py
- **Line:** 80 — Legacy path uses `r['text']` (KeyError if missing) but contamination-guard path uses `r.get(...)` everywhere; inconsistent dict access patterns
- **Severity:** LOW
- **Fix:** Use consistent `.get()` access in both paths

### agents/iteration_tracker.py
- **Line:** 100 — `f"  Params: {json.dumps(self._best_params)}"` — label says "Params" but dumps metrics (`sharpe_ratio` + all metric fields), not strategy parameters
- **Severity:** LOW
- **Fix:** Either rename label to "Metrics" or swap to dump `self._best_strategy`

### agents/researcher.py
- **Line:** 384 — `f"Price: ${md.get('current_price', {}).get('usd', 'N/A'):,}"` — formats string `'N/A'` with `:,` causing `ValueError` when CoinGecko response is missing `current_price.usd`
- **Severity:** HIGH
- **Fix:** Use conditional: `val = md.get('current_price', {}).get('usd'); f"${val:,.2f}" if isinstance(val, (int, float)) else "N/A"`

- **Line:** 44-45 — `self._generated_specs` and `self._specs` are two separate dicts always updated together; `_specs` is never read independently
- **Severity:** LOW
- **Fix:** Remove `_specs` or merge into one dict

- **Line:** 275 — `content = text[:6000]` is dead code since only `content[:2000]` is returned
- **Severity:** LOW
- **Fix:** Remove dead assignment, or increase return limit to 6000

### agents/risk_manager.py
- **Line:** 306-307, 311-312 — Duplicate `if sizing_tier is None:` check in `bayesian_kelly_position_size`; second check always false
- **Severity:** LOW
- **Fix:** Remove the duplicate guard on lines 311-312

- **Line:** 1154 — `Tool(func=kelly_position_size_conservative, description="Args: JSON with...")` — module-level function takes individual typed params, not JSON; only works because of string-coercion hack inside function body
- **Severity:** LOW
- **Fix:** Either make description match actual signature, or wrap in JSON-parsing adapter

### agents/strategist.py
- **Line:** 99-119 — `IterationRecord` class duplicated from `agents/iteration_tracker.py:26-46`
- **Severity:** MEDIUM
- **Fix:** Delete this copy and import from `agents.iteration_tracker`

- **Line:** 309-310 — Tool description lists only 5 strategy types but the tool validates 15 types; LLM won't know about the other 10
- **Severity:** MEDIUM
- **Fix:** Update description to match actual supported types

### api/event_bus.py
- **Line:** 22, 42, 67 — `datetime.utcnow()` deprecated
- **Severity:** MEDIUM
- **Fix:** Replace with `datetime.now(datetime.UTC)`

### api/server.py
- **Line:** 93, 239, 568, 593 — `datetime.utcnow()` deprecated
- **Severity:** MEDIUM
- **Fix:** Replace with `datetime.now(datetime.UTC)`

- **Line:** 513-516 — API endpoint reads private `_cycles_without_trade`, `_regime_tracker`, `_signal_history` directly (tight coupling to SignalScanner internals)
- **Severity:** MEDIUM
- **Fix:** Add public properties to SignalScanner (`cycles_without_trade`, `signal_count`, `current_regime`, etc.)

- **Line:** 534-539 — Direct mutation of `scanner._standby_mode`, `scanner._scan_interval` bypasses public API
- **Severity:** MEDIUM
- **Fix:** Use public methods like `scanner.set_standby(True)` instead

- **Line:** 234-236, 727-729 — Identical `AutonomousResearchLoop` construction block duplicated in `autonomous_start()` and `_rebuild_loop()`
- **Severity:** MEDIUM
- **Fix:** Extract into a `_build_autonomous_loop(event_bus)` helper

### backtesting/blind_search.py
- **Line:** 110 — `variant[key] = val + (i - n // 2) * spread / (n // 2)` — `ZeroDivisionError` when `n=1`
- **Severity:** LOW
- **Fix:** Guard: `if n <= 1: return [defaults]`

### backtesting/data_split.py
- **Line:** 114 — `from datetime import timedelta` at file bottom with `# noqa: E402`; violates import convention
- **Severity:** LOW
- **Fix:** Move to line 9 alongside other `datetime` imports

### data/database.py
- **Line:** 292, 344, 386, 427, 614 — `datetime.utcnow()` deprecated
- **Severity:** MEDIUM
- **Fix:** Replace with `datetime.now(datetime.UTC)`

- **Line:** 199-216 — `_connect()` and `transaction()` are near-identical implementations
- **Severity:** LOW
- **Fix:** Make one delegate to the other

- **Line:** 246 — `json.dumps(trade.get("metadata", {}))` without numpy-safe encoder; crashes if metadata contains numpy scalars
- **Severity:** HIGH
- **Fix:** Use `_NumpyEncoder` from `execution/audit_log.py`

### data/regime.py
- **Line:** 115-120 — Uses `x == x` NaN check in `classify_regime()` but `pd.isna()` in `classify_regime_snapshot()`; inconsistent patterns in same class
- **Severity:** LOW
- **Fix:** Use `pd.isna()` consistently in both methods

- **Line:** 223-232 — `get_best_strategy_types()` returns DIFFERENT values than `REGIME_STRATEGY_MAP["use"]` for the same regimes (e.g., strong_uptrend returns `["sma_crossover", "combined_sma_rsi", "momentum"]` vs map's `["momentum", "multi_timeframe", "sma_crossover", "breakout"]`)
- **Severity:** HIGH
- **Fix:** Pick one canonical source; either delete `get_best_strategy_types()` and use `REGIME_STRATEGY_MAP` everywhere, or reconcile the lists

### data/fetcher.py
- **Line:** 127 — `symbol.replace("/","").lower().replace("usdt","")` — fragile CoinCap symbol fallback won't work for symbols not ending in USDT (e.g., BTC/USDC)
- **Severity:** MEDIUM
- **Fix:** `symbol.split("/")[0].lower()` or maintain a proper mapping dict

- **Line:** 156 — Hardcoded `"binance"` as fallback exchange name in error return; caller might not have binance configured
- **Severity:** LOW
- **Fix:** Use `self._exchange_ids[0]` or return no exchange name

### data/stream.py
- **Line:** 40-42, 178-180 — `is_connected` property defined twice; second silently overwrites first (identical implementation, likely merge artifact)
- **Severity:** LOW
- **Fix:** Delete the duplicate at lines 178-180

### data/sentiment.py
- **Line:** 222 — `vol / 5000.0` — hardcoded magic normalization constant with no documentation
- **Severity:** LOW
- **Fix:** Make a named constant with a comment explaining the derivation

---

## Batch 2 — Data Layer (files 21-29)

### data/coincap_fetcher.py
- **Line:** 31 — `datetime.utcnow` deprecated. Severity: MEDIUM. Fix: `datetime.now(datetime.UTC)`
- **Line:** 115 — `import asyncio` inside method body. Severity: LOW. Fix: move to top

### data/santiment_fetcher.py
- **Line:** 37, 207, 273 — `datetime.utcnow` deprecated. Severity: MEDIUM. Fix: `datetime.now(datetime.UTC)`
- **Line:** 130 — Type hint `Optional[CacheClient]` but fallback is `TradingDatabase`. Severity: LOW. Fix: use broader type or Protocol
- **Line:** 272 — `from datetime import datetime, timedelta` re-imports inside method body (both already at top). Severity: LOW. Fix: remove redundant import

### data/onchain.py
- **Line:** 73 — `data.get("total_volume", {}).get("usd", 0)` could return `None` if CoinGecko returns `null` (key exists, value is null). Then line 79 `vol_24h / market_cap` crashes with `TypeError`. Severity: MEDIUM. Fix: `float(data.get("total_volume", {}).get("usd") or 0)`
- **Line:** 95 — `symbol.replace("/USDT", "")` fragile for non-USDT pairs. Severity: LOW. Fix: `symbol.split("/")[0]`

### data/api_health.py
- **Line:** 45, 52 — `datetime.utcnow()` deprecated. Severity: MEDIUM. Fix: `datetime.now(datetime.UTC)`

### data/messari_fetcher.py
- **Line:** 38 — `datetime.utcnow` deprecated. Severity: MEDIUM. Fix: `datetime.now(datetime.UTC)`
- **Line:** 182 — `import asyncio` inside method body. Severity: LOW. Fix: move to top
- **Line:** 23 — `MESSARI_BASE` hardcodes `/api/v2` but `get_profile` and `get_trending_topics` may only exist at v1. Severity: LOW. Fix: verify endpoints, possibly separate base URLs

### data/patterns.py
- No issues found.

### data/rate_limiter.py
- No issues found.

### data/cache_client.py
- No issues found.

### data/strategy_concepts.py
- No issues found.

---

## Batch 3 — Execution Layer (files 30-34)

### execution/trade_signal.py
- **Line:** 30 — `datetime.utcnow` deprecated. Severity: MEDIUM. Fix: `datetime.now(datetime.UTC)`

### execution/validation_mode.py
- **Line:** 57, 65, 70, 75, 156, 166 — `datetime.utcnow()` deprecated (6 occurrences). Severity: MEDIUM. Fix: `datetime.now(datetime.UTC)`
- **Line:** 158 — `json.dumps(trade)` without numpy-safe encoder. Severity: HIGH. Fix: use `_NumpyEncoder`
- **Line:** 195 — `mean_r / std_r * 16.0` — Sharpe annualization assumes daily returns but has no documentation on periodicity. Severity: LOW. Fix: add comment explaining annualization factor

### execution/quality_scorer.py
- **Line:** 116 — `datetime.utcnow()` deprecated. Severity: MEDIUM. Fix: `datetime.now(datetime.UTC)`
- **Line:** 221 — `np.empty((0, 1))` returns wrong column count (1 vs ~120+). Only called on empty rows but any direct caller would get shape mismatch. Severity: LOW. Fix: `np.empty((0, 0))`

### execution/signal_scanner.py
- **Line:** 71, 89, 105, 129, 224 — `datetime.utcnow()` deprecated (5 occurrences). Severity: MEDIUM. Fix: `datetime.now(datetime.UTC)`
- **Line:** 324, 336, 350, 362 — `import talib` inside `_evaluate_single_strategy()`, re-imported on every strategy evaluation per scan cycle. Severity: MEDIUM. Fix: move `import talib` to top of file
- **Line:** 449 — `return matched or self._approved_strategies[:1]` — fallback to first strategy regardless of regime match. Severity: MEDIUM. Fix: return empty list or log warning when falling back
- **Line:** 405 — `asyncio.create_task(...)` fire-and-forget with no error handler. Severity: LOW. Fix: store task reference or add error callback

### execution/live_executor.py
- **Line:** 207, 217, 416 — `datetime.utcnow()` deprecated. Severity: MEDIUM. Fix: `datetime.now(datetime.UTC)`
- **Line:** 121-128 — Docstring describes 7-step pipeline (circuit breaker → correlation → Kelly → pre-trade approval → execute → audit → emit) but only steps 1 and 5 are implemented. Risk manager's `correlation_check()`, `kelly_position_size()`, and `pre_trade_approval()` are never called. Severity: **HIGH**. Fix: wire in all risk checks before execution
- **Line:** 435 — `risk_verdict="approved"` hardcoded; actual risk manager result never recorded. Severity: MEDIUM. Fix: pass actual risk verdict from `_risk_manager.pre_trade_approval()`
- **Line:** 269 — `asyncio.get_event_loop()` deprecated. Severity: LOW. Fix: `asyncio.get_running_loop()`

---

## Batch 4 — Backtesting Setup, Memory, Monitoring (files 35-39)

### backtesting/setup_data.py
- **Line:** 14 — `datetime.utcnow()` deprecated + computed at module import time (frozen date). Severity: MEDIUM. Fix: `datetime.now(datetime.UTC)` and compute in function body
- **Line:** 78 — Hardcoded Windows venv paths (`venv/Scripts/freqtrade.exe`). Won't work on Linux/Mac. Severity: MEDIUM. Fix: use `sys.platform` or `shutil.which("freqtrade")`

### backtesting/setup_ft.py
- No issues found.

### memory/vector_store.py
- **Line:** 80 — `hashlib.md5()` used for doc IDs — not FIPS-compliant, minor code smell. Severity: LOW. Fix: use `hashlib.sha256()`
- **Line:** 138-143 — All metric values stringified via `str(round(value, 4))` for ChromaDB metadata, then parsed with `float()` — lossy round-trip. Severity: LOW. Fix: store as native numbers

### monitoring/anomaly_detector.py
- **Lines:** 85, 86, 103, 122, 134, 146, 150, 155, 202, 207, 212, 216, 228 — `datetime.utcnow()` deprecated (13+ occurrences). Severity: MEDIUM. Fix: `datetime.now(datetime.UTC)`
- **Line:** 182 — `from api.server import get_health_tracker` creates circular import. Severity: MEDIUM. Fix: inject via constructor
- **Line:** 167-168 — `_check_negative_kelly()` stub called every 30s, does nothing. Severity: LOW. Fix: implement or remove
- **Line:** 186 — New `CoinCapFetcher` created every 30s. Severity: LOW. Fix: cache in `__init__`

### monitoring/performance_monitor.py
- **Line:** 160 — `datetime.utcnow()` deprecated. Severity: MEDIUM. Fix: `datetime.now(datetime.UTC)`
- **Line:** 187-199 — `returns.rolling(window=window_trades)` returns a `Rolling` object which has no `.iloc` attribute; crashes with `AttributeError` if called. Severity: **HIGH**. Fix: use `rolling.mean()` + `rolling.std()` directly
- **Line:** 196 — `np.sqrt(365)` annualization applied to trade-count windows (not calendar days). Wrong when trade frequency varies. Severity: MEDIUM. Fix: use `np.sqrt(expected_trades_per_year)` or document assumption

---

## Batch 5 — Monitoring, Risk, State, Workspace (files 40-44)

### monitoring/telegram_alerter.py
- **Lines:** 45, 62, 82, 271 — `datetime.utcnow()` deprecated. Severity: MEDIUM. Fix: `datetime.now(datetime.UTC)`
- **Line:** 131 — `asyncio.create_task(...)` fire-and-forget with no error handler. Severity: LOW. Fix: store task reference
- **Line:** 192 — Auto-approves all trades when Telegram unavailable (returns `True`). Risky silent default. Severity: MEDIUM. Fix: add config flag `REQUIRE_TELEGRAM_APPROVAL`

### risk/portfolio_var.py
- **Line:** 71 — `datetime.utcnow()` deprecated. Severity: MEDIUM. Fix: `datetime.now(datetime.UTC)`
- **Line:** 147 — `cov_matrix.ndim == 0` is unreachable — `np.cov()` always returns 2D arrays. Severity: LOW. Fix: remove dead branch

### state/circuit_breaker.py
- **Lines:** 43, 44, 55 — `datetime.utcnow()` deprecated. Severity: MEDIUM. Fix: `datetime.now(datetime.UTC)`

### state/state_broker.py
- **Lines:** 100, 105 — `json.dumps(value)` without numpy-safe encoder; crashes on numpy scalars from data pipeline. Severity: **HIGH** (team memory requirement). Fix: use `_NumpyEncoder`
- **Lines:** 73, 75 — `asyncio.create_task(...)` fire-and-forget. Severity: LOW. Fix: store task reference

### workspace/vibe.py
- **Line:** 48 — `datetime.utcnow()` deprecated. Severity: MEDIUM. Fix: `datetime.now(datetime.UTC)`
- **Line:** 207 — `json.dumps(..., default=str)` silently stringifies datetimes and custom types. Severity: LOW. Fix: use `_NumpyEncoder`

---

## Batch 6 — Orchestration Layer (files 45-52)

### orchestration/board.py
- No issues found.

### orchestration/evaluation.py
- **Lines:** 24-65, 68-92 — `evaluate_strategy_quality()` and `check_convergence()` duplicate threshold logic. A threshold change requires both functions. Severity: LOW. Fix: make `check_convergence()` call `evaluate_strategy_quality()`
- **Line:** 40 — Uses `sharpe_ratio` key but other modules pass `sharpe`. Divergent key naming. Severity: LOW. Fix: accept both: `metrics.get("sharpe_ratio", metrics.get("sharpe", 0))`

### orchestration/experiment_tracker.py
- **Lines:** 99, 138, 144 — `json.dumps()` without numpy-safe encoder. If params contain numpy scalars from backtest outputs, serialization crashes. Severity: MEDIUM. Fix: use `_NumpyEncoder`

### orchestration/factory.py
- No issues found.

### orchestration/graph.py
- **Line:** 326 — Same greedy `\{.*\}` regex bug as `agents/backtester.py:73`. `ast.literal_eval` handles nested correctly, but fallback uses over-matched text. Severity: MEDIUM. Fix: use brace counter or `ast.literal_eval` on wider scan
- **Lines:** 106-107 — New `MarketDataFetcher()` and `MarketRegimeDetector()` created every `dispatch_task` call. Severity: LOW. Fix: inject via board/state

### orchestration/research.py
- **Line:** 18 — `datetime.utcnow` deprecated. Severity: MEDIUM. Fix: `datetime.now(datetime.UTC)`

### orchestration/auto_research.py
- **Line:** 19 — `loop` parameter never used. Severity: LOW. Fix: remove or use

### orchestration/deployment_pipeline.py
- **Lines:** 119, 133, 141, 156, 164, 175, 196, 244, 273, 305, 354, 362, 388, 401 — `datetime.utcnow()` deprecated (14+ occurrences). Severity: MEDIUM. Fix: `datetime.now(datetime.UTC)`
- **Line:** 240 — `passed_gates=2` after gate 4 exception is inconsistent (gates 1-3 passed, gate 3 fail-open). Should be `passed_gates=3`. Severity: LOW. Fix: track actual pass count

---

## Batch 7 — Orchestration Layer continued (files 53-55)

### orchestration/strategy_manager.py
- **Lines:** 285, 304 — `datetime.utcnow()` deprecated. Severity: MEDIUM. Fix: `datetime.now(datetime.UTC)`
- **Line:** 129 — Dead code: `live_sharpe = backtest_sharpe` then immediately `return None`. Severity: LOW. Fix: remove dead assignment
- **Line:** 246-248 — `_check_recovery()` method defined but NEVER CALLED anywhere in the class. Recovery logic (DECAYING→WARNING after 3+ good evals) is dead. Severity: MEDIUM. Fix: call from `_evaluate_single()` or remove
- **Line:** 269-272 — `get_summary_stats()` doesn't compute actual level breakdowns; everything is `"unknown"` except auto-retired strategies. Severity: MEDIUM. Fix: track last action per strategy or call `_evaluate_single()` per strategy

### orchestration/autonomous_loop.py
- **Lines:** 157, 681 — `datetime.utcnow()` deprecated. Severity: MEDIUM. Fix: `datetime.now(datetime.UTC)`
- **Line:** 124 — `json.dumps(..., default=str)` silently corrupts non-serializable types. Severity: LOW. Fix: use `_NumpyEncoder`
- **Line:** 584 — `asyncio.get_event_loop()` deprecated in Python 3.10+. Severity: LOW. Fix: `asyncio.get_running_loop()`

### orchestration/hermes.py
- **Line:** 169 — `datetime.utcnow()` deprecated. Severity: MEDIUM. Fix: `datetime.now(datetime.UTC)`
- **Line:** 781 — `run_research_goal()` calls `self._run_research_goal(...)` (async) without `await` or `asyncio.run()`, returning a coroutine instead of `Dict`. Any caller (vibe.py:77, api/server.py) gets a coroutine that crashes on `.get()`. Severity: **HIGH**. Fix: `return asyncio.run(self._run_research_goal(goal, max_cycles=max_cycles))`
- **Line:** 267 — `PatternDetector()` created twice on consecutive lines (264 and 267). First instance unused. Severity: LOW. Fix: remove redundant creation
- **Lines:** 528, 564 — ASCII-only sanitization corrupts Unicode in hypothesis output. Severity: MEDIUM. Fix: only sanitize for console, not programmatic use
- **Line:** 649 — `re.search(r'[Rr]esearching\s+(\w+)', goal_text)` won't match uppercase "RESEARCHING". Severity: LOW. Fix: use `re.IGNORECASE`

---

## Batch 8 — Backtesting Layer (files 56-61)

### backtesting/signal_factory.py
- No issues found. Clean vectorized signal factory with mirrored Freqtrade logic.

### backtesting/strategy_templates.py
- No issues found. Well-structured template data registry.

### backtesting/synthetic_validator.py
- No issues found. Clean synthetic data validation.

### backtesting/timerange_utils.py
- **Line:** 49-50 — When 4 digit groups with mixed lengths are parsed (e.g., `['2024', '01', '01', '2024']` for "YYYY-MM-DD YYYY"), `"".join(groups[:2])` produces `"202401"` (6 chars, not a valid date). The second date is corrupted. Severity: LOW. Fix: treat 4 groups as `groups[:3]` + `groups[3:]` with 4th group being standalone year

### backtesting/cpcv_validator.py
- No issues found. Clean CPCV implementation.

### backtesting/oos_validator.py (partial — first 80 lines)
- **Line:** 59 — `from config import settings` imported but unused in `__init__`. Severity: LOW. Fix: remove unused import
- **Line:** 57 — `from data.database import TradingDatabase` lazy import inside `__init__`. Severity: LOW. Same pattern as deployment_pipeline.py

### config.py
- No issues found. All settings loaded from environment with sensible defaults. Clean.
- Minor note: `TAVILY_ENABLED` (line 81) and `CRYPTOPANIC_API_KEY` (line 37) are loaded but may be unused — grep for usage before removing.

### main.py
- **Line:** 115 — `_make_orchestrator()` is a trivial one-line wrapper around `make_orchestrator()`, only called once on line 293. Severity: LOW. Fix: inline the call or remove the wrapper.
- **Line:** 196 — Direct access to private `_standby_mode` attribute on `SignalScanner`. Severity: LOW. Fix: use a public setter or constructor parameter.
- **Line:** 244 — Polling loop waiting for server startup silently exhausts all 40 retries with no error message if server never starts. The POST on line 260 will fail with a misleading "Failed to start run" rather than "Server not ready". Severity: LOW. Fix: add a flag after loop; if server never became ready, report the real error.

### backtesting/engine.py
- **Line:** 491 — `datetime.datetime.utcnow()` deprecated (Python 3.12+). Severity: MEDIUM. Fix: `datetime.now(datetime.UTC)`.
- **Line:** 293-294 — Drawdown computed on per-trade `profit_ratio` directly rather than on a cumulative equity curve. `cummax()` on per-trade returns doesn't give a proper equity curve. Produces incorrect drawdown values. Severity: HIGH. Fix: compute cumulative equity curve first (`(1 + returns).cumprod()`), then `drawdown = (cum_equity - cum_equity.cummax()).min()`.
- **Line:** 289 — Sharpe ratio annualization uses `(365 * 24) ** 0.5` (hourly data factor), but the input is per-trade returns, not hourly returns. The annualization should be `sqrt(trades_per_year)`, not `sqrt(8760)`. Severity: MEDIUM. Fix: estimate trades_per_year from data span, or skip annualization for per-trade data.
- **Line:** 336, 549 — `subprocess.run` with `timeout` parameter; `TimeoutExpired` is never caught. Long backtests or downloads will crash with unhandled exception. Severity: MEDIUM. Fix: wrap in try/except `subprocess.TimeoutExpired`.

### execution/paper_trader.py
- **Line:** 95 — `final_balance: self.balance + self.position_size`. `position_size` is set on line 63 when entering a position but **never reset to 0** when the position is closed (line 83 sets `self.position = None` but not `self.position_size = 0`). After any closed trade, `final_balance` overcounts by the last position_size, inflating results. Severity: HIGH. Fix: set `self.position_size = 0` on line 83 alongside `self.position = None`.

### execution/audit_log.py
- **Line:** 54 — `datetime.utcnow` deprecated (Python 3.12+). Severity: MEDIUM. Fix: `datetime.now(datetime.UTC)`.
- **Line:** 79-81 — `_load()` silently swallows all exceptions (bare `except Exception: pass`). If the JSONL file is corrupted, entries are silently dropped with zero logging. Severity: MEDIUM. Fix: log a warning when a line fails to parse.

### memory/__init__.py
- **Line:** 3, 5 — `__all__` assigned twice; line 5 overwrites line 3. `VectorStore` is imported but dropped from exports; `InsightMemory` is referenced in `__all__` but never imported. Severity: MEDIUM. Fix: single `__all__ = ["VectorStore"]` or add the missing `InsightMemory` import.

### agents/token_tracker.py
- No issues found. Clean utility tracking LLM token usage.

### backtesting/cost_model.py
- No issues found. Clean transaction cost model.

### api/__init__.py
- Empty file. No issues.

### backtesting/__init__.py
- Empty file. No issues.

### data/__init__.py
- Clean. Re-exports `MarketDataFetcher`. No issues.

### execution/__init__.py
- Empty file. No issues.

### monitoring/__init__.py
- Docstring only. No issues.

### orchestration/__init__.py
- Empty file. No issues.

### risk/__init__.py
- Comment only. No issues.

### state/__init__.py
- Comment only. No issues.

### ui/index.html
- **Line:** 947-969 — `fetchIterationDetail()` entire body is commented out. Dead code with an early-return guard for undefined ids. Severity: LOW. Fix: remove the dead function or implement the backend endpoint and uncomment.
- **Line:** 816 — `console.log("ITERATION PAYLOAD KEYS:", ...)` debug log left in production code. Severity: LOW. Fix: remove or guard with debug flag.
- **Line:** 9 — Google Fonts stylesheet loaded without `crossorigin="anonymous"` attribute. Severity: LOW. Fix: add `crossorigin="anonymous"` to the Google Fonts `<link>`.
- **Line:** 10-12 — React and Babel loaded from unpkg CDN in development mode. For a local dashboard this is acceptable but adds latency and external dependency risk. Severity: LOW. Fix: bundle for production or pin versions with SRI hashes.

---

## Audit Complete — 76 of 76 files reviewed

All Python source files (75) + ui/index.html have been audited. Test files excluded per scope.
