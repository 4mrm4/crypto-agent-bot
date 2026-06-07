# Audit: Agents Layer

Generated 2026-06-06. Analyzed 9 files, ~60 issues found.

## `agents/base.py`

### Design Issues
- **No timeout on LLM invoke (line 65):** Can hang indefinitely on network issues
- **No retry logic:** Transient failures propagate to caller
- **IndexError on empty messages (line 69):** `result["messages"][-1].content` crashes if empty
- **LangChain-specific internal format for step extraction (line 83):** Breaks across LangChain versions
- **TokenUsageHandler swallows all exceptions (lines 29-30):** `except Exception: pass` hides token tracking bugs
- **`ChatOpenAI` has no API key validation:** Crashes at invoke time, not construction
- **`recursion_limit=50` hardcoded (line 67):** No configuration surface

### Dead Code
- **`get_tool` method (lines 87-89):** Defined but never called

## `agents/strategist.py`

### Bugs
- **`generate_strategy` tool mutates caller's `params` dict via `.pop()` (line 168):** Side effect on shared state
- **`"custom"` strategy type in `valid_types` despite docstrings telling LLM to avoid it (line 173):** Contradictory

### Design Issues
- **`IterationRecord` class duplicated in `iteration_tracker.py` (lines 84-124):** 90-line exact copy
- **Threshold constants duplicated in 3 places (lines 79-81)**
- **`_current_regime` and `_current_sentiment` set but never read (lines 134-135)**
- **`suggest_next_params` only covers 4 of 11 strategy types (lines 234-247)**
- **`get_research_history` creates new `VectorStore()` on every call (line 294)**
- **`get_research_history` accesses `r["metadata"].get("iteration", "?")` — no type guard (line 300)**
- **`uuid.uuid4().hex[:8]` — 4-byte IDs have 1-in-4B collision chance (line 206)**
- **`list(self._generated_strategies.keys())[-1]` relies on insertion-order dicts (line 229):** Fragile

## `agents/backtester.py`

### Bugs
- **`download_data` tool ignores `timeframe` parameter (lines 302-303):** Possible bug
- **`_generated_strategies` not populated by direct backtest path (line 582):** Backtests via `run()` override invisible to other tools

### Design Issues
- **`_evaluate_metrics` is third duplicate of evaluation logic (lines 131-156)**
- **Threshold constants duplicated third time (lines 141-143)**
- **Two parallel backtest implementations:** `run_backtest` tool (line 190) vs `run()` override (line 60)
- **Regex `r'strategy_type[=:]\s*(\w+)'` won't match hyphens (line 70)**
- **`timerange` and `pairs` popped, `timeframe` uses `setdefault` — inconsistent (lines 82-86)**
- **`compare_strategies` selects winner purely by Sharpe, ignoring drawdown (lines 359-365)**
- **`run_hyperopt` builds raw `subprocess.run` to freqtrade CLI, bypassing `BacktestEngine` (lines 418-433)**
- **`result.stdout[-2000:]` truncation cuts critical output (line 435)**
- **`blind_search` accesses private `_generate_default_variants` (line 538)**
- **Hardcoded timerange dates go stale (lines 83, 207, 299, 415)**

## `agents/researcher.py`

### Bugs
- **`self._generated_specs` initialized but never written to (line 44):** `get_specs()` returns empty dict forever
- **`self._specs` written (line 328) but never read — dead attribute (line 45)**

### Design Issues
- **Tavily error handling inconsistency:** `ImportError` falls through to DuckDuckGo, API errors return error string
- **Search relevance scoring code duplicated between Tavily and DuckDuckGo paths (lines 88-112 vs 174-210)**
- **DuckDuckGo uses Instant Answer semantic API, not general web search (line 140):** Extremely limited results
- **`read_paper` uses naive regex `<[^>]+>` for HTML stripping (line 242)**
- **Double truncation in `read_paper`: 6000 chars then 2000 (lines 246, 252)**
- **`generate_custom_strategy_spec` hardcodes `keyword_map` duplicating `strategy_concepts` data (lines 286-296)**
- **`get_asset_fundamentals` uses `urllib.request` instead of `httpx` (line 345):** Inconsistent
- **`get_asset_fundamentals` has no User-Agent header or timeout (line 345)**
- **Tavily search results cached into ChromaDB but never queried (lines 115-129):** Dead cache writes

## `agents/analyst.py`

### Bugs
- **`sentiment_fn` is a hardcoded stub (lines 54-59):** Always returns "NEUTRAL (score 0.0/1.0)" with a TODO comment
- **`fetch_ohlcv_fn` splits input by space, crashes with IndexError if <3 parts (lines 40-44)**
- **`df.tail(20).to_string()` without checking if df is None or empty — AttributeError (lines 44-45)**
- **`f"Current price: ${price:,.2f}"` crashes if price is None or NaN (line 51)**

## `agents/risk_manager.py`

### Bugs
- **`assess_strategy_risk` rejects strategies with ZERO concerns (line 819):** `if risk_score >= 0.5 or not concerns:` — when `concerns` is empty (strategy is fine), `not concerns` is True, verdict = "reject"
- **`bayesian_kelly_position_size` has dead code after early return (lines 341-342)**

### Design Issues
- **`CircuitBreakerState` has no threading locks (lines 59-94):** Thread-unsafe
- **`is_halted()` is a getter with side effect (auto-clears expired halt) — violates command-query separation (lines 82-84)**
- **`datetime.utcnow()` used throughout (deprecated)**
- **4 separate Kelly sizing implementations with massive duplication (lines 133-453, 471-557, 946-987)**
- **Type-coercion for JSON-string-as-first-arg duplicated in 6 functions (lines 173-186, 315-328, 387-400, 485, 949, 970)**
- **`check_position_correlation` fetches 30 days of OHLCV for EVERY open position on EVERY check (line 616):** Inefficient
- **`pre_trade_approval` is 104 lines (lines 861-911):** Overly long
- **`_clamp_limit` inner function redefined on every `circuit_breaker_check` call (lines 706-720)**

## `agents/iteration_tracker.py`

### Design Issues
- **`IterationRecord` class is exact duplicate from `strategist.py` (lines 31-71)**
- **Threshold constants duplicated from `strategist.py` (lines 26-28)**
- **`_current_regime` and `_current_sentiment` never read (lines 84-85)**
- **`VectorStore` imported inside function bodies (lines 189, 226)**

## `agents/curator.py`

### Design Issues
- **`from backtesting.data_split import DATA_SPLIT` — dead import (line 52)**
- **`inject_context` uses different key names for same semantic content: `r.get("document", "")` vs `r["text"]` (lines 69 vs 81)**
- **`inject_context` legacy path uses `meta['goal_id']` with direct key access — KeyError if missing (line 85)**

## `execution/signal_scanner.py`

### Bugs
- **`asyncio.ensure_future()` deprecated (line 354):** Should be `asyncio.create_task()`
- **Regime gating bypassed: empty match falls back to first approved strategy (line 398)**
- **6 of 11 strategy types silently unimplemented (lines 256-319):** Always return None

### Design Issues
- **`import talib` repeated 4 times inside `_evaluate_single_strategy` (lines 273, 284, 299, 310)**
- **`_regime_cache` grows unboundedly — stale pairs never evicted (line 151)**
- **Signal confidence values hardcoded (0.65, 0.60, etc.) with no basis in backtest performance**
- **`_update_approved_strategy_validated_regimes` mutates caller's data (lines 153-158):** Side effect

## Cross-Cutting Agent Issues

- **Timerange defaults are hardcoded dates** in backtester.py (lines 83, 207, 299, 415)
- **`IterationRecord` class and `.evaluate()` method copy-pasted** across strategist.py and iteration_tracker.py
- **Metric evaluation thresholds hardcoded in 3 places:** strategist.py, iteration_tracker.py, backtester.py
- **`import json` inside function bodies in 9+ places** across the codebase
- **`VectorStore` instantiated directly in 6 places** — no dependency injection, coupled to ChromaDB
- **No agent has timeout on LLM `invoke()`** — all inherit from BaseAgent's bare `_agent.invoke()`
- **No agent has retry logic** for transient failures
- **All agents depend on LLM choosing correct tools** — no validation layer
