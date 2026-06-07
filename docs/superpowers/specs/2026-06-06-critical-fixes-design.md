# Critical Fixes Design — 2026-06-06

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix 9 critical + 20 high-priority issues identified in 2026-06-06 codebase audit, grouped into 5 fix batches.

**Architecture:** Each batch targets independent subsystems (data, backtesting, agents, architecture, monitoring). Batches ordered by blast radius — security/data first, then logic bugs, then design debt.

**Tech Stack:** Python 3.12, asyncio, SQLite, CCXT, TA-Lib, LangGraph, ChromaDB

---

## Batch 1: Data & Security (5 tasks)

### C1: Fix OpenAI Key Used as Exchange API Key

**File:** `execution/live_executor.py:106`

**Root cause:** `"apiKey": settings.OPENAI_API_KEY` passes the OpenAI API key to the CCXT exchange constructor. The exchange gets the wrong key entirely — won't authenticate.

**Fix:** The exchange API key settings don't exist yet. Add them:
1. Add `EXCHANGE_API_KEY` and `EXCHANGE_SECRET` to `config.py` settings class
2. In `live_executor.py:106`, use `settings.EXCHANGE_API_KEY` instead of `settings.OPENAI_API_KEY`, and add `"secret": settings.EXCHANGE_SECRET`
3. Add both to `.env.example`

**Verification:** Start bot with `--live`, confirm exchange connection succeeds without auth errors.

### C5/C6: asyncio.run() Crashes in Sentiment & Regime

**Files:** `data/sentiment.py:145`, `data/regime.py:74`

**Root cause:** `_fetch_santiment()` and `_get_social_signal()` call `asyncio.run()` from synchronous methods. When called from a running event loop (autonomous loop, API handlers), Python raises `RuntimeError: asyncio.run() cannot be called from a running event loop`.

**Fix pattern for both:**

```python
# Option: Create async version, sync wrapper checks for running loop
async def _fetch_santiment_async(self, ...) -> Dict:
    async with httpx.AsyncClient(timeout=30) as client:
        # ... await-based logic ...

def _fetch_santiment(self, ...) -> Dict:
    try:
        loop = asyncio.get_running_loop()
        # Already in async context — run directly
        return asyncio.create_task(self._fetch_santiment_async(...))
    except RuntimeError:
        # No running loop — use asyncio.run()
        return asyncio.run(self._fetch_santiment_async(...))
```

Actually, simpler approach: make the callers (`get_combined_sentiment`, `classify_regime_snapshot`) async, and make the SantimentFetcher calls properly async with `await`. The synchronous wrappers just become `await`-based.

**Verification:** Run sentiment fetch from within the autonomous loop. Confirm no RuntimeError. Run `get_combined_sentiment()` and `classify_regime_snapshot()` in both sync and async contexts.

### C7: SQL Injection in table_count

**File:** `data/database.py:632-637`

**Root cause:** `f"SELECT COUNT(*) FROM {table_name}"` uses direct string interpolation. `clear_all` has same pattern.

**Fix:** Validate `table_name` against a whitelist of known table names. If not in whitelist, raise ValueError.

```python
VALID_TABLES = {"trades", "experiments", "pipeline_results", "oos_results", "validation_trades", "api_cache", "migrations"}

def table_count(self, table_name: str) -> int:
    if table_name not in self.VALID_TABLES:
        raise ValueError(f"Invalid table name: {table_name}")
    with self._connect() as conn:
        return conn.execute(f"SELECT COUNT(*) FROM [{table_name}]").fetchone()[0]
```

Same pattern for `clear_all`.

**Verification:** Test with valid table name returns correct count. Test with invalid table name raises ValueError. Test with SQL injection string raises ValueError (doesn't execute).

### C8: CDLENGULFING Double-Count Bug

**File:** `data/patterns.py:11,14`

**Root cause:** `"CDLENGULFING"` appears in both `BULLISH_PATTERNS` and `BEARISH_PATTERNS` lists. `pattern_to_signal` counts it on both sides, neutralizing its signal.

**Fix:** Remove `"CDLENGULFING"` from one list depending on interpretation context. An engulfing candle is bearish when it appears at resistance (red engulfing green) and bullish at support (green engulfing red) — but the TA-Lib function returns a signed value. `CDLENGULFING` returns positive for bullish, negative for bearish. The `pattern_to_signal` function already handles the sign via `int(signals[-1])`. Since the sign is intrinsic to the pattern output, it belongs only in one named list. Keep it in `BULLISH_PATTERNS` (the sign determines bearish/bullish at runtime via the TA-Lib return value).

**Verification:** Run pattern detection. CDLENGULFING signals correctly contribute to one side. Signals are no longer always neutral for engulfing.

### C9: .env Tracked in Git

**File:** `.gitignore`, `.env`

**Root cause:** `.env` is tracked in git, exposing API keys for CoinGecko, HuggingFace, Messari, Santiment, CoinCap.

**Fix:** 
1. Add `.env` to `.gitignore`
2. Remove `.env` from git tracking: `git rm --cached .env`
3. Create `.env.example` with placeholder values
4. Rotate all exposed API keys

**Verification:** `git status` shows `.env` as untracked. `.env.example` exists with placeholder values. Keys rotated at respective providers.

---

## Batch 2: Backtesting Integrity (3 tasks)

### C2/C3: Synthetic Validator + Permutation Test No-Op

**Files:** `backtesting/synthetic_validator.py:155-165, 249-257`, `backtesting/engine.py`

**Root cause:** `generate_random_walk(seed=i)` creates synthetic data but the backtest engine call uses a hardcoded timerange `"20170101-20231231"`. The engine has no mechanism to accept pre-loaded DataFrames — it always runs Freqtrade as a subprocess.

**Fix approach — simplest path:** Add an optional `dataframe_override` parameter to `BacktestEngine.__init__` or `run_backtest()`. When provided, skip the Freqtrade subprocess entirely and use FastMetrics + signal_factory to evaluate the strategy on the synthetic DataFrame.

```python
class BacktestEngine:
    def __init__(self, ..., dataframe_override: Optional[Dict[str, pd.DataFrame]] = None):
        self._dataframe_override = dataframe_override or {}
    
    def run_backtest(self, strategy_name, strategy_code, timerange, pairs, timeframe, ...):
        if self._dataframe_override:
            return self._run_fastmetrics_backtest(strategy_name, strategy_code)
        return self._run_freqtrade_backtest(...)
    
    def _run_fastmetrics_backtest(self, strategy_name, strategy_code):
        # Use SignalFactory + FastMetrics directly on self._dataframe_override
        # Avoid the Freqtrade subprocess entirely
```

This is the minimal change to make the validator function correctly. The alternative (refactoring the engine to accept DataFrames throughout) is a larger change that belongs in a separate task.

**Verification:** `SyntheticValidator.validate_strategy()` passes synthetic data and gets results back. `SyntheticValidator.run_permutation_test()` also works with synthetic data. The validator actually validates against random data as designed.

### H6: Data Directory Hardcoded to Binance

**File:** `backtesting/setup_data.py:32`

**Root cause:** `data_path = ft_path / "data" / "binance"` hardcodes "binance" but `.env` uses Kraken.

**Fix:** `data_path = ft_path / "data" / settings.EXCHANGE_ID`

**Verification:** Data check finds Kraken data directory correctly. No unnecessary re-downloads on startup.

### H8: No Timeout on LangGraph Invoke

**File:** `orchestration/hermes.py:325` (or wherever `self._graph.invoke(initial_state)` is called)

**Root cause:** No timeout parameter on LangGraph `invoke()`. Can hang indefinitely.

**Fix:** Use `asyncio.wait_for()` or `functools.partial` to add a timeout:

```python
try:
    result = await asyncio.wait_for(
        self._graph.ainvoke(initial_state),
        timeout=settings.LANGGRAPH_TIMEOUT or 300
    )
except asyncio.TimeoutError:
    logger.error("LangGraph invoke timed out after %ds", timeout)
    # Return error state
    initial_state["error"] = "Research cycle timed out"
    return initial_state
```

**Verification:** Add a test that mocks a slow agent and confirms timeout fires. Confirm normal execution still works.

---

## Batch 3: Agent Logic Bugs (4 tasks)

### H1: assess_strategy_risk Rejects Good Strategies

**File:** `agents/risk_manager.py:819`

**Root cause:** `if risk_score >= 0.5 or not concerns:` — when `concerns` list is empty (meaning the strategy has no identified risks), `not concerns` is True, so the condition evaluates to True for ANY risk_score. Good strategies with zero concerns get rejected.

**Fix:** Change `or` to `and`:
```python
if risk_score >= 0.5 and concerns:
```

Now rejection only happens when BOTH there are concerns AND risk score exceeds threshold.

**Verification:** Test with empty concerns list + any risk_score → strategy passes. Test with non-empty concerns + risk_score < 0.5 → passes. Test with non-empty concerns + risk_score >= 0.5 → rejected.

### H2: sentiment_fn Hardcoded Stub

**File:** `agents/analyst.py:54-59`

**Root cause:** `sentiment_fn` is a closure that returns `"NEUTRAL (score 0.0/1.0)"` with a TODO comment.

**Fix:** Wire the existing `SentimentFetcher` (or `get_combined_sentiment`) into the closure. The analyst already has access to the symbol — pass it to the sentiment fetcher and return real data.

```python
def sentiment_fn(symbol: str) -> str:
    try:
        from data.sentiment import get_combined_sentiment
        result = get_combined_sentiment(symbol)
        score = result.get("score", 0.5)
        label = result.get("label", "NEUTRAL")
        return f"{label} (score {score:.2f}/1.0)"
    except Exception as e:
        return f"UNAVAILABLE (error: {e})"
```

**Verification:** Call `analyst.run("Analyze BTC/USDT sentiment")`. Response includes real sentiment data, not "NEUTRAL (score 0.0/1.0)".

### H3: _generated_specs Never Written

**File:** `agents/researcher.py:44`

**Root cause:** `self._generated_specs` is initialized as `{}` in `__init__` but no method ever writes to it. `get_specs()` always returns empty dict.

**Fix:** Trace where strategy specs are generated (likely in `generate_custom_strategy_spec`) and add the write to `self._generated_specs`:

```python
def generate_custom_strategy_spec(self, ...) -> str:
    spec = self._generate_spec(...)
    if spec.get("name"):
        self._generated_specs[spec["name"]] = spec
    return spec
```

**Verification:** Call `researcher.run("Generate a momentum strategy")`, then call `researcher.get_specs()`. Returns the generated spec instead of empty dict.

### H4/H5: PaperTrader State Lost Every Signal

**File:** `execution/live_executor.py:217-224`

**Root cause:** `_execute_paper` creates a new `PaperTrader(max_candles=5)` for each signal. Paper trading state resets on every signal, and `max_candles=5` means ~5 minutes of data.

**Fix:** Store `PaperTrader` instance as `self._paper_trader` and reuse it. Initialize once. Remove `max_candles=5` limit or use a reasonable default.

```python
def __init__(self, ...):
    ...
    self._paper_trader: Optional[PaperTrader] = None

async def _initialize_paper_trader(self):
    if self._paper_trader is None:
        self._paper_trader = PaperTrader(...)

async def _execute_paper(self, signal: TradeSignal) -> Dict[str, Any]:
    await self._initialize_paper_trader()
    # reuse self._paper_trader for all signals
```

Also fix the fill_price bug at line 238: `fill_price=result.get("final_balance", 0)` should get the actual fill price from the trade execution instead of the balance.

**Verification:** Multiple signals in sequence show cumulative paper trading state. Fill prices are actual trade prices, not balances.

---

## Batch 4: Architecture Debt (5 tasks)

### H7: Board Reassigned Mid-Method

**File:** `orchestration/hermes.py:156`

**Root cause:** `self.board = StateGraph(MarketIntelligence)` creates a new board mid-method, discarding previously queued tasks.

**Fix:** Remove the mid-method assignment. Initialize board once in `__init__` or at the start of the method, not in the middle after queuing work.

### LLM Timeout/Retry on BaseAgent

**File:** `agents/base.py:65`

**Fix:** Add timeout parameter to `_agent.invoke()` and retry logic for transient failures:

```python
async def _invoke_with_retry(self, messages, max_retries=3, timeout=120):
    for attempt in range(max_retries):
        try:
            return await asyncio.wait_for(
                self._agent.ainvoke(messages),
                timeout=timeout
            )
        except (asyncio.TimeoutError, APIError) as e:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)  # exponential backoff
```

### VectorStore Dependency Injection

**Files:** Multiple (strategist.py, backtester.py, iteration_tracker.py, curator.py)

**Fix:** Accept an optional `vector_store` parameter in constructor. Default to creating a new one (backward compatible). Tests can inject mocks.

### CircuitBreakerState Thread Safety

**File:** `agents/risk_manager.py:59-94`

**Fix:** Add `threading.Lock` to all state-mutating methods.

### EventBus Stale Loop + Unsubscribe

**File:** `core/event_bus.py`

**Fix:** Remove import-time event loop capture. Store loop reference per-subscription or resolve at publish time. Add `unsubscribe()` method.

---

## Batch 5: Monitoring & Deployment (3 tasks)

### C4: Telegram Approvals Non-Functional

**File:** `monitoring/telegram_alerter.py:117-130`

**Fix:** Register callback query handler for Approve/Reject buttons. Start polling when needed. Wire `resolve_approval` to the handler.

### Deployment Pipeline Gate Bugs (H9-H11)

**File:** `orchestration/deployment_pipeline.py`

**Fixes:**
- H9: Fix permutation test to pass synthetic data to engine
- H10: Increase CPCV limit from 1000 to meaningful window
- H11: Remove hardcoded avg_win_pct/avg_loss_pct fallback — compute from backtest results

### H19: Rolling Sharpe Window

**File:** `monitoring/performance_monitor.py:183`

**Fix:** Replace `returns.rolling(window=window_days)` with time-based window using `pd.Grouper(freq='D')` or equivalent.

---

## Implementation Order

```
Batch 1 (Data & Security) → Batch 2 (Backtesting Integrity) → Batch 3 (Agent Logic) → Batch 4 (Architecture) → Batch 5 (Monitoring)
```

Each batch produces working, testable changes independently.

**Test strategy:** Each fix gets a regression test before implementation (TDD). Run full test suite after each batch.
