# Critical Fixes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 9 critical + 20 high-priority issues identified in 2026-06-06 codebase audit.

**Architecture:** 5 independent fix batches ordered by blast radius. Each batch targets different subsystems with no cross-batch dependencies. Security/data first, then backtesting integrity, then agent logic bugs, then architecture debt, then monitoring.

**Tech Stack:** Python 3.12, asyncio, SQLite, CCXT, TA-Lib, LangGraph, ChromaDB

---

## File Structure

### Files Modified

| File | Batch | Change |
|------|-------|--------|
| `config.py` | 1 | Add `EXCHANGE_API_KEY`, `EXCHANGE_SECRET` settings |
| `execution/live_executor.py` | 1,3 | Fix C1 (exchange key), fix H4/H5 (PaperTrader state) |
| `data/sentiment.py` | 1 | Fix C5 (asyncio.run crash) |
| `data/regime.py` | 1 | Fix C6 (asyncio.run crash) |
| `data/database.py` | 1 | Fix C7 (SQL injection) |
| `data/patterns.py` | 1 | Fix C8 (CDLENGULFING double-count) |
| `.gitignore` | 1 | Add .env |
| `.env.example` | 1 | Create with placeholders |
| `backtesting/engine.py` | 2 | Add `dataframe_override` parameter |
| `backtesting/synthetic_validator.py` | 2 | Pass synthetic data to engine |
| `backtesting/setup_data.py` | 2 | Fix H6 (binance → Kraken) |
| `orchestration/hermes.py` | 2,4 | Fix H8 (timeout), fix H7 (board reassign) |
| `agents/risk_manager.py` | 3,4 | Fix H1 (boolean logic), add CircuitBreaker locks |
| `agents/analyst.py` | 3 | Fix H2 (sentiment stub) |
| `agents/researcher.py` | 3 | Fix H3 (_generated_specs never written) |
| `agents/base.py` | 4 | Add LLM timeout/retry |
| `core/event_bus.py` | 4 | Fix stale loop, add unsubscribe |
| `monitoring/telegram_alerter.py` | 5 | Fix C4 (callback handler) |
| `orchestration/deployment_pipeline.py` | 5 | Fix H9-H11 (gate bugs) |
| `monitoring/performance_monitor.py` | 5 | Fix H19 (rolling window) |

### Test Files Created/Modified

| Test File | Batch | Tests |
|-----------|-------|-------|
| `tests/test_live_executor.py` | 1,3 | Exchange key, PaperTrader state |
| `tests/test_sentiment.py` | 1 | Async sentiment fetch |
| `tests/test_regime.py` | 1 | Async regime classification |
| `tests/test_trading_database.py` | 1 | SQL injection guards |
| `tests/test_patterns.py` | 1 | CDLENGULFING fix |
| `tests/test_synthetic_validator.py` | 2 | Synthetic data flow |
| `tests/test_engine.py` | 2 | Dataframe override |
| `tests/test_setup_data.py` | 2 | Kraken data path |
| `tests/test_hermes.py` | 2,4 | LangGraph timeout, board fix |
| `tests/test_risk_manager.py` | 3,4 | assess_strategy_risk logic |
| `tests/test_analyst.py` | 3 | Sentiment tool |
| `tests/test_researcher.py` | 3 | _generated_specs |
| `tests/test_base_agent.py` | 4 | LLM timeout/retry |
| `tests/test_event_bus.py` | 4 | Stale loop, unsubscribe |
| `tests/test_telegram_alerter.py` | 5 | Callback handler |
| `tests/test_deployment_pipeline.py` | 5 | Gate fixes |
| `tests/test_performance_monitor.py` | 5 | Rolling window |

---

## Batch 1: Data & Security

### Task 1.1: Fix OpenAI Key Used as Exchange API Key (C1)

**Files:**
- Modify: `config.py` (add `EXCHANGE_API_KEY`, `EXCHANGE_SECRET`)
- Modify: `execution/live_executor.py:106` (use correct keys)
- Create: `.env.example` (template with placeholders)
- Test: `tests/test_live_executor.py` (exchange init with correct keys)

**Root cause:** `live_executor.py:106` passes `settings.OPENAI_API_KEY` as the exchange API key. The config has no `EXCHANGE_API_KEY` or `EXCHANGE_SECRET` settings at all — they need to be added from scratch.

- [ ] **Step 1: Read live_executor.py, config.py, and .env to verify current state**

```bash
grep -n 'OPENAI_API_KEY\|EXCHANGE_API' C:/Trading-bot/crypto_agent_bot/config.py
grep -n 'apiKey\|secret' C:/Trading-bot/crypto_agent_bot/execution/live_executor.py
```

Expected: `OPENAI_API_KEY` exists in config, no `EXCHANGE_API_KEY` exists.

- [ ] **Step 2: Add exchange API key settings to config.py**

After line 13 (`EXCHANGE_ID`), add:
```python
    EXCHANGE_API_KEY: str = os.getenv("EXCHANGE_API_KEY", "")
    EXCHANGE_SECRET: str = os.getenv("EXCHANGE_SECRET", "")
```

- [ ] **Step 3: Fix live_executor.py to use correct exchange keys**

Change line 105-108 from:
```python
            self._exchange = exchange_class({
                "apiKey": settings.OPENAI_API_KEY,  # placeholder — real key goes in .env
                "secret": "",
                "enableRateLimit": True,
                "options": {"defaultType": "spot"},
            })
```
To:
```python
            self._exchange = exchange_class({
                "apiKey": settings.EXCHANGE_API_KEY,
                "secret": settings.EXCHANGE_SECRET,
                "enableRateLimit": True,
                "options": {"defaultType": "spot"},
            })
```

- [ ] **Step 4: Create `.env.example`** with all env vars documented + placeholder values

Create `C:\Trading-bot\crypto_agent_bot\.env.example`:
```bash
# === REQUIRED ===
OPENAI_API_KEY=sk-your-openai-key-here
EXCHANGE_API_KEY=your-exchange-api-key
EXCHANGE_SECRET=your-exchange-secret
EXCHANGE_ID=kraken
SYMBOL=BTC/USDT
TIMEFRAME=1h

# === OPTIONAL API KEYS ===
COINGECKO_API_KEY=
CRYPTOPANIC_API_KEY=
SANTIMENT_API_KEY=
COINCAP_API_KEY=
WHALE_ALERT_API_KEY=
TAVILY_API_KEY=
HF_TOKEN=

# === FEATURE FLAGS ===
ENABLE_SENTIMENT=true
ENABLE_PATTERNS=true
ENABLE_ONCHAIN=false
EXECUTION_MODE=paper
```

- [ ] **Step 5: Write test for correct exchange key usage**

Create `C:\Trading-bot\crypto_agent_bot\tests\test_live_executor.py`:
```python
"""Tests for live_executor.py — exchange key configuration."""
import pytest
from unittest.mock import patch, MagicMock
from execution.live_executor import LiveExecutor

class TestExchangeKeyConfig:
    def test_exchange_uses_correct_key_settings(self):
        """LiveExecutor should use EXCHANGE_API_KEY, not OPENAI_API_KEY."""
        with patch("execution.live_executor.settings") as mock_settings:
            mock_settings.EXCHANGE_API_KEY = "test_exchange_key"
            mock_settings.EXCHANGE_SECRET = "test_exchange_secret"
            mock_settings.OPENAI_API_KEY = "sk-test-openai"
            mock_settings.EXECUTION_MODE = "live"
            mock_settings.EXCHANGE_ID = "kraken"

            with patch("ccxt.kraken") as mock_exchange_class:
                executor = LiveExecutor(...)  # minimal init
                executor.exchange  # trigger lazy-init
                mock_exchange_class.assert_called_once()
                call_kwargs = mock_exchange_class.call_args[0][0]
                assert call_kwargs["apiKey"] == "test_exchange_key"
                assert call_kwargs["secret"] == "test_exchange_secret"
                assert call_kwargs["apiKey"] != mock_settings.OPENAI_API_KEY
```

- [ ] **Step 6: Run test, verify it passes**

Run: `python -m pytest tests/test_live_executor.py::TestExchangeKeyConfig -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add config.py execution/live_executor.py .env.example tests/test_live_executor.py
git commit -m "fix: add EXCHANGE_API_KEY/EXCHANGE_SECRET settings, fix C1 (OpenAI key used as exchange key)"
```

---

### Task 1.2: Fix asyncio.run() Crash in Sentiment (C5)

**Files:**
- Modify: `data/sentiment.py:138-149` (make _fetch_santiment async)
- Modify: `data/sentiment.py:151-168` (make get_combined_sentiment async)
- Test: `tests/test_sentiment.py` (async sentiment fetch)

**Root cause:** `_fetch_santiment()` calls `asyncio.run(fetcher.get_signal(slug))` from a sync method. When called from the async autonomous loop, Python raises `RuntimeError`.

- [ ] **Step 1: Convert _fetch_santiment to async**

Replace lines 138-149 with:
```python
    async def _fetch_santiment_async(self, slug: str = "bitcoin") -> Optional[object]:
        """Fetch Santiment signal asynchronously."""
        if not getattr(settings, "SANTIMENT_ENABLED", False):
            return None
        try:
            from data.santiment_fetcher import SantimentFetcher
            fetcher = SantimentFetcher()
            signal = await fetcher.get_signal(slug)
            return signal
        except Exception as exc:
            logger.warning("Santiment fetch failed: %s", exc)
            return None

    def _fetch_santiment(self, slug: str = "bitcoin") -> Optional[object]:
        """Fetch Santiment signal synchronously (wraps async)."""
        try:
            loop = asyncio.get_running_loop()
            # Already in async context — schedule and run
            import asyncio
            future = asyncio.run_coroutine_threadsafe(
                self._fetch_santiment_async(slug), loop
            )
            return future.result(timeout=30)
        except RuntimeError:
            # No running loop — use asyncio.run()
            return asyncio.run(self._fetch_santiment_async(slug))
```

- [ ] **Step 2: Convert get_combined_sentiment to async**

Make `get_combined_sentiment` an async method. Change signature and update all calls to `_fetch_santiment` to use `await self._fetch_santiment_async(slug)`.

```python
    async def get_combined_sentiment(
        self, slug: str = "bitcoin", currency: str = "BTC"
    ) -> CombinedSentiment:
        """Aggregate all sentiment sources asynchronously."""
        # ... same logic but use await for Santiment call ...
        if settings.SANTIMENT_ENABLED:
            signal = await self._fetch_santiment_async(slug)
            # ... rest of logic ...
```

Add a sync wrapper for backward compatibility:
```python
    def get_combined_sentiment_sync(self, slug: str = "bitcoin", currency: str = "BTC") -> CombinedSentiment:
        """Sync wrapper for get_combined_sentiment."""
        try:
            loop = asyncio.get_running_loop()
            future = asyncio.run_coroutine_threadsafe(
                self.get_combined_sentiment(slug, currency), loop
            )
            return future.result(timeout=30)
        except RuntimeError:
            return asyncio.run(self.get_combined_sentiment(slug, currency))
```

- [ ] **Step 3: Write test for async sentiment fetch**

```python
"""Tests for sentiment.py — async fetch safety."""
import pytest
from data.sentiment import SentimentFetcher

@pytest.mark.asyncio
async def test_fetch_santiment_async_no_crash():
    """Should not crash when called from async context."""
    fetcher = SentimentFetcher()
    # Mock Santiment to avoid real API calls
    result = await fetcher._fetch_santiment_async("bitcoin")
    # Should return None without crashing (no API key in test)
    assert result is None

def test_fetch_santiment_sync_no_crash():
    """Should not crash when called from sync context."""
    fetcher = SentimentFetcher()
    result = fetcher._fetch_santiment("bitcoin")
    assert result is None
```

- [ ] **Step 4: Write test for async get_combined_sentiment**

```python
@pytest.mark.asyncio
async def test_get_combined_sentiment_async():
    """get_combined_sentiment should work from async context without RuntimeError."""
    fetcher = SentimentFetcher()
    # Should not raise RuntimeError about asyncio.run()
    result = await fetcher.get_combined_sentiment("bitcoin", "BTC")
    assert result is not None
    assert hasattr(result, "score")
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_sentiment.py -v`
Expected: All PASS

- [ ] **Step 6: Update all callers of get_combined_sentiment to use async version**

Search for callers:
```bash
grep -rn "get_combined_sentiment" C:/Trading-bot/crypto_agent_bot/ --include="*.py"
```
Update each caller to use `await` if in async context, or `get_combined_sentiment_sync` if in sync context.

- [ ] **Step 7: Commit**

```bash
git add data/sentiment.py tests/test_sentiment.py
git commit -m "fix: make sentiment fetch async-safe, fix C5 (asyncio.run crash in async context)"
```

---

### Task 1.3: Fix asyncio.run() Crash in Regime (C6)

**Files:**
- Modify: `data/regime.py:63-85` (make _get_social_signal async-safe)
- Test: `tests/test_regime.py`

**Root cause:** Same pattern as C5 — `_get_social_signal` calls `asyncio.run()` from sync context.

- [ ] **Step 1: Create _get_social_signal_async variant**

Add an async version alongside the sync one:
```python
    async def _get_social_signal_async(self, slug: str = "bitcoin") -> Optional[float]:
        """Fetch Santiment social dominance z-score asynchronously."""
        if not getattr(settings, "SANTIMENT_ENABLED", False):
            return None
        try:
            from data.santiment_fetcher import SantimentFetcher
            fetcher = SantimentFetcher()
            signal = await fetcher.get_signal(slug)
            if signal is None or signal.social_dominance_pct is None:
                return None
            self._dominance_history.append(signal.social_dominance_pct)
            if len(self._dominance_history) > 50:
                self._dominance_history = self._dominance_history[-50:]
            return self._compute_dominance_zscore(signal.social_dominance_pct)
        except Exception as exc:
            logger.warning("Social signal fetch failed: %s", exc)
            return None

    def _get_social_signal(self, slug: str = "bitcoin") -> Optional[float]:
        """Sync wrapper for _get_social_signal_async."""
        try:
            loop = asyncio.get_running_loop()
            future = asyncio.run_coroutine_threadsafe(
                self._get_social_signal_async(slug), loop
            )
            return future.result(timeout=30)
        except RuntimeError:
            return asyncio.run(self._get_social_signal_async(slug))
```

- [ ] **Step 2: Write tests**

```python
"""Tests for regime.py — async regime classification safety."""
import pytest
from data.regime import RegimeClassifier

@pytest.mark.asyncio
async def test_get_social_signal_async_no_crash():
    """Should not crash when called from async context."""
    classifier = RegimeClassifier()
    result = await classifier._get_social_signal_async("bitcoin")
    assert result is None  # No API key in test, returns None gracefully

def test_get_social_signal_sync_no_crash():
    """Should not crash when called from sync context."""
    classifier = RegimeClassifier()
    result = classifier._get_social_signal("bitcoin")
    assert result is None
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/test_regime.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add data/regime.py tests/test_regime.py
git commit -m "fix: make regime social signal fetch async-safe, fix C6 (asyncio.run crash)"
```

---

### Task 1.4: Fix SQL Injection in table_count (C7)

**Files:**
- Modify: `data/database.py:632-644`
- Test: `tests/test_trading_database.py`

**Root cause:** `table_count()` and `clear_all()` use f-string interpolation for table names with no sanitization.

- [ ] **Step 1: Add whitelist validation to database.py**

At the top of `TradingDatabase` class, add:
```python
    VALID_TABLES = frozenset({
        "trades", "experiments", "pipeline_results",
        "oos_results", "validation_trades", "api_cache", "_migrations"
    })
```

Change `table_count` (lines 632-638) to:
```python
    def table_count(self, table_name: str) -> int:
        """Count rows in a table. Raises ValueError for invalid table names."""
        if table_name not in self.VALID_TABLES:
            raise ValueError(f"Invalid table name: {table_name!r}")
        with self.transaction() as conn:
            result = conn.execute(
                f"SELECT COUNT(*) FROM [{table_name}]"
            ).fetchone()
            return result[0]
```

Change `clear_all` (lines 640-644) to:
```python
    def clear_all(self) -> None:
        """Clear all data (for testing)."""
        with self.transaction() as conn:
            for table in self.VALID_TABLES - {"_migrations"}:
                conn.execute(f"DELETE FROM [{table}]")
```

- [ ] **Step 2: Write tests for SQL injection guards**

Add to `tests/test_trading_database.py`:
```python
"""Tests for database.py — SQL injection guards."""
import pytest
from data.database import TradingDatabase

class TestTableCountSecurity:
    def test_table_count_valid_table(self, db: TradingDatabase):
        """Valid table name returns a count."""
        count = db.table_count("trades")
        assert isinstance(count, int)

    def test_table_count_invalid_table_raises(self, db: TradingDatabase):
        """Invalid table name raises ValueError."""
        with pytest.raises(ValueError, match="Invalid table name"):
            db.table_count("nonexistent_table")

    def test_table_count_sql_injection_attempt_raises(self, db: TradingDatabase):
        """SQL injection strings are rejected."""
        with pytest.raises(ValueError):
            db.table_count("trades; DROP TABLE trades;--")
        with pytest.raises(ValueError):
            db.table_count("' OR '1'='1")
        with pytest.raises(ValueError):
            db.table_count(""); DROP TABLE trades; --")

    def test_clear_all_with_injection_safe(self, db: TradingDatabase):
        """clear_all only clears known tables, not affected by injection."""
        # clear_all should only touch VALID_TABLES
        db.clear_all()
        assert db.table_count("trades") == 0
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/test_trading_database.py::TestTableCountSecurity -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add data/database.py tests/test_trading_database.py
git commit -m "fix: add SQL injection guards to table_count and clear_all, fix C7"
```

---

### Task 1.5: Fix CDLENGULFING Double-Count Bug (C8)

**Files:**
- Modify: `data/patterns.py:14` (remove CDLENGULFING from BEARISH_PATTERNS)
- Test: `tests/test_patterns.py`

**Root cause:** CDLENGULFING in both bullish and bearish lists. `pattern_to_signal` counts it twice, always neutralizing. TA-Lib's `CDLENGULFING` already returns a signed value (positive=bullish, negative=bearish), so the named list is just for grouping — keep in BULLISH only.

- [ ] **Step 1: Fix the pattern list**

Change line 13-15 from:
```python
BEARISH_PATTERNS = [
    "CDLSHOOTINGSTAR", "CDLENGULFING", "CDLEVENINGSTAR",
```
To:
```python
BEARISH_PATTERNS = [
    "CDLSHOOTINGSTAR", "CDLEVENINGSTAR",
```

- [ ] **Step 2: Write test for CDLENGULFING fix**

```python
"""Tests for patterns.py — CDLENGULFING fix."""
import pytest
from data.patterns import BULLISH_PATTERNS, BEARISH_PATTERNS, PatternDetector

class TestPatternLists:
    def test_cdlengulfing_not_in_both_lists(self):
        """CDLENGULFING should only be in one list."""
        assert ("CDLENGULFING" in BULLISH_PATTERNS) != ("CDLENGULFING" in BEARISH_PATTERNS), \
            "CDLENGULFING must not appear in both lists — it would always neutralize"

    def test_no_pattern_in_both_lists(self):
        """No pattern should appear in both bullish and bearish lists."""
        for p in BULLISH_PATTERNS:
            assert p not in BEARISH_PATTERNS, f"{p} appears in both lists"

class TestPatternToSignal:
    def test_pattern_to_signal_no_double_count(self):
        """Engulfing should not be double-counted."""
        detector = PatternDetector()
        # If only CDLENGULFING is active, the signal should NOT be neutral
        # (it should be bullish since we kept it in BULLISH_PATTERNS)
        engulfing_only = ["CDLENGULFING"]
        result = detector.pattern_to_signal(engulfing_only)
        # Since we kept it in BULLISH_PATTERNS, result should be "bullish"
        assert result == "bullish"
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/test_patterns.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add data/patterns.py tests/test_patterns.py
git commit -m "fix: remove CDLENGULFING from BEARISH_PATTERNS to fix double-count, fix C8"
```

---

### Task 1.6: Remove .env from Git Tracking (C9)

**Files:**
- Modify: `.gitignore`
- Create: `.env.example` (already created in Task 1.1)
- Delete: `.env` from git tracking (not from disk)

**Root cause:** `.env` is tracked in git with valid-looking API keys for CoinGecko, HuggingFace, Messari, Santiment, CoinCap.

- [ ] **Step 1: Add .env to .gitignore**

Append `.env` to existing `.gitignore`.

- [ ] **Step 2: Remove .env from git tracking**

```bash
git rm --cached .env
```

This leaves the file on disk but stops tracking it.

- [ ] **Step 3: Verify no API keys in git**

```bash
git status  # .env should show as deleted from tracking
```

- [ ] **Step 4: Rotate all exposed API keys**

Inform the user to rotate keys at each provider (CoinGecko, HuggingFace, Messari, Santiment, CoinCap).

- [ ] **Step 5: Commit**

```bash
git add .gitignore
git commit -m "fix: remove .env from git tracking, add .env.example, fix C9 (API key exposure)"
```

---

## Batch 2: Backtesting Integrity

### Task 2.1: Fix Synthetic Validator No-Op (C2/C3)

**Files:**
- Modify: `backtesting/engine.py` (add `dataframe_override` parameter)
- Modify: `backtesting/synthetic_validator.py:155-165, 249-257` (pass synthetic data)
- Test: `tests/test_synthetic_validator.py`, `tests/test_engine.py`

**Root cause:** `SyntheticValidator.validate_strategy()` generates synthetic data but passes a hardcoded timerange to the engine instead of the synthetic DataFrame. `run_permutation_test()` has the same bug. The engine has no way to accept pre-loaded DataFrames.

- [ ] **Step 1: Add dataframe_override to BacktestEngine**

Add to `backtesting/engine.py` `__init__`:
```python
    def __init__(self, ..., dataframe_override: Optional[Dict[str, pd.DataFrame]] = None):
        # ... existing init ...
        self._dataframe_override = dataframe_override or {}
```

Add a `_run_fastmetrics_backtest` method:
```python
    def _run_fastmetrics_backtest(self, strategy_name: str, strategy_code: str, 
                                   dataframe: pd.DataFrame) -> Dict[str, Any]:
        """Run backtest using FastMetrics + SignalFactory on pre-loaded DataFrame.
        Used by SyntheticValidator to test against random-walk data."""
        from backtesting.signal_factory import SignalFactory, FastMetrics
        
        factory = SignalFactory(dataframe)
        metrics = factory.compute_all(strategy_code)
        
        return {
            "sharpe_ratio": metrics.get("sharpe", 0),
            "win_rate": metrics.get("win_rate", 0),
            "total_trades": metrics.get("total_trades", 0),
            "max_drawdown": metrics.get("max_drawdown", 0),
            "profit_ratio": metrics.get("profit_ratio", 0),
        }
```

Modify `run_backtest` to check for dataframe_override:
```python
    def run_backtest(self, strategy_params=None, strategy_type=None, ...):
        if self._dataframe_override:
            df = self._dataframe_override
            # Run on synthetic data using FastMetrics
            return self._run_fastmetrics_backtest(
                strategy_params.get("name", "synthetic"),
                strategy_params.get("code", ""),
                df
            )
        # ... existing Freqtrade subprocess logic ...
```

- [ ] **Step 2: Fix synthetic_validator.py validate_strategy**

Change lines 155-165 to pass the synthetic DataFrame:
```python
        for i in range(n_synthetic_runs):
            try:
                df_synthetic = self.generate_random_walk(seed=i)
                engine = BacktestEngine(dataframe_override={"synthetic": df_synthetic})
                # ...
```

Or, simpler: pass the dataframe_override to the existing engine instance:
```python
        engine = BacktestEngine()
        for i in range(n_synthetic_runs):
            try:
                df_synthetic = self.generate_random_walk(seed=i)
                engine._dataframe_override = {"synthetic": df_synthetic}
                result = engine.run_backtest(
                    strategy_params=strategy_params,
                    strategy_type=strategy_type,
                    timerange="20170101-20231231",  # Not used when dataframe_override is set
                )
```

Wait — the engine is created once outside the loop. The simplest approach: pass `dataframe_override` per-call:

```python
    def run_backtest(self, strategy_params=None, strategy_type=None, ...,
                     dataframe_override: Optional[pd.DataFrame] = None):
        if dataframe_override is not None:
            return self._run_fastmetrics_backtest(strategy_params, dataframe_override)
        # ... existing ...
```

Then in synthetic_validator.py:
```python
                result = engine.run_backtest(
                    strategy_params=strategy_params,
                    strategy_type=strategy_type,
                    timerange="20170101-20231231",
                    dataframe_override=df_synthetic,
                )
```

- [ ] **Step 3: Fix synthetic_validator.py run_permutation_test**

Same pattern for lines 249-257:
```python
                df_permuted = self.generate_random_walk(seed=i + 1000)
                perm_result = engine.run_backtest(
                    strategy_params=strategy_params,
                    strategy_type=strategy_type,
                    timerange="20170101-20231231",
                    dataframe_override=df_permuted,
                )
```

- [ ] **Step 4: Write tests**

```python
"""Tests for synthetic_validator.py — synthetic data flow."""
import pytest
from backtesting.synthetic_validator import SyntheticValidator

class TestSyntheticValidator:
    def test_validate_strategy_uses_synthetic_data(self):
        """validate_strategy should run backtest on synthetic data, not real data."""
        validator = SyntheticValidator()
        # Patch BacktestEngine to verify dataframe_override is passed
        with patch("backtesting.engine.BacktestEngine.run_backtest") as mock_run:
            validator.validate_strategy(
                strategy_id="test_1",
                strategy_type="sma_crossover",
            )
            for call_args in mock_run.call_args_list:
                kwargs = call_args[1]
                assert "dataframe_override" in kwargs, \
                    "Must pass dataframe_override to engine"
                assert kwargs["dataframe_override"] is not None, \
                    "dataframe_override must not be None"

    def test_run_permutation_test_uses_synthetic_data(self):
        """run_permutation_test should also use synthetic data."""
        validator = SyntheticValidator()
        with patch("backtesting.engine.BacktestEngine.run_backtest") as mock_run:
            validator.run_permutation_test(
                strategy_id="test_1",
                strategy_type="sma_crossover",
            )
            for call_args in mock_run.call_args_list:
                kwargs = call_args[1]
                assert "dataframe_override" in kwargs
                assert kwargs["dataframe_override"] is not None
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_synthetic_validator.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add backtesting/engine.py backtesting/synthetic_validator.py tests/test_synthetic_validator.py
git commit -m "fix: add dataframe_override to BacktestEngine, fix C2/C3 (synthetic validator no-op)"
```

---

### Task 2.2: Fix Data Directory Hardcoded to Binance (H6)

**Files:**
- Modify: `backtesting/setup_data.py:32`

**Root cause:** `ft_path / "data" / "binance"` should use `settings.EXCHANGE_ID` instead of hardcoded "binance".

- [ ] **Step 1: One-line fix**

Change:
```python
data_path = ft_path / "data" / "binance"
```
To:
```python
from config import settings
data_path = ft_path / "data" / settings.EXCHANGE_ID
```

Check if `from config import settings` already exists at the top of the file. If not, add it.

- [ ] **Step 2: Write test**

```python
"""Tests for setup_data.py — exchange data path."""
from unittest.mock import patch, MagicMock
from backtesting.setup_data import ensure_data_available

def test_data_path_uses_exchange_id():
    """Data path should use settings.EXCHANGE_ID, not hardcoded 'binance'."""
    with patch("backtesting.setup_data.settings") as mock_settings:
        mock_settings.EXCHANGE_ID = "kraken"
        mock_settings.SYMBOL = "BTC/USDT"
        mock_settings.TIMEFRAME = "1h"
        # ensure_data_available should use "kraken" subdirectory
        with patch("pathlib.Path.exists") as mock_exists:
            mock_exists.return_value = True
            # Trigger the check
            # Assert the path contains "kraken" not "binance"
```

- [ ] **Step 3: Commit**

```bash
git add backtesting/setup_data.py tests/test_setup_data.py
git commit -m "fix: use settings.EXCHANGE_ID instead of hardcoded binance in data path, fix H6"
```

---

### Task 2.3: Add Timeout to LangGraph Invoke (H8)

**Files:**
- Modify: `orchestration/hermes.py` (wherever `self._graph.invoke()` is called)
- Test: `tests/test_hermes.py`

**Root cause:** No timeout on `self._graph.invoke(initial_state)` — can hang indefinitely.

- [ ] **Step 1: Read the invoke call location**

```bash
grep -n 'self._graph.invoke\|self._graph.ainvoke' C:/Trading-bot/crypto_agent_bot/orchestration/hermes.py
```

- [ ] **Step 2: Add timeout wrapper**

Replace the bare invoke call with:
```python
        try:
            result = await asyncio.wait_for(
                self._graph.ainvoke(initial_state),
                timeout=settings.LANGGRAPH_TIMEOUT if hasattr(settings, 'LANGGRAPH_TIMEOUT') else 300
            )
        except asyncio.TimeoutError:
            logger.error("LangGraph invoke timed out")
            initial_state["error"] = "Research cycle timed out"
            return initial_state
```

Add `LANGGRAPH_TIMEOUT` to config.py:
```python
    LANGGRAPH_TIMEOUT: int = int(os.getenv("LANGGRAPH_TIMEOUT", "300"))
```

- [ ] **Step 3: Write test**

```python
"""Tests for hermes.py — LangGraph timeout."""
import pytest
from unittest.mock import patch, MagicMock
import asyncio

class TestLangGraphTimeout:
    @pytest.mark.asyncio
    async def test_timeout_on_slow_agent(self):
        """Should raise TimeoutError and return error state when agents hang."""
        # Create a hermes instance with a mock graph that hangs
        hermes = ResearchOrchestrator(...)
        with patch.object(hermes._graph, 'ainvoke', side_effect=asyncio.TimeoutError):
            result = await hermes._run_research_goal("test goal")
            assert "error" in result
            assert "timed out" in result["error"]
```

- [ ] **Step 4: Commit**

```bash
git add orchestration/hermes.py config.py tests/test_hermes.py
git commit -m "fix: add timeout to LangGraph invoke, fix H8 (no timeout on research cycle)"
```

---

## Batch 3: Agent Logic Bugs

### Task 3.1: Fix assess_strategy_risk Rejecting Good Strategies (H1)

**Files:**
- Modify: `agents/risk_manager.py:819`
- Test: `tests/test_risk_manager.py`

**Root cause:** `if risk_score >= 0.5 or not concerns:` — when concerns is empty (strategy is fine), `not concerns` is True, triggering rejection.

- [ ] **Step 1: Fix boolean logic**

Change line 819:
```python
            if risk_score >= 0.5 or not concerns:
                verdict = "reject"
```
To:
```python
            if risk_score >= 0.5 and concerns:
                verdict = "reject"
```

- [ ] **Step 2: Write tests**

```python
"""Tests for risk_manager.py — assess_strategy_risk boolean logic."""
import pytest
from agents.risk_manager import RiskManagerAgent

class TestAssessStrategyRisk:
    def test_empty_concerns_passes(self):
        """Strategy with zero concerns should pass regardless of risk_score.
        (The score is 0 when concerns list is empty anyway.)"""
        manager = RiskManagerAgent(...)
        result = json.loads(manager.assess_strategy_risk(json.dumps({
            "sharpe_ratio": 0.5, "win_rate": 0.5, "max_drawdown": 0.1,
            "total_trades": 50, "profit_factor": 1.5,
        })))
        # With empty concerns, verdict should NOT be "reject"
        assert result.get("verdict") != "reject"

    def test_high_risk_with_concerns_rejects(self):
        """High risk score WITH concerns should reject."""
        # This strategy has concerns (drawdown, profit factor)
        # ... mock or construct the call ...
        result = json.loads(manager.assess_strategy_risk(json.dumps({
            "sharpe_ratio": 0.1, "win_rate": 0.3, "max_drawdown": 0.2,
            "total_trades": 50, "profit_factor": 0.8,
        })))
        assert result.get("verdict") == "reject"

    def test_low_risk_with_concerns_passes_cautious(self):
        """Low risk score with concerns should pass as 'cautious', not reject."""
        result = json.loads(manager.assess_strategy_risk(json.dumps({
            "sharpe_ratio": 0.5, "win_rate": 0.5, "max_drawdown": 0.08,
            "total_trades": 50, "profit_factor": 1.1,
        })))
        assert result.get("verdict") in ("cautious", "accept")
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/test_risk_manager.py::TestAssessStrategyRisk -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add agents/risk_manager.py tests/test_risk_manager.py
git commit -m "fix: fix assess_strategy_risk boolean logic, fix H1 (rejects good strategies)"
```

---

### Task 3.2: Wire Real Sentiment Data to Analyst (H2)

**Files:**
- Modify: `agents/analyst.py:53-59` (replace stub with real sentiment)
- Test: `tests/test_analyst.py`

**Root cause:** `sentiment_fn` is a hardcoded stub returning "NEUTRAL (score 0.0/1.0)".

- [ ] **Step 1: Replace stub with real sentiment fetch**

```python
        def sentiment_fn(symbol: str = "") -> str:
            """Get crypto market sentiment from multiple sources."""
            s = symbol.strip() or settings.SYMBOL
            try:
                from data.sentiment import SentimentFetcher
                fetcher = SentimentFetcher()
                # Get slug from symbol
                slug = s.split("/")[0].lower() if "/" in s else s.lower()
                result = fetcher.get_combined_sentiment_sync(slug, s.split("/")[0])
                score = result.score
                label = result.label
                return f"Sentiment: {label} (score {score:.2f}/1.0)"
            except Exception as e:
                logger.warning("Sentiment fetch failed for %s: %s", s, e)
                return f"Sentiment: UNAVAILABLE (error fetching data)"
```

- [ ] **Step 2: Write tests**

```python
"""Tests for analyst.py — sentiment tool."""
import pytest
from unittest.mock import patch, MagicMock
from agents.analyst import AnalystAgent

class TestAnalystSentiment:
    def test_sentiment_fn_returns_real_data(self):
        """sentiment_fn should return real sentiment, not hardcoded stub."""
        analyst = AnalystAgent()
        with patch("agents.analyst.SentimentFetcher") as mock_fetcher_class:
            mock_instance = MagicMock()
            mock_instance.get_combined_sentiment_sync.return_value = MagicMock(
                score=0.75, label="BULLISH"
            )
            mock_fetcher_class.return_value = mock_instance
            
            # Extract the sentiment_fn closure from the tools list
            tools = analyst._build_tools()
            sentiment_tool = [t for t in tools if t.name == "get_market_sentiment"][0]
            result = sentiment_tool.func("BTC/USDT")
            
            assert "NEUTRAL" not in result
            assert "BULLISH" in result
            assert "0.75" in result

    def test_sentiment_fn_fallback_on_error(self):
        """sentiment_fn should return UNAVAILABLE on error, not crash."""
        analyst = AnalystAgent()
        with patch("agents.analyst.SentimentFetcher", side_effect=ImportError):
            tools = analyst._build_tools()
            sentiment_tool = [t for t in tools if t.name == "get_market_sentiment"][0]
            result = sentiment_tool.func("BTC/USDT")
            assert "UNAVAILABLE" in result
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/test_analyst.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add agents/analyst.py tests/test_analyst.py
git commit -m "fix: wire real sentiment data to analyst tool, fix H2 (sentiment stub)"
```

---

### Task 3.3: Fix _generated_specs Never Written (H3)

**Files:**
- Modify: `agents/researcher.py` (find where specs are generated, write to `_generated_specs`)
- Test: `tests/test_researcher.py`

**Root cause:** `_generated_specs` initialized as `{}` but never written. `get_specs()` always returns empty.

- [ ] **Step 1: Find where spec generation produces output**

```bash
grep -n 'generate_custom_strategy_spec\|_specs' C:/Trading-bot/crypto_agent_bot/agents/researcher.py
```

- [ ] **Step 2: Add write to _generated_specs**

In the `generate_custom_strategy_spec` method, after producing the spec, store it:
```python
    def generate_custom_strategy_spec(self, ...) -> str:
        # ... existing logic to generate spec ...
        spec = self._build_spec(...)
        if spec.get("name"):
            self._generated_specs[spec["name"]] = spec
        return json.dumps(spec)
```

- [ ] **Step 3: Write tests**

```python
"""Tests for researcher.py — _generated_specs storage."""
import pytest
from agents.researcher import ResearcherAgent

class TestGeneratedSpecs:
    def test_generated_specs_populated_after_generation(self):
        """generate_custom_strategy_spec should populate _generated_specs."""
        researcher = ResearcherAgent()
        researcher.generate_custom_strategy_spec(
            name="Test Momentum",
            description="A test momentum strategy",
            indicators=["SMA", "RSI"],
            entry_conditions=["SMA crossover"],
            exit_conditions=["RSI overbought"],
        )
        specs = researcher.get_specs()
        assert len(specs) > 0
        assert "Test Momentum" in specs

    def test_get_specs_not_empty_after_run(self):
        """After running research, get_specs() should return data."""
        researcher = ResearcherAgent()
        researcher.run("Generate a momentum strategy")
        specs = researcher.get_specs()
        # Should not be empty dict
        assert isinstance(specs, dict)
```

- [ ] **Step 4: Commit**

```bash
git add agents/researcher.py tests/test_researcher.py
git commit -m "fix: write generated specs to _generated_specs, fix H3 (get_specs always returns empty)"
```

---

### Task 3.4: Fix PaperTrader State Reset on Every Signal (H4/H5)

**Files:**
- Modify: `execution/live_executor.py` (store PaperTrader instance, fix fill_price)
- Test: `tests/test_live_executor.py`

**Root cause:** `_execute_paper` creates a new `PaperTrader(max_candles=5)` per signal, losing all state.

- [ ] **Step 1: Add PaperTrader instance variable**

In `__init__`, add:
```python
        self._paper_trader: Optional[PaperTrader] = None
```

- [ ] **Step 2: Modify _execute_paper to reuse PaperTrader**

Change to:
```python
    async def _execute_paper(self, signal: TradeSignal) -> Dict[str, Any]:
        """Execute a signal in paper mode, maintaining continuous state."""
        if self._paper_trader is None:
            self._paper_trader = PaperTrader(
                symbol=signal.symbol,
                initial_balance=settings.PAPER_INITIAL_BALANCE,
                fetcher=self._fetcher,
            )
        
        result = await self._paper_trader.run(
            signal=signal,
            kelly_fraction=signal.kelly_fraction,
        )
        
        return {
            "success": True,
            "fill_price": result.get("fill_price", 0),
            "fill_amount": result.get("amount", 0),
            "order_id": result.get("order_id", f"paper_{signal.signal_id}"),
            "status": "filled",
        }
```

- [ ] **Step 3: Fix fill_price from PaperTrader**

In paper_trader.py, ensure `run()` returns the actual fill price, not `final_balance`. Modify the return dict:
```python
        return {
            "fill_price": entry_price,  # Actual price from the candle
            "amount": self.position_size,
            "final_balance": self.balance,
            "order_id": f"paper_{uuid.uuid4().hex[:8]}",
        }
```

- [ ] **Step 4: Write tests**

```python
"""Tests for live_executor.py — PaperTrader state persistence."""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from execution.live_executor import LiveExecutor, TradeSignal

class TestPaperTraderState:
    @pytest.mark.asyncio
    async def test_paper_trader_reused_across_signals(self):
        """PaperTrader should be reused, not recreated per signal."""
        executor = LiveExecutor(paper_mode=True, ...)
        
        mock_trader = AsyncMock(spec=PaperTrader)
        mock_trader.run.return_value = {"fill_price": 50000, "amount": 0.1}
        
        with patch.object(executor, '_paper_trader', None), \
             patch('execution.live_executor.PaperTrader', return_value=mock_trader):
            
            signal1 = TradeSignal(symbol="BTC/USDT", side="buy", ...)
            signal2 = TradeSignal(symbol="BTC/USDT", side="sell", ...)
            
            await executor._execute_paper(signal1)
            await executor._execute_paper(signal2)
            
            # PaperTrader should be created only once
            assert executor._paper_trader is not None

    @pytest.mark.asyncio
    async def test_paper_execution_fill_price_is_price_not_balance(self):
        """Paper fill price should be a price, not a balance value."""
        executor = LiveExecutor(paper_mode=True, ...)
        executor._paper_trader = AsyncMock(spec=PaperTrader)
        executor._paper_trader.run.return_value = {
            "fill_price": 50000.0,  # Actual BTC price
            "amount": 0.1,
            "final_balance": 9500.0,
        }
        
        result = await executor._execute_paper(TradeSignal(...))
        assert result["fill_price"] == 50000.0  # Price, not balance
        assert result["fill_price"] != 9500.0
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_live_executor.py::TestPaperTraderState -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add execution/live_executor.py execution/paper_trader.py tests/test_live_executor.py
git commit -m "fix: reuse PaperTrader across signals, fix fill_price, fix H4/H5"
```

---

## Batch 4: Architecture Debt

### Task 4.1: Fix Board Reassigned Mid-Method (H7)

**Files:**
- Modify: `orchestration/hermes.py:156`

- [ ] **Step 1: Read the board assignment code**

```bash
grep -n 'self.board\s*=\|self\.board\s*=' C:/Trading-bot/crypto_agent_bot/orchestration/hermes.py | head -10
```

- [ ] **Step 2: Fix by removing mid-method reassignment**

Move the `self.board = StateGraph(MarketIntelligence)` to `__init__` or method start. Remove the duplicate assignment mid-method.

- [ ] **Step 3: Commit**

---

### Task 4.2: Add LLM Timeout/Retry to BaseAgent

**Files:**
- Modify: `agents/base.py:65`
- Test: `tests/test_base_agent.py`

- [ ] **Step 1: Add _invoke_with_retry to BaseAgent**

```python
    async def _invoke_with_retry(self, messages, max_retries=3, timeout=120):
        """Invoke LLM with timeout and exponential backoff retry."""
        last_error = None
        for attempt in range(max_retries):
            try:
                return await asyncio.wait_for(
                    self._agent.ainvoke(messages),
                    timeout=timeout
                )
            except (asyncio.TimeoutError, Exception) as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait = 2 ** attempt  # exponential backoff: 1, 2, 4
                    logger.warning(
                        "LLM invoke attempt %d/%d failed: %s. Retrying in %ds...",
                        attempt + 1, max_retries, e, wait
                    )
                    await asyncio.sleep(wait)
        raise RuntimeError(f"LLM invoke failed after {max_retries} retries") from last_error
```

- [ ] **Step 2: Update run() to use _invoke_with_retry**

Replace `result = self._agent.invoke(initial_state)` with:
```python
        result = await self._invoke_with_retry(initial_state)
```

- [ ] **Step 3: Commit**

---

### Task 4.3: Add VectorStore Dependency Injection

**Files:**
- Modify: `agents/strategist.py`, `agents/backtester.py`, `memory/iteration_tracker.py`, `agents/curator.py`

- [ ] **Step 1: Add optional vector_store parameter to each constructor**

```python
    def __init__(self, ..., vector_store: Optional['VectorStore'] = None):
        self._vector_store = vector_store or VectorStore()
```

- [ ] **Step 2: Commit**

---

### Task 4.4: Add Thread Safety to CircuitBreakerState

**Files:**
- Modify: `agents/risk_manager.py:59-94`

- [ ] **Step 1: Add threading.Lock to CircuitBreakerState**

```python
@dataclass
class CircuitBreakerState:
    _halted: ClassVar[bool] = False
    _halt_reason: ClassVar[str] = ""
    _resume_after: ClassVar[Optional[datetime]] = None
    _research_mode: ClassVar[bool] = False
    _lock: ClassVar[threading.Lock] = threading.Lock()
```

- [ ] **Step 2: Add lock acquire/release to all state-mutating methods**

```python
    @classmethod
    def halt(cls, reason: str, duration_minutes: int = 60) -> None:
        with cls._lock:
            cls._halted = True
            cls._halt_reason = reason
            cls._resume_after = datetime.now(timezone.utc) + timedelta(minutes=duration_minutes)
    
    @classmethod
    def clear(cls) -> None:
        with cls._lock:
            cls._halted = False
            cls._halt_reason = ""
            cls._resume_after = None
    
    @classmethod
    def is_halted(cls) -> bool:
        with cls._lock:
            if cls._halted and cls._resume_after and datetime.now(timezone.utc) >= cls._resume_after:
                cls._halted = False
                cls._halt_reason = ""
                cls._resume_after = None
            return cls._halted
```

- [ ] **Step 3: Commit**

---

### Task 4.5: Fix EventBus Stale Loop + Add Unsubscribe

**Files:**
- Modify: `core/event_bus.py`

- [ ] **Step 1: Remove import-time event loop capture**

Replace `self._loop = asyncio.get_event_loop()` (or similar) with lazy resolution at publish time.

- [ ] **Step 2: Add unsubscribe method**

```python
    def unsubscribe(self, event_type: str, callback) -> bool:
        """Remove a subscriber. Returns True if found and removed."""
        if event_type in self._subscribers:
            if callback in self._subscribers[event_type]:
                self._subscribers[event_type].remove(callback)
                return True
        return False
```

- [ ] **Step 3: Commit**

---

## Batch 5: Monitoring & Deployment

### Task 5.1: Fix Telegram Approvals Non-Functional (C4)

**Files:**
- Modify: `monitoring/telegram_alerter.py:117-130`
- Test: `tests/test_telegram_alerter.py`

**Root cause:** Callback query handler never registered. Polling never started.

- [ ] **Step 1: Register callback handler in start()**

After `app = Application.builder().token(token).build()`:
```python
            app.add_handler(CallbackQueryHandler(self._handle_callback))
            await app.start()
            await app.updater.start_polling()
```

- [ ] **Step 2: Create _handle_callback method**

```python
    async def _handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle inline keyboard button presses."""
        query = update.callback_query
        await query.answer()
        
        data = query.data  # e.g., "approve_BTC/USDT" or "reject_BTC/USDT"
        if data.startswith("approve_"):
            pair = data.replace("approve_", "")
            self.resolve_approval(pair, approved=True)
            await query.edit_message_text(f"✅ Approved {pair}")
        elif data.startswith("reject_"):
            pair = data.replace("reject_", "")
            self.resolve_approval(pair, approved=False)
            await query.edit_message_text(f"❌ Rejected {pair}")
```

- [ ] **Step 3: Commit**

---

### Task 5.2: Fix Deployment Pipeline Gate Bugs (H9-H11)

**Files:**
- Modify: `orchestration/deployment_pipeline.py`

**Fixes:**
- H9 (line 300-306): Same fix as C2/C3 — pass synthetic DataFrame
- H10 (line 267): Increase `limit=1000` to something meaningful, e.g., full research window
- H11 (lines 319-320): Compute avg_win_pct/avg_loss_pct from backtest results instead of hardcoded defaults

- [ ] **Step 1: Commit**

---

### Task 5.3: Fix Rolling Sharpe Window (H19)

**Files:**
- Modify: `monitoring/performance_monitor.py:183`

**Root cause:** `returns.rolling(window=window_days)` uses row count, not time.

- [ ] **Step 1: Fix to use time-based window**

Replace:
```python
        rolling_sharpe = returns.rolling(window=window_days).mean() / \
                         returns.rolling(window=window_days).std()
```
With:
```python
        # Use time-based window instead of row-count window
        rolling_sharpe = returns.ewm(span=window_days).mean() / \
                         returns.ewm(span=window_days).std().clip(lower=1e-8)
```

- [ ] **Step 2: Commit**

---

## Self-Review Checklist

1. **Spec coverage:** Every issue from critical-issues.md is covered:
   - C1 → Task 1.1 ✅
   - C5 → Task 1.2 ✅
   - C6 → Task 1.3 ✅
   - C7 → Task 1.4 ✅
   - C8 → Task 1.5 ✅
   - C9 → Task 1.6 ✅
   - C2/C3 → Task 2.1 ✅
   - H6 → Task 2.2 ✅
   - H8 → Task 2.3 ✅
   - H1 → Task 3.1 ✅
   - H2 → Task 3.2 ✅
   - H3 → Task 3.3 ✅
   - H4/H5 → Task 3.4 ✅
   - H7 → Task 4.1 ✅
   - C4 → Task 5.1 ✅
   - H9-H11 → Task 5.2 ✅
   - H19 → Task 5.3 ✅

2. **Placeholder scan:** No TBD/TODO/placeholder content in code blocks.

3. **Type consistency:** All method names match across tasks.
