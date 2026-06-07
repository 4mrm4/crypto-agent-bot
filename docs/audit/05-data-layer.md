# Audit: Data Layer

Generated 2026-06-06.

Summary: 43 bugs, 32 code smells, 40 design issues, 46 missing tests across ~15 files.

---

## `data/database.py`

### Critical Bugs
- **SQL injection in `table_count` (lines 632-637):** `f"SELECT COUNT(*) FROM {table_name}"` — direct string interpolation, no sanitization
- **`clear_all` uses f-string for table names (lines 640-644):** Same injection pattern

### Bugs
- **Singleton `__new__` race condition (lines 38-53):** Another thread could get half-initialized instance
- **No connection pooling:** Two `transaction()` calls open two separate SQLite connections
- **Migration idempotency gap (lines 457-465):** Lock released before migration work begins

### Code Smells
- **`import uuid` inside 5 methods, 15 times total (lines 219, 273, 321, 370, 412)**
- **SQL query built by concatenating list elements (lines 248-267)**
- **3 repetitive query methods following identical pattern**
- **`migrate_jsonl` is a monolithic 147-line method (lines 468-614)**

### Design Issues
- **Hardcoded DB_PATH (line 26):** `"./workspace/trading.db"` — relative to CWD, not configurable via env
- **Singleton pattern makes unit testing difficult**
- **No connection timeout/retry:** No `PRAGMA busy_timeout`
- **No schema versioning:** Only tracks migration names, no version integer

### Missing Tests
`migrate_jsonl`, `integrity_check`, `is_wal_mode`, `table_count`, `clear_all`, `query_oos_results` (no strategy_id), `query_pipeline_results` (no strategy_id), `query_validation_trades` (no strategy_id)

---

## `data/coincap_fetcher.py`

### Bugs
- **Symbol mapping covers only 15 assets (lines 170-186):** Unknown tokens silently return None
- **`symbol_to_coincap_id()` returns None with no fallback (lines 192-197):** Fetcher.py's version has fallback, coincap version doesn't

### Code Smells
- Lazy client init without connection pooling reuse (lines 52-55)
- `import asyncio` inside method body (line 116)
- Hardcoded 10s timeout (line 54)
- All errors swallowed — caller gets None, can't distinguish failure types (lines 35-37)

---

## `data/fetcher.py`

### Bugs
- **`getattr(ccxt, self.exchange_id)` crash (line 32):** AttributeError on unknown exchange — `MarketDataFetcher` has no try/except
- **`self._exchange.load_markets()` blocks event loop (lines 40-44):** Synchronous call in async context
- **`self.exchange.sleep(...)` blocks event loop (line 71):** Called from async `fetch_ohlcv_merged` (line 160)
- **`symbol_to_coincap_id()` function duplicated with DIFFERENT behavior (lines 123-127):** Version here has fallback, coincap version returns None
- **Hardcoded default exchanges (line 134):** `["kraken", "binance"]` ignores `settings.EXCHANGES`
- **`fetch_ohlcv_merged` creates new `CoinCapFetcher()` per call (lines 160-191):** New TCP connection every time

### Design Issues
- `exchange` property not thread-safe (lines 28-45)
- No health tracking integration — unlike coincap/messari/santiment
- Returns empty DataFrame on failure — callers may not check

### Missing Tests
`MarketDataFetcher` has no direct tests at all

---

## `data/messari_fetcher.py`

### Bugs
- **API error wrapped in HTTP 200 gets cached (lines 74-107):** `raise_for_status()` only checks HTTP codes
- **Endpoint URL mismatch (lines 166-175):** `/news/topics` (v1 endpoint) appended to v2 base URL — might not exist

### Code Smells
- `import asyncio` inside method (line 183)

---

## `data/santiment_fetcher.py`

### Bugs
- **Free tier date range logic is wrong (lines 140-144):** Comment says "data up to ~30 days ago" but code requests 37+ days ago to 30 days ago. Actually always requests data that is 30+ days old
- **Cache key collision (lines 146-148):** `santiment:{slug}:{metric}:{days}` doesn't include `from_dt`/`to_dt` — different `days` parameters collide

### Code Smells
- Inline caching duplicates `_get` pattern from Messari (different caching pattern)
- `import asyncio` inside method (line 219)

---

## `data/sentiment.py`

### CRITICAL BUGS
- **`asyncio.run()` crash in async context (line 145 → line 168):** `_fetch_santiment` uses `asyncio.run()` in `get_combined_sentiment`. If called from a running event loop (which it will be, from the autonomous loop), raises `RuntimeError`
- **Two competing weight systems (lines 82-105):** `SENTIMENT_WEIGHTS` dict (40/20/25/15) defined but ignored; hardcoded 0.6/0.4 weights used instead

### Bugs
- **`get_fear_greed_index` uses synchronous httpx (lines 43-51):** Blocks event loop
- **Silent error mask (line 43):** Returns `{"value": 50, "classification": "Neutral"}` on API failure — callers can't distinguish "down" from "neutral"

### Design Issues
- `_fetch_santiment` creates NEW `SantimentFetcher` every call — expensive
- No health tracking integration
- `_fetch_santiment` doesn't close the client — resource leak

---

## `data/regime.py`

### CRITICAL BUGS
- **`asyncio.run()` crash in async context (line 74):** `_get_social_signal` uses `asyncio.run()` from a sync method. When `classify_regime_snapshot` is called from async context, crashes with `RuntimeError`

### Bugs
- **`classify_regime` and `classify_regime_snapshot` duplicate identical TA-Lib calculations (lines 87-123):** Snapshot calls classify_regime but re-does all calculations anyway (lines 127-142)

### Code Smells
- NaN checking using `val == val` pattern — works but fragile (lines 102-107)
- `import statistics` inside method (lines 52-61)
- `import asyncio` inside method (line 71)
- `get_best_strategy_types` duplicates `REGIME_STRATEGY_MAP` (lines 184-192)

### Design Issues
- Social dominance z-score uses rolling window of 50 — unreliable until 50 points accumulate
- `_dominance_history` not persisted — lost on restart
- No health tracking integration

---

## `data/onchain.py`

### Bugs
- **"Exchange netflow" actually measures volume-to-market-cap (lines 56-87):** Misleading name — does NOT measure exchange inflow/outflow. It's a volume analysis.
- **`vol_ratio > 0.05` means "volume > 5% of market cap" (line 80):** Extreme threshold — almost always returns neutral
- **Synchonous httpx (lines 26-54, 56-87):** Blocks event loop
- **No error handling on sub-calls in `get_onchain_report` (lines 88-106)**

### Design Issues
- Entire file is sync in an async codebase
- No APIHealthTracker integration
- No caching — calls made fresh every time
- `.env` has `WHALE_ALERT_API_KEY=` (empty) — wastes network calls

### Missing Tests
**Zero tests** for onchain.py at all

---

## `data/patterns.py`

### Bugs
- **CDLENGULFING in BOTH bullish and bearish lists (lines 11, 14):** `pattern_to_signal` counts it on both sides simultaneously — always neutralizes the signal. Most reliable reversal pattern effectively disabled.
- **`signals[-1]` can be NaN (line 30):** TA-Lib returns NaN for first N-1 candles depending on pattern lookback

### Code Smells
- Only 13 of 61+ TA-Lib patterns defined — many common patterns missing

---

## `data/stream.py`

### Bugs
- **Stream count may exceed Binance's 1024 limit (line 60):** 50 pairs × 6 stream types = 300 — no check/warning
- **Queue drops messages silently (lines 72-78):** `maxsize=100`, uses `q.get_nowait()` on full — silent data loss
- **`disconnect` uses `asyncio.ensure_future()` without awaiting (lines 171-175):** May never execute during shutdown

### Design Issues
- Hardcoded for Binance — won't work with Kraken
- **No reconnection logic (line 106-123):** `max_reconnects=3` variable defined but never used
- No authentication support
- No health tracking

### Missing Tests
Zero tests for stream.py

---

## `memory/vector_store.py`

### Bugs
- **`hash()` for doc IDs not stable across restarts (line 79):** Python uses randomized hash seeds — same text produces different IDs each run. Causes duplicates and orphaned documents
- **`hash()` collision risk (lines 75-84):** First 16 chars of hash string → reduced entropy, collisions possible
- **`store_strategy_result` uses `add()` not `upsert()` (line 146):** uuid4 collision (extremely unlikely but possible) = ChromaDB error
- **`store_strategy_result` imports from `backtesting.data_split` at runtime (lines 122-151):** Import cycle risk

### Code Smells
- `_get_embedding_function()` sets `HF_TOKEN` via `os.environ` — global side effect, no cleanup (lines 20-37)
- `query_similar` calls `self.count()` twice (lines 89-92): Two ChromaDB round-trips

### Design Issues
- No cleanup of old strategies — unbounded collection growth
- `query_deployable_by_regime` searches for `"DEPLOYABLE"` — fragile embedding-dependent search
- `tag_as_deployable` and `update_live_performance` generate new IDs every call — duplicates accumulate

### Missing Tests
`store_strategy_result`, `get_best_strategies` (with regime filtering), `query_strategies_excluding_window`, `tag_as_deployable`, `query_deployable_by_regime`, `update_live_performance`

---

## Cross-Cutting Issues

### `datetime.utcnow()` Deprecation
14+ locations across: api_health.py, coincap_fetcher.py, database.py, messari_fetcher.py, santiment_fetcher.py, regime.py, sentiment.py

### `import asyncio` Inside Method Bodies
coincap_fetcher.py:116, messari_fetcher.py:183, santiment_fetcher.py:219, regime.py:71

### `import uuid` Inside Method Bodies
database.py: 15 occurrences across 5 methods

### `asyncio.run()` From Sync Context (CRASHES)
sentiment.py:145, regime.py:74 — both crash when called from running event loop

### Missing APIHealthTracker Integration
fetcher.py (MarketDataFetcher), sentiment.py, onchain.py, stream.py

### Empty Response Handling Inconsistency
7 different patterns: None (coincap/messari/santiment), empty DataFrame (fetcher), [] or {} (onchain), {"value": 50} (sentiment), None (stream)

### Hardcoded Relative Paths
- `"./workspace/trading.db"` (database.py:26)
- `"./ft_userdata/config.json"` (config.py:18)
- `"./chroma_db"` (config.py:20)
- `"./ft_userdata"` (setup_data.py:18)

### API Keys in Git-Tracked `.env`
CoinGecko, HuggingFace, Messari, Santiment, CoinCap keys are committed to the repository. Security exposure.
