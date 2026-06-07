# Critical Issues — Consolidated & Prioritized

Generated 2026-06-06. All critical and high-severity issues across all subsystems, prioritized for action.

---

## CRITICAL PRIORITY (active bug, security, or data loss)

| # | Severity | File | Line | Issue |
|---|----------|------|------|-------|
| C1 | CRITICAL | `execution/live_executor.py` | 106 | **OPENAI_API_KEY used as exchange API key** — `"apiKey": settings.OPENAI_API_KEY` sent to exchange as CCXT auth credential |
| C2 | CRITICAL | `backtesting/synthetic_validator.py` | 155-165 | **Synthetic validator is a complete no-op** — generates random-walk data but never passes it to the backtest engine. Runs on real data. All overfitting detection is meaningless |
| C3 | CRITICAL | `backtesting/synthetic_validator.py` | 249-257 | **Permutation test also a no-op** — same bug as C2 |
| C4 | CRITICAL | `monitoring/telegram_alerter.py` | 117-130 | **Human-in-the-loop approvals non-functional** — inline keyboard buttons sent but no callback handler registered, no polling started |
| C5 | CRITICAL | `data/sentiment.py` | 145 | **`asyncio.run()` crashes from async context** — `_fetch_santiment` used in `get_combined_sentiment` crashes with `RuntimeError` when called from running event loop |
| C6 | CRITICAL | `data/regime.py` | 74 | **`asyncio.run()` crashes from async context** — same bug pattern in `_get_social_signal` |
| C7 | CRITICAL | `data/database.py` | 632-637 | **SQL injection in `table_count`** — `f"SELECT COUNT(*) FROM {table_name}"` with direct string interpolation, no sanitization |
| C8 | CRITICAL | `data/patterns.py` | 17 | **CDLENGULFING in both bullish and bearish lists** — signal always neutralized. Most reliable reversal pattern effectively disabled |
| C9 | CRITICAL | `.env` file | — | **API keys committed to git-tracked `.env`** — CoinGecko, HuggingFace, Messari, Santiment, CoinCap keys exposed |

## HIGH PRIORITY (incorrect behavior, silent failure, untested critical path)

| # | Severity | File | Line | Issue |
|---|----------|------|------|-------|
| H1 | HIGH | `agents/risk_manager.py` | 819 | **`assess_strategy_risk` rejects good strategies** — `if risk_score >= 0.5 or not concerns:` means empty concerns (strategy is fine) triggers rejection |
| H2 | HIGH | `agents/analyst.py` | 54-59 | **`sentiment_fn` is a hardcoded stub** — always returns "NEUTRAL (score 0.0/1.0)" with a TODO comment |
| H3 | HIGH | `agents/researcher.py` | 44 | **`_generated_specs` never written** — `get_specs()` returns empty dict forever |
| H4 | HIGH | `execution/live_executor.py` | 217-224 | **PaperTrader state reset on every signal** — new instance created per signal with max_candles=5 |
| H5 | HIGH | `execution/live_executor.py` | 238 | **Paper execution fill_price set to final_balance** — not actual fill price |
| H6 | HIGH | `backtesting/setup_data.py` | 32 | **Data directory hardcoded to "binance" but exchange is Kraken** — file check always fails, triggers unnecessary re-downloads every cycle |
| H7 | HIGH | `orchestration/hermes.py` | 156 | **Board reassigned mid-method** — `self.board = StateGraph(...)` discards previously queued tasks |
| H8 | HIGH | `orchestration/hermes.py` | 325 | **No timeout on LangGraph invoke** — can hang indefinitely |
| H9 | HIGH | `orchestration/deployment_pipeline.py` | 300-306 | **Permutation test runs on real data** — meaningless for overfitting detection (same root cause as C2) |
| H10 | HIGH | `orchestration/deployment_pipeline.py` | 267 | **CPCV evaluates on only 1000 candles** — ~41 days at 1h, not representative |
| H11 | HIGH | `orchestration/deployment_pipeline.py` | 319-320 | **Kelly sizing uses hardcoded 2%/1% avg_win/avg_loss** |
| H12 | HIGH | `data/sentiment.py` | 82-105 | **Two competing weight systems** — `SENTIMENT_WEIGHTS` dict (40/20/25/15) defined but ignored; hardcoded 0.6/0.4 used instead |
| H13 | HIGH | `data/onchain.py` | 56-87 | **"Exchange netflow" is actually volume analysis** — misleading method name = misleading signals |
| H14 | HIGH | `memory/vector_store.py` | 79 | **`hash()` doc IDs non-deterministic across restarts** — Python randomized hash seeds cause duplicates and orphaned docs |
| H15 | HIGH | `data/stream.py` | 72-78 | **WebSocket queue drops messages silently** — maxsize=100, get_nowait() on full |
| H16 | HIGH | `data/stream.py` | 106-123 | **No reconnection logic** — max_reconnects=3 defined but never used |
| H17 | HIGH | `core/event_bus.py` | — | **Stale event loop capture at import time** — crashes if loop replaced; no unsubscribe = memory leak |
| H18 | HIGH | `agents/experiment_tracker.py` | — | **Silent data loss on concurrent writes** — no file locking on experiments.jsonl |
| H19 | HIGH | `monitoring/performance_monitor.py` | 183 | **Rolling Sharpe uses trade-count window, not time window** — "30-day" is actually "last 30 trades" |
| H20 | HIGH | `backtesting/cpcv_validator.py` | 165-181 | **Purge/embargo logic incorrect for multi-fold** — later folds undo earlier removals |

## MEDIUM (design debt, code smell, incomplete feature)

Too numerous to list here. See subsystem-specific audit files for full details.

- `agents/base.py` — no timeout/retry on LLM invoke, IndexError on empty messages, dead get_tool method
- `agents/strategist.py` — duplicated IterationRecord, hardcoded thresholds in 3 places, only 4/11 strategy types parameterized
- `agents/backtester.py` — duplicated evaluation logic, two parallel backtest paths, ignores timeframe param
- `orchestration/graph.py` — global mutable state, fragile ast.literal_eval parsing, hardcoded agent routing strings
- `orchestration/autonomous_loop.py` — off-by-one sleep, no graceful shutdown
- `execution/signal_scanner.py` — deprecated ensure_future, regime gating bypassed, 6/11 strategies unimplemented
- `execution/audit_log.py` — swallows all exceptions on load, inconsistent field name mapping
- `monitoring/anomaly_detector.py` — `_check_negative_kelly` is a no-op
- `data/fetcher.py` — duplicate symbol_to_coincap_id with different behavior, blocks event loop, no health tracking
- `data/coincap_fetcher.py` — only 15 assets mapped, all errors swallowed
- `data/santiment_fetcher.py` — free tier date range always 30+ days old, cache key collision
- `data/database.py` — no connection timeout, no schema versioning, 147-line migrate_jsonl
- `backtesting/oos_validator.py` — inline degradation diverges from static method, pass criteria inconsistency
- `backtesting/blind_search.py` — non-integer TA-Lib params from spread formula
- `backtesting/signal_factory.py` — unvectorized for-loop on 5000+ rows, wrong trade sequences for multi-entry strategies
- `backtesting/cost_model.py` — fee comments inverted relative to market convention
- `backtesting/engine.py` — race condition in strategy file cleanup, Windows-only paths, temp file orphaning

## Architecture-Level Concerns

1. **No agent has timeout or retry on LLM calls** — all inherit bare `_agent.invoke()` from BaseAgent
2. **All agents depend on LLM choosing correct tools** — no validation layer
3. **`VectorStore` instantiated directly in 6 places** — no dependency injection, coupled to ChromaDB
4. **`CircuitBreakerState` is a singleton with no locks** — thread-unsafe
5. **Hardcoded relative paths throughout** — `./workspace/`, `./ft_userdata/`, `./chroma_db` — all break if CWD differs
6. **7 different empty-response patterns** — None, empty DataFrame, [], {}, {"value": 50}
7. **`datetime.utcnow()` deprecated in 20+ locations**
8. **20+ files with zero test coverage**
