# Audit: Test Coverage

Generated 2026-06-06. Function-level coverage analysis of 58 source modules (~847 functions).

## Overall Coverage: ~58%

## Critical Zero-Coverage Modules (no tests exist)

| Module | Risk | Notes |
|--------|------|-------|
| `api/server.py` | HIGH | All 12 API endpoints untested |
| `execution/live_executor.py` | CRITICAL | Live trading path — zero tests |
| `execution/signal_scanner.py` | HIGH | Signal scanning logic untested |
| `execution/audit_log.py` | MEDIUM | Audit persistence untested |
| `execution/trade_signal.py` | MEDIUM | Signal data model untested |
| `execution/price_feed.py` | MEDIUM | Protocol definition untested |
| `agents/analyst.py` | HIGH | Research agent untested |
| `agents/base.py` | HIGH | Foundation agent class untested |
| `agents/curator.py` | MEDIUM | Memory curation untested |
| `agents/iteration_tracker.py` | MEDIUM | Iteration tracking untested |
| `orchestration/hermes.py` | HIGH | Orchestration engine untested |
| `orchestration/graph.py` | HIGH | LangGraph state graph untested |
| `risk/portfolio_manager.py` | MEDIUM | Portfolio logic untested |
| `backtesting/engine.py` | HIGH | _run_freqtrade_backtest, _parse_results untested |
| `backtesting/cost_model.py` | LOW | Cost computation helper untested |
| `backtesting/timerange_utils.py` | LOW | Date parsing untested |
| `backtesting/strategy_templates.py` | LOW | Strategy templates untested |
| `backtesting/setup_data.py` | LOW | Data setup untested |
| `data/onchain.py` | LOW | Onchain data untested |
| `data/sentiment.py` | MEDIUM | Sentiment scoring untested |

## Known-Broken Tests

- `test_cpcv.py` — 3 tests consistently fail (CPCV index mismatch bug — passing but broken)

## Files with Partial Coverage

- `execution/validation_mode.py` — tests exist but miss `apply_position_cap`, `apply_tight_circuit_breaker`
- `data/fetcher.py` — only MultiExchangeFetcher tested through coincap tests; MarketDataFetcher has no direct tests
- `data/database.py` — most CRUD tested but `migrate_jsonl`, `integrity_check`, `table_count`, `clear_all`, `query_*` without strategy_id are untested
- `memory/vector_store.py` — basic CRUD tested, but `store_strategy_result`, `get_best_strategies`, `query_deployable_by_regime`, `tag_as_deployable`, `update_live_performance` all untested
