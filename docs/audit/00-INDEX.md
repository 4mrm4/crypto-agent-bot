# Codebase Audit — Index

Generated 2026-06-06. Comprehensive limitations analysis across all subsystems.

| File | Coverage | Scope |
|------|----------|-------|
| [01-test-coverage.md](01-test-coverage.md) | 58 modules | Function-level coverage gaps, zero-coverage modules |
| [02-orchestration.md](02-orchestration.md) | 13 files | hermes, autonomous_loop, graph, experiment_tracker, deployment_pipeline, event_bus |
| [03-agents.md](03-agents.md) | 9 files | base, strategist, backtester, researcher, analyst, risk_manager, iteration_tracker, curator, signal_scanner |
| [04-risk-exec-monitor-backtest.md](04-risk-exec-monitor-backtest.md) | ~20 files | portfolio_var, live_executor, paper_trader, audit_log, validation_mode, signal_scanner, anomaly_detector, performance_monitor, telegram_alerter, engine, data_split, cpcv_validator, blind_search, oos_validator, synthetic_validator, signal_factory, cost_model, timerange_utils, setup_data, deployment_pipeline |
| [05-data-layer.md](05-data-layer.md) | ~15 files | fetcher, coincap, messari, santiment, sentiment, regime, onchain, patterns, stream, database, vector_store, api_health, rate_limiter, strategy_concepts, data_split, setup_data, timerange_utils |
| [06-critical-issues.md](06-critical-issues.md) | — | Consolidated critical & high severity issues across all layers, prioritized for action |

## Summary Totals

- **Total files analyzed:** ~60+
- **Bugs found:** ~100+
- **Code smells:** ~80+
- **Design issues:** ~80+
- **Files with zero tests:** ~20
- **Deprecated API usages:** 20+
