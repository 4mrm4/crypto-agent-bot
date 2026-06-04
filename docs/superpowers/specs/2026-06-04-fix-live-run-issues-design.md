# Fix 7 Issues from Autonomous Live Run — Design Doc

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all 7 issues observed in a live `--ui --autonomous` run — the backtester never runs, risk is misrouted, CoinGecko 401 noise, regime confidence display, discouraged strategy false positives, database reinit noise, and researcher 47-char output.

**Architecture:** Fixes are localized to individual files following the existing patterns. The backtester contract changes from `strategy_id` lookup to direct `strategy_type+params` input — this is the correct architectural boundary after the Task 4 agent split. All other fixes are minimal targeted changes with no design changes.

**Tech Stack:** Python, LangGraph, Freqtrade, httpx, SQLite, TA-Lib

---

## Files Changed

| File | Change |
|------|--------|
| `agents/backtester.py` | `run_backtest` accepts `strategy_type`+`params` directly; strategy_id lookup removed |
| `agents/strategist.py` | Prompt clarified: `next:` line must include full JSON params, not reference ID |
| `orchestration/graph.py` | `_extract_child_tasks()` uses `ast.literal_eval` for robust JSON parsing; discouraged warning checks only `strategy_type=X` |
| `orchestration/hermes.py` | Researcher output length check; `_agent_capabilities` keywords fixed for risk routing |
| `data/sentiment.py` | CoinGecko news replaced with status_updates; singleton pattern for SentimentFetcher |
| `data/database.py` | `TradingDatabase` singleton enforcement |
| `data/regime.py` | Confidence debug logging; `conf_threshold` config check |
| `config.py` | Add `REGIME_CONF_THRESHOLD` setting |
| New: `test_regression_live_run.py` | Tests for backtester contract, extraction parsing, routing, pre-filter logging |

---

## Issue 1 (CRITICAL): Backtester Never Runs

**Root cause:** After Task 4 split, `run_backtest` tool requires `strategy_id` looked up in `backtester._generated_strategies`, which is always empty (strategist stores strategies in its own instance). Task description only carries `strategy_type=X params={...}`.

**Fix:** Replace `run_backtest` tool with a new signature that takes `strategy_type` and `params` directly. Remove strategy_id lookup. The BacktestEngine already accepts `(strategy_params, strategy_type, timerange, pairs)` directly — this was a wrapper issue, not an engine issue.

New tool signature:
```python
def run_backtest(params_json: str) -> str:
    """Run backtest for a strategy type with parameters.
    Pass JSON: {"strategy_type": "sma_crossover", "params": {"fast_ma": 10, "slow_ma": 30}}
    Returns performance metrics with keep/discard verdict.
    Also accepts flat format: {"strategy_type": "sma_crossover", "fast_ma": 10, "slow_ma": 30}
    """
```

Also add a log line at the START of `BacktestEngine.run_backtest()` to immediately confirm subprocess calls.

## Issue 1b: _extract_child_tasks() JSON Parsing

**Fix:** `_extract_child_tasks()` in graph.py currently takes everything after `next:` literally. For robust JSON extraction from unstructured text, use `ast.literal_eval()` to extract the `params={...}` block. Handle nested braces and spaces by finding brace pairs rather than naive substring matching.

```python
import ast
m = re.search(r'params=(\{.*\})', line, re.DOTALL)
if m:
    try:
        params = ast.literal_eval(m.group(1))
        desc = f"backtest strategy_type={strategy_type} params={json.dumps(params)}"
    except (ValueError, SyntaxError):
        # fallback to raw text
        desc = line
```

## Issue 2 (HIGH): Risk Misrouted

**Root cause:** `_pick_agent()` tie-breaking — "strategies" matches strategist keywords, "risk" matches risk_manager. Dict iteration order puts strategist first.

**Fix:** Two changes:
1. Add keywords to `risk_manager`: `"assess", "assessment", "position_sizing", "risk", "approval", "kelly", "circuit"` — ensures higher score for risk tasks
2. Remove `"strategies"` from strategist keywords (it's a substring of "strategy" which is already there) — reduces false matches

## Issue 3 (MEDIUM): CoinGecko 401

**Root cause:** `_get_coingecko_news()` calls Pro-only `/api/v3/news` endpoint.

**Fix:** Replace with free `/api/v3/coins/{id}/status_updates` endpoint (no auth required). Also add a one-time warning log so it fires once per session, not every cycle.

## Issue 4 (INVESTIGATE): regime confidence=1%

**Root cause:** Unknown from code reading alone — all code paths in `classify_regime_snapshot()` produce {0.9, 0.7, 0.6, 0.8, 0.5}. Likely the running code differs from current file state.

**Fix:** Add `conf_threshold` config var (default 0.3). Add WARNING when regime confidence falls below it: `"Regime confidence {:.0%} below threshold {:.0%}"`. Add debug log at confidence calculation point.

## Issue 5 (LOW): Discouraged Strategy False Positives

**Root cause:** Warning fires on the full task description string, which includes injected memory context mentioning strategy names. Not just the requested `strategy_type=X`.

**Fix:** Extract `strategy_type=X` from task description with regex. Only check the extracted type against discouraged list.

## Issue 6 (LOW): Database Reinit Noise

**Root cause:** `TradingDatabase.__init__()` has `_instance`/`_lock` class attrs but never checks them — no singleton enforcement.

**Fix:** Add `_instance` check in `__init__`:
```python
if TradingDatabase._instance is not None:
    self.db_path = TradingDatabase._instance.db_path
    return  # skip re-init
TradingDatabase._instance = self
```

## Issue 7 (MEDIUM): Researcher 47-char Output

**Root cause:** DuckDuckGo returns 202 (no results) for most queries. 47 chars = "No results found for that query."

**Fix:** Add retry in `web_search` tool: on 202, strip quotes/operators and retry with simpler query. Also add warning in hermes.py when researcher output < 100 chars.

---

## Test Strategy

New file `tests/test_regression_live_run.py`:

1. `test_backtester_run_backtest_accepts_strategy_type_params` — direct call with strategy_type+params returns dict with sharpe_ratio, win_rate, max_drawdown, total_trades
2. `test_backtester_run_backtest_without_strategy_id` — no strategy_id in params does NOT raise error
3. `test_extract_child_tasks_parses_params_json` — "next: backtest strategy_type=sma_crossover params={\"fast_ma\": 10}" extracts correct params dict
4. `test_extract_child_tasks_nested_braces` — params with nested dicts/strings handle correctly
5. `test_extract_child_tasks_spaces_in_values` — params values with spaces parsed correctly
6. `test_pick_agent_risk_assessment` — "Assess risk for strategies targeting X" returns "risk_manager"
7. `test_pick_agent_backtest` — "backtest strategy_type=X params=Y" returns "backtester"
8. `test_pick_agent_generate_strategy` — "Generate strategies for X" returns "strategist"
9. `test_discouraged_strategy_only_checks_requested_type` — task mentioning discouraged names in context does NOT trigger warning
10. `test_database_singleton` — TradingDatabase() called twice does not double-init

---

## Execution Order

1. Issue 1 + 1b (CRITICAL) — backtester contract + extraction parsing
2. Issue 2 (HIGH) — risk routing fix
3. Issue 3 (MEDIUM) — CoinGecko 401
4. Issue 7 (MEDIUM) — researcher 47-char / DuckDuckGo 202
5. Issue 4 (INVESTIGATE) — regime confidence logging
6. Issue 6 (LOW) — database singleton
7. Issue 5 (LOW) — discouraged strategy false positives
8. Final verification — full test suite, demo smoke test
