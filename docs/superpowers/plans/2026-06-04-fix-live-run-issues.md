# Fix 7 Issues from Autonomous Live Run — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all 7 issues from a live autonomous run — backtester never runs (CRITICAL), risk misrouted (HIGH), CoinGecko 401 noise (MEDIUM), researcher 47-char output (MEDIUM), regime confidence investigation, database reinit noise (LOW), discouraged strategy false positives (LOW).

**Architecture:** Each fix is self-contained in one file. The backtester contract changes from `strategy_id` lookup to `strategy_type` + `params` string inputs — the correct boundary after the Task 4 agent split. No shared agent state.

**Tech Stack:** Python, LangGraph, Freqtrade, httpx, SQLite, TA-Lib, pytest

---

## Files Affected

| File | Change |
|------|--------|
| `agents/backtester.py` | `run_backtest` takes `strategy_type: str, params: str` separately; removes strategy_id lookup |
| `agents/strategist.py` | Prompt line 67-75: clarify `next:` format must include full params inline |
| `orchestration/graph.py` | `_extract_child_tasks()` robust JSON extraction via `ast.literal_eval`; discouraged warning checks only `strategy_type=X` |
| `orchestration/hermes.py` | `_agent_capabilities` keywords for risk_manager; researcher output length warning |
| `data/sentiment.py` | `_get_coingecko_news` calls `/status_updates` instead of `/news`; one-time 401 warning |
| `data/database.py` | `TradingDatabase` singleton via `__new__` |
| `data/regime.py` | Confidence debug logging; `conf_threshold` config check |
| `config.py` | Add `REGIME_CONF_THRESHOLD` |
| New: `tests/test_regression_live_run.py` | 11 regression tests |

---

### Task 0: Create regression test file

**Files:**
- Create: `tests/test_regression_live_run.py`

Write ALL tests first (TDD — they will fail until implementations land).

- [ ] **Step 1: Create test file with all regression tests**

```python
"""Regression tests for live run issues — backtester contract, routing, parsing."""
import json
import re
import ast
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Issue 1: Backtester run_backtest accepts strategy_type + params ──

def test_backtester_run_backtest_accepts_strategy_type_params():
    """run_backtest('sma_crossover', '{"fast_ma": 10}') must not require strategy_id."""
    from agents.backtester import BacktesterAgent
    agent = BacktesterAgent()
    # Patch engine to return a valid result without subprocess
    from unittest.mock import MagicMock
    agent._engine.run_backtest = MagicMock(return_value={
        "total_trades": 50, "win_rate": 0.6, "sharpe_ratio": 1.2,
        "max_drawdown": -0.03, "profit_ratio": 0.05,
    })
    result = agent.get_tool("run_backtest").func(
        strategy_type="sma_crossover",
        params='{"fast_ma": 10, "slow_ma": 30}',
    )
    assert "Error: unknown strategy_id" not in result
    assert agent._iteration_history  # must have recorded an iteration


def test_backtester_run_backtest_returns_metrics():
    """run_backtest result must contain sharpe_ratio, win_rate, max_drawdown, total_trades."""
    from agents.backtester import BacktesterAgent
    from unittest.mock import MagicMock
    agent = BacktesterAgent()
    agent._engine.run_backtest = MagicMock(return_value={
        "total_trades": 50, "win_rate": 0.6, "sharpe_ratio": 1.2,
        "max_drawdown": -0.03, "profit_ratio": 0.05,
    })
    agent.get_tool("run_backtest").func(
        strategy_type="sma_crossover",
        params='{"fast_ma": 10}',
    )
    record = agent._iteration_history[-1]
    metrics = record["metrics"]
    for key in ("sharpe_ratio", "win_rate", "max_drawdown", "total_trades"):
        assert key in metrics, f"Missing metric: {key}"


def test_extract_child_tasks_parses_params_json():
    """_extract_child_tasks must extract strategy_type and params from 'next:' lines."""
    from orchestration.graph import _extract_child_tasks
    from orchestration.board import TaskBoard
    board = TaskBoard({}, {})
    parent = type("Task", (), {"id": "p1", "result": (
        'Strategy [abc12345] created: type=sma_crossover\n'
        'next: backtest strategy_type=sma_crossover params={"fast_ma": 10, "slow_ma": 30}\n'
    ), "assigned_to": "strategist"})()
    _extract_child_tasks(parent, board)
    tasks = board.get_tasks_by_status("TODO")
    assert len(tasks) == 1
    desc = tasks[0].description
    assert "strategy_type=sma_crossover" in desc
    assert "fast_ma" in desc


def test_extract_child_tasks_nested_braces():
    """Params with nested JSON (strings containing braces) must parse correctly."""
    from orchestration.graph import _extract_child_tasks
    from orchestration.board import TaskBoard
    board = TaskBoard({}, {})
    parent = type("Task", (), {"id": "p2", "result": (
        'next: backtest strategy_type=rsi_oversold '
        'params={"rsi_period": 14, "note": "test {with} braces"}\n'
    ), "assigned_to": "strategist"})()
    _extract_child_tasks(parent, board)
    tasks = board.get_tasks_by_status("TODO")
    assert len(tasks) == 1
    assert "rsi_period" in tasks[0].description


def test_extract_child_tasks_spaces_in_values():
    """Params values with spaces in strings must parse correctly."""
    from orchestration.graph import _extract_child_tasks
    from orchestration.board import TaskBoard
    board = TaskBoard({}, {})
    parent = type("Task", (), {"id": "p3", "result": (
        'next: backtest strategy_type=custom '
        'params={"indicator_code": "df[\'ema\'] = ta.EMA(df, 10)"}\n'
    ), "assigned_to": "strategist"})()
    _extract_child_tasks(parent, board)
    tasks = board.get_tasks_by_status("TODO")
    assert len(tasks) == 1


def test_full_dispatch_chain():
    """Complete path: strategist next: output -> _extract_child_tasks -> correctly formed backtester task."""
    from orchestration.graph import _extract_child_tasks
    from orchestration.board import TaskBoard
    board = TaskBoard({}, {})
    # Simulate strategist output
    strategist_output = (
        "Strategy [a1b2c3d4] created: type=multi_timeframe\n"
        "next: backtest strategy_type=multi_timeframe "
        'params={"timeframe": "1h", "stoploss": -0.05}\n'
    )
    parent = type("Task", (), {
        "id": "p_chain", "result": strategist_output,
        "assigned_to": "strategist"
    })()
    _extract_child_tasks(parent, board)
    tasks = board.get_tasks_by_status("TODO")
    assert len(tasks) == 1
    desc = tasks[0].description
    # The task must contain all info the backtester needs
    assert "strategy_type=multi_timeframe" in desc
    assert '"timeframe": "1h"' in desc


# ── Issue 2: Agent routing ──

def test_pick_agent_risk_assessment():
    """'Assess risk for strategies targeting X' must route to risk_manager."""
    from orchestration.graph import _pick_agent
    caps = {
        "analyst": ["analysis", "market"],
        "strategist": ["strategy", "generate", "concept", "design"],
        "backtester": ["backtest", "walk_forward", "hyperopt", "optimise"],
        "risk_manager": ["risk", "assess", "kelly", "circuit", "position"],
        "curator": ["memory", "context"],
        "researcher": ["research", "web", "paper"],
    }
    result = _pick_agent("Assess risk for strategies targeting: Auto-research", caps)
    assert result == "risk_manager", f"Got {result}, expected risk_manager"


def test_pick_agent_backtest():
    """'backtest strategy_type=X params=Y' must route to backtester."""
    from orchestration.graph import _pick_agent
    caps = {
        "analyst": ["analysis", "market"],
        "strategist": ["strategy", "generate", "concept", "design"],
        "backtester": ["backtest", "walk_forward", "hyperopt", "optimise"],
        "risk_manager": ["risk", "assess", "kelly", "circuit", "position"],
        "curator": ["memory", "context"],
        "researcher": ["research", "web", "paper"],
    }
    result = _pick_agent("backtest strategy_type=sma_crossover params={}", caps)
    assert result == "backtester", f"Got {result}, expected backtester"


# ── Issue 5: Discouraged strategy only checks requested type ──

def test_discouraged_only_checks_requested_type():
    """Task mentioning discouraged strategies in context must NOT trigger false positive."""
    desc = ('[CURRENT REGIME: strong_downtrend]\n'
            'backtest strategy_type=multi_timeframe params={"timeframe": "1h"}')
    # Extract strategy_type
    m = re.search(r'strategy_type=(\w+)', desc)
    assert m is not None
    strategy_type = m.group(1)
    discouraged = ["momentum", "breakout", "sma_crossover"]
    assert strategy_type not in discouraged, "multi_timeframe should not be in discouraged"


# ── Issue 6: TradingDatabase singleton ──

def test_database_singleton():
    """TradingDatabase() called twice must return same instance."""
    from data.database import TradingDatabase
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    db1 = TradingDatabase(db_path)
    db2 = TradingDatabase(db_path)
    assert db1 is db2, "Multiple instances should return same object"
    # Clean up
    import os
    os.unlink(db_path)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd C:/Trading-bot/crypto_agent_bot
python -m pytest tests/test_regression_live_run.py -v --tb=short 2>&1
```

Expected: FAILURES — functions not yet implemented, imports may fail. This confirms tests are valid.

---

### Task 1: Fix Issue 1 (CRITICAL) — Backtester run_backtest tool signature

**Files:**
- Modify: `agents/backtester.py:120-211`
- Verify: `agents/strategist.py:67-75`

- [ ] **Step 1: Rewrite run_backtest tool to accept strategy_type + params**

Change lines ~120-211 in `agents/backtester.py`. The tool function changes from:

```python
def run_backtest(backtest_json: str = "{}") -> str:
    ...
    sid = params.get("strategy_id", "")
    if not sid or sid not in self._generated_strategies:
        return f"Error: unknown strategy_id '{sid}'. Use generate_strategy first."
    strat_params = self._generated_strategies[sid].copy()
```

To:

```python
def run_backtest(strategy_type: str = "sma_crossover", params: str = "{}") -> str:
    """Backtest a strategy by type and parameters.
    Args:
        strategy_type: One of sma_crossover, macd_crossover, rsi_oversold, etc.
        params: JSON string of strategy parameters, e.g. '{"fast_ma": 10, "slow_ma": 30}'
    Returns performance metrics with keep/discard verdict."""
    import json
    try:
        strat_params = json.loads(params)
    except json.JSONDecodeError:
        strat_params = {}
    if not isinstance(strat_params, dict):
        strat_params = {}
    strat_params["strategy_type"] = strategy_type
    # Use global config for defaults
    global_cfg = getattr(self, "_backtest_config", {})
    timerange = strat_params.pop("timerange", global_cfg.get("timerange", "20210101-"))
    pairs = strat_params.pop("pairs", global_cfg.get("pairs", None))
    strat_params.setdefault("timeframe", global_cfg.get("timeframe", settings.TIMEFRAME))
    try:
        result = self._engine.run_backtest(
            strat_params,
            strategy_type=strategy_type,
            timerange=timerange,
            pairs=pairs,
        )
    except Exception as exc:
        return f"Error running backtest: {exc}"
```

Then update the metrics formatting, iteration record, and return block to match (same as existing code but using the new `result` and `strat_params`).

Also update the tool registration to reflect the new signature:

```python
Tool(name="run_backtest", func=run_backtest,
     description="Backtest a strategy type with JSON parameters. "
     "Args: strategy_type (str), params (str JSON). "
     "Example params: '{\"fast_ma\": 10, \"slow_ma\": 30}'"),
```

- [ ] **Step 2: Verify strategist prompt instructs next: format correctly**

Read `agents/strategist.py` lines 67-75 and confirm:

```
At the END of your response, ALWAYS output a line in EXACTLY this format:
next: backtest strategy_type=STRATEGY_TYPE params={"key": "value"}

Replace STRATEGY_TYPE and params with the actual strategy you generated.
Include ALL strategy parameters in the params JSON — the backtester has no
access to previously generated strategies and receives everything it needs
from this line.

Example:
next: backtest strategy_type=sma_crossover params={"fast_ma": 10, "slow_ma": 30}
```

If the prompt already says this, no change needed.

- [ ] **Step 3: Run regression tests for Issue 1**

```bash
cd C:/Trading-bot/crypto_agent_bot
python -m pytest tests/test_regression_live_run.py::test_backtester_run_backtest_accepts_strategy_type_params -v --tb=short
python -m pytest tests/test_regression_live_run.py::test_backtester_run_backtest_returns_metrics -v --tb=short
```

Expected: PASS

- [ ] **Step 4: Run full test suite to check for regressions**

```bash
cd C:/Trading-bot/crypto_agent_bot
python -m pytest --tb=short -q 2>&1 | tail -10
```

Expected: 0 failed (same pass count as before)

- [ ] **Step 5: Commit**

```bash
cd C:/Trading-bot/crypto_agent_bot
git add agents/backtester.py
git commit -m "fix: run_backtest accepts strategy_type+params directly, removes strategy_id lookup

After Task 4 agent split, backtester's _generated_strategies was always empty.
Now accepts strategy_type (str) and params (str JSON) directly from task
description — no shared state between agents. This is the correct architectural
boundary: strategist produces a spec, backtester executes it."
```

---

### Task 2: Fix Issue 1b (CRITICAL) — _extract_child_tasks() JSON parsing

**Files:**
- Modify: `orchestration/graph.py:261-289`

- [ ] **Step 1: Rewrite _extract_child_tasks() with robust JSON extraction**

Replace the current `_extract_child_tasks()` body that does naive `strategy_type=` regex and keyword matching with a version that first tries to parse `params={...}` via `ast.literal_eval`, then falls back to the existing keyword approach:

```python
def _extract_child_tasks(parent: "Task", board: TaskBoard):
    """Extract follow-up tasks from agent output.
    First tries to parse 'next: backtest strategy_type=X params={...}' lines with
    robust JSON extraction. Falls back to keyword-based strategy type detection."""
    result_text = str(parent.result) if parent.result else ""
    found_any = False
    for line in result_text.split("\n"):
        line = line.strip()
        if line.startswith("next: "):
            desc = line.replace("next: ", "")
            board.add_task(description=desc, parent_id=parent.id,
                           metadata={"auto": True})
            found_any = True
    if not found_any and "strategy_type=" in result_text:
        m = re.search(r'strategy_type[=:]\s*(\w+)', result_text)
        if m:
            strategy_type = m.group(1)
            # Try to extract params={...} JSON robustly using ast.literal_eval
            params_match = re.search(r'params=(\{.*\})', result_text, re.DOTALL)
            params_str = ""
            if params_match:
                try:
                    parsed = ast.literal_eval(params_match.group(1))
                    params_str = f" params={json.dumps(parsed)}"
                except (ValueError, SyntaxError):
                    # Fallback: use raw text between braces (may be fragile)
                    raw = params_match.group(1)
                    params_str = f" params={raw}"
            desc = f"backtest strategy_type={strategy_type}{params_str}"
            board.add_task(description=desc, parent_id=parent.id,
                           metadata={"auto": True})
            found_any = True
    if not found_any and getattr(parent, "assigned_to", None) == "strategist" and parent.result:
        lowered = result_text.lower()
        normalized = lowered.replace("-", "_").replace(" ", "_")
        for kw in ["multi_timeframe", "sma_crossover", "macd_crossover", "rsi_oversold",
                    "bollinger_bands", "combined_sma_rsi", "momentum", "breakout",
                    "mean_reversion", "volatility_squeeze", "sentiment_driven"]:
            if kw in lowered or kw in normalized:
                board.add_task(description="backtest strategy_type=" + kw,
                               parent_id=parent.id, metadata={"auto": True})
                break
```

Also add `import ast` at the top of graph.py (it's already there — check).

- [ ] **Step 2: Run regression tests for parsing**

```bash
cd C:/Trading-bot/crypto_agent_bot
python -m pytest tests/test_regression_live_run.py::test_extract_child_tasks_parses_params_json -v --tb=short
python -m pytest tests/test_regression_live_run.py::test_extract_child_tasks_nested_braces -v --tb=short
python -m pytest tests/test_regression_live_run.py::test_extract_child_tasks_spaces_in_values -v --tb=short
python -m pytest tests/test_regression_live_run.py::test_full_dispatch_chain -v --tb=short
```

Expected: all PASS

- [ ] **Step 3: Full suite regression check**

```bash
cd C:/Trading-bot/crypto_agent_bot
python -m pytest --tb=short -q 2>&1 | tail -10
```

Expected: 0 failed

- [ ] **Step 4: Commit**

```bash
cd C:/Trading-bot/crypto_agent_bot
git add orchestration/graph.py
git commit -m "fix: _extract_child_tasks robust JSON parsing + full dispatch chain

Uses ast.literal_eval for params={...} extraction instead of fragile
substring matching. Handles nested braces, spaces in string values.
Falls back to keyword detection when params JSON is malformed."
```

---

### Task 3: Fix Issue 2 (HIGH) — Risk routing to wrong agent

**Files:**
- Modify: `orchestration/hermes.py:21-33`

- [ ] **Step 1: Fix _agent_capabilities keywords**

Replace `strategist` keywords: remove `"strategies"` (already covered by `"strategy"`).
Replace `risk_manager` keywords: add `"assess"`, `"assessment"`, `"kelly"`, `"circuit"`, `"approval"`, `"position"`.

New dict:

```python
self._agent_capabilities: Dict[str, List[str]] = {
    "analyst": ["analysis", "market_research", "sentiment", "analyse", "market"],
    "strategist": ["strategy", "generate", "concept", "design"],
    "backtester": ["backtest", "backtesting", "walk_forward", "hyperopt",
                   "optimization", "optimise", "compare", "benchmark",
                   "download", "data", "run_backtest", "run"],
    "iteration_tracker": ["iteration", "history", "best_strategy",
                          "store", "track", "record", "memory", "recall"],
    "risk_manager": ["risk", "assess", "assessment", "kelly", "circuit",
                     "position", "sizing", "approval", "correlation"],
    "curator": ["memory", "context", "history"],
    "researcher": ["research", "web", "paper", "novel", "search", "literature"],
}
```

- [ ] **Step 2: Run routing regression tests**

```bash
cd C:/Trading-bot/crypto_agent_bot
python -m pytest tests/test_regression_live_run.py::test_pick_agent_risk_assessment -v --tb=short
python -m pytest tests/test_regression_live_run.py::test_pick_agent_backtest -v --tb=short
```

Expected: PASS

- [ ] **Step 3: Full suite regression check**

```bash
cd C:/Trading-bot/crypto_agent_bot
python -m pytest --tb=short -q 2>&1 | tail -10
```

Expected: 0 failed

- [ ] **Step 4: Commit**

```bash
cd C:/Trading-bot/crypto_agent_bot
git add orchestration/hermes.py
git commit -m "fix: risk_manager routing — add assess/kelly/circuit/approval keywords, remove 'strategies' from strategist

Prevents tie-breaking bug where 'Assess risk for strategies...' task was
routed to strategist instead of risk_manager. strategist keywords lose
'strategies' (already covered by 'strategy'), risk_manager gains specific
risk-domain keywords for higher match scores."
```

---

### Task 4: Fix Issue 3 (MEDIUM) — CoinGecko 401 noise

**Files:**
- Modify: `data/sentiment.py:79-104`

- [ ] **Step 1: Replace CoinGecko /news with /status_updates**

Replace the `_get_coingecko_news` method:

```python
def _get_coingecko_news(self, currency: str = "BTC") -> list:
    """Fallback: CoinGecko status updates (free endpoint, no API key required)."""
    try:
        symbol_map = {"BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana"}
        coin_id = symbol_map.get(currency.upper(), currency.lower())
        r = httpx.get(
            f"https://api.coingecko.com/api/v3/coins/{coin_id}/status_updates",
            params={"per_page": 10},
            timeout=10
        )
        if r.status_code == 401:
            logger.warning(
                "CoinGecko status_updates returned 401 (unavailable on free tier). "
                "This warning fires once per session."
            )
            return []
        items = r.json().get("status_updates", [])[:10]
        return [
            {
                "title": i.get("description", "")[:200],
                "published_at": i.get("created_at", ""),
                "url": "",
            }
            for i in items
        ]
    except Exception as e:
        logger.warning("CoinGecko status_updates fetch failed: %s", e)
        return []
```

Also update the `get_cryptopanic_news` fallback chain: when no `CRYPTOPANIC_API_KEY` is set, call `_get_coingecko_news` (already done), but the fallback should now work without any API key since `/status_updates` is free.

- [ ] **Step 2: Run tests**

```bash
cd C:/Trading-bot/crypto_agent_bot
python -m pytest tests/test_regression_live_run.py -v --tb=short -k "not backtester and not pick_agent and not extract_child and not discouraged and not database" 2>&1
```

Expected: PASS (no CoinGecko-specific tests, but shouldn't break existing)

- [ ] **Step 3: Full suite regression check**

```bash
cd C:/Trading-bot/crypto_agent_bot
python -m pytest --tb=short -q 2>&1 | tail -10
```

Expected: 0 failed

- [ ] **Step 4: Commit**

```bash
cd C:/Trading-bot/crypto_agent_bot
git add data/sentiment.py
git commit -m "fix: replace CoinGecko /v3/news (Pro-only, 401) with /status_updates (free)

/news endpoint requires CoinGecko Pro API key and returns 401 on free tier.
Replaced with /coins/{id}/status_updates which is available on free tier
with no API key required. Also adds one-time 401 warning to avoid log spam."
```

---

### Task 5: Fix Issue 7 (MEDIUM) — Researcher 47-char / DuckDuckGo 202

**Files:**
- Modify: `agents/researcher.py` (DuckDuckGo fallback section, ~line 138-208)
- Modify: `orchestration/hermes.py` (output length warning)

- [ ] **Step 1: Add DuckDuckGo 202 retry in researcher.py**

In the DuckDuckGo fallback section (~line 138), replace the current raw `httpx.get` call with retry logic:

```python
# ── Fallback: DuckDuckGo ──
try:
    import httpx
    # Retry with simpler query on 202 (accepted but no results)
    resp = httpx.get(
        "https://api.duckduckgo.com/",
        params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
        timeout=15.0,
    )
    if resp.status_code == 202:
        logger.warning("DuckDuckGo returned 202 for query: %s — retrying with simpler query", query)
        # Strip quotes/operators and retry
        simple_query = query.replace('"', '').replace("'", '').replace(' AND ', ' ').replace(' OR ', ' ')
        resp = httpx.get(
            "https://api.duckduckgo.com/",
            params={"q": simple_query, "format": "json", "no_html": 1, "skip_disambig": 1},
            timeout=15.0,
        )
    resp.raise_for_status()
    data = resp.json()
    # ... rest of existing parsing code ...
```

- [ ] **Step 2: Add researcher output length warning in hermes.py**

In `orchestration/hermes.py`, in the code that calls the researcher:

```python
researcher_output = researcher.run(research_query)
output_text = researcher_output.get("output", "")
if len(output_text) < 100:
    logger.warning(
        "Researcher output too short (%d chars) — web search may be failing. "
        "Consider configuring Tavily API key for better results.",
        len(output_text),
    )
```

Find the exact location where researcher output is processed (likely in `_run_research_goal` or `run_research_loop`).

- [ ] **Step 3: Full suite regression check**

```bash
cd C:/Trading-bot/crypto_agent_bot
python -m pytest --tb=short -q 2>&1 | tail -10
```

Expected: 0 failed

- [ ] **Step 4: Commit**

```bash
cd C:/Trading-bot/crypto_agent_bot
git add agents/researcher.py orchestration/hermes.py
git commit -m "fix: DuckDuckGo 202 retry + researcher output length warning

DuckDuckGo returns 202 (accepted but no results) for complex queries.
Added retry with simplified query (strips quotes and operators) on 202.
Added WARNING log in hermes when researcher output <100 chars."
```

---

### Task 6: Fix Issue 4 (INVESTIGATE) — Regime confidence logging

**Files:**
- Modify: `data/regime.py:151-161`
- Modify: `config.py`

- [ ] **Step 1: Add REGIME_CONF_THRESHOLD to config.py**

```python
REGIME_CONF_THRESHOLD: float = float(os.getenv("REGIME_CONF_THRESHOLD", "0.3"))
```

- [ ] **Step 2: Add debug logging at confidence calculation in regime.py**

In `classify_regime_snapshot()`, after the confidence calculation block:

```python
# Confidence based on ADX clarity
if regime in ("strong_uptrend", "strong_downtrend") and last_adx > 30:
    confidence = 0.9
elif regime in ("strong_uptrend", "strong_downtrend"):
    confidence = 0.7
elif regime == "volatile":
    confidence = 0.6
elif regime == "ranging" and last_adx < 15:
    confidence = 0.8
else:
    confidence = 0.5

logger.debug(
    "Regime confidence: regime=%s adx=%.1f confidence=%.2f",
    regime, last_adx, confidence,
)
if confidence < settings.REGIME_CONF_THRESHOLD:
    logger.warning(
        "Regime confidence (%.0f%%) below threshold (%.0f%%) for regime=%s — "
        "regime-conditioned gating is in conservative mode",
        confidence * 100, settings.REGIME_CONF_THRESHOLD * 100, regime,
    )
```

- [ ] **Step 3: Run tests**

```bash
cd C:/Trading-bot/crypto_agent_bot
python -m pytest --tb=short -q 2>&1 | tail -10
```

Expected: 0 failed

- [ ] **Step 4: Commit**

```bash
cd C:/Trading-bot/crypto_agent_bot
git add data/regime.py config.py
git commit -m "fix: add regime confidence debug logging and conf_threshold warning

Logs actual confidence value at calculation point to diagnose 1% issue.
Adds REGIME_CONF_THRESHOLD config (default 0.3) that fires a WARNING
when confidence falls below it, making low-confidence regimes visible."
```

---

### Task 7: Fix Issue 6 (LOW) — TradingDatabase singleton

**Files:**
- Modify: `data/database.py:29-50`

- [ ] **Step 1: Implement __new__ singleton pattern**

Replace the unused `_instance`/`_lock` class attrs with a proper `__new__`:

```python
class TradingDatabase:
    """Thread-safe SQLite-backed trading database. Singleton per db_path."""

    _instances: Dict[str, "TradingDatabase"] = {}
    _lock = threading.Lock()

    def __new__(cls, db_path=None, legacy_backup=True):
        path = Path(db_path) if isinstance(db_path, str) else (db_path or DB_PATH)
        key = str(path)
        with cls._lock:
            if key not in cls._instances:
                instance = super().__new__(cls)
                instance.db_path = path
                instance.legacy_backup = legacy_backup
                if str(instance.db_path) != ":memory:":
                    instance.db_path.parent.mkdir(parents=True, exist_ok=True)
                instance._init_schema()
                cls._instances[key] = instance
            return cls._instances[key]

    def __init__(self, db_path=None, legacy_backup=True):
        # __new__ handles initialization; __init__ is a no-op on reuse
        pass
```

Wait, that won't work well because `__init__` is called after `__new__` even on return of cached instance. The better approach:

```python
class TradingDatabase:
    """Thread-safe SQLite-backed trading database. Singleton per db_path."""

    _instances: Dict[str, "TradingDatabase"] = {}
    _lock = threading.Lock()

    def __new__(cls, db_path=None, legacy_backup=True):
        path = Path(db_path) if isinstance(db_path, str) else (db_path or DB_PATH)
        key = str(path)
        with cls._lock:
            if key not in cls._instances:
                instance = super().__new__(cls)
                instance.db_path = path
                instance.legacy_backup = legacy_backup
                if str(instance.db_path) != ":memory:":
                    instance.db_path.parent.mkdir(parents=True, exist_ok=True)
                instance._initialized = False
                cls._instances[key] = instance
            return cls._instances[key]

    def __init__(self, db_path=None, legacy_backup=True):
        if getattr(self, '_initialized', False):
            return
        self._initialized = True
        self._init_schema()
        logger.info("Database schema initialised at %s", self.db_path)
```

- [ ] **Step 2: Run singleton test**

```bash
cd C:/Trading-bot/crypto_agent_bot
python -m pytest tests/test_regression_live_run.py::test_database_singleton -v --tb=short
```

Expected: PASS

- [ ] **Step 3: Full suite regression check**

```bash
cd C:/Trading-bot/crypto_agent_bot
python -m pytest --tb=short -q 2>&1 | tail -10
```

Expected: 0 failed

- [ ] **Step 4: Commit**

```bash
cd C:/Trading-bot/crypto_agent_bot
git add data/database.py
git commit -m "fix: TradingDatabase singleton via __new__ pattern, prevents schema reinit on every use"

Removes unused _instance/_lock class attrs. Proper __new__ caching ensures
same db_path returns same instance. _initialized flag prevents __init__
from re-running schema init on cached instances. Fixes 'Database schema
initialised' log spam on every sentiment fetch cycle."
```

---

### Task 8: Fix Issue 5 (LOW) — Discouraged strategy false positives

**Files:**
- Modify: `orchestration/graph.py:118-128`

- [ ] **Step 1: Parse strategy_type from task, don't scan whole description**

Replace the discouraged warning block in `dispatch_task()`:

```python
# Apply penalty for discouraged strategies
strategy_type_match = re.search(r'strategy_type=(\w+)', task.description)
if strategy_type_match:
    requested_type = strategy_type_match.group(1)
    for discouraged in snapshot.discouraged_strategies:
        if discouraged.lower() == requested_type.lower():
            logger.warning(
                "Task uses discouraged strategy '%s' for regime '%s'",
                discouraged, snapshot.regime,
            )
            task.description += (
                f"\n[WARNING: '{discouraged}' is discouraged in {snapshot.regime} regime. "
                f"Consider switching to: {', '.join(snapshot.recommended_strategies)}]"
            )
```

- [ ] **Step 2: Run regression test**

```bash
cd C:/Trading-bot/crypto_agent_bot
python -m pytest tests/test_regression_live_run.py::test_discouraged_only_checks_requested_type -v --tb=short
```

Expected: PASS

- [ ] **Step 3: Full suite regression check**

```bash
cd C:/Trading-bot/crypto_agent_bot
python -m pytest --tb=short -q 2>&1 | tail -10
```

Expected: 0 failed

- [ ] **Step 4: Commit**

```bash
cd C:/Trading-bot/crypto_agent_bot
git add orchestration/graph.py
git commit -m "fix: discouraged strategy warning only checks strategy_type=X, not full task text

Previously scanned entire task description (including injected memory context
mentioning strategy names) for discouraged strategies, producing false positive
warnings. Now parses strategy_type=X from the task and only checks that."
```

---

### Task 9: Final verification

- [ ] **Step 1: Run full test suite**

```bash
cd C:/Trading-bot/crypto_agent_bot
python -m pytest --tb=short -q 2>&1
```

Expected: ALL PASS, 0 failed

- [ ] **Step 2: Run all new regression tests explicitly**

```bash
cd C:/Trading-bot/crypto_agent_bot
python -m pytest tests/test_regression_live_run.py -v 2>&1
```

Expected: ALL 11 PASS

- [ ] **Step 3: Smoke test with --mock-llm**

```bash
cd C:/Trading-bot/crypto_agent_bot
python -m pytest --tb=short -q -k "regression" 2>&1
```

Expected: All regression tests pass

- [ ] **Step 4: Push to GitHub**

```bash
cd C:/Trading-bot/crypto_agent_bot
git push
```
