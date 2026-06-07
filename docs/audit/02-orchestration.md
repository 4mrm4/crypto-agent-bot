# Audit: Orchestration Layer

Generated 2026-06-06. Analyzed 13 files, ~85 issues found.

## `orchestration/hermes.py` (~15 issues)

### Bugs
- **Board reassigned mid-method (line 156):** `self.board = StateGraph(MarketIntelligence)` creates a NEW board mid-method, discarding previously queued tasks
- **`_extract_metrics` Level 3 missing `return` (lines 556-589):** After the inner loop processes experiments.jsonl, there's no `return` — falls through to default 0 values

### Design Issues
- **No timeout on `self._graph.invoke(initial_state)` (line 325):** LangGraph invoke can hang indefinitely with no timeout
- **`CircuitBreakerState._research_mode` mutated via private attribute access (lines 153, 327):** Fragile — refactoring CircuitBreakerState breaks hermes
- **Hardcoded `"./workspace/experiments.jsonl"` path (line 566):** Breaks if CWD differs
- **Deprecated `datetime.utcnow()` usage**
- **Regex `r'[Rr]esearching\s+(\w+)'` brittleness (line 561):** Fragile parsing of LLM output

## `orchestration/autonomous_loop.py`

### Bugs
- **Off-by-one in sleep timing:** Sleeps _after_ first iteration check, not before — delays first cycle

### Design Issues
- **No graceful shutdown during active cycle:** Can leave state inconsistent
- **No circuit breaker state persistence across restarts**

## `orchestration/graph.py`

### Bugs
- **Global mutable state (`_agent_executors`):** Not thread-safe
- **No timeout on individual agent execution:** One slow agent blocks the entire graph
- **Fragile child task parsing:** `ast.literal_eval(params)` silently swallows malformed JSON
- **Child task routing uses hardcoded agent name strings:** Brittle

## `agents/experiment_tracker.py`

### Bugs
- **Silent data loss on concurrent writes:** No file locking on `experiments.jsonl`
- **Non-transactional writes:** Partial writes on crash corrupt the file
- **Sentinel value `-999` mixed with real scores:** Pollutes averages

## `memory/iteration_tracker.py`

### Bugs
- **Latent `AttributeError` crash on first invocation:** When `_best_params=None`

## `core/event_bus.py`

### Design Issues
- **Stale event loop capture:** Stores loop reference at import time — crashes if loop is replaced
- **No `unsubscribe()` mechanism:** Memory leak on dynamic subscribers

## `deployment/deployment_pipeline.py`

### Bugs
- **Fail-open validation gates:** Returns `True` on exception instead of `False`
- **Off-by-one in gate indexing**
- **Gate 5 labeled "ExperimentTracker" but imports `Experiment` class (line 226):** Potential import failure
- **Gate 6 CPCV uses hardcoded 1000 candles (line 267):** ~41 days at 1h, not representative
- **Gate 7 permutation test runs on real data, not synthetic (line 300-306):** Completely meaningless
- **Gate 8 Kelly sizing uses hardcoded 2%/1% avg_win/avg_loss fallback (lines 319-320)**
- **`run_full_pipeline` is 280+ lines (lines 91-386):** Impossible to unit test, deeply nested try/except
- **Pipeline components imported inline:** Anti-pattern
