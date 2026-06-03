# StrategistAgent Split — 3 Focused Agents

## Motivation

The original `StrategistAgent` had 13 tools covering three distinct concerns:
strategy design, backtesting execution, and iteration tracking. This made
the agent's prompt bloated, its state complex, and its testing surface large.

Splitting into 3 focused agents improves:
- **Prompt quality**: each agent has a narrow, precise system prompt
- **State isolation**: each agent owns its own state, no cross-contamination
- **Testability**: each agent can be tested independently
- **Routing**: the graph dispatches tasks to the right specialist

## New Agent Architecture

```
                    Task Board (orchestration)
                         │
          ┌──────────────┼──────────────┐
          │              │              │
     Strategist      Backtester    IterationTracker
   (design only)   (execute only)  (record only)
          │              │              │
     Concepts        BacktestEngine   History
     Generation      Hyperopt         Best Strategy
     Params          Walk-Forward     Store/Load
```

## Agent Boundaries

### StrategistAgent (trimmed, kept name)

| Tool | Purpose |
|------|---------|
| `generate_strategy` | Create strategy spec from type + params |
| `suggest_next_params` | Suggest param tweaks based on past results |
| `get_strategy_concepts` | Retrieve proven concept templates |
| `get_research_history` | Query past research from ChromaDB |

State owned: `_generated_strategies: Dict[str, Dict]`

Prompt trimmed to focus on strategy design only — no backtesting instructions,
no memory instructions, no hyperopt instructions.

### BacktestAgent (new)

| Tool | Purpose |
|------|---------|
| `run_backtest` | Run a single backtest |
| `run_hyperopt` | Parameter optimization with Freqtrade |
| `walk_forward_validate` | Multi-window robustness check |
| `blind_search` | Automatic parameter search |
| `compare_strategies` | Side-by-side comparison |
| `set_backtest_config` | Global timerange/pairs/timeframe |
| `download_data` | Fetch new historical data |

State owned: `_engine: BacktestEngine`, `_tracker: ExperimentTracker`

Prompt: Backtesting specialist — no strategy design, no memory.

### IterationTrackerAgent (new)

| Tool | Purpose |
|------|---------|
| `get_best_strategy` | Retrieve best strategy from history |
| `get_iteration_history` | View all past attempts |
| `store_strategy_result` | Persist a strategy to memory |
| `store_strategy_insight` | Save a strategic observation |

State owned: `_iteration_history: List[IterationRecord]`,
`_best_strategy: Dict`, `_best_params: Dict`

Prompt: Strategy record-keeper — no design, no execution.

## Wiring Changes

### factory.py

```python
def build_orchestrator() -> HermesOrchestrator:
    return HermesOrchestrator(agents={
        "analyst": AnalystAgent(),
        "strategist": StrategistAgent(),
        "backtester": BacktesterAgent(),
        "iteration_tracker": IterationTrackerAgent(),
        "risk_manager": RiskManagerAgent(),
        "curator": CuratorAgent(),
        "researcher": ResearcherAgent(),
    })
```

### hermes.py

Add to `_agent_capabilities`:
```python
"backtester": ["backtest", "backtesting", "walk_forward", "hyperopt",
               "optimization", "compare", "benchmark", "download", "data"],
"iteration_tracker": ["iteration", "history", "best_strategy",
                      "store", "track", "record", "memory", "recall"],
```

Trim `strategist` keywords to only design-related:
```python
"strategist": ["strategy", "strategies", "generate", "concept",
               "parameter", "params", "design"],
```

The graph keyword router (`_pick_agent`) handles this automatically —
no graph.py changes needed.

### Workflow (how agents coordinate)

The strategist generates a strategy, then creates follow-up tasks:
```
Strategist generates strategy spec
  → task result mentions "next: backtest sma_crossover params..."
  → board extracts as child task
  → routed to BacktesterAgent by keyword matching
  → Backtester runs backtest, produces metrics
  → task result mentions "next: store this result..."
  → routed to IterationTrackerAgent
```

This already works via `_extract_child_tasks()` in graph.py.

## Files

| File | Action |
|------|--------|
| `agents/strategist.py` | Trim tools, update prompt, remove engine/history state |
| `agents/backtester.py` | New — BacktesterAgent with backtesting tools |
| `agents/iteration_tracker.py` | New — IterationTrackerAgent with memory tools |
| `orchestration/factory.py` | Add 2 new agents |
| `orchestration/hermes.py` | Add capabilities + task routing |
| `test_backtester.py` | New tests |
| `test_iteration_tracker.py` | New tests |
