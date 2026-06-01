# Anti-Overfitting & Data Integrity — Brainstorm

## 1. Contamination Loop Map

Every path where data leaks between training and evaluation:

### 1a. Curator injects past winners into new research
- **File:** `orchestration/hermes.py` lines 128-134
- **Mechanism:** `curator.inject_context(goal, k=3)` is called at the start of every research run, injecting top-3 similar past results into the LLM's context window
- **Risk:** The LLM sees "this strategy scored Sharpe 1.8 on BTC" and converges toward similar parameters instead of exploring novel space
- **Severity:** HIGH — this is the primary contamination vector

### 1b. ExperimentTracker.suggest_next_params()
- **File:** `orchestration/experiment_tracker.py` lines 124-188
- **Mechanism:** After each backtest, `suggest_next_params()` moves parameters 30% toward the best known result
- **Risk:** Parameter hill-climbing based on past results = manual curve fitting. After enough cycles, parameters fit historical noise
- **Severity:** HIGH — this is explicit parameter optimization against past results

### 1c. ChromaDB stores and retrieves strategy_results
- **File:** `memory/vector_store.py` lines 103-178
- **Mechanism:** Every backtest result stored with full metrics, retrieved by `get_best_strategies()` and `query_similar()`
- **Risk:** ChromaDB returns strategies from overlapping data windows, creating feedback where past "winners" influence new research
- **Severity:** MEDIUM — mitigated by semantic search not being exact, but still a risk

### 1d. Research iteration persistence loop
- **File:** `orchestration/hermes.py` lines 491-514
- **Mechanism:** Each research iteration is stored to ChromaDB with verdict, metrics, critique
- **Risk:** Next research cycle retrieves and re-reads past iteration records, anchoring on previous approaches
- **Severity:** MEDIUM

### 1e. Backtest timerange contamination
- **File:** `backtesting/engine.py` line 462, default timerange `"20210101-"`
- **Mechanism:** Default timerange goes through all available data, no guard against holdout window
- **Risk:** LLM could request a timerange that includes holdout data. `_sanitize_timerange()` converts any format but doesn't check against holdout boundaries
- **Severity:** CRITICAL — if holdout data leaks, every validation result is invalid

### 1f. Walk-forward auto-detected data range
- **File:** `backtesting/engine.py` lines 603-627
- **Mechanism:** WFV auto-detects data range from file timestamps, not respecting split boundaries
- **Risk:** WFV could silently include holdout data in its test windows
- **Severity:** HIGH — WFV would report false robustness

### 1g. Autonomous loop coverage gaps read from ChromaDB
- **File:** `orchestration/autonomous_loop.py` lines 340-362
- **Mechanism:** `_compute_coverage_gaps()` queries ChromaDB for best strategies per regime
- **Risk:** ChromaDB contains strategies from all data windows, so gaps reflect full-history performance, not research-window performance
- **Severity:** MEDIUM — affects goal priority, not strategy parameters directly

### 1h. Kelly sizing uses backtest metrics at face value
- **File:** `agents/risk_manager.py` lines 101-168
- **Mechanism:** Kelly formula directly uses win_rate, avg_win, avg_loss from backtest
- **Risk:** Backtest metrics are typically 40-50% optimistic. Feeding them raw into Kelly produces over-sized positions
- **Severity:** HIGH — financial risk from overconfidence

## 2. Conservative Position Sizing — Concrete Definition

**Current state:** Kelly fraction at 25% of full Kelly, capped at 10% portfolio per trade.

**What "conservative" means in this context:**

- **Pessimism haircut:** Reduce backtest win_rate and avg_win by a fixed factor (e.g., 45% reduction) before Kelly calculation
- **Validation mode:** First 90 days live, cap at 2% portfolio regardless of Kelly output
- **Tiered sizing based on evidence:**
  - 0-90 days live: max 2% (validation mode)
  - 90-180 days or live Sharpe < 0.6: max 5% (cautious)
  - 180+ days with live Sharpe >= 0.6: max 10% (normal = current state)
- **Degradation-adjusted Kelly:** Expected OOS degradation (default 40%) factored into inputs

## 3. Validation Mode — Behavioral Specification

**What's different vs normal operation:**

| Aspect | Normal | Validation Mode |
|--------|--------|-----------------|
| Max position | 10% portfolio | 2% portfolio |
| CB daily limit | -3% | -1.5% |
| CB weekly limit | -8% | -4% |
| OOS required for deploy | No | Yes |
| Trade review cadence | Every 30 trades | Every 10 trades |
| Live results logging | Main audit log | Separate `validation_trades.jsonl` |
| Graduation check | N/A | After 90 days, need Sharpe > 0.6 and 50+ trades |

**User experience:** UI shows amber "VALIDATION MODE" bar with day countdown, graduation criteria checklist, rolling Sharpe. Non-alarming — this is expected healthy behavior.

**What happens on graduation:** Auto-transition if criteria met. If not met, extend validation until all criteria pass with a warning.

## 4. Edge Cases

### 4a. Holdout window runs out
- DataSplitConfig has fixed holdout end date. When current date passes holdout_end, OOS validation still works on the holdout window (it's historical data, not live)
- If user wants to extend, they must **manually** update DataSplitConfig and re-validate all strategies on the extended window
- Autonomous loop should never reference holdout window dates

### 4b. Strategy passes holdout, decays immediately in live
- **Scenario:** OOS Sharpe 0.9, live Sharpe 0.2 after 2 weeks
- **Response:** PerformanceMonitor detects abnormal degradation (>50% threshold). Strategy suspended (not retired). Regime mismatch check runs. If regime changed, suspension is correct — strategy may recover when regime returns. If same regime, retire after 14 days of suspension.
- **Root cause:** Could be overfitting that OOS didn't catch, or regime change between holdout and live periods

### 4c. Synthetic sanity check fails
- Strategy claims Sharpe 2.0 on real data but shows Sharpe 0.8 on random walk
- **Response:** Reject immediately. Do not proceed to WFV or OOS. The strategy is clearly overfit — it found patterns in noise.
- This is a fast, cheap gate that saves time and prevents the most egregious overfitting

### 4d. LLM bypasses restrictions
- The LLM is instructed to use `generate_parameter_search_space()` for blind search, but could call `run_backtest()` directly with specific parameters if the tool exists
- **Mitigation:** During the blind search phase, remove `run_backtest()` and `interpret_metrics()` from the strategist's tool list. Only add them back during the direction phase

### 4e. ChromaDB contamination from same research window
- Two strategies discovered on the same research window should not be compared for "best" — they share the same data and are not independent
- **Mitigation:** Tag every stored strategy with `discovered_on_window`. When generating context for a new research cycle, exclude strategies from the same window

## 5. Implementation Order

1. DataSplitConfig (S1a) — foundation, everything depends on this
2. Holdout guard in engine.py (S1b) — immediate safety
3. OOSValidator (S1c) — enables separation of concerns
4. SyntheticValidator (S6) — fast cheap gate early in pipeline
5. BlindParameterSearch (S2) — changes LLM workflow
6. ChromaDB contamination guard (S7) — breaks feedback loop
7. Conservative Kelly (S3) — financial safety
8. PerformanceMonitor (S4) — live monitoring infrastructure
9. Validation mode (S5) — wraps execution
10. Deployment pipeline (S8) — wires everything together
11. Tests (S9) — verify integrity
12. Documentation (S10) — explain the system

## 6. Data Integrity Invariants

These must NEVER be violated:
- `DATA_SPLIT.validate()` MUST be called at import time
- `run_backtest()` MUST raise ValueError if timerange includes holdout window
- `OOSValidator` MUST never write to ChromaDB
- `SyntheticValidator` MUST run before WFV in the pipeline
- Kelly MUST apply degradation haircut before position sizing
- ChromaDB entries MUST be tagged with `discovered_on_window`
- `meets_deploy_criteria()` MUST include `synthetic_sanity_passed`
