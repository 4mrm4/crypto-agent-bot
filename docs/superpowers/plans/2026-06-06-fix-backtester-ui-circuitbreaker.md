# Fix: Backtester Missing, UI Not Showing Results, Circuit Breaker False Halt

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 3 bugs — backtester agent missing in autonomous mode (Sharpe always -0.81), UI doesn't show autonomous research results (blank charts), circuit breaker false halts on daily drawdown

**Architecture:** All 3 fixes are independent. Bug 1 is a one-line add in main.py. Bug 2 touches autonomous loop state, API server, and UI. Bug 3 adds input clamping in risk_manager.py.

**Tech Stack:** Python 3.12, FastAPI, React (Babel standalone), Freqtrade

---

### Task 1: Fix circuit breaker false halt

**Files:**
- Modify: `agents/risk_manager.py:663-710`

The LLM passes raw PnL values as tool parameters without knowing actual trading state. When `daily_pnl_pct` is -0.50 (-50%) but the bot has never executed a trade, the circuit breaker falsely halts.

**Root cause:** The LLM hallucinates PnL values. The `_clamp_limit()` function already handles raw percentages for `daily_limit`/`weekly_limit` (>1.0 → divide by 100, ≤0 → use default). But `daily_pnl_pct` has no clamping. When the LLM passes a decimal like -0.50 and the limit is clamped to 0.03, `abs(-0.50) > 0.03` triggers a false halt.

**Fix:** Add the same >1.0 clamping for `daily_pnl_pct` and `weekly_pnl_pct`. Also add a sanity check that PnL is not unreasonably large (>90% daily) which would indicate a hallucinated value.

- [ ] **Step 1: Modify circuit_breaker_check to clamp PnL values**

Edit `agents/risk_manager.py`, in the `circuit_breaker_check` function around line 663-666. Add clamping for `daily_pnl` and `weekly_pnl` after they're parsed:

```python
daily_pnl = float(params.get("daily_pnl_pct", 0.0))
weekly_pnl = float(params.get("weekly_pnl_pct", 0.0))
daily_limit_raw = float(params.get("daily_limit", -0.03))
weekly_limit_raw = float(params.get("weekly_limit", -0.08))

# Clamp PnL values the same way we clamp limits
# LLMs often pass raw percentages (e.g. -50 for -50%) instead of decimals
if abs(daily_pnl) > 1.0:
    daily_pnl /= 100.0
    logger.debug("Clamped daily_pnl from raw percentage to %.4f", daily_pnl)
if abs(weekly_pnl) > 1.0:
    weekly_pnl /= 100.0
    logger.debug("Clamped weekly_pnl from raw percentage to %.4f", weekly_pnl)

# Sanity guard: reject PnL values >90% daily drawdown (clearly hallucinated)
if abs(daily_pnl) > 0.90:
    logger.warning("Rejecting implausible daily_pnl=%.4f (>90%%), treating as 0", daily_pnl)
    daily_pnl = 0.0
if abs(weekly_pnl) > 0.90:
    logger.warning("Rejecting implausible weekly_pnl=%.4f (>90%%), treating as 0", weekly_pnl)
    weekly_pnl = 0.0
```

- [ ] **Step 2: Run existing circuit breaker tests**

```bash
cd C:/Trading-bot/crypto_agent_bot && python -m pytest test_risk_kelly.py -v -x 2>&1
```

Expected: All tests pass (no regressions).

- [ ] **Step 3: Commit**

```bash
cd C:/Trading-bot && git add crypto_agent_bot/agents/risk_manager.py && git commit -m "fix: clamp hallucinated PnL values in circuit_breaker_check"
```

---

### Task 2: Add backtester to autonomous mode agents

**Files:**
- Modify: `main.py:148-155`

- [ ] **Step 1: Import BacktesterAgent and add it to the agents dict**

Edit `main.py`, in the `_run_autonomous()` function around line 130. Add the import and add the backtester to the agents dict:

Add import near line 130 (among the existing imports):
```python
from agents.backtester import BacktesterAgent
```

Update the agents dict (around line 148-155):
```python
agents = {
    "analyst": AnalystAgent(),
    "strategist": StrategistAgent(),
    "backtester": BacktesterAgent(),
    "iteration_tracker": IterationTrackerAgent(),
    "risk_manager": RiskManagerAgent(),
    "curator": CuratorAgent(),
    "researcher": ResearcherAgent(),
}
```

Note: `IterationTrackerAgent` also needs to be imported. Check if it's already imported or add:

```python
from agents.iteration_tracker import IterationTrackerAgent
```

Also add this import if not already there:
```python
from agents.iteration_tracker import IterationTrackerAgent
```

- [ ] **Step 2: Verify main.py starts without ImportError**

```bash
cd C:/Trading-bot/crypto_agent_bot && python -c "from main import _run_autonomous; print('Import OK')"
```

Expected: `Import OK` (no errors).

- [ ] **Step 3: Commit**

```bash
cd C:/Trading-bot && git add crypto_agent_bot/main.py && git commit -m "fix: add missing backtester agent to autonomous mode agents dict"
```

---

### Task 3: Track autonomous loop iteration results

**Files:**
- Modify: `orchestration/autonomous_loop.py` (add `iteration_results` list to `AutonomousLoopState`, capture results in `_run_research_cycle`)
- Modify: `api/server.py` (add `/api/autonomous/iterations` endpoint)
- Modify: `ui/index.html` (poll new endpoint, display on Dashboard)

- [ ] **Step 1: Add iteration_results to AutonomousLoopState**

In `orchestration/autonomous_loop.py`, add iteration results tracking to the dataclass (around line 44):

```python
@dataclass
class AutonomousLoopState:
    """Current state of the autonomous research loop."""
    is_running: bool = False
    is_paused: bool = False
    current_goal: Optional[str] = None
    last_goal_generated: Optional[datetime] = None
    next_cycle_eta: Optional[datetime] = None
    total_cycles: int = 0
    total_goals_generated: int = 0
    consecutive_failures: int = 0
    last_error: Optional[str] = None
    last_regime: str = "unknown"
    coverage_gaps: Dict[str, float] = field(default_factory=dict)
    # Iteration results for UI charting
    iteration_results: List[Dict[str, Any]] = field(default_factory=list)
```

- [ ] **Step 2: Capture iteration results after each research cycle**

In `orchestration/autonomous_loop.py`, modify `_run_research_cycle` to extract and store iteration results. Find the method around line 452 and add after the `logger.info(...)` at the end:

Add after line ~484:
```python
        # Capture iteration results for UI
        iterations = result.get("iterations", [])
        for it in iterations:
            metrics = it.get("metrics", {})
            self.state.iteration_results.append({
                "cycle": self.state.total_cycles,
                "iteration": it.get("iteration", 0),
                "verdict": it.get("verdict", "unknown"),
                "strategy_id": it.get("strategy_id", ""),
                "sharpe_ratio": metrics.get("sharpe_ratio", 0),
                "win_rate": metrics.get("win_rate", 0),
                "max_drawdown": metrics.get("max_drawdown", 0),
                "total_trades": metrics.get("total_trades", 0),
            })

        # Keep only last 100 results
        if len(self.state.iteration_results) > 100:
            self.state.iteration_results = self.state.iteration_results[-100:]
```

Also add the `best_metrics` to the state for quick access. In the same section:

```python
        # Store current best metrics for UI
        best_metrics = result.get("best_metrics", {})
        self.state.current_sharpe = best_metrics.get("sharpe_ratio", 0)
```

Add the two new fields to the dataclass:
```python
    current_sharpe: float = 0.0
    current_best_sharpe: float = 0.0
```

And update the best value:
```python
        # Track best sharpe across all cycles
        for it in iterations:
            metrics = it.get("metrics", {})
            s = metrics.get("sharpe_ratio", 0)
            if isinstance(s, (int, float)) and s > self.state.current_best_sharpe:
                self.state.current_best_sharpe = s
```

- [ ] **Step 3: Update get_state to include iteration info**

In `orchestration/autonomous_loop.py`, update the `get_state()` method (around line 186) to include the new fields:

```python
    def get_state(self) -> dict:
        """Return current state dict for API/UI."""
        return {
            "is_running": self.state.is_running,
            "is_paused": self.state.is_paused,
            "current_goal": self.state.current_goal,
            "last_goal_generated": (
                self.state.last_goal_generated.isoformat()
                if self.state.last_goal_generated else None
            ),
            "next_cycle_eta": (
                self.state.next_cycle_eta.isoformat()
                if self.state.next_cycle_eta else None
            ),
            "total_cycles": self.state.total_cycles,
            "total_goals_generated": self.state.total_goals_generated,
            "consecutive_failures": self.state.consecutive_failures,
            "last_error": self.state.last_error,
            "last_regime": self.state.last_regime,
            "coverage_gaps": self.state.coverage_gaps,
            "current_sharpe": self.state.current_sharpe,
            "current_best_sharpe": self.state.current_best_sharpe,
        }
```

- [ ] **Step 4: Add /api/autonomous/iterations endpoint**

In `api/server.py`, add a new endpoint after `autonomous_status()` (around line 177):

```python
@app.get("/api/autonomous/iterations")
async def autonomous_iterations():
    """Return iteration results from autonomous research cycles for UI charting."""
    loop = _autonomous_loop_ref or getattr(app.state, "autonomous_loop", None)
    if loop and hasattr(loop.state, "iteration_results"):
        results = loop.state.iteration_results
        # Compute summary stats
        discarded = sum(1 for r in results if r.get("verdict") == "discarded")
        kept = sum(1 for r in results if r.get("verdict") == "converged" or r.get("verdict") == "kept")
        best = max(
            (r.get("sharpe_ratio", 0) for r in results if isinstance(r.get("sharpe_ratio"), (int, float))),
            default=0,
        )
        return {
            "results": results,
            "discarded_count": discarded,
            "kept_count": kept,
            "best_sharpe": best,
        }
    return {"results": [], "discarded_count": 0, "kept_count": 0, "best_sharpe": 0}
```

- [ ] **Step 5: Update UI Dashboard to show iteration results**

In `ui/index.html`, modify the `DashboardView` to accept and display iteration data. First update the component call to pass iteration data:

Find where `DashboardView` is rendered (around line 680-688):
```jsx
{DashboardView && (
  <DashboardView
    autonomousState={autonomousState}
    circuitBreaker={circuitBreaker}
    regimeSnapshot={regimeSnapshot}
    positions={positions}
    tokens={tokens}
  />
)}
```

Add the missing props:
```jsx
{DashboardView && (
  <DashboardView
    autonomousState={autonomousState}
    circuitBreaker={circuitBreaker}
    regimeSnapshot={regimeSnapshot}
    positions={positions}
    tokens={tokens}
    iterationData={iterationData}
  />
)}
```

Add the `iterationData` state variable near the other state variables:
```jsx
const [iterationData, setIterationData] = useState({ results: [], discarded_count: 0, kept_count: 0, best_sharpe: 0 });
```

Add the fetch in the dashboard poll effect (around line 486):
```jsx
fetch("/api/autonomous/iterations")
  .then((r) => r.json())
  .then(setIterationData)
  .catch(() => {});
```

Now update the `DashboardView` function signature (line 1408):
```jsx
function DashboardView({ autonomousState, circuitBreaker, regimeSnapshot, positions, tokens, iterationData }) {
```

Add a Sharpe Progress section in the Dashboard layout. After the "Live Positions" section (around line 1700), add a research results panel:

```jsx
      {/* Bottom-right: Research Results */}
      <div
        className="r4"
        style={{
          background: T.card,
          borderRadius: "0 8px 0 8px",
          border: `1px solid ${T.border}`,
          padding: 14,
          position: "relative",
          gridRow: "2",
          maxHeight: 250,
          overflow: "auto",
        }}
      >
        <div
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            right: 0,
            height: 2,
            background: `linear-gradient(90deg,${T.gold},transparent)`,
          }}
        />
        <h2
          style={{
            fontFamily: "'Syne',sans-serif",
            fontSize: 10,
            fontWeight: 600,
            color: T.muted,
            textTransform: "uppercase",
            letterSpacing: "2px",
            marginBottom: 10,
          }}
        >
          ◈ Research Results
        </h2>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
          <MiniStat
            label="Best Sharpe"
            value={iterationData?.best_sharpe?.toFixed(2) || "—"}
            color={T.amber}
          />
          <MiniStat
            label="Discarded"
            value={iterationData?.discarded_count || 0}
            color={T.coral}
          />
          <MiniStat
            label="Kept"
            value={iterationData?.kept_count || 0}
            color={T.sage}
          />
          <MiniStat
            label="Total Iterations"
            value={iterationData?.results?.length || 0}
            color={T.teal}
          />
        </div>
        {iterationData?.results?.length > 0 && (
          <div
            style={{
              marginTop: 10,
              maxHeight: 120,
              overflowY: "auto",
              fontSize: 9.5,
              fontFamily: "'DM Mono',monospace",
            }}
          >
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead>
                <tr style={{ color: T.muted, fontSize: 8, textTransform: "uppercase" }}>
                  <th style={{ textAlign: "left", padding: "2px 4px" }}>It</th>
                  <th style={{ textAlign: "left", padding: "2px 4px" }}>Sharpe</th>
                  <th style={{ textAlign: "left", padding: "2px 4px" }}>WR</th>
                  <th style={{ textAlign: "left", padding: "2px 4px" }}>DD</th>
                  <th style={{ textAlign: "left", padding: "2px 4px" }}>Trades</th>
                  <th style={{ textAlign: "left", padding: "2px 4px" }}>Verdict</th>
                </tr>
              </thead>
              <tbody>
                {iterationData.results.slice(-20).reverse().map((r, i) => (
                  <tr key={i} style={{ borderTop: `1px solid ${T.border}20` }}>
                    <td style={{ padding: "2px 4px", color: T.muted }}>{r.iteration}</td>
                    <td style={{
                      padding: "2px 4px",
                      color: r.sharpe_ratio >= 0.8 ? T.sage : T.coral
                    }}>{r.sharpe_ratio?.toFixed(2)}</td>
                    <td style={{ padding: "2px 4px", color: T.text }}>
                      {r.win_rate ? `${(r.win_rate * 100).toFixed(0)}%` : "—"}
                    </td>
                    <td style={{ padding: "2px 4px", color: T.text }}>
                      {r.max_drawdown ? `${(Math.abs(r.max_drawdown) * 100).toFixed(1)}%` : "—"}
                    </td>
                    <td style={{ padding: "2px 4px", color: T.text }}>{r.total_trades || 0}</td>
                    <td style={{
                      padding: "2px 4px",
                      color: r.verdict === "kept" || r.verdict === "converged" ? T.sage : T.coral
                    }}>{r.verdict}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
```

- [ ] **Step 6: Run existing tests to check for regressions**

```bash
cd C:/Trading-bot/crypto_agent_bot && python -m pytest test_autonomous_loop.py -v -x 2>&1
```

Expected: All tests pass.

- [ ] **Step 7: Commit**

```bash
cd C:/Trading-bot && git add crypto_agent_bot/orchestration/autonomous_loop.py crypto_agent_bot/api/server.py crypto_agent_bot/ui/index.html && git commit -m "feat: wire autonomous research results to UI (iterations endpoint + dashboard panel)"
```
