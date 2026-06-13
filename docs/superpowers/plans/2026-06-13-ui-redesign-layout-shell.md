# UI Redesign — Layout Shell Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development to implement this plan task-by-task.

**Goal:** Replace the existing 952-line React UI with a redesigned layout shell using OLED dark mode palette, sharp 0px corners, Orbitron/Exo2/DM Mono typography, and a 4-zone grid (Topbar / Left Sidebar / Main Chart / Right Sidebar / Bottom Strip).

**Architecture:** Single-file `ui/index.html`, React 18 + Babel standalone via CDN. All charts are hand-built SVGs with viewBox scaling. No external charting libraries. WebSocket at `/ws/autonomous`, API base at `/api`.

**Tech Stack:** React 18 UMD, Babel standalone, Google Fonts (Orbitron, Exo2, DM Mono), vanilla CSS custom properties, hand-built SVG components.

---

### Task 1: CSS Variables & Global Reset

**Files:**
- Modify: `ui/index.html:12-36` (replace `<style>` block)

Replace the entire CSS block with the new OLED dark palette. All border-radius values must be 0. Add new utility classes for Orbitron headings, DM Mono data, and the scanline overlay.

- [ ] **Write the new CSS block**

Replace lines 12-36 in `<style>`:
- Variables: `--bg: #060A12`, `--surface: #0C1220`, `--elevated: #111827`, `--border: #1E2D45`, `--border-bright: #2A4060`, `--text: #E2EBF5`, `--secondary: #7A95B0`, `--muted: #3D5470`, `--green: #00FF88`, `--green-dim: #003D22`, `--red: #FF3B5C`, `--red-dim: #3D0012`, `--amber: #FFB830`, `--amber-dim: #3D2A00`, `--blue: #3B82F6`, `--purple: #8B5CF6`, `--cyan: #06B6D4`
- Reset: `*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}`
- Body: `background:var(--bg); color:var(--text); font-family:'Exo 2',sans-serif; font-size:13px; line-height:1.6; overflow:hidden`
- Scrollbar: 4px thin, transparent track, `--border` thumb
- Animations: `pulse` (opacity), `fadeIn`, `slideUp` (translateY 6px), `countUp`
- Scanline: `.scanline{position:absolute;inset:0;background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,255,136,0.015)2px,rgba(0,255,136,0.015)4px);pointer-events:none;z-index:10}`
- Utility: `.truncate{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}`
- Root: `#root{display:flex;flex-direction:column;height:100vh}`
- Main: `.main{display:flex;flex:1;min-height:0}`
- NO border-radius anywhere — verify 0px throughout

- [ ] **Verify CSS renders**

Open `ui/index.html` in browser after full build and check computed styles.

---

### Task 2: Theme Object (T) & Utility Functions

**Files:**
- Modify: `ui/index.html:43-56` (replace T object and utilities)

Write the `const T` theme object mapping CSS variable names to hex values for inline React styles. Keep field names backward-compatible with existing code. Write utility functions: `fmtPnl`, `fmtTime`, `clamp`, `agentColors` map.

- [ ] **Write the T object and utilities**

```javascript
const T = {
  bg:"#060A12", surface:"#0C1220", elevated:"#111827",
  border:"#1E2D45", borderBright:"#2A4060",
  text:"#E2EBF5", secondary:"#7A95B0", muted:"#3D5470",
  accent:"#00FF88", coral:"#FF3B5C", teal:"#06B6D4",
  gold:"#FFB830", purple:"#8B5CF6", blue:"#3B82F6",
};
const fmtPnl = (v) => v != null ? `${v >= 0 ? "+" : ""}${(v * 100).toFixed(2)}%` : "\u2014";
const fmtTime = (ts) => ts ? new Date(ts).toLocaleTimeString() : "\u2014";
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const agentColors = {strategist:T.purple,analyst:T.cyan,risk_manager:T.gold,backtester:T.muted,researcher:T.blue};
```

---

### Task 3: Layout Shell Structure (Empty Panels)

**Files:**
- Modify: `ui/index.html:697-948` (replace App component render section)

Build the App.jsx render tree with 4 empty panel components:

- `Topbar({status, regime, confidence, goal, tokens, onNewResearch})` — 40px fixed, flex row space-between, Orbitron brand mark "CAB" with gold text-shadow, live/standby status dot, regime badge with confidence bar, goal text, token counter, gold "NEW RESEARCH" button
- `LeftSidebar` — 220px fixed width, surface background, border-right, Orbitron "AGENTS" heading, scrollable event list (empty container), collapsible "HYPOTHESIS" details panel
- `RightSidebar` — 260px fixed, surface background, border-left, flex-col gap 8px, padding 10px. Container divs for SharpeGauge, WinRateDonut, DrawdownBar, TradeCounter, CircuitBreakerPill
- `BottomStrip` — 200px fixed, grid 1fr 280px 1fr, border-top. Container divs for "SHARPE PROGRESSION", "STRATEGY RADAR", "ITERATIONS"
- `App` — orchestrator: topbar, `.main` (left + chartArea + right), bottomStrip

Each panel initially renders a placeholder text ("Sharpe Gauge", "Win Rate", etc.) in muted DM Mono.

- [ ] **Build the full layout shell**

Result: `App()` returns:

```jsx
<div style={{height:"100%",display:"flex",flexDirection:"column"}}>
  <Topbar ... />
  <div className="main">
    <LeftSidebar events={events} />
    <StrategyChart ... />
    <RightSidebar ... />
  </div>
  <BottomStrip ... />
</div>
```

- [ ] **Verify 100vh layout**

Check that body has no scrollbar, main area fills remaining height, all borders align.

---

### Task 4: Topbar Component

**Files:**
- Modify: `ui/index.html` (Topbar section)

Props: `{status, regime, confidence, goal, tokens, onNewResearch}`

Left group: Orbitron "CAB" (14px, gold, text-shadow), separator line, status dot (8px square, green pulsing if "running"), DM Mono "LIVE"/"STANDBY", separator, regime badge (Orbitron 10px blue), confidence bar (80px wide, 3px tall), separator, goal text (Exo2 10px secondary, italic, 280px max, ellipsis)

Right group: DM Mono token counter "TK 0", gold "NEW RESEARCH" button (Orbitron 9px, #000 text, 6px 14px padding)

- [ ] **Implement Topbar**

---

### Task 5: Preserve All Mock Data Generators, SVG Helpers & Chart Primitives

**Files:**
- Modify: `ui/index.html:57-365`

Keep these unchanged (they are chart primitives that downstream components depend on):
- `generateMockCandles()`, `generateMockTrades()` — mock data generators
- `arcPath()` — SVG arc path helper
- `SharpeGauge({value})` — SVG gauge component
- `WinRateDonut({value})` — SVG donut component
- `DrawdownBar({value})` — SVG bar component
- `TradeCounter({value, prev})` — numeric display
- `CircuitBreakerPill({state, onHalt, onClear})` — status pill
- `CandlestickSVG({candles, trades})` — candlestick SVG
- `SharpeLineChart({results, selectedIteration, onSelect})` — line chart
- `StrategyRadar({currentStrategy, regimeSnapshot})` — hex radar
- `IterationTable({results, selected, onSelect})` — data table

- [ ] **Preserve all chart components**

---

### Task 6: Preserve Weboo Hook, API Hook, Persistence & ResearchModal

**Files:**
- Modify: `ui/index.html` (preserve hooks section)

Keep unchanged:
- `useWebSocket(url, handlers)` — WS hook with auto-reconnect
- `useApi()` — fetch wrapper
- `loadPersistedState()`, `persistState()` — localStorage helpers
- `ResearchModal({goal, cycles, iters, onStart, onClose})` — research modal

- [ ] **Preserve all hooks and modal**

---

### Task 7: Wire App State & Verify Layout

**Files:**
- Modify: `ui/index.html:697-948` (App component logic)

The App component preserves ALL existing useState declarations, ALL existing useEffect hooks (initial fetch, persistence, WS handlers), and ALL existing callback actions (startAutonomous, stopAutonomous, etc.).

The render tree is the new layout shell with empty panels. All existing WS event handler cases remain intact. The `StrategyChart` component wraps `CandlestickSVG` with its header and iteration selector.

- [ ] **Wire App state management**

---

### Task 8: Verify Layout Renders Correctly

- [ ] **Open the HTML file in browser**

Open `file:///C:/Trading-bot/crypto_agent_bot/ui/index.html` in a browser.

Expected:
- No scrollbar on body or html
- Topbar visible at top with CAB brand, status dot, regime, confidence bar, goal text, "NEW RESEARCH" button
- Left sidebar visible with "AGENTS" heading and "Waiting for events..." placeholder
- Main chart area shows scanline overlay and StrategyChart with mock candles
- Right sidebar shows SharpeGauge, WinRateDonut, DrawdownBar, TradeCounter, CircuitBreakerPill
- Bottom strip shows 3-column grid: SharpeLineChart, StrategyRadar, IterationTable
- All dividers and borders crisp, no rounded corners anywhere
- ResearchModal opens on "NEW RESEARCH" click with correct form fields
