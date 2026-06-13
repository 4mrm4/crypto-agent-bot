# ADR 002: SignalFactory Must Match Freqtrade TA-Lib Calls

**Status:** Accepted  
**Context:** Backtest results diverged from live signals because pre-filter functions used independently written indicator logic instead of replicating Freqtrade strategy templates.  
**Decision:** SignalFactory pre-filter functions must use identical TA-Lib calls as the corresponding Freqtrade strategy templates in `engine.py`. Signals are derived from, not re-written.  
**Consequences:** Eliminates signal mismatch. Requires maintaining lockstep between SignalFactory and strategy templates.  
**Date:** 2026-06-03
