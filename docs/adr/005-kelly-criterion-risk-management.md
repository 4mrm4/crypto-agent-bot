# ADR 005: Kelly Criterion Position Sizing with Risk Profiles

**Status:** Accepted  
**Context:** Need systematic position sizing that adapts to user risk tolerance while preventing over-leverage.  
**Decision:** Use Kelly Criterion with three profiles: conservative (0.25 fraction), compounding (0.5), aggressive (0.75). Circuit breaker halts trading on drawdown or anomaly thresholds.  
**Consequences:** Mathematically grounded sizing. Requires accurate win-rate and volatility estimates from backtest metrics.  
**Date:** 2026-06-01
