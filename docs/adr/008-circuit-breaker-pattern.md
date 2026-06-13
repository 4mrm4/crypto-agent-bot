# ADR 008: Circuit Breaker for Trading Safety

**Status:** Accepted  
**Context:** No automated protection against runaway losses during autonomous trading. Once triggered, the system would continue executing despite adverse conditions.  
**Decision:** Implement two-layer circuit breaker: (1) drawdown-based — halts if portfolio drops below threshold, (2) anomaly-based — halts on unusual market conditions. UI panel shows breaker state with manual reset.  
**Consequences:** Safety net for autonomous mode. Breaker state persisted in StateBroker. Research loop also halted when breaker is tripped (fixed in v13).  
**Date:** 2026-06-01
