# ADR 001: Multi-Agent LangGraph Architecture

**Status:** Accepted  
**Context:** Need autonomous trading research loop that can decompose goals, generate strategies, backtest, and execute without human intervention.  
**Decision:** Use LangGraph ReAct agents with specialized sub-agents (Strategist, Backtester, Risk, Execution) coordinated by an OrchestratorAgent.  
**Consequences:** Enables complex multi-step research loops. Requires careful recursion limit management and async event handling.  
**Date:** 2026-05-15
