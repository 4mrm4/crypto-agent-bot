# ADR 004: EventBus + StateBroker for Inter-Agent Communication

**Status:** Accepted  
**Context:** Agents needed loosely coupled communication without direct dependencies or shared mutable state.  
**Decision:** Implement async EventBus (pub/sub) for event-driven coordination and StateBroker (key-value + pub/sub) for shared state. StateBroker supports in-memory fallback and optional Redis backend.  
**Consequences:** Decoupled agents, easier testing (mock EventBus), but requires careful async/await handling — asyncio event loops must not be cached across coroutine calls.  
**Date:** 2026-06-03
