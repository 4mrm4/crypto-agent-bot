# ADR 003: SQLite Singleton with __new__ Enforcement

**Status:** Accepted  
**Context:** Multiple agents creating TradingDatabase instances caused connection conflicts and state inconsistency.  
**Decision:** Use `__new__`-based singleton pattern (not `__init__` guard) to ensure only one database connection exists.  
**Consequences:** Thread-safe single connection. Second constructor call returns existing instance without re-initializing.  
**Date:** 2026-06-05
