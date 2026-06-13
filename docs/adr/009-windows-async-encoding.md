# ADR 009: Windows Compatibility — Async, Encoding, and Process Mgmt

**Status:** Accepted  
**Context:** Development and deployment on Windows requires handling cp1252 encoding, asyncio event loop lifecycle, and process management quirks.  
**Decision:** Always use `encoding='utf-8'` for file I/O. Never cache event loop references (asyncio.run() destroys its loop — query `get_running_loop()` per call). Use `async def` wrappers for monkey-patched coroutines.  
**Consequences:** Cross-platform portability. Required 3 rounds of fixes for async/await in event handlers and WebSocket connect-close cycles.  
**Date:** 2026-06-07
