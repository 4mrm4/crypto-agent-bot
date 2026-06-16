---
name: risk-manager-import-order
description: PositionSizingTier must be defined before _kelly_core in risk_manager.py
type: project
---

# risk_manager.py import order bug

`PositionSizingTier` enum was defined at line 163 but used at line 76 by `_kelly_core()`, causing `NameError` at import time. This was a pre-existing bug (not from the MEDIUM audit fixes).

**Fix applied (2026-06-16):** Moved `PositionSizingTier` class definition to before `_kelly_core()` function.
**Why:** Class used in function signature must be defined first.
**How to apply:** In `agents/risk_manager.py`, `PositionSizingTier` must appear before `_kelly_core()`. The `from config import settings` and `BACKTEST_OPTIMISM_FACTOR` stayed in their original position after the enum move.
