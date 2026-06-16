---
name: datetime-utcnow-fix-lessons
description: datetime.utcnow() deprecation fix for this project has critical pitfalls
type: feedback
---

# datetime.utcnow() → datetime.now() fix has critical pitfalls on this system

## The problem
When replacing `datetime.utcnow()` with `datetime.now(datetime.UTC)`, two bugs emerged:

1. **`datetime.UTC` does NOT work with `from datetime import datetime`** on this system (Python 3.12.0, Windows). `datetime.UTC` is only available with `import datetime` (module-level). The class import pattern `from datetime import datetime` makes `datetime.UTC` unavailable.

2. **`datetime.timezone.utc` also doesn't work with `from datetime import datetime`** alone — `datetime` resolves to the `datetime.datetime` class, which has no `timezone` attribute.

## The correct fix pattern
```python
from datetime import datetime, timezone  # MUST import timezone explicitly
datetime.now(timezone.utc)               # Use timezone.utc, NOT datetime.UTC or datetime.timezone.utc
```

## For `default_factory` in dataclass fields:
```python
# WRONG — calls function at class-def time, returns datetime instance (not callable):
field(default_factory=datetime.now(timezone.utc))

# CORRECT — lambda defers execution:
field(default_factory=lambda: datetime.now(timezone.utc))
```

**Why:** `default_factory=` needs a callable, not a datetime instance. The lambda wraps it correctly.
**How to apply:** In any file using `from datetime import datetime` + `datetime.now(datetime.UTC)` or `datetime.now(datetime.timezone.utc)`, add `, timezone` to the import and use `timezone.utc`. For `field(default_factory=...)` patterns, add `lambda:` wrapper.
