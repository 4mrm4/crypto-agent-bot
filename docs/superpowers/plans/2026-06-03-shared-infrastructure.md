# Shared Infrastructure: api_cache + APIHealthTracker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add SQLite api_cache table for data source caching and an APIHealthTracker for monitoring all external API health.

**Architecture:** Extend TradingDatabase with `get_cached`/`set_cached` methods using a new `api_cache` table. Create a standalone `data/api_health.py` module with an `APIHealthTracker` class that records success/failure per source. Wire a `/api/data/health` endpoint into the FastAPI server.

**Tech Stack:** sqlite3 (stdlib), Python dataclasses, FastAPI

---
## Files

- Modify: `data/database.py` — add api_cache table + cache methods (insert after line ~130)
- Create: `data/api_health.py` — APIHealthTracker class
- Modify: `api/server.py` — add GET /api/data/health endpoint
- Test: `test_api_health.py` — APIHealthTracker unit tests

---

### Task 1: Add api_cache table to TradingDatabase

- [ ] **Step 1: Write the failing test first**

Create `test_database.py` (new file):

```python
"""Tests for TradingDatabase cache methods."""
import json
import time
import pytest
from data.database import TradingDatabase


@pytest.fixture
def db():
    d = TradingDatabase(db_path=":memory:")
    # Ensure api_cache table exists
    d.conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='api_cache'")
    return d


def test_set_and_get_cached(db):
    """Test basic set_cached then get_cached round-trip."""
    db.set_cached("test:key", {"price": 50000}, "test_source", ttl_seconds=300)
    result = db.get_cached("test:key")
    assert result == {"price": 50000}


def test_get_cached_expired_ttl(db):
    """Test expired TTL returns None and cleans up row."""
    db.set_cached("test:expired", {"data": 1}, "test_source", ttl_seconds=1)
    time.sleep(1.1)
    result = db.get_cached("test:expired")
    assert result is None
    # Row should be deleted
    row = db.conn.execute(
        "SELECT * FROM api_cache WHERE cache_key = ?", ("test:expired",)
    ).fetchone()
    assert row is None


def test_get_cached_missing_key(db):
    """Test missing key returns None."""
    result = db.get_cached("nonexistent")
    assert result is None


def test_set_cached_overwrites_existing(db):
    """Test set_cached with same key overwrites existing row."""
    db.set_cached("test:key", {"version": 1}, "test_source", 300)
    db.set_cached("test:key", {"version": 2}, "test_source", 300)
    result = db.get_cached("test:key")
    assert result == {"version": 2}


def test_cache_roundtrip_handles_complex_nested_data(db):
    """Test complex dicts (lists, nested dicts) survive JSON round-trip."""
    data = {"metrics": {"sharpe": 1.2, "trades": [1, 2, 3]}, "tags": ["a", "b"]}
    db.set_cached("test:complex", data, "test_source", 300)
    assert db.get_cached("test:complex") == data
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest test_database.py -v --tb=short`
Expected: FAIL with `sqlite3.OperationalError` (table api_cache doesn't exist) or attribute errors

- [ ] **Step 3: Add SCHEMA_API_CACHE and cache methods to TradingDatabase**

In `data/database.py`, at line ~115 (after SCHEMA_MIGRATIONS), add:

```python
SCHEMA_API_CACHE = """
CREATE TABLE IF NOT EXISTS api_cache (
    cache_key TEXT PRIMARY KEY,
    data TEXT NOT NULL,
    source TEXT NOT NULL,
    cached_at INTEGER NOT NULL,
    ttl_seconds INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_api_cache_source ON api_cache(source, cached_at);
"""
```

In `_init_schema()`, add `self.SCHEMA_API_CACHE` to the schemas list (after `SCHEMA_MIGRATIONS`).

Add methods (insert after `is_wal_mode()` around line 602):

```python
import json
import time

def get_cached(self, cache_key: str) -> Optional[dict]:
    """Return cached data dict, or None if missing/expired."""
    row = self.conn.execute(
        "SELECT data, cached_at, ttl_seconds FROM api_cache WHERE cache_key = ?",
        (cache_key,),
    ).fetchone()
    if not row:
        return None
    if time.time() - row["cached_at"] > row["ttl_seconds"]:
        self.conn.execute(
            "DELETE FROM api_cache WHERE cache_key = ?", (cache_key,)
        )
        self.conn.commit()
        return None
    return json.loads(row["data"])

def set_cached(
    self, cache_key: str, data: dict, source: str, ttl_seconds: int
):
    """Insert or replace a cached entry."""
    self.conn.execute(
        """INSERT OR REPLACE INTO api_cache
           (cache_key, data, source, cached_at, ttl_seconds)
           VALUES (?, ?, ?, ?, ?)""",
        (cache_key, json.dumps(data), source, int(time.time()), ttl_seconds),
    )
    self.conn.commit()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest test_database.py -v --tb=short`
Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add data/database.py test_database.py
git commit -m "feat: add api_cache table and get_cached/set_cached methods to TradingDatabase"
```

---

### Task 2: Create APIHealthTracker

- [ ] **Step 1: Write the failing test first**

Create `test_api_health.py`:

```python
"""Tests for APIHealthTracker."""
import time
from datetime import datetime
import pytest
from data.api_health import APIHealthTracker, APIHealth


@pytest.fixture
def tracker():
    return APIHealthTracker()


def test_record_success_creates_healthy_entry(tracker):
    tracker.record_success("messari")
    health = tracker.get_health("messari")
    assert health.source == "messari"
    assert health.last_success is not None
    assert health.last_failure is None
    assert health.consecutive_failures == 0
    assert health.is_healthy is True


def test_record_failure_increments_counter(tracker):
    tracker.record_failure("coincap", "Timeout")
    health = tracker.get_health("coincap")
    assert health.consecutive_failures == 1
    assert health.last_failure is not None


def test_consecutive_failures_three_marks_unhealthy(tracker):
    for _ in range(3):
        tracker.record_failure("santiment", "Error")
    health = tracker.get_health("santiment")
    assert health.consecutive_failures == 3
    assert health.is_healthy is False


def test_success_resets_consecutive_failures(tracker):
    tracker.record_failure("messari", "Error 1")
    tracker.record_failure("messari", "Error 2")
    tracker.record_success("messari")
    health = tracker.get_health("messari")
    assert health.consecutive_failures == 0
    assert health.is_healthy is True


def test_get_all_health_returns_all_sources(tracker):
    tracker.record_success("messari")
    tracker.record_success("coincap")
    all_h = tracker.get_all_health()
    assert "messari" in all_h
    assert "coincap" in all_h


def test_get_degraded_returns_only_unhealthy(tracker):
    tracker.record_success("messari")
    tracker.record_failure("coincap", "Err")
    tracker.record_failure("coincap", "Err")
    tracker.record_failure("coincap", "Err")
    degraded = tracker.get_degraded_sources()
    assert "coincap" in degraded
    assert "messari" not in degraded


def test_unknown_source_returns_default_healthy(tracker):
    health = tracker.get_health("nonexistent")
    assert health.source == "nonexistent"
    assert health.is_healthy is True
    assert health.consecutive_failures == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest test_api_health.py -v --tb=short`
Expected: FAIL with `ModuleNotFoundError: data.api_health`

- [ ] **Step 3: Implement APIHealthTracker**

Create `data/api_health.py`:

```python
"""Shared health tracker for all external API integrations."""
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Consecutive failures threshold to mark a source unhealthy
UNHEALTHY_THRESHOLD = 3


@dataclass
class APIHealth:
    """Health state for a single API source."""
    source: str
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    consecutive_failures: int = 0

    @property
    def is_healthy(self) -> bool:
        """True if no recent failures or success reset the counter."""
        return self.consecutive_failures < UNHEALTHY_THRESHOLD


class APIHealthTracker:
    """Tracks health of all external API sources.

    Stores state in-memory (simple dict). If Task 3 Redis is available,
    can be extended to use StateBroker for persistence.
    """

    def __init__(self):
        self._state: Dict[str, APIHealth] = {}

    def _ensure(self, source: str) -> APIHealth:
        if source not in self._state:
            self._state[source] = APIHealth(source=source)
        return self._state[source]

    def record_success(self, source: str):
        """Record a successful API call — resets consecutive failures."""
        health = self._ensure(source)
        health.last_success = datetime.utcnow()
        health.consecutive_failures = 0
        logger.debug("APIHealth[%s]: success recorded", source)

    def record_failure(self, source: str, error: str):
        """Record an API failure — increments consecutive failures."""
        health = self._ensure(source)
        health.last_failure = datetime.utcnow()
        health.consecutive_failures += 1
        logger.warning(
            "APIHealth[%s]: failure #%d: %s",
            source, health.consecutive_failures, error,
        )

    def get_health(self, source: str) -> APIHealth:
        """Return current health for a source."""
        return self._ensure(source)

    def get_all_health(self) -> Dict[str, APIHealth]:
        """Return health for all tracked sources."""
        return dict(self._state)

    def get_degraded_sources(self) -> List[str]:
        """Return list of source names that are unhealthy."""
        return [
            name for name, h in self._state.items()
            if not h.is_healthy
        ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest test_api_health.py -v --tb=short`
Expected: all 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add data/api_health.py test_api_health.py
git commit -m "feat: add APIHealthTracker for monitoring external API health"
```

---

### Task 3: Wire /api/data/health endpoint into FastAPI

- [ ] **Step 1: Write the failing test**

Create `test_server_health.py`:

```python
"""Tests for /api/data/health endpoint."""
import pytest
from fastapi.testclient import TestClient
from api.server import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint_returns_json(client):
    resp = client.get("/api/data/health")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict)
    # Should have a health key per source
    for key in data:
        assert "source" in data[key]
        assert "consecutive_failures" in data[key]
        assert "is_healthy" in data[key]


def test_health_endpoint_includes_known_sources(client):
    resp = client.get("/api/data/health")
    data = resp.json()
    sources = [data[k]["source"] for k in data]
    # At least the core sources should be tracked
    for s in ["kraken", "binance"]:
        assert s in sources
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest test_server_health.py -v --tb=short`
Expected: FAIL (404 on the endpoint)

- [ ] **Step 3: Add the endpoint to api/server.py**

Add import at top:
```python
from data.api_health import APIHealthTracker
```

Add a global health tracker (near top with other globals):
```python
_health_tracker = APIHealthTracker()
```

Add the endpoint (before the WebSocket handler):

```python
@app.get("/api/data/health")
async def get_api_health():
    """Return health status for all external API sources."""
    all_health = _health_tracker.get_all_health()
    return {
        name: {
            "source": h.source,
            "last_success": h.last_success.isoformat() if h.last_success else None,
            "last_failure": h.last_failure.isoformat() if h.last_failure else None,
            "consecutive_failures": h.consecutive_failures,
            "is_healthy": h.is_healthy,
        }
        for name, h in all_health.items()
    }
```

Export `_health_tracker` so integration modules can access it:

```python
# At module level, add a getter
def get_health_tracker() -> APIHealthTracker:
    return _health_tracker
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest test_server_health.py -v --tb=short`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add api/server.py test_server_health.py
git commit -m "feat: add /api/data/health endpoint with APIHealthTracker integration"
```
