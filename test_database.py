"""Tests for TradingDatabase cache methods."""
import json
import time
import pytest
from pathlib import Path
from data.database import TradingDatabase


@pytest.fixture
def db(tmp_path: Path) -> TradingDatabase:
    """Create a TradingDatabase backed by a temp file (not :memory:)."""
    db_path = tmp_path / "test_trading.db"
    d = TradingDatabase(db_path=db_path)
    return d


def test_set_and_get_cached(db):
    db.set_cached("test:key", {"price": 50000}, "test_source", ttl_seconds=300)
    result = db.get_cached("test:key")
    assert result == {"price": 50000}


def test_get_cached_expired_ttl(db):
    db.set_cached("test:expired", {"data": 1}, "test_source", ttl_seconds=1)
    time.sleep(1.1)
    result = db.get_cached("test:expired")
    assert result is None


def test_get_cached_missing_key(db):
    result = db.get_cached("nonexistent")
    assert result is None


def test_set_cached_overwrites_existing(db):
    db.set_cached("test:key", {"version": 1}, "test_source", 300)
    db.set_cached("test:key", {"version": 2}, "test_source", 300)
    result = db.get_cached("test:key")
    assert result == {"version": 2}


def test_cache_roundtrip_handles_complex_nested_data(db):
    data = {"metrics": {"sharpe": 1.2, "trades": [1, 2, 3]}, "tags": ["a", "b"]}
    db.set_cached("test:complex", data, "test_source", 300)
    assert db.get_cached("test:complex") == data
