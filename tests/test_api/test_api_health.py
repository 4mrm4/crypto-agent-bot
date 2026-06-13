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
