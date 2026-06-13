"""Tests for token bucket rate limiter."""
import time
import pytest
from data.rate_limiter import RateLimiter


def test_initial_tokens_equals_capacity():
    rl = RateLimiter(rpm=10)
    assert rl._tokens == 10.0


def test_consume_allows_within_limit():
    rl = RateLimiter(rpm=5)
    for _ in range(5):
        assert rl.acquire() is True


def test_consume_blocks_over_limit():
    rl = RateLimiter(rpm=3)
    for _ in range(3):
        assert rl.acquire() is True
    # 4th should block (or return False if non-blocking)
    assert rl.acquire(block=False) is False


def test_acquire_blocking_wait():
    rl = RateLimiter(rpm=60)  # 1 per second
    for _ in range(60):
        rl.acquire()
    start = time.time()
    rl.acquire(block=True, timeout=5)
    elapsed = time.time() - start
    assert elapsed >= 0.9  # Should have waited ~1 second


def test_context_manager():
    rl = RateLimiter(rpm=10)
    with rl as r:
        assert r is rl
        assert rl._tokens == 9.0


def test_refill_over_time():
    rl = RateLimiter(rpm=60)
    # Drain all tokens
    for _ in range(60):
        rl.acquire()
    assert rl.acquire(block=False) is False
    # Wait for refill
    time.sleep(1.05)
    assert rl.acquire(block=False) is True
