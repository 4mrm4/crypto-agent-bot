"""Tests for agents/token_tracker.py"""

import threading

from agents.token_tracker import TokenTracker


class TestTokenTracker:
    def test_get_returns_same_instance(self):
        t1 = TokenTracker.get()
        t2 = TokenTracker.get()
        assert t1 is t2

    def test_initial_usage_zero(self):
        t = TokenTracker.get()
        # Use a fresh instance to avoid cross-test contamination
        fresh = TokenTracker()
        usage = fresh.get_usage()
        assert usage["prompt_tokens"] == 0
        assert usage["completion_tokens"] == 0
        assert usage["total_tokens"] == 0

    def test_add_usage_accumulates(self):
        t = TokenTracker()
        t.add_usage(prompt=100, completion=50)
        usage = t.get_usage()
        assert usage["prompt_tokens"] == 100
        assert usage["completion_tokens"] == 50
        assert usage["total_tokens"] == 150

    def test_multiple_adds(self):
        t = TokenTracker()
        t.add_usage(prompt=100, completion=50)
        t.add_usage(prompt=200, completion=100)
        usage = t.get_usage()
        assert usage["prompt_tokens"] == 300
        assert usage["completion_tokens"] == 150
        assert usage["total_tokens"] == 450

    def test_add_usage_zero(self):
        t = TokenTracker()
        t.add_usage(prompt=0, completion=0)
        usage = t.get_usage()
        assert usage["total_tokens"] == 0

    def test_add_usage_defaults(self):
        t = TokenTracker()
        t.add_usage(prompt=50)
        usage = t.get_usage()
        assert usage["prompt_tokens"] == 50
        assert usage["completion_tokens"] == 0

    def test_reset_clears(self):
        t = TokenTracker()
        t.add_usage(prompt=500, completion=300)
        t.reset()
        usage = t.get_usage()
        assert usage["prompt_tokens"] == 0
        assert usage["completion_tokens"] == 0
        assert usage["total_tokens"] == 0

    def test_reset_then_add(self):
        t = TokenTracker()
        t.add_usage(prompt=500, completion=300)
        t.reset()
        t.add_usage(prompt=10, completion=5)
        usage = t.get_usage()
        assert usage["prompt_tokens"] == 10
        assert usage["completion_tokens"] == 5

    def test_large_numbers(self):
        t = TokenTracker()
        t.add_usage(prompt=1000000, completion=500000)
        usage = t.get_usage()
        assert usage["prompt_tokens"] == 1000000
        assert usage["total_tokens"] == 1500000

    def test_thread_safety(self):
        t = TokenTracker()
        errors = []

        def add_many():
            try:
                for _ in range(100):
                    t.add_usage(prompt=10, completion=5)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_many) for _ in range(10)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        assert len(errors) == 0
        usage = t.get_usage()
        assert usage["prompt_tokens"] == 10 * 100 * 10  # 10 threads * 100 adds * 10
        assert usage["completion_tokens"] == 5 * 100 * 10
        assert usage["total_tokens"] == 15000

    def test_get_usage_atomic(self):
        t = TokenTracker()
        t.add_usage(prompt=100, completion=50)
        usage = t.get_usage()
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage
        assert len(usage) == 3
