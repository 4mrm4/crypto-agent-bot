"""Tests for StateBroker — shared key-value + pub/sub."""

import asyncio
import pytest
import time
from unittest.mock import patch

from state.state_broker import (
    StateBroker,
    KEY_POSITION, KEY_SIGNAL, KEY_STATUS,
    TTL_POSITION, TTL_SIGNAL, TTL_STATUS,
)


# ── Key-value tests ──

def test_set_and_get():
    broker = StateBroker(redis_url=None)
    async def run():
        await broker.set("test:key", {"value": 42})
        val = await broker.get("test:key")
        assert val == {"value": 42}
    asyncio.run(run())


def test_get_missing():
    broker = StateBroker()
    async def run():
        val = await broker.get("nonexistent")
        assert val is None
    asyncio.run(run())


def test_delete():
    broker = StateBroker()
    async def run():
        await broker.set("test:del", "hello")
        await broker.delete("test:del")
        val = await broker.get("test:del")
        assert val is None
    asyncio.run(run())


def test_exists():
    broker = StateBroker()
    async def run():
        await broker.set("test:exists", 1)
        assert await broker.exists("test:exists") is True
        await broker.delete("test:exists")
        assert await broker.exists("test:exists") is False
    asyncio.run(run())


def test_ttl_expiry():
    broker = StateBroker()
    async def run():
        await broker.set("test:ttl", "expires", ttl=1)
        assert await broker.get("test:ttl") == "expires"
        await asyncio.sleep(1.5)
        val = await broker.get("test:ttl")
        assert val is None
    asyncio.run(run())


def test_multiple_keys():
    broker = StateBroker()
    async def run():
        await broker.set("k1", "a")
        await broker.set("k2", "b")
        assert await broker.get("k1") == "a"
        assert await broker.get("k2") == "b"
    asyncio.run(run())


def test_overwrite():
    broker = StateBroker()
    async def run():
        await broker.set("overwrite", "old")
        await broker.set("overwrite", "new")
        assert await broker.get("overwrite") == "new"
    asyncio.run(run())


def test_complex_values():
    broker = StateBroker()
    async def run():
        data = {"nested": [1, 2, 3], "bool": True, "null": None}
        await broker.set("complex", data)
        assert await broker.get("complex") == data
    asyncio.run(run())


def test_zero_ttl():
    """TTL=0 means no expiry."""
    broker = StateBroker()
    async def run():
        await broker.set("no_expiry", "stays", ttl=0)
        val = await broker.get("no_expiry")
        assert val == "stays"
    asyncio.run(run())


# ── Pub/sub tests ──

def test_publish_subscribe():
    broker = StateBroker()
    async def run():
        q = await broker.subscribe("test_channel")
        await broker.publish("test_channel", "hello")
        msg = await asyncio.wait_for(q.get(), timeout=1)
        assert msg == "hello"
    asyncio.run(run())


def test_multiple_subscribers():
    broker = StateBroker()
    async def run():
        q1 = await broker.subscribe("multi")
        q2 = await broker.subscribe("multi")
        await broker.publish("multi", "broadcast")
        msg1 = await asyncio.wait_for(q1.get(), timeout=1)
        msg2 = await asyncio.wait_for(q2.get(), timeout=1)
        assert msg1 == "broadcast"
        assert msg2 == "broadcast"
    asyncio.run(run())


def test_unsubscribe():
    broker = StateBroker()
    async def run():
        q = await broker.subscribe("unsub")
        await broker.unsubscribe("unsub", q)
        await broker.publish("unsub", "should_not_arrive")
        import asyncio
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(q.get(), timeout=0.3)
    asyncio.run(run())


def test_publish_no_subscribers():
    broker = StateBroker()
    async def run():
        await broker.publish("empty", "no listeners")
        # no crash = success
    asyncio.run(run())


def test_queue_full_behavior():
    broker = StateBroker()
    async def run():
        q = await broker.subscribe("full")
        # Fill the queue
        for i in range(100):
            try:
                q.put_nowait(i)
            except asyncio.QueueFull:
                break
        # Publish another — should drop oldest
        await broker.publish("full", "newest")
        msg = await asyncio.wait_for(q.get(), timeout=0.5)
        assert msg is not None
    asyncio.run(run())


# ── State helper tests ──

def test_position_lifecycle():
    broker = StateBroker()
    async def run():
        await broker.set_position("BTC/USDT", {"size": 0.5, "entry_price": 50000})
        pos = await broker.get_position("BTC/USDT")
        assert pos is not None
        assert pos["size"] == 0.5
        assert pos["entry_price"] == 50000
    asyncio.run(run())


def test_get_all_positions():
    broker = StateBroker()
    async def run():
        await broker.set_position("BTC/USDT", {"size": 0.5})
        await broker.set_position("ETH/USDT", {"size": 10})
        positions = await broker.get_all_positions()
        assert len(positions) >= 2
        pairs = [p.get("pair") for p in positions]
        assert "BTC/USDT" in pairs or any("BTC/USDT" in str(p) for p in positions)
    asyncio.run(run())


def test_signal_lifecycle():
    broker = StateBroker()
    async def run():
        sig = {"signal": "buy", "confidence": 0.75, "strategy_type": "momentum"}
        await broker.set_signal("BTC/USDT", sig)
        result = await broker.get_signal("BTC/USDT")
        assert result is not None
        assert result["signal"] == "buy"
        assert result["confidence"] == 0.75
    asyncio.run(run())


def test_system_status():
    broker = StateBroker()
    async def run():
        status = {"running": True, "active_tasks": 5}
        await broker.set_system_status(status)
        result = await broker.get_system_status()
        assert result is not None
        assert result["running"] is True
    asyncio.run(run())


def test_get_missing_position():
    broker = StateBroker()
    async def run():
        pos = await broker.get_position("NONEXISTENT/USDT")
        assert pos is None
    asyncio.run(run())


def test_get_missing_signal():
    broker = StateBroker()
    async def run():
        sig = await broker.get_signal("NONEXISTENT/USDT")
        assert sig is None
    asyncio.run(run())


def test_get_missing_system_status():
    broker = StateBroker()
    async def run():
        status = await broker.get_system_status()
        assert status is None
    asyncio.run(run())


# ── Redis fallback tests ──

def test_redis_fallback_on_import_error():
    """When redis-py is not installed, StateBroker uses in-memory."""
    broker = StateBroker(redis_url="redis://localhost:6379")
    async def run():
        await broker.set("fallback", "memory")
        val = await broker.get("fallback")
        assert val == "memory"
    asyncio.run(run())


def test_redis_fallback_on_connection_error():
    """When Redis connection fails, StateBroker uses in-memory."""
    broker = StateBroker(redis_url="redis://nonexistent:6379")
    async def run():
        await broker.set("conn_fail", "still_works")
        val = await broker.get("conn_fail")
        assert val == "still_works"
    asyncio.run(run())


# ── Key namespace tests ──

def test_key_namespaces():
    """Verify key namespace format constants."""
    assert KEY_POSITION == "broker:position:{}"
    assert KEY_SIGNAL == "broker:signal:{}"
    assert KEY_STATUS == "broker:status:system"

    assert KEY_POSITION.format("BTC/USDT") == "broker:position:BTC/USDT"
    assert KEY_SIGNAL.format("ETH/USDT") == "broker:signal:ETH/USDT"


# ── Start/stop tests ──

def test_start_stop():
    broker = StateBroker()
    async def run():
        await broker.start()
        assert broker._started is True
        assert broker._cleanup_task is not None
        await broker.stop()
        assert broker._started is False
    asyncio.run(run())


def test_double_start():
    broker = StateBroker()
    async def run():
        await broker.start()
        await broker.start()  # second call should be no-op
        assert broker._started is True
        await broker.stop()
    asyncio.run(run())
