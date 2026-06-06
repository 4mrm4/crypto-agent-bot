"""
StateBroker — shared key-value store + pub/sub for cross-component state.

Uses in-memory dicts by default. Optionally backs with Redis when
REDIS_URL is configured. Graceful degradation: if Redis is unavailable,
falls back to in-memory without crashing.

Key namespaces:
  broker:position:{pair}  — TTL 24h (LiveExecutor)
  broker:signal:{pair}    — TTL 1h  (SignalScanner)
  broker:status:system    — TTL 5min (Orchestrator)
"""

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Key namespaces ──
KEY_POSITION = "broker:position:{}"
KEY_SIGNAL = "broker:signal:{}"
KEY_STATUS = "broker:status:system"
KEY_PUBSUB = "broker:pubsub:{}"

# ── Default TTLs (seconds) ──
TTL_POSITION = 86400     # 24h
TTL_SIGNAL = 3600        # 1h
TTL_STATUS = 300         # 5min
CLEANUP_INTERVAL = 60    # cleanup expired keys every 60s

REDIS_URL = os.getenv("REDIS_URL", "")


class StateBroker:
    """Shared state broker with key-value store and pub/sub.

    In-memory by default. If REDIS_URL is set and redis-py is installed,
    uses Redis for cross-process sharing. Falls back to in-memory if Redis
    connection fails.
    """

    def __init__(self, redis_url: Optional[str] = None) -> None:
        self._store: Dict[str, Tuple[Any, float]] = {}  # key -> (value, expiry_ts)
        self._pubsub: Dict[str, List[asyncio.Queue]] = {}
        self._redis_url = redis_url or REDIS_URL
        self._redis = None
        self._redis_listener_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._started = False

        if self._redis_url:
            self._try_connect_redis()

    def _try_connect_redis(self):
        """Attempt to connect to Redis. Falls back silently."""
        try:
            from redis.asyncio.client import Redis
            self._redis = Redis.from_url(self._redis_url, decode_responses=True)
            logger.info("StateBroker: connected to Redis at %s", self._redis_url)
        except Exception as exc:
            logger.warning("StateBroker: Redis unavailable (%s), using in-memory", exc)
            self._redis = None

    async def start(self) -> None:
        """Start background tasks (cleanup, Redis listener)."""
        if self._started:
            return
        self._started = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        if self._redis:
            self._redis_listener_task = asyncio.create_task(self._redis_listen_loop())

    async def stop(self) -> None:
        """Stop background tasks."""
        self._started = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            self._cleanup_task = None
        if self._redis_listener_task:
            self._redis_listener_task.cancel()
            self._redis_listener_task = None
        if self._redis:
            try:
                await self._redis.aclose()
            except Exception:
                pass

    # ── Key-value operations ──

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a key-value pair with optional TTL (seconds)."""
        expiry = (time.time() + ttl) if ttl else 0.0
        self._store[key] = (value, expiry)
        if self._redis and ttl:
            try:
                await self._redis.setex(key, ttl, json.dumps(value))
            except Exception:
                pass
        elif self._redis:
            try:
                await self._redis.set(key, json.dumps(value))
            except Exception:
                pass

    async def get(self, key: str) -> Optional[Any]:
        """Get a value by key. Returns None if missing or expired."""
        # Check in-memory first
        if key in self._store:
            value, expiry = self._store[key]
            if expiry == 0.0 or time.time() < expiry:
                return value
            else:
                del self._store[key]
        # Fall back to Redis
        if self._redis:
            try:
                raw = await self._redis.get(key)
                if raw is not None:
                    return json.loads(raw)
            except Exception:
                pass
        return None

    async def delete(self, key: str) -> None:
        """Delete a key."""
        self._store.pop(key, None)
        if self._redis:
            try:
                await self._redis.delete(key)
            except Exception:
                pass

    async def exists(self, key: str) -> bool:
        """Check if a key exists and is not expired."""
        val = await self.get(key)
        return val is not None

    # ── Pub/sub operations ──

    async def publish(self, channel: str, message: Any) -> None:
        """Publish a message to a channel. All subscribers receive it."""
        # In-memory pubsub
        if channel in self._pubsub:
            for q in self._pubsub[channel]:
                try:
                    q.put_nowait(message)
                except asyncio.QueueFull:
                    try:
                        q.get_nowait()
                        q.put_nowait(message)
                    except asyncio.QueueEmpty:
                        pass
        # Redis pubsub
        if self._redis:
            try:
                await self._redis.publish(channel, json.dumps(message))
            except Exception:
                pass

    async def subscribe(self, channel: str) -> asyncio.Queue:
        """Subscribe to a channel. Returns an asyncio.Queue for receiving messages."""
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        if channel not in self._pubsub:
            self._pubsub[channel] = []
        self._pubsub[channel].append(q)
        return q

    async def unsubscribe(self, channel: str, queue: asyncio.Queue) -> None:
        """Remove a queue from a channel's subscriber list."""
        if channel in self._pubsub:
            self._pubsub[channel] = [q for q in self._pubsub[channel] if q is not queue]
            if not self._pubsub[channel]:
                del self._pubsub[channel]

    # ── State helpers ──

    async def set_position(self, pair: str, data: dict) -> None:
        """Publish current position state for a pair."""
        key = KEY_POSITION.format(pair)
        await self.set(key, data, TTL_POSITION)
        await self.publish(f"position:{pair}", data)

    async def get_position(self, pair: str) -> Optional[dict]:
        """Get the latest position for a pair."""
        return await self.get(KEY_POSITION.format(pair))

    async def get_all_positions(self) -> List[dict]:
        """Get all known positions from the store."""
        positions = []
        prefix = "broker:position:"
        for key in list(self._store.keys()):
            if key.startswith(prefix):
                val = await self.get(key)
                if val is not None:
                    if isinstance(val, dict):
                        pair = key[len(prefix):]
                        val["pair"] = pair
                    positions.append(val)
        return positions

    async def set_signal(self, pair: str, data: dict) -> None:
        """Publish the latest trade signal for a pair."""
        key = KEY_SIGNAL.format(pair)
        await self.set(key, data, TTL_SIGNAL)
        await self.publish(f"signal:{pair}", data)

    async def get_signal(self, pair: str) -> Optional[dict]:
        """Get the latest signal for a pair."""
        return await self.get(KEY_SIGNAL.format(pair))

    async def set_system_status(self, data: dict) -> None:
        """Publish system health status."""
        await self.set(KEY_STATUS, data, TTL_STATUS)
        await self.publish("status:system", data)

    async def get_system_status(self) -> Optional[dict]:
        """Get the latest system status."""
        return await self.get(KEY_STATUS)

    # ── Internal ──

    async def _cleanup_loop(self):
        """Periodically remove expired keys from in-memory store."""
        while self._started:
            now = time.time()
            expired = [k for k, (v, exp) in self._store.items()
                       if exp != 0.0 and now >= exp]
            for k in expired:
                del self._store[k]
            if expired:
                logger.debug("StateBroker: cleaned %d expired keys", len(expired))
            await asyncio.sleep(CLEANUP_INTERVAL)

    async def _redis_listen_loop(self):
        """Background task to forward Redis pub/sub messages to in-memory subscribers."""
        if not self._redis:
            return
        try:
            pubsub = self._redis.pubsub()
            await pubsub.subscribe("__keyevent@0__:expired")
            async for msg in pubsub.listen():
                if msg["type"] == "message":
                    channel = msg["channel"].decode() if isinstance(msg["channel"], bytes) else msg["channel"]
                    data = msg["data"].decode() if isinstance(msg["data"], bytes) else msg["data"]
                    await self.publish(channel, {"event": "expired", "key": data})
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.debug("StateBroker Redis listener stopped: %s", exc)
