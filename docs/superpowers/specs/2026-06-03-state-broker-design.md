# State Broker — shared key-value + pub/sub

## Motivation

The bot has components that need to share state across modules:
LiveExecutor, SignalScanner, Web UI, and the orchestrator. Currently
each component holds its own in-memory state, which is lost on restart
and invisible to components that connect later (e.g., a UI page load).
A lightweight state broker fills this gap.

## Design

### File: `state/state_broker.py`

Single class, no external dependencies required. Redis is optional.

### Interface

```python
class StateBroker:
    def __init__(self, redis_url: Optional[str] = None):
        # In-memory dicts by default
        # If redis_url set and redis-py installed, use Redis

    # Key-value (set with optional TTL)
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None
    async def get(self, key: str) -> Optional[Any]
    async def delete(self, key: str) -> None
    async def exists(self, key: str) -> bool

    # Pub/sub
    async def publish(self, channel: str, message: Any) -> None
    async def subscribe(self, channel: str) -> asyncio.Queue
    async def unsubscribe(self, channel: str, queue: asyncio.Queue) -> None

    # State helpers (named keys, not stored in separate Redis DBs)
    async def set_position(self, pair: str, data: dict) -> None
    async def get_position(self, pair: str) -> Optional[dict]
    async def get_all_positions(self) -> List[dict]
    async def set_signal(self, pair: str, data: dict) -> None
    async def get_signal(self, pair: str) -> Optional[dict]
    async def set_system_status(self, data: dict) -> None
    async def get_system_status(self) -> Optional[dict]
```

### Key namespaces (internal, not exposed to consumers)

```
broker:position:{pair}      — TTL 24h
broker:signal:{pair}        — TTL 1h
broker:status:system        — TTL 5min
broker:pubsub:{channel}     — no TTL (managed by subscribe/unsubscribe)
```

### In-memory backend (default)

- `_store: Dict[str, Tuple[Any, float]]` — (value, expiry_timestamp)
- `_pubsub: Dict[str, List[asyncio.Queue]]` — subscribers per channel
- `set()` stores value + timestamp, starts cleanup loop
- `get()` checks expiry, returns None if expired
- Background task periodically removes expired keys

### Redis backend (optional, when REDIS_URL is set)

- Uses `redis.asyncio.client.Redis` for async key-value ops
- Uses `redis.asyncio.client.PubSub` for pub/sub
- Falls back to in-memory on connection error or import error

### Config (`config.py`)

```python
REDIS_URL: str = os.getenv("REDIS_URL", "")
```

Empty string = in-memory only.

### Integration points (included in this task)

| Component | Change |
|-----------|--------|
| `execution/live_executor.py` | Set position on open/close |
| `execution/signal_scanner.py` | Set latest signal after evaluation |
| `orchestration/hermes.py` | Set system status in main loop |

### Testing

- 100% testable without Redis (all tests use in-memory backend)
- TTL expiry: mock time to verify cleanup
- Pub/sub: publish → subscribe → verify message received
- State helpers: set_position → get_position → verify fields
- Edge cases: missing key, expired key, concurrent subscribe
- Redis path: mocked with patch (no live Redis server needed)
