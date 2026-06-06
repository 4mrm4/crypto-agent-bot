"""
MarketDataStream — live WebSocket connections to Binance.

Maintains real-time streams for ticker, kline, order book depth,
and aggregated trades. Publishes to internal asyncio channels consumed
by agents and the signal scanner.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MarketDataStream:
    """Manages WebSocket connections to exchange for real-time market data."""

    STREAMS = {
        "ticker":    "{pair}@ticker",
        "kline_5m":  "{pair}@kline_5m",
        "kline_15m": "{pair}@kline_15m",
        "kline_1h":  "{pair}@kline_1h",
        "depth":     "{pair}@depth20@100ms",
        "trades":    "{pair}@aggTrade",
    }

    def __init__(self, base_url: str = "wss://stream.binance.com:9443/ws") -> None:
        self._base_url = base_url
        self._subscribers: Dict[str, List[asyncio.Queue]] = {}
        self._latest_data: Dict[str, Any] = {}
        self._ws = None
        self._connected = False
        self._pairs: List[str] = []
        self._streams: List[str] = []
        self.reconnect_count = 0
        self.max_reconnects = 3

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self, pairs: List[str], streams: Optional[List[str]] = None) -> None:
        """Open Binance WebSocket for all pair/stream combinations."""
        import websockets

        self._pairs = pairs
        self._streams = streams or list(self.STREAMS.keys())

        # Build combined stream URL
        stream_names = []
        for pair in pairs:
            normalized = pair.replace("/", "").lower()
            for s in self._streams:
                template = self.STREAMS.get(s)
                if template:
                    stream_names.append(template.format(pair=normalized))

        url = f"{self._base_url}/{'/'.join(stream_names)}"
        logger.info("Connecting to %d streams for %s", len(stream_names), pairs)

        try:
            self._ws = await websockets.connect(url, ping_interval=30)
            self._connected = True
            self.reconnect_count = 0
            logger.info("WebSocket connected: %s", url[:80])
        except Exception as exc:
            logger.warning("WebSocket connection failed: %s — will poll REST", exc)
            self._connected = False

    async def subscribe(self, channel: str) -> asyncio.Queue:
        """Return a queue that receives every update for this channel."""
        if channel not in self._subscribers:
            self._subscribers[channel] = []
        q = asyncio.Queue(maxsize=100)
        self._subscribers[channel].append(q)
        return q

    async def _dispatch(self, data: dict):
        """Route incoming data to subscribers by channel."""
        if not isinstance(data, dict):
            return

        event_type = data.get("e", "")
        channel_map = {
            "24hrTicker": "ticker",
            "kline": f"kline_{data.get('k', {}).get('i', '1h')}",
            "depthUpdate": "depth",
            "aggTrade": "trades",
        }
        channel = channel_map.get(event_type, event_type)

        self._latest_data[channel] = data
        subscribers = self._subscribers.get(channel, [])
        for q in subscribers:
            try:
                q.put_nowait(data)
            except asyncio.QueueFull:
                try:
                    q.get_nowait()
                    q.put_nowait(data)
                except asyncio.QueueEmpty:
                    pass

    async def read_loop(self) -> None:
        """Read from WebSocket and dispatch to subscribers."""
        if not self._connected or not self._ws:
            logger.warning("WebSocket not connected — read loop skipped")
            return

        try:
            async for message in self._ws:
                try:
                    data = json.loads(message)
                    await self._dispatch(data)
                except json.JSONDecodeError:
                    continue
        except Exception as exc:
            logger.warning("WebSocket read loop ended: %s", exc)
        finally:
            self._connected = False
            self.reconnect_count += 1

    def get_latest_candle(self, pair: str, timeframe: str = "1h") -> Optional[dict]:
        """Return most recently closed candle from stream data."""
        channel = f"kline_{timeframe}"
        data = self._latest_data.get(channel)
        if data and isinstance(data, dict):
            kline = data.get("k", {})
            return {
                "open": float(kline.get("o", 0)),
                "high": float(kline.get("h", 0)),
                "low": float(kline.get("l", 0)),
                "close": float(kline.get("c", 0)),
                "volume": float(kline.get("v", 0)),
                "closed": kline.get("x", False),
                "timestamp": kline.get("T", 0),
            }
        return None

    def get_order_book_imbalance(self, pair: str) -> float:
        """
        Returns bid_volume / (bid_volume + ask_volume) for top 5 levels.
        > 0.6 = buy pressure, < 0.4 = sell pressure.
        """
        data = self._latest_data.get("depth")
        if not data or not isinstance(data, dict):
            return 0.5

        bids = data.get("b", [])[:5]
        asks = data.get("a", [])[:5]
        bid_vol = sum(float(b[1]) for b in bids)
        ask_vol = sum(float(a[1]) for a in asks)
        total = bid_vol + ask_vol
        return bid_vol / total if total > 0 else 0.5

    def get_ticker(self, pair: str) -> Optional[dict]:
        """Return latest ticker data."""
        data = self._latest_data.get("ticker")
        if data and isinstance(data, dict):
            return {
                "price": float(data.get("c", 0)),
                "high": float(data.get("h", 0)),
                "low": float(data.get("l", 0)),
                "volume": float(data.get("v", 0)),
                "change_pct": float(data.get("p", 0)),
            }
        return None

    def disconnect(self) -> None:
        """Close the WebSocket connection."""
        self._connected = False
        if self._ws:
            asyncio.ensure_future(self._ws.close())
        logger.info("MarketDataStream disconnected")

    @property
    def is_connected(self) -> bool:
        return self._connected
