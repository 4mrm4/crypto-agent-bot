"""CoinCap v3 REST API client — backup price feed and OHLCV fallback.

Designed as a tertiary source when Binance and Bybit are both unavailable.
Only enabled when COINCAP_ENABLED=true and COINCAP_API_KEY is set.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import httpx
import pandas as pd

from config import settings

logger = logging.getLogger(__name__)

COINCAP_BASE = "https://rest.coincap.io/v3"


@dataclass
class CoinCapPrice:
    """Current price snapshot from CoinCap."""
    asset_id: str
    symbol: str
    price_usd: float
    market_cap_usd: Optional[float] = None
    volume_24h_usd: Optional[float] = None
    change_pct_24h: Optional[float] = None
    supply_circulating: Optional[float] = None
    fetched_at: datetime = field(default_factory=datetime.utcnow)
    source: str = "coincap"


class CoinCapFetcher:
    """Async REST client for CoinCap v3 API.

    Gracefully degrades: returns None on any error, never raises.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        enabled: Optional[bool] = None,
        health_tracker=None,
    ):
        self._api_key = api_key or settings.COINCAP_API_KEY
        self._enabled = enabled if enabled is not None else settings.COINCAP_ENABLED
        self._health_tracker = health_tracker
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=10)
        return self._client

    async def _get(self, path: str, params: Optional[dict] = None) -> Optional[dict]:
        if not self._enabled or not self._api_key:
            return None
        client = await self._get_client()
        url = f"{COINCAP_BASE}{path}"
        request_params = {"apiKey": self._api_key}
        if params:
            request_params.update(params)
        try:
            resp = await client.get(url, params=request_params)
            resp.raise_for_status()
            if self._health_tracker:
                self._health_tracker.record_success("coincap")
            return resp.json()
        except Exception as exc:
            logger.warning("CoinCap API error [%s]: %s", path, exc)
            if self._health_tracker:
                self._health_tracker.record_failure("coincap", str(exc))
            return None

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    async def get_price(self, asset_id: str) -> Optional[CoinCapPrice]:
        """Fetch current price for a single asset."""
        data = await self._get(f"/assets/{asset_id}")
        if not data or "data" not in data:
            return None
        d = data["data"]
        try:
            return CoinCapPrice(
                asset_id=d.get("id", asset_id),
                symbol=d.get("symbol", ""),
                price_usd=float(d.get("priceUsd", 0)),
                market_cap_usd=(
                    float(d["marketCapUsd"]) if d.get("marketCapUsd") else None
                ),
                volume_24h_usd=(
                    float(d["volumeUsd24Hr"]) if d.get("volumeUsd24Hr") else None
                ),
                change_pct_24h=(
                    float(d["changePercent24Hr"])
                    if d.get("changePercent24Hr") else None
                ),
                supply_circulating=(
                    float(d["supply"]) if d.get("supply") else None
                ),
            )
        except (ValueError, TypeError) as exc:
            logger.warning("CoinCap parse error for %s: %s", asset_id, exc)
            return None

    async def get_batch_prices(
        self, asset_ids: List[str],
    ) -> Dict[str, CoinCapPrice]:
        """Fetch prices for multiple assets concurrently."""
        import asyncio
        tasks = [self.get_price(aid) for aid in asset_ids]
        results = await asyncio.gather(*tasks)
        return {
            aid: result
            for aid, result in zip(asset_ids, results)
            if result is not None
        }

    async def get_ohlcv_fallback(
        self,
        asset_id: str,
        interval: str = "h1",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> Optional[pd.DataFrame]:
        """Fetch OHLCV history from CoinCap as a fallback data source.

        Returns a DataFrame with columns: timestamp, open, high, low, close, volume.
        Returns None on any error.
        """
        path = f"/assets/{asset_id}/history"
        params = {"interval": interval}
        if start:
            params["start"] = str(int(start.timestamp() * 1000))
        if end:
            params["end"] = str(int(end.timestamp() * 1000))

        data = await self._get(path, params=params)
        if not data or "data" not in data:
            return None
        rows_data = data["data"]
        if not rows_data:
            return None
        try:
            rows = []
            for d in rows_data:
                rows.append({
                    "timestamp": d["time"],
                    "open": float(d["open"]),
                    "high": float(d["high"]),
                    "low": float(d["low"]),
                    "close": float(d["close"]),
                    "volume": float(d["volume"]),
                })
            df = pd.DataFrame(rows)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            return df
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning("CoinCap OHLCV parse error [%s]: %s", asset_id, exc)
            return None

# ── Symbol mapping: CCXT pair format -> CoinCap asset ID ──

SYMBOL_TO_COINCAP: Dict[str, str] = {
    "BTC/USDT": "bitcoin",
    "ETH/USDT": "ethereum",
    "SOL/USDT": "solana",
    "BNB/USDT": "binancecoin",
    "XRP/USDT": "xrp",
    "ADA/USDT": "cardano",
    "DOGE/USDT": "dogecoin",
    "DOT/USDT": "polkadot",
    "MATIC/USDT": "matic",
    "LINK/USDT": "chainlink",
    "UNI/USDT": "uniswap",
    "ATOM/USDT": "cosmos",
    "LTC/USDT": "litecoin",
    "BCH/USDT": "bitcoin-cash",
    "AVAX/USDT": "avalanche",
}

# Inverse: CoinCap asset ID -> CCXT symbol
COINCAP_TO_SYMBOL: Dict[str, str] = {v: k for k, v in SYMBOL_TO_COINCAP.items()}


def symbol_to_coincap_id(symbol: str) -> Optional[str]:
    """Convert CCXT pair like 'BTC/USDT' to CoinCap asset ID like 'bitcoin'.

    Returns None if the symbol is not in the known mapping.
    """
    return SYMBOL_TO_COINCAP.get(symbol)


def coincap_id_to_symbol(asset_id: str) -> Optional[str]:
    """Reverse lookup: CoinCap asset ID -> CCXT symbol."""
    return COINCAP_TO_SYMBOL.get(asset_id)
