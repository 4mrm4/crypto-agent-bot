"""On-chain and whale data fetchers."""
import logging
from typing import Optional
import httpx
from config import settings

logger = logging.getLogger(__name__)


class OnChainFetcher:
    """
    Fetches on-chain signals. Requires WHALE_ALERT_API_KEY for whale
    transactions. Exchange netflow uses a public endpoint.
    """

    def get_whale_transactions(
        self,
        min_usd: float = 1_000_000,
        limit: int = 10,
    ) -> list:
        """
        Fetch recent large transactions via Whale Alert.
        Returns list of {blockchain, symbol, amount_usd, from, to, timestamp}.
        Falls back to empty list if no API key.
        """
        api_key = getattr(settings, "WHALE_ALERT_API_KEY", "")
        if not api_key:
            logger.debug("WHALE_ALERT_API_KEY not set — skipping whale data")
            return []
        try:
            r = httpx.get(
                "https://api.whale-alert.io/v1/transactions",
                params={
                    "api_key": api_key,
                    "min_value": int(min_usd),
                    "limit": limit,
                },
                timeout=10,
            )
            txns = r.json().get("transactions", [])
            return [
                {
                    "blockchain": t.get("blockchain", ""),
                    "symbol": t.get("symbol", "").upper(),
                    "amount_usd": t.get("amount_usd", 0),
                    "from_owner": t.get("from", {}).get("owner", "unknown"),
                    "to_owner": t.get("to", {}).get("owner", "unknown"),
                    "timestamp": t.get("timestamp", 0),
                }
                for t in txns
            ]
        except Exception as e:
            logger.warning("Whale Alert fetch failed: %s", e)
            return []

    def get_exchange_netflow_signal(self, symbol: str = "BTC") -> dict:
        """
        Derive a simple exchange netflow signal from CoinGecko volume data.
        Negative netflow (coins leaving exchanges) = bullish accumulation signal.
        This is a proxy — real netflow needs Glassnode/CryptoQuant premium.
        Returns {'signal': 'accumulation'|'distribution'|'neutral', 'confidence': float}
        """
        try:
            coin_map = {"BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana"}
            coin_id = coin_map.get(symbol.upper(), symbol.lower())
            r = httpx.get(
                f"https://api.coingecko.com/api/v3/coins/{coin_id}",
                params={"localization": "false", "tickers": "false",
                        "market_data": "true", "community_data": "false"},
                timeout=10,
            )
            data = r.json().get("market_data", {})
            vol_24h = data.get("total_volume", {}).get("usd", 0)
            market_cap = data.get("market_cap", {}).get("usd", 1)
            price_change_24h = data.get("price_change_percentage_24h", 0)

            # High volume + price drop = distribution (selling into strength)
            # High volume + price rise = accumulation
            vol_ratio = vol_24h / market_cap if market_cap else 0
            if vol_ratio > 0.05 and price_change_24h > 1:
                return {"signal": "accumulation", "confidence": min(vol_ratio * 10, 1.0)}
            elif vol_ratio > 0.05 and price_change_24h < -1:
                return {"signal": "distribution", "confidence": min(vol_ratio * 10, 1.0)}
            return {"signal": "neutral", "confidence": 0.3}
        except Exception as e:
            logger.warning("CoinGecko netflow proxy failed: %s", e)
            return {"signal": "neutral", "confidence": 0.0}

    def get_onchain_report(self, symbol: str = "BTC") -> dict:
        """Full on-chain snapshot."""
        if not getattr(settings, "ENABLE_ONCHAIN", False):
            return {}
        whales = self.get_whale_transactions()
        btc_whales = [w for w in whales if w["symbol"] == symbol.replace("/USDT", "")]
        netflow = self.get_exchange_netflow_signal(symbol.replace("/USDT", ""))
        return {
            "symbol": symbol,
            "whale_transactions": btc_whales[:5],
            "large_whale_count": len(btc_whales),
            "netflow": netflow,
            "summary": (
                f"{len(btc_whales)} large {symbol} transactions detected. "
                f"Exchange flow signal: {netflow['signal']} "
                f"(confidence={netflow['confidence']:.0%})"
            ),
        }
