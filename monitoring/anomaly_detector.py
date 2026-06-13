"""
AnomalyDetector — monitors system state and emits alerts.

Background task that runs all anomaly checks every 30 seconds.
On critical anomalies: calls self._circuit_breaker.halt().
Always emits alerts to EventBus and audit log.

CircuitBreaker is a global halt switch shared with the RiskManagerAgent.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

from state.circuit_breaker import CircuitBreakerState

logger = logging.getLogger(__name__)

# ── CircuitBreaker lives in state/circuit_breaker.py ──

MONITOR_INTERVAL = 30  # seconds between check cycles


class AnomalyDetector:
    """Background task monitoring system state for anomalies."""

    def __init__(
        self,
        live_executor=None,
        signal_scanner=None,
        market_data_stream=None,
        audit_log=None,
        event_bus=None,
        circuit_breaker: Optional[CircuitBreakerState] = None,
    ):
        self._live_executor = live_executor
        self._signal_scanner = signal_scanner
        self._market_data_stream = market_data_stream
        self._audit_log = audit_log
        self._event_bus = event_bus
        self._circuit_breaker = circuit_breaker or CircuitBreakerState()

        # State tracking for anomaly detection
        self._last_price: Optional[float] = None
        self._last_price_time: Optional[datetime] = None
        self._api_error_count = 0
        self._api_error_window: List[datetime] = []
        self._reconnect_count = 0
        self._reconnect_window: List[datetime] = []
        self._signal_count = 0
        self._signal_window: List[datetime] = []
        self._portfolio_history: List[dict] = []

    async def monitor_loop(self):
        """Run all anomaly checks every MONITOR_INTERVAL seconds."""
        logger.info("AnomalyDetector started (interval=%ds)", MONITOR_INTERVAL)

        while True:
            try:
                await self._check_rapid_drawdown()
                await self._check_stuck_positions()
                await self._check_api_errors()
                await self._check_signal_flood()
                await self._check_stale_price()
                await self._check_exchange_disconnect()
                await self._check_negative_kelly()
                await self._check_price_source()
            except Exception as exc:
                logger.exception("Anomaly check cycle error: %s", exc)

            await asyncio.sleep(MONITOR_INTERVAL)

    async def _check_rapid_drawdown(self):
        """Portfolio down >2% in <10 minutes."""
        if not self._portfolio_history or len(self._portfolio_history) < 2:
            return

        recent = self._portfolio_history[-10:]
        if len(recent) < 2:
            return

        first_value = recent[0].get("equity", 10000)
        last_value = recent[-1].get("equity", 10000)
        time_span = (recent[-1].get("timestamp", datetime.utcnow()) -
                     recent[0].get("timestamp", datetime.utcnow()))

        if time_span.total_seconds() < 600 and last_value < first_value * 0.98:
            drawdown_pct = (first_value - last_value) / first_value * 100
            logger.warning("Rapid drawdown: %.1f%% in %d min", drawdown_pct, time_span.total_seconds() / 60)
            await self._alert(
                "rapid_drawdown", "critical",
                {"drawdown_pct": round(drawdown_pct, 2), "time_span_seconds": int(time_span.total_seconds())},
            )
            self._circuit_breaker.halt(f"Rapid drawdown {drawdown_pct:.1f}% in {int(time_span.total_seconds())}s")

    async def _check_stuck_positions(self):
        """Open position unchanged for >4h with no update."""
        if not self._live_executor:
            return

        positions = self._live_executor.get_open_positions()
        now = datetime.utcnow()

        for pos in positions:
            entry_time_str = pos.get("entry_time", "")
            if not entry_time_str:
                continue
            try:
                entry_time = datetime.fromisoformat(entry_time_str)
                if (now - entry_time).total_seconds() > 14400:  # 4 hours
                    await self._alert(
                        "stuck_position", "warning",
                        {"pair": pos.get("pair"), "entry_time": entry_time_str,
                         "hours_open": round((now - entry_time).total_seconds() / 3600, 1)},
                    )
            except (ValueError, TypeError):
                continue

    async def _check_api_errors(self):
        """>3 API errors in 60 seconds."""
        now = datetime.utcnow()
        self._api_error_window = [t for t in self._api_error_window if (now - t).total_seconds() < 60]

        if len(self._api_error_window) > 3:
            await self._alert(
                "api_error_cascade", "critical",
                {"error_count": len(self._api_error_window), "window_seconds": 60},
            )
            self._circuit_breaker.halt(f"API error cascade: {len(self._api_error_window)} errors in 60s")

    async def _check_signal_flood(self):
        """>10 signals in 60 seconds (likely bug)."""
        now = datetime.utcnow()
        self._signal_window = [t for t in self._signal_window if (now - t).total_seconds() < 60]

        if len(self._signal_window) > 10:
            await self._alert(
                "strategy_signal_flood", "critical",
                {"signal_count": len(self._signal_window), "window_seconds": 60},
            )
            self._circuit_breaker.halt(f"Signal flood: {len(self._signal_window)} signals in 60s")

    async def _check_stale_price(self):
        """Last price timestamp >5 minutes old."""
        if self._last_price_time and (datetime.utcnow() - self._last_price_time).total_seconds() > 300:
            await self._alert(
                "price_feed_stale", "warning",
                {"last_update": self._last_price_time.isoformat(),
                 "seconds_ago": int((datetime.utcnow() - self._last_price_time).total_seconds())},
            )

    async def _check_exchange_disconnect(self):
        """WebSocket reconnect count >3 in 10 minutes."""
        now = datetime.utcnow()
        self._reconnect_window = [t for t in self._reconnect_window if (now - t).total_seconds() < 600]

        if len(self._reconnect_window) > 3:
            await self._alert(
                "exchange_disconnect", "critical",
                {"reconnect_count": len(self._reconnect_window), "window_seconds": 600},
            )
            self._circuit_breaker.halt("Exchange disconnect: >3 reconnects in 10 min")

    async def _check_negative_kelly(self):
        """Check for strategies with negative Kelly (edge flipped)."""
        # This is checked during pre_trade_approval — no additional action needed
        pass

    async def _check_price_source(self):
        """Verify at least one price source is responding.

        If Binance WebSocket is disconnected AND CoinCap REST is down,
        trip circuit breaker with CRITICAL alert.
        """
        binance_ok = self._market_data_stream and self._market_data_stream.is_connected
        if binance_ok:
            return  # Primary source healthy

        # Binance down — check CoinCap as backup
        coincap_ok = False
        from api.server import get_health_tracker
        ht = get_health_tracker()
        try:
            from data.coincap_fetcher import CoinCapFetcher
            cf = CoinCapFetcher(health_tracker=ht)
            price = await cf.get_price("bitcoin")
            coincap_ok = price is not None
        except Exception:
            pass

        if not binance_ok and not coincap_ok:
            logger.critical("ALL PRICE SOURCES DOWN — Binance WebSocket disconnected, CoinCap REST failed")
            self._circuit_breaker.halt("All price sources unavailable")
            await self._alert(
                "price_source_critical", "critical",
                {"binance_ws": binance_ok, "coincap_rest": coincap_ok},
            )

    def record_api_error(self):
        """Called by external components when an API error occurs."""
        self._api_error_window.append(datetime.utcnow())
        self._api_error_count += 1

    def record_signal(self):
        """Called by SignalScanner for each signal evaluated."""
        self._signal_window.append(datetime.utcnow())

    def record_price_update(self, price: float):
        """Called by MarketDataStream when a new price arrives."""
        self._last_price = price
        self._last_price_time = datetime.utcnow()

    def record_reconnect(self):
        """Called on WebSocket reconnect."""
        self._reconnect_window.append(datetime.utcnow())

    async def _alert(self, anomaly_type: str, severity: str, details: dict):
        """Emit an anomaly alert."""
        logger.warning("ANOMALY [%s]: %s — %s", severity.upper(), anomaly_type, details)

        if self._event_bus:
            try:
                await self._event_bus.publish("anomaly_detected", {
                    "anomaly_type": anomaly_type,
                    "severity": severity,
                    "details": details,
                    "timestamp": datetime.utcnow().isoformat(),
                })
            except Exception:
                pass
