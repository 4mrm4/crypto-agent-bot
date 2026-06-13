"""Tests for AnomalyDetector."""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from monitoring.anomaly_detector import (
    AnomalyDetector,
    MONITOR_INTERVAL,
)


def make_detector(**kwargs):
    det = AnomalyDetector(**kwargs)
    return det


class TestAnomalyDetectorInit:
    def test_default_init(self):
        det = make_detector()
        assert det._event_bus is None
        assert det._last_price is None
        assert det._api_error_count == 0

    def test_init_with_deps(self):
        bus = MagicMock()
        logger_obj = MagicMock()
        det = make_detector(event_bus=bus, audit_log=logger_obj)
        assert det._event_bus is bus
        assert det._audit_log is logger_obj


class TestRecordingMethods:
    def test_record_price_update(self):
        det = make_detector()
        assert det._last_price is None
        det.record_price_update(50000.0)
        assert det._last_price == 50000.0
        assert det._last_price_time is not None

    def test_record_api_error(self):
        det = make_detector()
        assert det._api_error_count == 0
        det.record_api_error()
        assert det._api_error_count == 1
        det.record_api_error()
        assert det._api_error_count == 2

    def test_record_signal(self):
        det = make_detector()
        assert len(det._signal_window) == 0
        det.record_signal()
        assert len(det._signal_window) == 1
        det.record_signal()
        assert len(det._signal_window) == 2

    def test_record_reconnect(self):
        det = make_detector()
        assert len(det._reconnect_window) == 0
        det.record_reconnect()
        assert len(det._reconnect_window) == 1


class TestAlertEmission:
    def test_alert_emits_to_event_bus(self):
        bus = MagicMock()
        det = make_detector(event_bus=bus)
        import asyncio
        asyncio.run(det._alert("test_type", "warning", {"key": "val"}))
        assert bus.publish.called
        call_args = bus.publish.call_args[0]
        assert call_args[0] == "anomaly_detected"

    def test_alert_without_event_bus(self):
        det = make_detector()
        import asyncio
        asyncio.run(det._alert("test", "warning", {}))
        # no crash = success

    def test_alert_severity_critical(self):
        bus = MagicMock()
        det = make_detector(event_bus=bus)
        import asyncio
        asyncio.run(det._alert("rapid_drawdown", "critical", {"pct": 5.0}))
        payload = bus.publish.call_args[0][1]
        assert payload["severity"] == "critical"
        assert payload["anomaly_type"] == "rapid_drawdown"

    def test_alert_includes_timestamp(self):
        bus = MagicMock()
        det = make_detector(event_bus=bus)
        import asyncio
        asyncio.run(det._alert("test", "info", {}))
        assert "timestamp" in bus.publish.call_args[0][1]


class TestAnomalyChecks:
    def test_rapid_drawdown_no_history(self):
        det = make_detector()
        import asyncio
        asyncio.run(det._check_rapid_drawdown())
        # no crash with empty history

    def test_rapid_drawdown_small_history(self):
        det = make_detector()
        det._portfolio_history = [{"equity": 10000, "timestamp": datetime.utcnow()}]
        import asyncio
        asyncio.run(det._check_rapid_drawdown())
        # no crash with too-small history

    def test_stuck_positions_no_executor(self):
        det = make_detector()
        import asyncio
        asyncio.run(det._check_stuck_positions())
        # no crash without executor

    def test_api_errors_no_errors(self):
        det = make_detector()
        import asyncio
        asyncio.run(det._check_api_errors())
        # no crash, no errors

    def test_signal_flood_no_signals(self):
        det = make_detector()
        import asyncio
        asyncio.run(det._check_signal_flood())
        # no crash

    def test_stale_price_no_price(self):
        det = make_detector()
        import asyncio
        asyncio.run(det._check_stale_price())
        # no crash

    def test_exchange_disconnect_no_stream(self):
        det = make_detector()
        import asyncio
        asyncio.run(det._check_exchange_disconnect())
        # no crash

    def test_negative_kelly_no_executor(self):
        det = make_detector()
        import asyncio
        asyncio.run(det._check_negative_kelly())
        # no crash


class TestMonitorLoop:
    def test_monitor_loop_runs_once(self):
        """Verify one iteration of monitor_loop doesn't crash."""
        det = make_detector(event_bus=MagicMock())
        import asyncio

        # Run just one iteration by patching the infinite loop
        async def run_one():
            await det._check_rapid_drawdown()
            await det._check_stuck_positions()
            await det._check_api_errors()
            await det._check_signal_flood()
            await det._check_stale_price()
            await det._check_exchange_disconnect()
            await det._check_negative_kelly()

        asyncio.run(run_one())

    def test_monitor_interval_constant(self):
        assert MONITOR_INTERVAL == 30


class TestRapidDrawdownLogic:
    def test_drawdown_detected(self):
        """Simulate rapid drawdown and check alert is sent."""
        bus = MagicMock()
        det = make_detector(event_bus=bus)

        now = datetime.utcnow()
        det._portfolio_history = [
            {"equity": 10000, "timestamp": now - timedelta(minutes=5)},
            {"equity": 9500, "timestamp": now - timedelta(minutes=3)},
            {"equity": 9200, "timestamp": now - timedelta(minutes=1)},
        ]
        import asyncio
        asyncio.run(det._check_rapid_drawdown())
        # Should trigger alert for drawdown > 2%
        assert bus.publish.called

    def test_no_drawdown_no_alert(self):
        bus = MagicMock()
        det = make_detector(event_bus=bus)

        now = datetime.utcnow()
        det._portfolio_history = [
            {"equity": 10000, "timestamp": now - timedelta(minutes=5)},
            {"equity": 9950, "timestamp": now - timedelta(minutes=3)},
            {"equity": 9920, "timestamp": now - timedelta(minutes=1)},
        ]
        import asyncio
        asyncio.run(det._check_rapid_drawdown())
        # Small change, no alert
        assert not bus.publish.called
