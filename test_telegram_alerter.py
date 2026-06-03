"""Tests for Telegram Alerter (Task 11)."""

import os
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from monitoring.telegram_alerter import (
    TelegramAlerter,
    _format_alert,
    _format_trade_signal,
    _format_position_escalation,
    TRADE_APPROVAL_TIMEOUT_SECONDS,
)


# ── Formatting Tests ──

def test_format_alert_critical():
    text = _format_alert("rapid_drawdown", "critical", {"drawdown_pct": 5.2})
    assert "CRITICAL" in text
    assert "rapid_drawdown" in text
    assert "drawdown_pct" in text


def test_format_alert_warning():
    text = _format_alert("stuck_position", "warning", {"pair": "BTC/USDT", "hours_open": 6})
    assert "WARNING" in text
    assert "BTC/USDT" in text


def test_format_trade_signal_buy():
    signal = {
        "pair": "ETH/USDT", "signal": "buy", "confidence": 0.85,
        "strategy_type": "momentum", "regime": "uptrend",
    }
    text = _format_trade_signal(signal)
    assert "BUY" in text
    assert "ETH/USDT" in text
    assert "momentum" in text
    assert "/approve" in text


def test_format_trade_signal_sell():
    signal = {
        "pair": "BTC/USDT", "signal": "sell", "confidence": 0.72,
        "strategy_type": "rsi", "regime": "ranging",
    }
    text = _format_trade_signal(signal)
    assert "SELL" in text
    assert "BTC/USDT" in text


def test_format_position_escalation():
    pos = {
        "pair": "SOL/USDT", "size": 5000, "unrealized_pnl": -250,
        "hours_open": 8.5, "reason": "max_drawdown_breach",
    }
    text = _format_position_escalation(pos)
    assert "SOL/USDT" in text
    assert "5000" in text
    assert "8.5" in text


# ── TelegramAlerter State Tests ──

def test_alerter_not_enabled_without_env():
    """Without env vars, TELEGRAM_ENABLED is False."""
    with patch.dict(os.environ, {}, clear=True):
        from monitoring.telegram_alerter import TELEGRAM_ENABLED
        assert not TELEGRAM_ENABLED


def test_alerter_start_no_config():
    """start() should not crash when Telegram is not configured."""
    with patch.dict(os.environ, {}, clear=True):
        alerter = TelegramAlerter()
        asyncio.run(alerter.start())
        assert not alerter._running


def test_send_message_not_running():
    """send_message returns False when not running."""
    alerter = TelegramAlerter()
    result = asyncio.run(alerter.send_message("test"))
    assert result is False


def test_send_message_with_mock():
    """send_message returns True when bot sends successfully."""
    alerter = TelegramAlerter()
    alerter._running = True
    alerter._app = MagicMock()
    alerter._app.bot.send_message = AsyncMock(return_value=True)
    result = asyncio.run(alerter.send_message("test message"))
    assert result is True


def test_send_message_failure():
    """send_message returns False on send exception."""
    alerter = TelegramAlerter()
    alerter._running = True
    alerter._app = MagicMock()
    alerter._app.bot.send_message = AsyncMock(side_effect=Exception("Network error"))
    result = asyncio.run(alerter.send_message("test"))
    assert result is False


def test_send_alert_disabled():
    """send_alert does not crash when disabled."""
    alerter = TelegramAlerter()
    asyncio.run(alerter.send_alert("test_type", "warning", {"key": "value"}))


def test_send_alert_enabled():
    """send_alert sends formatted message."""
    alerter = TelegramAlerter()
    alerter._running = True
    alerter._app = MagicMock()
    alerter._app.bot.send_message = AsyncMock(return_value=True)
    asyncio.run(alerter.send_alert("test_type", "critical", {"key": "val"}))
    assert alerter._app.bot.send_message.called


def test_send_trade_approval_disabled():
    """send_trade_approval auto-approves when Telegram disabled."""
    alerter = TelegramAlerter()
    signal = {"pair": "BTC/USDT", "signal": "buy", "confidence": 0.85}
    result = asyncio.run(alerter.send_trade_approval(signal))
    assert result is True  # auto-approved


def test_send_trade_approval_with_mock():
    """send_trade_approval sends inline keyboard and waits."""
    alerter = TelegramAlerter()
    alerter._running = True
    alerter._app = MagicMock()
    alerter._app.bot.send_message = AsyncMock()
    mock_msg = MagicMock()
    alerter._app.bot.send_message.return_value = mock_msg

    signal = {"pair": "BTC/USDT", "signal": "buy", "confidence": 0.85,
              "strategy_type": "momentum", "regime": "uptrend"}
    with patch("monitoring.telegram_alerter.TRADE_APPROVAL_TIMEOUT_SECONDS", 0.1):
        result = asyncio.run(alerter.send_trade_approval(signal))
        assert result is None  # timed out


def test_send_critical_alert():
    """send_critical_alert sends formatted critical alert."""
    alerter = TelegramAlerter()
    alerter._running = True
    alerter._app = MagicMock()
    alerter._app.bot.send_message = AsyncMock(return_value=True)
    asyncio.run(alerter.send_critical_alert("test_critical", {"reason": "testing"}))
    assert alerter._app.bot.send_message.called


def test_send_position_escalation():
    """send_position_escalation sends formatted escalation."""
    alerter = TelegramAlerter()
    alerter._running = True
    alerter._app = MagicMock()
    alerter._app.bot.send_message = AsyncMock(return_value=True)
    asyncio.run(alerter.send_position_escalation(
        {"pair": "SOL/USDT", "size": 5000, "unrealized_pnl": -250, "hours_open": 8, "reason": "test"}
    ))
    assert alerter._app.bot.send_message.called


def test_send_status():
    """send_status sends a status update."""
    alerter = TelegramAlerter()
    alerter._running = True
    alerter._app = MagicMock()
    alerter._app.bot.send_message = AsyncMock(return_value=True)
    asyncio.run(alerter.send_status("Bot is running"))
    assert alerter._app.bot.send_message.called


# ── Event Callback Tests ──

def test_make_event_callback():
    """make_event_callback returns a callable that processes events."""
    alerter = TelegramAlerter()
    alerter._running = True
    alerter._app = MagicMock()
    alerter._app.bot.send_message = AsyncMock(return_value=True)
    callback = alerter.make_event_callback()
    assert callable(callback)
    asyncio.run(callback("anomaly_detected", {
        "severity": "warning",
        "anomaly_type": "stuck_position",
        "details": {"pair": "BTC/USDT"},
    }))
    assert alerter._app.bot.send_message.called


def test_event_callback_critical():
    """Critical anomaly events trigger critical alert."""
    alerter = TelegramAlerter()
    alerter._running = True
    alerter._app = MagicMock()
    alerter._app.bot.send_message = AsyncMock(return_value=True)
    callback = alerter.make_event_callback()
    asyncio.run(callback("anomaly_detected", {
        "severity": "critical",
        "anomaly_type": "rapid_drawdown",
        "details": {"drawdown_pct": 5.0},
    }))
    assert alerter._app.bot.send_message.called


def test_event_callback_circuit_breaker():
    """Circuit breaker halt events trigger critical alert."""
    alerter = TelegramAlerter()
    alerter._running = True
    alerter._app = MagicMock()
    alerter._app.bot.send_message = AsyncMock(return_value=True)
    callback = alerter.make_event_callback()
    asyncio.run(callback("circuit_breaker_halt", {"reason": "Rapid drawdown"}))
    assert alerter._app.bot.send_message.called


def test_resolve_approval():
    """resolve_approval sets the approval result and triggers the event."""
    alerter = TelegramAlerter()
    event = asyncio.Event()
    alerter._pending_approvals["BTC/USDT"] = event
    alerter.resolve_approval("BTC/USDT", True)
    assert event.is_set()
    assert alerter._approval_results.get("BTC/USDT") is True


def test_resolve_approval_reject():
    """resolve_approval also works for reject."""
    alerter = TelegramAlerter()
    event = asyncio.Event()
    alerter._pending_approvals["ETH/USDT"] = event
    alerter.resolve_approval("ETH/USDT", False)
    assert alerter._approval_results.get("ETH/USDT") is False


def test_resolve_approval_nonexistent():
    """resolve_approval does not crash for unknown pair."""
    alerter = TelegramAlerter()
    alerter.resolve_approval("NONEXISTENT", True)
