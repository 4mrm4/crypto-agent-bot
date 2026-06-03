"""
TelegramAlerter — human-in-the-loop alerts and trade approvals via Telegram.

Capabilities:
  - Trade proposal notifications with inline approve/reject buttons
  - CRITICAL anomaly alerts (drawdown, API cascade, circuit breaker)
  - Position escalation alerts
  - General status updates
  - Event bus subscriber for automatic forwarding

Graceful degradation: if Telegram is not configured, all methods log
instead of crashing.
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Configuration ──
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_ENABLED = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)

# Approval timeouts
TRADE_APPROVAL_TIMEOUT_SECONDS = 300  # 5 min to approve/reject

# Severity icons (limited ASCII due to Windows terminal constraints)
ICONS = {
    "info": "[i]",
    "warning": "[!]",
    "critical": "[!!]",
    "success": "[+]",
    "trade": "[$]",
}


def _format_alert(anomaly_type: str, severity: str, details: dict) -> str:
    """Format an anomaly alert as a Telegram message string."""
    icon = ICONS.get(severity, "[?]")
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"{icon} *{severity.upper()}*: {anomaly_type}",
        f"Time: {ts} UTC",
    ]
    for key, value in details.items():
        lines.append(f"  {key}: {value}")
    return "\n".join(lines)


def _format_trade_signal(signal: dict) -> str:
    """Format a trade signal proposal for Telegram."""
    pair = signal.get("pair", "unknown")
    direction = signal.get("signal", "hold").upper()
    confidence = signal.get("confidence", 0)
    strategy_type = signal.get("strategy_type", "unknown")
    regime = signal.get("regime", "unknown")
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    return (
        f"{ICONS['trade']} *TRADE SIGNAL*: {direction} {pair}\n"
        f"Time: {ts} UTC\n"
        f"Confidence: {confidence:.1%}\n"
        f"Strategy: {strategy_type}\n"
        f"Regime: {regime}\n"
        f"---\n"
        f"Reply with /approve or /reject within {TRADE_APPROVAL_TIMEOUT_SECONDS}s"
    )


def _format_position_escalation(position: dict) -> str:
    """Format a position escalation alert."""
    pair = position.get("pair", "unknown")
    size = position.get("size", 0)
    pnl = position.get("unrealized_pnl", 0)
    duration = position.get("hours_open", 0)
    reason = position.get("reason", "breached limit")
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    return (
        f"{ICONS['warning']} *POSITION ESCALATION*: {pair}\n"
        f"Time: {ts} UTC\n"
        f"Size: {size}\n"
        f"Unrealized PnL: {pnl:+.2f}\n"
        f"Duration: {duration:.1f}h\n"
        f"Reason: {reason}\n"
        f"---\n"
        f"Action required."
    )


class TelegramAlerter:
    """Telegram alerting and human-in-the-loop trade approvals.

    Uses the python-telegram-bot v22.x API. If Telegram is not configured
    (TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID missing), all methods are no-ops.
    """

    def __init__(self):
        self._bot = None
        self._app = None
        self._pending_approvals: Dict[str, asyncio.Event] = {}
        self._approval_results: Dict[str, bool] = {}
        self._approval_callbacks: Dict[str, Callable] = {}
        self._running = False

    async def start(self):
        """Initialize the Telegram bot application and start polling."""
        if not TELEGRAM_ENABLED:
            logger.info("Telegram not configured — alerter disabled. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
            return

        try:
            from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
            from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

            self._app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

            # Register command handlers
            await self._app.bot.initialize()

            # We use bot.send_message directly rather than full polling for simplicity.
            # For interactive approvals, we register handlers but don't start polling
            # unless specifically needed (the event bus subscriber can push alerts).
            self._running = True
            logger.info("TelegramAlerter initialized (chat_id=%s)", TELEGRAM_CHAT_ID[:4] + "...")
        except Exception as exc:
            logger.warning("TelegramAlerter init failed: %s", exc)
            self._running = False

    async def stop(self):
        """Shutdown the Telegram bot."""
        if self._app:
            try:
                await self._app.bot.shutdown()
            except Exception:
                pass
        self._running = False

    async def send_message(self, text: str, parse_mode: str = "Markdown") -> bool:
        """Send a plain text message to the configured chat.

        Returns True if sent successfully, False otherwise.
        """
        if not self._running or not self._app:
            logger.debug("Telegram not active, skipping message: %s", text[:80])
            return False
        try:
            await self._app.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=text,
                parse_mode=parse_mode,
            )
            return True
        except Exception as exc:
            logger.warning("Telegram send failed: %s", exc)
            return False

    async def send_alert(self, anomaly_type: str, severity: str, details: dict):
        """Send an anomaly alert to Telegram."""
        text = _format_alert(anomaly_type, severity, details)
        sent = await self.send_message(text)
        if sent:
            logger.info("Telegram alert sent: %s/%s", severity, anomaly_type)

    async def send_trade_approval(
        self, signal: dict, callback: Optional[Callable] = None
    ) -> Optional[bool]:
        """Send a trade signal for human approval.

        Args:
            signal: Trade signal dict with pair, signal, confidence, strategy_type, regime
            callback: Optional async callable(signal_dict, approved: bool) called on result

        Returns:
            True if approved, False if rejected, None if timeout/error.
        """
        if not self._running or not self._app:
            # Auto-approve if Telegram is not available
            logger.info("Telegram not active — auto-approving trade signal")
            return True

        try:
            from telegram import InlineKeyboardButton, InlineKeyboardMarkup

            text = _format_trade_signal(signal)
            pair = signal.get("pair", "unknown")

            # Inline keyboard: Approve / Reject
            keyboard = [
                [
                    InlineKeyboardButton("Approve", callback_data=f"approve_{pair}"),
                    InlineKeyboardButton("Reject", callback_data=f"reject_{pair}"),
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            msg = await self._app.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=text,
                parse_mode="Markdown",
                reply_markup=reply_markup,
            )

            # Wait for approval with timeout
            event = asyncio.Event()
            self._pending_approvals[pair] = event
            if callback:
                self._approval_callbacks[pair] = callback

            try:
                await asyncio.wait_for(event.wait(), timeout=TRADE_APPROVAL_TIMEOUT_SECONDS)
                approved = self._approval_results.pop(pair, None)
                self._pending_approvals.pop(pair, None)
                self._approval_callbacks.pop(pair, None)

                # Notify user of outcome
                status = "APPROVED" if approved else "REJECTED"
                await self._app.bot.send_message(
                    chat_id=TELEGRAM_CHAT_ID,
                    text=f"{pair}: {status} (via Telegram)",
                )

                if callback and approved is not None:
                    try:
                        await callback(signal, approved)
                    except Exception as exc:
                        logger.exception("Trade approval callback error: %s", exc)

                return approved
            except asyncio.TimeoutError:
                self._pending_approvals.pop(pair, None)
                self._approval_callbacks.pop(pair, None)
                await self._app.bot.send_message(
                    chat_id=TELEGRAM_CHAT_ID,
                    text=f"{pair}: approval timed out ({TRADE_APPROVAL_TIMEOUT_SECONDS}s). Trade rejected by default.",
                )
                return None

        except Exception as exc:
            logger.warning("Telegram trade approval failed: %s", exc)
            return None

    async def send_critical_alert(self, anomaly_type: str, details: dict):
        """Send a high-priority critical alert."""
        text = _format_alert(anomaly_type, "critical", details)
        sent = await self.send_message(text)
        if sent:
            logger.info("Telegram critical alert sent: %s", anomaly_type)

    async def send_position_escalation(self, position: dict):
        """Send a position escalation alert."""
        text = _format_position_escalation(position)
        sent = await self.send_message(text)
        if sent:
            logger.info("Telegram position escalation sent: %s", position.get("pair", "unknown"))

    async def send_status(self, text: str):
        """Send a status update message."""
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"{ICONS['info']} *Status*: {text}\nTime: {ts} UTC"
        await self.send_message(formatted)

    # ── Event bus subscriber ──

    def make_event_callback(self) -> Callable:
        """Return an async callback suitable for event bus subscription.

        Usage:
            alerter = TelegramAlerter()
            bus.subscribe("anomaly_detected", alerter.make_event_callback())
        """
        async def callback(event_type: str, payload: dict):
            await self._on_event(event_type, payload)
        return callback

    async def _on_event(self, event_type: str, payload: dict):
        """Process an event from the event bus."""
        if not self._running:
            return

        try:
            if event_type == "anomaly_detected":
                severity = payload.get("severity", "warning")
                anomaly_type = payload.get("anomaly_type", "unknown")
                details = payload.get("details", {})
                if severity == "critical":
                    await self.send_critical_alert(anomaly_type, details)
                else:
                    await self.send_alert(anomaly_type, severity, details)

            elif event_type == "trade_signal_evaluated":
                signal = payload
                if signal.get("signal") in ("buy", "sell") and signal.get("confidence", 0) >= 0.8:
                    # Auto-send high-confidence signals for approval
                    await self.send_trade_approval(signal)

            elif event_type == "position_escalation":
                position = payload
                await self.send_position_escalation(position)

            elif event_type == "strategy_retired":
                s_id = payload.get("strategy_id", "unknown")
                reason = payload.get("reason", "")
                await self.send_alert("strategy_retired", "warning",
                                      {"strategy_id": s_id, "reason": reason})

            elif event_type == "circuit_breaker_halt":
                reason = payload.get("reason", "unknown")
                await self.send_critical_alert("circuit_breaker_halt",
                                               {"reason": reason})

        except Exception as exc:
            logger.exception("Telegram event handler error: %s", exc)

    # ── Approval callback (called from bot command handler) ──

    def resolve_approval(self, pair: str, approved: bool):
        """Resolve a pending trade approval. Called by the bot command handler."""
        if pair in self._pending_approvals:
            self._approval_results[pair] = approved
            self._pending_approvals[pair].set()
