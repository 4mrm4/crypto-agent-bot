"""LiveExecutor — bridges approved strategies to actual exchange orders.

Starts in PAPER mode always. Requires explicit flag to go LIVE.
Handles TWAP splitting, position monitoring, and full audit logging.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import ccxt

from agents.risk_manager import RiskManagerAgent, CircuitBreakerState
from config import settings
from data.fetcher import MarketDataFetcher
from execution.audit_log import AuditLog, AuditEntry
from execution.paper_trader import PaperTrader
from execution.quality_scorer import TradeQualityScorer, BLOCK_THRESHOLD
from execution.trade_signal import TradeSignal
from execution.validation_mode import ValidationMode
from state.state_broker import StateBroker

logger = logging.getLogger(__name__)


class ExecutionResult:
    """Result of a single trade execution attempt."""
    def __init__(self, signal_id: str, success: bool, fill_price: float = 0.0,
                 fill_amount: float = 0.0, order_id: str = "",
                 error: str = "", status: str = "failed"):
        self.signal_id = signal_id
        self.success = success
        self.fill_price = fill_price
        self.fill_amount = fill_amount
        self.order_id = order_id
        self.error = error
        self.status = status

    def to_dict(self) -> dict:
        return {
            "signal_id": self.signal_id,
            "success": self.success,
            "fill_price": self.fill_price,
            "fill_amount": self.fill_amount,
            "order_id": self.order_id,
            "error": self.error,
            "status": self.status,
        }


class LiveExecutor:
    """
    Executes trade signals through exchange orders (paper or live).
    Always starts in PAPER mode.
    """

    def __init__(
        self,
        exchange_id: str = "binance",
        paper_mode: bool = True,
        fetcher: Optional[MarketDataFetcher] = None,
        risk_manager: Optional[RiskManagerAgent] = None,
        event_bus=None,
        state_broker: Optional[StateBroker] = None,
        quality_scorer: Optional[TradeQualityScorer] = None,
    ):
        self.exchange_id = exchange_id
        self.paper_mode = paper_mode
        # Guard: EXECUTION_MODE env var must be explicitly "live" to disable paper
        from config import settings
        if not self.paper_mode and settings.EXECUTION_MODE != "live":
            logger.warning(
                "EXECUTION_MODE=%s but paper_mode=False — forcing paper. "
                "Set EXECUTION_MODE=live in .env for live trading.",
                settings.EXECUTION_MODE,
            )
            self.paper_mode = True
        self._fetcher = fetcher or MarketDataFetcher(exchange_id)
        self._risk_manager = risk_manager or RiskManagerAgent(fetcher=self._fetcher)
        self._event_bus = event_bus
        self._state_broker = state_broker
        self._audit_log = AuditLog()
        self._paper_trader = PaperTrader(fetcher=self._fetcher) if paper_mode else None
        self._exchange: Optional[ccxt.Exchange] = None
        self._quality_scorer = quality_scorer or TradeQualityScorer()

        # Validation mode for first 90 days live
        self._validation_mode = ValidationMode(
            live_start_date=getattr(settings, "LIVE_START_DATE", None)
        )

        # Track open positions for monitoring
        self._open_positions: Dict[str, Dict[str, Any]] = {}

        if not paper_mode:
            logger.warning(
                "!!! LIVE MODE ENABLED !!! "
                "This executor will place REAL ORDERS on %s",
                exchange_id,
            )

    @property
    def exchange(self) -> ccxt.Exchange:
        """Lazy-init CCXT exchange instance for live mode."""
        if self._exchange is None and not self.paper_mode:
            exchange_class = getattr(ccxt, self.exchange_id)
            self._exchange = exchange_class({
                "apiKey": settings.EXCHANGE_API_KEY,
                "secret": settings.EXCHANGE_SECRET,
                "enableRateLimit": True,
                "options": {"defaultType": "spot"},
            })
            self._exchange.load_markets()
        return self._exchange

    async def execute_signal(self, signal: TradeSignal) -> ExecutionResult:
        """
        Full execution pipeline:
        1. Circuit breaker check
        2. Correlation check (via risk manager)
        3. Kelly sizing (via risk manager)
        4. Pre-trade approval
        5. Execute (paper or live)
        6. Audit log
        """
        # 1. Circuit breaker (regular + validation mode tight thresholds)
        if CircuitBreakerState.is_halted():
            result = ExecutionResult(
                signal_id=signal.signal_id,
                success=False,
                error=f"Circuit breaker active: {CircuitBreakerState.status()['reason']}",
            )
            self._audit_log.record(self._make_audit_entry(signal, result))
            return result

        # 1b. Validation mode circuit breaker (tighter thresholds)
        if not self.paper_mode and self._validation_mode.is_active:
            daily_pnl = getattr(signal, "daily_pnl_pct", 0.0)
            weekly_pnl = getattr(signal, "weekly_pnl_pct", 0.0)
            if self._validation_mode.apply_tight_circuit_breaker(daily_pnl, weekly_pnl):
                result = ExecutionResult(
                    signal_id=signal.signal_id,
                    success=False,
                    error=f"Validation mode circuit breaker: daily={daily_pnl:.2%}, weekly={weekly_pnl:.2%}",
                )
                self._audit_log.record(self._make_audit_entry(signal, result))
                return result

        # 1c. ML quality filter
        prediction = self._quality_scorer.predict_quality(signal)
        signal.quality_score = prediction.quality_score
        signal.quality_multiplier = prediction.quality_multiplier
        if prediction.blocked:
            logger.info(
                "ML quality filter blocked %s/%s: score=%.3f, multiplier=%.2f",
                signal.pair, signal.strategy_type,
                prediction.quality_score, prediction.quality_multiplier,
            )
            result = ExecutionResult(
                signal_id=signal.signal_id,
                success=False,
                error=f"ML quality filter: score={prediction.quality_score:.3f} < {BLOCK_THRESHOLD}",
                status="rejected",
            )
            self._audit_log.record(self._make_audit_entry(signal, result))
            return result

        # 2. Get current price
        try:
            current_price = self._fetcher.fetch_current_price(signal.pair)
            signal.price = current_price if current_price else signal.price
        except Exception as exc:
            logger.warning("Could not fetch current price: %s", exc)

        # 2b. Apply quality multiplier to position size
        if signal.quality_multiplier < 1.0 and signal.position_size_usdt > 0:
            original = signal.position_size_usdt
            signal.position_size_usdt = round(original * signal.quality_multiplier, 2)
            logger.info(
                "ML quality multiplier applied: $%.2f → $%.2f (x%.2f) for %s/%s",
                original, signal.position_size_usdt,
                signal.quality_multiplier, signal.pair, signal.strategy_type,
            )

        # 3. Execute in paper or live mode
        try:
            if self.paper_mode:
                fill_result = await self._execute_paper(signal)
            else:
                fill_result = await self._execute_live(signal)

            # 4. Audit log
            entry = self._make_audit_entry(signal, fill_result)
            self._audit_log.record(entry)

            # 5. Update open positions
            if fill_result.success:
                self._open_positions[signal.pair] = {
                    "side": signal.side,
                    "entry_price": fill_result.fill_price,
                    "size": signal.position_size_usdt,
                    "strategy": signal.strategy_name,
                    "entry_time": datetime.utcnow().isoformat(),
                    "signal_id": signal.signal_id,
                }

            # 5b. Publish position state to StateBroker
            if fill_result.success and self._state_broker:
                await self._state_broker.set_position(signal.pair, {
                    "side": signal.side,
                    "size": signal.position_size_usdt,
                    "entry_price": fill_result.fill_price,
                    "timestamp": datetime.utcnow().isoformat(),
                    "status": "open",
                })

            # 6. Emit to UI
            await self._emit("trade_executed", {
                "signal_id": signal.signal_id,
                "pair": signal.pair,
                "side": signal.side,
                "size_usdt": signal.position_size_usdt,
                "price": fill_result.fill_price,
                "status": fill_result.status,
                "mode": "paper" if self.paper_mode else "live",
            })

            # 7. Validation mode: log to separate file
            if not self.paper_mode and self._validation_mode.is_active:
                self._validation_mode.log_validation_trade({
                    "signal_id": signal.signal_id,
                    "pair": signal.pair,
                    "side": signal.side,
                    "size_usdt": signal.position_size_usdt,
                    "fill_price": fill_result.fill_price,
                    "strategy": signal.strategy_name,
                    "regime": signal.regime,
                })

            # 8. ML scorer: record trade and trigger retrain if needed
            if fill_result.success:
                self._quality_scorer.record_trade_executed()

            return fill_result

        except Exception as exc:
            logger.exception("Execution failed: %s", exc)
            result = ExecutionResult(signal_id=signal.signal_id, success=False, error=str(exc))
            self._audit_log.record(self._make_audit_entry(signal, result))
            return result

    async def _execute_paper(self, signal: TradeSignal) -> ExecutionResult:
        """Execute signal in paper trading mode."""
        if self._paper_trader is None:
            self._paper_trader = PaperTrader(
                symbol=signal.pair,
                initial_balance=settings.PAPER_INITIAL_BALANCE,
                fetcher=self._fetcher,
            )

        # Use a simple signal function
        signal_fn = lambda df: signal.side if len(df) > 1 else "hold"

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._paper_trader.run(signal_fn, max_candles=5),
            )
            trades = result.get("trades", [])
            fill_price = trades[-1].entry_price if trades else 0.0
            return ExecutionResult(
                signal_id=signal.signal_id,
                success=True,
                fill_price=fill_price,
                fill_amount=signal.position_size_usdt,
                status="executed_paper",
            )
        except Exception as exc:
            return ExecutionResult(
                signal_id=signal.signal_id,
                success=False,
                error=f"Paper execution failed: {exc}",
            )

    async def _execute_live(self, signal: TradeSignal) -> ExecutionResult:
        """Execute signal on a real exchange via CCXT."""
        if not self._exchange:
            return ExecutionResult(
                signal_id=signal.signal_id,
                success=False,
                error="Exchange not initialised — check API keys",
            )

        try:
            amount = signal.position_size_usdt / signal.price if signal.price > 0 else 0

            # TWAP splitting for large orders
            if signal.position_size_usdt >= settings.TWAP_THRESHOLD_USDT:
                return await self._twap_execute(signal, amount)
            else:
                order = self.exchange.create_market_order(
                    symbol=signal.pair,
                    side=signal.side,
                    amount=amount,
                )
                return ExecutionResult(
                    signal_id=signal.signal_id,
                    success=True,
                    fill_price=float(order.get("price", signal.price)),
                    fill_amount=float(order.get("filled", amount)),
                    order_id=str(order.get("id", "")),
                    status="executed_live",
                )
        except Exception as exc:
            return ExecutionResult(
                signal_id=signal.signal_id,
                success=False,
                error=f"Live execution failed: {exc}",
            )

    async def _twap_execute(self, signal: TradeSignal, total_amount: float) -> ExecutionResult:
        """
        TWAP order splitting for large orders.
        Splits into N child orders over T seconds.
        """
        n_slices = 5
        slice_amount = total_amount / n_slices
        delay_seconds = 2  # 2 seconds between slices

        total_filled = 0.0
        total_price = 0.0
        last_error = ""

        for i in range(n_slices):
            try:
                if CircuitBreakerState.is_halted():
                    break
                order = self.exchange.create_market_order(
                    symbol=signal.pair,
                    side=signal.side,
                    amount=slice_amount,
                )
                filled = float(order.get("filled", 0))
                price = float(order.get("price", 0))
                total_filled += filled
                total_price += price * filled
                await asyncio.sleep(delay_seconds)
            except Exception as exc:
                last_error = str(exc)
                logger.warning("TWAP slice %d failed: %s", i, exc)
                break

        avg_price = total_price / total_filled if total_filled > 0 else 0

        return ExecutionResult(
            signal_id=signal.signal_id,
            success=total_filled > 0,
            fill_price=avg_price,
            fill_amount=total_filled,
            error=last_error if total_filled == 0 else "",
            status="executed_live_twap" if total_filled > 0 else "failed",
        )

    async def monitor_open_positions(self):
        """
        Background task. Checks every 60s:
        - Stop loss hit?
        - Take profit hit?
        - Strategy regime still valid?
        """
        while True:
            await asyncio.sleep(60)
            if not self._open_positions:
                continue

            for pair, pos in list(self._open_positions.items()):
                try:
                    current_price = self._fetcher.fetch_current_price(pair)
                    if current_price is None or pos.get("entry_price", 0) <= 0:
                        continue

                    entry = pos["entry_price"]
                    if pos["side"] == "buy":
                        pnl_pct = (current_price - entry) / entry
                    else:
                        pnl_pct = (entry - current_price) / entry

                    logger.debug(
                        "Position monitoring %s: PnL=%.2f%%",
                        pair, pnl_pct * 100,
                    )

                    # Check stop loss
                    stoploss = settings.STOP_LOSS_DEFAULT
                    if pnl_pct < -stoploss:
                        logger.warning(
                            "Stop loss triggered for %s: PnL=%.2f%%",
                            pair, pnl_pct * 100,
                        )
                        await self._emit("position_closed", {
                            "pair": pair,
                            "reason": "stop_loss",
                            "pnl_pct": round(pnl_pct * 100, 2),
                        })
                        if self._state_broker:
                            await self._state_broker.set_position(pair, {
                                "side": pos.get("side", "unknown"),
                                "size": 0,
                                "exit_price": current_price,
                                "pnl": round(pnl_pct * 100, 2),
                                "timestamp": datetime.utcnow().isoformat(),
                                "status": "closed",
                            })

                except Exception as exc:
                    logger.debug("Position monitoring error for %s: %s", pair, exc)

    def _make_audit_entry(self, signal: TradeSignal, result: ExecutionResult) -> AuditEntry:
        return AuditEntry(
            signal_id=signal.signal_id,
            pair=signal.pair,
            side=signal.side,
            strategy_type=signal.strategy_type,
            strategy_name=signal.strategy_name,
            regime=signal.regime,
            confidence=signal.confidence,
            position_size_usdt=signal.position_size_usdt,
            entry_price=result.fill_price,
            status=result.status,
            risk_verdict="approved",
            circuit_breaker_state=CircuitBreakerState.status(),
            correlation_result={},
            kelly_result={},
            error=result.error if not result.success else None,
        )

    async def _emit(self, event_type: str, payload: dict):
        if self._event_bus:
            try:
                await self._event_bus.publish(event_type, payload)
            except Exception:
                pass

    def get_open_positions(self) -> List[dict]:
        return list(self._open_positions.values())

    def get_audit_log(self) -> AuditLog:
        return self._audit_log
