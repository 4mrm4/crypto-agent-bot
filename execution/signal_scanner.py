"""
SignalScanner — continuously scans all pairs × approved strategies for entry signals.

Runs every N seconds. Only fires signals for strategies whose regime matches
the current detected regime. Passes approved signals to LiveExecutor.

Regime-Conditioned Gating (v10):
- Validated-regime matching: each strategy lists which regimes it was validated in
- Transition cooldown: regime shifts freeze signals for a configurable window
- Confidence gating: low regime confidence raises the signal confidence floor

Standby Mode (v11):
- When market is quiet (no regime transitions, no signals fired), the scanner
  enters standby: extends scan interval and skips strategy evaluation to save
  CPU/exchange API calls.
- Automatically wakes on regime transition, manual wake(), or after N idle cycles.
- Returns to standby after ACTIVE_CYCLES_WITHOUT_TRADE cycles with no signal execution.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

import pandas as pd

from config import settings
import talib
from data.fetcher import MarketDataFetcher
from data.regime import MarketRegimeDetector, RegimeSnapshot
from execution.live_executor import LiveExecutor
from execution.trade_signal import TradeSignal
from state.state_broker import StateBroker

logger = logging.getLogger(__name__)

SCAN_INTERVAL_SECONDS = 60   # Active mode: check every minute
STANDBY_INTERVAL_SECONDS = 300  # Standby mode: check every 5 minutes
ACTIVE_CYCLES_WITHOUT_TRADE = 10  # Return to standby after N silent active cycles

# ── Regime transition cooldown ──
TRANSITION_COOLDOWN_MINUTES = 30  # No signals for N minutes after regime shift
LOW_CONFIDENCE_THRESHOLD = 0.5    # Below this, raise signal floor
LOW_CONFIDENCE_SIGNAL_FLOOR = 0.75  # Harder entry when regime is uncertain


@dataclass
class RegimeTransition:
    """Record of a regime change event."""
    from_regime: str
    to_regime: str
    timestamp: datetime
    confidence: float


class RegimeTransitionTracker:
    """Tracks regime history and enforces cooldown on transitions."""

    def __init__(self, cooldown_minutes: int = TRANSITION_COOLDOWN_MINUTES):
        self._cooldown = timedelta(minutes=cooldown_minutes)
        self._history: List[RegimeTransition] = []
        self._current_regime: str = "unknown"
        self._current_confidence: float = 0.0

    def update(self, regime: str, confidence: float) -> Optional[RegimeTransition]:
        """Register a regime update. Returns a RegimeTransition if regime changed."""
        if regime != self._current_regime and self._current_regime != "unknown":
            transition = RegimeTransition(
                from_regime=self._current_regime,
                to_regime=regime,
                timestamp=datetime.now(timezone.utc),
                confidence=confidence,
            )
            self._history.append(transition)
            if len(self._history) > 50:
                self._history = self._history[-50:]
            self._current_regime = regime
            self._current_confidence = confidence
            return transition
        self._current_regime = regime
        self._current_confidence = confidence
        return None

    def is_in_cooldown(self) -> bool:
        """True if a recent transition is still within the cooldown window."""
        if not self._history:
            return False
        last_transition = self._history[-1]
        elapsed = datetime.now(timezone.utc) - last_transition.timestamp
        return elapsed < self._cooldown

    @property
    def current_regime(self) -> str:
        return self._current_regime

    @property
    def current_confidence(self) -> float:
        return self._current_confidence

    def last_transition(self) -> Optional[RegimeTransition]:
        return self._history[-1] if self._history else None

    def transition_count(self, window_hours: int = 24) -> int:
        """Count regime transitions in the last N hours. High counts indicate instability."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
        return sum(1 for t in self._history if t.timestamp >= cutoff)

    def effective_signal_floor(self, base_floor: float = 0.6) -> float:
        """Return the effective signal confidence floor adjusted for regime conditions."""
        if self.is_in_cooldown():
            return max(base_floor, LOW_CONFIDENCE_SIGNAL_FLOOR)
        if self._current_confidence < LOW_CONFIDENCE_THRESHOLD:
            return max(base_floor, LOW_CONFIDENCE_SIGNAL_FLOOR)
        # Flapping regime: many shifts in recent hours
        if self.transition_count(24) >= 5:
            return max(base_floor, 0.7)
        return base_floor


@dataclass
class SignalResult:
    """Result of evaluating a single strategy on a single pair."""
    signal: str          # "buy" | "sell" | "hold"
    confidence: float    # 0.0–1.0
    indicators: dict     # current indicator values
    pair: str
    strategy_type: str
    regime: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class SignalScanner:
    """
    Continuously scans all pairs for entry signals from approved strategies.
    """

    def __init__(
        self,
        pairs: Optional[List[str]] = None,
        approved_strategies: Optional[List[dict]] = None,
        regime_detector: Optional[MarketRegimeDetector] = None,
        live_executor: Optional[LiveExecutor] = None,
        fetcher: Optional[MarketDataFetcher] = None,
        vector_store=None,
        event_bus=None,
        scan_interval: int = SCAN_INTERVAL_SECONDS,
        standby_interval: int = STANDBY_INTERVAL_SECONDS,
        state_broker: Optional[StateBroker] = None,
    ):
        self._pairs = pairs or ["BTC/USDT", "ETH/USDT"]
        self._approved_strategies = approved_strategies or []
        self._regime_detector = regime_detector or MarketRegimeDetector()
        self._live_executor = live_executor
        self._fetcher = fetcher or MarketDataFetcher()
        self._vector_store = vector_store
        self._event_bus = event_bus
        self._state_broker = state_broker
        self._scan_interval = scan_interval
        self._standby_interval = standby_interval
        self._signal_history: List[SignalResult] = []
        self._regime_cache: Dict[str, str] = {}
        self._regime_tracker = RegimeTransitionTracker()
        # Standby mode state
        self._standby_mode: bool = True  # Start in standby (energy-saving)
        self._cycles_without_trade: int = 0

    def _update_approved_strategy_validated_regimes(self):
        """Ensure every approved strategy has a validated_regimes field."""
        for s in self._approved_strategies:
            if "validated_regimes" not in s:
                regime = s.get("regime", "unknown")
                s["validated_regimes"] = [regime] if regime != "unknown" else []

    async def scan_loop(self):
        """Main loop. Runs at standby interval by default; switches to active
        interval on regime transitions or after wake(). In standby, only
        regime detection runs (no strategy evaluation) to save CPU/API calls."""
        logger.info(
            "SignalScanner started (active_interval=%ds, standby_interval=%ds, pairs=%s)",
            self._scan_interval, self._standby_interval, self._pairs,
        )
        await self._emit("scanner_mode", {"mode": "standby", "reason": "initial"})

        while True:
            interval = self._standby_interval if self._standby_mode else self._scan_interval
            logger.debug("Scanner cycle: mode=%s, interval=%ds", "standby" if self._standby_mode else "active", interval)

            try:
                regime_changed = False
                for pair in self._pairs:
                    if self._standby_mode:
                        # Standby: only detect regime, no strategy evaluation
                        previous = self._regime_cache.get(pair)
                        current = await self._detect_pair_regime(pair)
                        if previous and current != previous and current != "unknown":
                            regime_changed = True
                            logger.info(
                                "Regime shift detected in standby: %s → %s — waking scanner",
                                previous, current,
                            )
                    else:
                        # Active: full scan with strategy evaluation
                        signals = await self._scan_pair(pair)
                        executed_any = False
                        for signal_result in signals:
                            self._signal_history.append(signal_result)
                            if len(self._signal_history) > 1000:
                                self._signal_history = self._signal_history[-1000:]

                            await self._emit("trade_signal_evaluated", {
                                "pair": signal_result.pair,
                                "signal": signal_result.signal,
                                "confidence": signal_result.confidence,
                                "strategy_type": signal_result.strategy_type,
                                "regime": signal_result.regime,
                            })

                            if self._state_broker:
                                await self._state_broker.set_signal(signal_result.pair, {
                                    "signal": signal_result.signal,
                                    "confidence": signal_result.confidence,
                                    "strategy_type": signal_result.strategy_type,
                                    "regime": signal_result.regime,
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                })

                            signal_floor = self._regime_tracker.effective_signal_floor(0.6)
                            if signal_result.signal in ("buy", "sell") and signal_result.confidence >= signal_floor:
                                await self._execute_signal(signal_result)
                                executed_any = True

                        # Track idle cycles to auto-return to standby
                        if executed_any:
                            self._cycles_without_trade = 0
                        else:
                            self._cycles_without_trade += 1

                # ── Standby ↔ active transitions ──
                if self._standby_mode and regime_changed:
                    self._standby_mode = False
                    self._cycles_without_trade = 0
                    await self._emit("scanner_mode", {"mode": "active", "reason": "regime_transition"})
                elif not self._standby_mode and self._cycles_without_trade >= ACTIVE_CYCLES_WITHOUT_TRADE:
                    self._standby_mode = True
                    self._cycles_without_trade = 0
                    logger.info("Scanner returning to standby after %d silent cycles", ACTIVE_CYCLES_WITHOUT_TRADE)
                    await self._emit("scanner_mode", {"mode": "standby", "reason": "idle_timeout"})

            except Exception as exc:
                logger.exception("Scan cycle error: %s", exc)

            await asyncio.sleep(interval)

    async def _scan_pair(self, pair: str) -> List[SignalResult]:
        """Evaluate all approved strategies on a single pair."""
        results = []

        # Get current regime for this pair
        regime = await self._detect_pair_regime(pair)

        # Get strategies approved for this regime
        strategies = self._get_strategies_for_regime(regime)
        if not strategies:
            return results

        # Get OHLCV data
        try:
            ohlcv = self._fetcher.fetch_ohlcv(pair, "1h", limit=100)
            if ohlcv is None or len(ohlcv) < 30:
                return results
        except Exception as exc:
            logger.debug("Could not fetch %s: %s", pair, exc)
            return results

        # Evaluate each strategy
        for strategy in strategies:
            signal_result = self._evaluate_single_strategy(strategy, ohlcv, regime, pair)
            if signal_result:
                results.append(signal_result)

        return results

    def _evaluate_single_strategy(
        self,
        strategy_config: dict,
        ohlcv: pd.DataFrame,
        regime: str,
        pair: str,
    ) -> Optional[SignalResult]:
        """Run a single strategy's entry logic on the latest candles."""
        strategy_type = strategy_config.get("strategy_type", "")
        params = strategy_config.get("params", {})

        try:
            # Compute basic indicators
            close = ohlcv["close"].astype(float)
            volume = ohlcv["volume"].astype(float) if "volume" in ohlcv else pd.Series([0] * len(ohlcv))

            latest_close = float(close.iloc[-1])
            latest_volume = float(volume.iloc[-1])

            # Strategy-specific signal evaluation
            signal = "hold"
            confidence = 0.0
            indicators = {"close": latest_close, "volume": latest_volume}

            if strategy_type == "sma_crossover":
                fast_period = int(params.get("fast_ma", 10))
                slow_period = int(params.get("slow_ma", 30))
                sma_fast = close.rolling(fast_period).mean()
                sma_slow = close.rolling(slow_period).mean()
                indicators["sma_fast"] = float(sma_fast.iloc[-1])
                indicators["sma_slow"] = float(sma_slow.iloc[-1])

                if len(sma_fast) > 1 and len(sma_slow) > 1:
                    if sma_fast.iloc[-1] > sma_slow.iloc[-1] and sma_fast.iloc[-2] <= sma_slow.iloc[-2]:
                        signal = "buy"
                        confidence = 0.65
                    elif sma_fast.iloc[-1] < sma_slow.iloc[-1] and sma_fast.iloc[-2] >= sma_slow.iloc[-2]:
                        signal = "sell"
                        confidence = 0.60

            elif strategy_type == "rsi_oversold":

                rsi = talib.RSI(close.values, timeperiod=int(params.get("rsi_period", 14)))
                indicators["rsi"] = float(rsi[-1]) if len(rsi) > 0 else 50

                if float(rsi[-1]) < float(params.get("rsi_buy_threshold", 30)):
                    signal = "buy"
                    confidence = 0.70
                elif float(rsi[-1]) > float(params.get("rsi_sell_threshold", 70)):
                    signal = "sell"
                    confidence = 0.65

            elif strategy_type == "bollinger_bands":

                period = int(params.get("bb_period", 20))
                upper, middle, lower = talib.BBANDS(close.values.astype(float), timeperiod=period, nbdevup=2, nbdevdn=2)
                indicators["bb_upper"] = float(upper[-1]) if len(upper) > 0 else 0
                indicators["bb_lower"] = float(lower[-1]) if len(lower) > 0 else 0

                if latest_close < float(lower[-1]):
                    signal = "buy"
                    confidence = 0.65
                elif latest_close > float(upper[-1]):
                    signal = "sell"
                    confidence = 0.60

            elif strategy_type == "mean_reversion":

                rsi = talib.RSI(close.values, timeperiod=14)
                period = int(params.get("bb_period", 20))
                upper, middle, lower = talib.BBANDS(close.values.astype(float), timeperiod=period, nbdevup=2, nbdevdn=2)
                indicators["rsi"] = float(rsi[-1]) if len(rsi) > 0 else 50
                indicators["bb_lower"] = float(lower[-1]) if len(lower) > 0 else 0

                if float(rsi[-1]) < 35 and latest_close < float(lower[-1]):
                    signal = "buy"
                    confidence = 0.75

            elif strategy_type == "momentum":

                roc = talib.ROC(close.values, timeperiod=int(params.get("roc_period", 10)))
                vol_sma = volume.rolling(20).mean()
                indicators["roc"] = float(roc[-1]) if len(roc) > 0 else 0
                indicators["vol_ratio"] = float(latest_volume / max(vol_sma.iloc[-1], 0.01)) if len(vol_sma) > 0 else 1

                if float(roc[-1]) > 2.0 and latest_volume > vol_sma.iloc[-1] * 1.5:
                    signal = "buy"
                    confidence = 0.70

            # Only return if we got a signal
            if signal != "hold":
                return SignalResult(
                    signal=signal,
                    confidence=confidence,
                    indicators=indicators,
                    pair=pair,
                    strategy_type=strategy_type,
                    regime=regime,
                )

        except Exception as exc:
            logger.debug("Strategy eval failed for %s/%s: %s", pair, strategy_type, exc)

        return None

    async def _detect_pair_regime(self, pair: str) -> str:
        """Detect current regime for a specific pair and update the regime tracker."""
        try:
            ohlcv = self._fetcher.fetch_ohlcv(pair, "1h", limit=250)
            if ohlcv is not None and len(ohlcv) > 200:
                snapshot = self._regime_detector.classify_regime_snapshot(ohlcv)
                regime = snapshot.regime
                confidence = snapshot.confidence
                self._regime_cache[pair] = regime

                # Update regime tracker and emit transition events
                transition = self._regime_tracker.update(regime, confidence)
                if transition:
                    logger.info(
                        "Regime transition: %s → %s (confidence=%.2f)",
                        transition.from_regime, transition.to_regime, transition.confidence,
                    )
                    asyncio.create_task(self._emit("regime_transition", {
                        "from_regime": transition.from_regime,
                        "to_regime": transition.to_regime,
                        "confidence": transition.confidence,
                        "timestamp": transition.timestamp.isoformat(),
                        "cooldown_minutes": TRANSITION_COOLDOWN_MINUTES,
                    }))
                return regime
        except Exception:
            pass
        return self._regime_cache.get(pair, "unknown")

    def _get_strategies_for_regime(self, regime: str) -> List[dict]:
        """Filter approved strategies to those matching the current regime.

        Two-layer gating:
        1. Strategy type must be in REGIME_STRATEGY_MAP[regime]["use"]
        2. Strategy must have the current regime in its validated_regimes list
        """
        self._update_approved_strategy_validated_regimes()

        from data.regime import REGIME_STRATEGY_MAP
        recommended = REGIME_STRATEGY_MAP.get(regime, {}).get("use", [])

        if not recommended:
            return []

        matched = []
        for s in self._approved_strategies:
            s_type = s.get("strategy_type", "")
            if s_type not in recommended:
                continue
            # Check validated_regimes gate
            validated = s.get("validated_regimes")
            if validated is not None:
                # validated_regimes key is present: empty = not validated, skip
                if regime not in validated:
                    logger.debug(
                        "Strategy %s validated for %s, not current regime %s — skipping",
                        s_type, validated, regime,
                    )
                    continue
            matched.append(s)

        return matched  # only return regime-matched strategies

    async def _execute_signal(self, signal_result: SignalResult):
        """Convert a SignalResult to a TradeSignal and execute it."""
        if not self._live_executor:
            logger.info("Signal %s/%s but no executor configured", signal_result.pair, signal_result.signal)
            return

        import uuid

        # Look up real strategy metrics from approved strategies
        sharpe = 1.0
        win_rate = 0.5
        max_drawdown = 0.05
        for s in self._approved_strategies:
            if s.get("strategy_type") == signal_result.strategy_type:
                sharpe = float(s.get("sharpe", s.get("metrics", {}).get("sharpe", 1.0)))
                win_rate = float(s.get("win_rate", s.get("metrics", {}).get("win_rate", 0.5)))
                max_drawdown = float(s.get("max_drawdown", s.get("metrics", {}).get("max_drawdown", 0.05)))
                break

        trade_signal = TradeSignal(
            pair=signal_result.pair,
            side=signal_result.signal,
            strategy_name=signal_result.strategy_type,
            strategy_type=signal_result.strategy_type,
            regime=signal_result.regime,
            confidence=signal_result.confidence,
            sharpe=sharpe,
            win_rate=win_rate,
            max_drawdown=max_drawdown,
            suggested_stoploss=settings.STOP_LOSS_DEFAULT,
            suggested_take_profit=settings.TAKE_PROFIT_DEFAULT,
            source_agent="signal_scanner",
            signal_id=uuid.uuid4().hex[:8],
        )
        await self._live_executor.execute_signal(trade_signal)

    async def _emit(self, event_type: str, payload: dict):
        if self._event_bus:
            try:
                await self._event_bus.publish(event_type, payload)
            except Exception:
                pass

    def get_signal_history(self, limit: int = 100) -> List[SignalResult]:
        return self._signal_history[-limit:]

    def update_approved_strategies(self, strategies: List[dict]):
        """Update the list of approved strategies (called by StrategyManager)."""
        self._approved_strategies = strategies
        self._update_approved_strategy_validated_regimes()
        logger.info("SignalScanner: %d approved strategies loaded", len(strategies))

    # ── Standby mode control ──

    def wake(self):
        """Force the scanner into active mode for the next cycle."""
        if self._standby_mode:
            self._standby_mode = False
            self._cycles_without_trade = 0
            logger.info("SignalScanner woken — switching to active mode")

    @property
    def is_standby(self) -> bool:
        """Whether the scanner is currently in power-saving standby mode."""
        return self._standby_mode

    @property
    def active_interval(self) -> int:
        return self._scan_interval

    @property
    def standby_interval(self) -> int:
        return self._standby_interval

    @property
    def cycles_without_trade(self) -> int:
        return self._cycles_without_trade

    @property
    def regime_tracker(self):
        return self._regime_tracker

    @property
    def signal_count(self) -> int:
        return len(self._signal_history)

    def enter_standby(self):
        """Put scanner into standby mode."""
        self._standby_mode = True

    def set_scan_interval(self, interval: int):
        """Set the active scan interval in seconds."""
        if interval > 0:
            self._scan_interval = interval
