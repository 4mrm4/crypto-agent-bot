"""
SignalScanner — continuously scans all pairs × approved strategies for entry signals.

Runs every N seconds. Only fires signals for strategies whose regime matches
the current detected regime. Passes approved signals to LiveExecutor.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from config import settings
from data.fetcher import MarketDataFetcher
from data.regime import MarketRegimeDetector, RegimeSnapshot
from execution.live_executor import LiveExecutor
from execution.trade_signal import TradeSignal

logger = logging.getLogger(__name__)

SCAN_INTERVAL_SECONDS = 60  # Check every minute


@dataclass
class SignalResult:
    """Result of evaluating a single strategy on a single pair."""
    signal: str          # "buy" | "sell" | "hold"
    confidence: float    # 0.0–1.0
    indicators: dict     # current indicator values
    pair: str
    strategy_type: str
    regime: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


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
    ):
        self._pairs = pairs or ["BTC/USDT", "ETH/USDT"]
        self._approved_strategies = approved_strategies or []
        self._regime_detector = regime_detector or MarketRegimeDetector()
        self._live_executor = live_executor
        self._fetcher = fetcher or MarketDataFetcher()
        self._vector_store = vector_store
        self._event_bus = event_bus
        self._scan_interval = scan_interval
        self._signal_history: List[SignalResult] = []
        self._regime_cache: Dict[str, str] = {}

    async def scan_loop(self):
        """Main loop. Runs every scan_interval seconds."""
        logger.info("SignalScanner started (interval=%ds, pairs=%s)", self._scan_interval, self._pairs)

        while True:
            try:
                for pair in self._pairs:
                    signals = await self._scan_pair(pair)
                    for signal_result in signals:
                        # Record for history
                        self._signal_history.append(signal_result)
                        if len(self._signal_history) > 1000:
                            self._signal_history = self._signal_history[-1000:]

                        # Emit signal for UI
                        await self._emit("trade_signal_evaluated", {
                            "pair": signal_result.pair,
                            "signal": signal_result.signal,
                            "confidence": signal_result.confidence,
                            "strategy_type": signal_result.strategy_type,
                            "regime": signal_result.regime,
                        })

                        # Execute if buy/sell with sufficient confidence
                        if signal_result.signal in ("buy", "sell") and signal_result.confidence >= 0.6:
                            await self._execute_signal(signal_result)

            except Exception as exc:
                logger.exception("Scan cycle error: %s", exc)

            await asyncio.sleep(self._scan_interval)

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
                import talib
                rsi = talib.RSI(close.values, timeperiod=int(params.get("rsi_period", 14)))
                indicators["rsi"] = float(rsi[-1]) if len(rsi) > 0 else 50

                if float(rsi[-1]) < float(params.get("rsi_buy_threshold", 30)):
                    signal = "buy"
                    confidence = 0.70
                elif float(rsi[-1]) > float(params.get("rsi_sell_threshold", 70)):
                    signal = "sell"
                    confidence = 0.65

            elif strategy_type == "bollinger_bands":
                import talib
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
                import talib
                rsi = talib.RSI(close.values, timeperiod=14)
                period = int(params.get("bb_period", 20))
                upper, middle, lower = talib.BBANDS(close.values.astype(float), timeperiod=period, nbdevup=2, nbdevdn=2)
                indicators["rsi"] = float(rsi[-1]) if len(rsi) > 0 else 50
                indicators["bb_lower"] = float(lower[-1]) if len(lower) > 0 else 0

                if float(rsi[-1]) < 35 and latest_close < float(lower[-1]):
                    signal = "buy"
                    confidence = 0.75

            elif strategy_type == "momentum":
                import talib
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
        """Detect current regime for a specific pair."""
        try:
            ohlcv = self._fetcher.fetch_ohlcv(pair, "1h", limit=250)
            if ohlcv is not None and len(ohlcv) > 200:
                regime = self._regime_detector.classify_regime(ohlcv)
                self._regime_cache[pair] = regime
                return regime
        except Exception:
            pass
        return self._regime_cache.get(pair, "unknown")

    def _get_strategies_for_regime(self, regime: str) -> List[dict]:
        """Filter approved strategies to those matching the current regime."""
        from data.regime import REGIME_STRATEGY_MAP
        recommended = REGIME_STRATEGY_MAP.get(regime, {}).get("use", [])

        if not recommended:
            return []

        matched = [s for s in self._approved_strategies
                   if s.get("strategy_type") in recommended]
        return matched or self._approved_strategies[:1]  # fallback to first

    async def _execute_signal(self, signal_result: SignalResult):
        """Convert a SignalResult to a TradeSignal and execute it."""
        if not self._live_executor:
            logger.info("Signal %s/%s but no executor configured", signal_result.pair, signal_result.signal)
            return

        import uuid
        trade_signal = TradeSignal(
            pair=signal_result.pair,
            side=signal_result.signal,
            strategy_name=signal_result.strategy_type,
            strategy_type=signal_result.strategy_type,
            regime=signal_result.regime,
            confidence=signal_result.confidence,
            sharpe=1.0,  # placeholder — real value from strategy metadata
            win_rate=0.5,
            max_drawdown=0.05,
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
        logger.info("SignalScanner: %d approved strategies loaded", len(strategies))
