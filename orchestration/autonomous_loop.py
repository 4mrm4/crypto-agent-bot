"""
AutonomousResearchLoop — continuous self-directing research engine.

Runs forever on a configurable interval. Never waits for human input.
Self-generates research goals based on:
  1. Coverage gaps (regimes with no good strategy in ChromaDB)
  2. Strategy decay (previously good strategies degrading)
  3. Regime changes (market shifted, existing strategies no longer match)
  4. Scheduled refresh (regimes not researched in >N days)
  5. Exploration (novel hypotheses combining top strategies)
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from orchestration.research import ResearchGoal

logger = logging.getLogger(__name__)

# Safety limits from autonomous-agents skill: every step compounds error,
# so we keep the loop simple and guardrailed.
MAX_CONSECUTIVE_FAILURES = 5   # halt loop if this many research cycles fail in a row
MIN_INTERVAL_SECONDS = 300      # 5 minutes — never loop faster than this


@dataclass
class AutonomousLoopState:
    """Current state of the autonomous research loop."""
    is_running: bool = False
    is_paused: bool = False
    current_goal: Optional[str] = None
    last_goal_generated: Optional[datetime] = None
    next_cycle_eta: Optional[datetime] = None
    total_cycles: int = 0
    total_goals_generated: int = 0
    consecutive_failures: int = 0
    last_error: Optional[str] = None
    last_regime: str = "unknown"
    coverage_gaps: Dict[str, float] = field(default_factory=dict)  # regime -> best_sharpe


class AutonomousResearchLoop:
    """
    Runs forever. Self-directs based on market state.
    Designed for reliability first — every step has validation and error recovery.
    """

    def __init__(
        self,
        orchestrator,
        regime_detector=None,
        experiment_tracker=None,
        vector_store=None,
        interval_minutes: int = 30,
        event_bus=None,
    ):
        self._orchestrator = orchestrator
        self._regime_detector = regime_detector
        self._experiment_tracker = experiment_tracker
        self._vector_store = vector_store
        self._interval_seconds = max(interval_minutes * 60, MIN_INTERVAL_SECONDS)
        self._event_bus = event_bus

        # Lazy imports (avoid circular deps at module level)
        self._sentiment_fetcher = None
        self._fetcher = None
        self._santiment_fetcher = None

        self._shutdown = False
        self.state = AutonomousLoopState()
        logger.info(
            "AutonomousResearchLoop initialized (interval=%ds)",
            self._interval_seconds,
        )

    # ── Public API ──

    async def run_forever(self):
        """Main loop. Runs until shutdown event is set."""
        self.state.is_running = True
        logger.info("=== AutonomousResearchLoop started ===")

        while not self._shutdown:
            try:
                self.state.total_cycles += 1
                logger.info(
                    "--- Autonomous cycle %d ---",
                    self.state.total_cycles,
                )

                # 1. Detect current market regime
                regime = await self._detect_regime()
                self.state.last_regime = regime

                # 2. Generate next research goal autonomously
                goal = await self._generate_next_goal()
                if goal is None:
                    logger.info("No research needed right now. Sleeping.")
                    await self._sleep()
                    continue

                self.state.current_goal = goal.motivation
                self.state.last_goal_generated = datetime.utcnow()
                self.state.total_goals_generated += 1
                self.state.consecutive_failures = 0

                # Emit goal to UI
                await self._emit("autonomous_goal_generated", {
                    "regime": goal.regime,
                    "strategy_type_hint": goal.strategy_type_hint,
                    "motivation": goal.motivation,
                    "priority_score": goal.priority_score,
                    "triggered_by": goal.triggered_by,
                })

                # 3. Run the research cycle
                logger.info(
                    "Researching: [%s] %s (priority=%.2f, triggered_by=%s)",
                    goal.regime, goal.motivation[:80],
                    goal.priority_score, goal.triggered_by,
                )
                try:
                    result = await self._run_research_cycle(goal)
                except Exception as exc:
                    logger.exception("Research cycle failed: %s", exc)
                    self.state.consecutive_failures += 1
                    self.state.last_error = str(exc)
                    if self.state.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        logger.critical(
                            "%d consecutive failures — halting autonomous loop",
                            MAX_CONSECUTIVE_FAILURES,
                        )
                        await self._emit("autonomous_halted", {
                            "reason": f"{MAX_CONSECUTIVE_FAILURES} consecutive failures",
                            "last_error": str(exc),
                        })
                        break
                    await self._sleep()
                    continue

                # 4. Detect strategy decay
                await self._detect_strategy_decay()

                # 5. Update coverage gaps
                self.state.coverage_gaps = await self._compute_coverage_gaps()

                # Emit cycle complete
                await self._emit("autonomous_cycle_complete", {
                    "cycle": self.state.total_cycles,
                    "regime": regime,
                    "goals_generated": self.state.total_goals_generated,
                    "coverage_gaps": self.state.coverage_gaps,
                })

            except Exception as exc:
                logger.exception("Autonomous loop caught top-level error: %s", exc)
                self.state.last_error = str(exc)
                self.state.consecutive_failures += 1

            # Sleep until next cycle
            await self._sleep()

        self.state.is_running = False
        logger.info("=== AutonomousResearchLoop stopped ===")

    def shutdown(self):
        """Signal the loop to stop gracefully."""
        self._shutdown = True
        logger.info("Shutdown requested — will stop after current cycle.")

    def pause(self):
        """Pause research (trading continues)."""
        self.state.is_paused = True
        logger.info("Autonomous research paused.")

    def resume(self):
        """Resume research."""
        self.state.is_paused = False
        logger.info("Autonomous research resumed.")

    def get_state(self) -> dict:
        """Return current state dict for API/UI."""
        return {
            "is_running": self.state.is_running,
            "is_paused": self.state.is_paused,
            "current_goal": self.state.current_goal,
            "last_goal_generated": (
                self.state.last_goal_generated.isoformat()
                if self.state.last_goal_generated else None
            ),
            "next_cycle_eta": (
                self.state.next_cycle_eta.isoformat()
                if self.state.next_cycle_eta else None
            ),
            "total_cycles": self.state.total_cycles,
            "total_goals_generated": self.state.total_goals_generated,
            "consecutive_failures": self.state.consecutive_failures,
            "last_error": self.state.last_error,
            "last_regime": self.state.last_regime,
            "coverage_gaps": self.state.coverage_gaps,
        }

    # ── Internal: Goal generation ──

    async def _generate_next_goal(self) -> Optional[ResearchGoal]:
        """
        Determine what to research next based on priority scoring.

        Priority order:
        1. Current regime has no strategy with Sharpe > 0.8 → research for that regime
        2. Strategy decay detected → re-research the decaying strategy's regime
        3. Regime not researched in >7 days → refresh
        4. Otherwise → explore novel hypothesis
        """
        # 1. Check coverage gaps for current regime
        coverage = await self._compute_coverage_gaps()
        current_regime = self.state.last_regime

        # 1a. If current regime has no good strategy → highest priority
        best_sharpe = coverage.get(current_regime, 0.0)
        if best_sharpe < 0.8:
            # Find a recommended strategy type for this regime
            from data.regime import REGIME_STRATEGY_MAP
            recommended = REGIME_STRATEGY_MAP.get(current_regime, {}).get("use", [])
            hint = recommended[0] if recommended else "sma_crossover"
            return ResearchGoal(
                regime=current_regime,
                strategy_type_hint=hint,
                motivation=(
                    f"Coverage gap: {current_regime} has no strategy with "
                    f"Sharpe > 0.8 (best={best_sharpe:.2f}). Researching {hint}."
                ),
                priority_score=1.0,
                triggered_by="coverage_gap",
            )

        # 1b. Check Santiment trending assets and boost priority if trending
        try:
            trending = await self._check_trending_assets()
            if trending:
                trending_slugs = ", ".join(trending[:3])
                logger.info(
                    "Trending assets: %s — boosting research priority",
                    trending_slugs,
                )
                return ResearchGoal(
                    regime=current_regime,
                    strategy_type_hint="momentum",
                    motivation=(
                        f"Trending asset research: {trending_slugs} surging on social. "
                        f"Researching momentum strategies for {current_regime}."
                    ),
                    priority_score=0.85,
                    triggered_by="trending_assets",
                )
        except Exception as exc:
            logger.debug("Trending assets check skipped: %s", exc)

        # 2. Check for decaying strategies
        decay_report = await self._detect_strategy_decay()
        if decay_report:
            worst = decay_report[0]
            return ResearchGoal(
                regime=worst["regime"],
                strategy_type_hint=worst["strategy_type"],
                motivation=(
                    f"Strategy decay: {worst['strategy_type']} performance "
                    f"dropped by {worst['decay_pct']:.0%}. Re-researching for {worst['regime']}."
                ),
                priority_score=0.8,
                triggered_by="decay",
            )

        # 3. Check for stale regimes (not researched in >7 days)
        for regime in coverage:
            if coverage[regime] < 0.5:  # no data for this regime
                recommended = REGIME_STRATEGY_MAP.get(regime, {}).get("use", [])
                hint = recommended[0] if recommended else "sma_crossover"
                return ResearchGoal(
                    regime=regime,
                    strategy_type_hint=hint,
                    motivation=(
                        f"Scheduled refresh: {regime} has no research data. "
                        f"Researching {hint}."
                    ),
                    priority_score=0.6,
                    triggered_by="scheduled",
                )

        # 4. Explore — combine two top strategies
        return ResearchGoal(
            regime=current_regime,
            strategy_type_hint="combined_sma_rsi",  # safe default exploration
            motivation=(
                f"Exploration: all regimes have coverage. "
                f"Testing combined strategy variant for {current_regime}."
            ),
            priority_score=0.3,
            triggered_by="exploration",
        )

    async def _check_trending_assets(self) -> List[str]:
        """Fetch Santiment trending asset slugs. Returns empty list if disabled/unavailable."""
        from config import settings
        if not getattr(settings, "SANTIMENT_ENABLED", False):
            return []
        if self._santiment_fetcher is None:
            from data.santiment_fetcher import SantimentFetcher
            self._santiment_fetcher = SantimentFetcher()
        try:
            trending = await self._santiment_fetcher.get_trending_assets()
            return trending or []
        except Exception as exc:
            logger.debug("Santiment trending fetch failed: %s", exc)
            return []

    async def _detect_strategy_decay(self) -> List[Dict[str, Any]]:
        """
        Query ChromaDB for all active strategies.
        Compare backtest baseline to (simulated) live performance.
        Returns list of decaying strategies sorted by severity.
        """
        if not self._vector_store:
            return []

        try:
            # Get best strategies from memory
            strategies = self._vector_store.get_best_strategies(min_sharpe=0.5, k=20)
            decaying = []

            for strat in strategies:
                meta = strat.get("metadata", {})
                if not meta:
                    continue

                backtest_sharpe = float(meta.get("sharpe", 0))
                strategy_type = meta.get("strategy_type", "unknown")
                regime = meta.get("regime", "unknown")

                # Check if we have live performance data in ChromaDB
                live_sharpe_key = f"live_sharpe_{strategy_type}"
                live_sharpe = float(meta.get(live_sharpe_key, backtest_sharpe))

                if backtest_sharpe > 0:
                    decay_pct = (backtest_sharpe - live_sharpe) / backtest_sharpe
                    if decay_pct > 0.20:  # DECAY_THRESHOLD
                        decaying.append({
                            "strategy_type": strategy_type,
                            "regime": regime,
                            "backtest_sharpe": backtest_sharpe,
                            "live_sharpe": live_sharpe,
                            "decay_pct": decay_pct,
                        })
                        logger.warning(
                            "Decay detected: %s/%s dropped %.0f%% "
                            "(backtest=%.2f, live=%.2f)",
                            strategy_type, regime, decay_pct * 100,
                            backtest_sharpe, live_sharpe,
                        )
                        await self._emit("strategy_decay_detected", {
                            "strategy_type": strategy_type,
                            "regime": regime,
                            "decay_pct": decay_pct,
                            "backtest_sharpe": backtest_sharpe,
                            "live_sharpe": live_sharpe,
                        })

            decaying.sort(key=lambda d: d["decay_pct"], reverse=True)
            return decaying

        except Exception as exc:
            logger.warning("Strategy decay detection failed: %s", exc)
            return []

    async def _compute_coverage_gaps(self) -> Dict[str, float]:
        """
        Returns {regime: best_sharpe_in_memory} for all 5 regimes.
        Used to identify which regimes are underserved.
        """
        if not self._vector_store:
            return {}

        regimes = ["strong_uptrend", "weak_trend", "ranging", "volatile", "low_liquidity"]
        gaps = {}

        for regime in regimes:
            try:
                best = self._vector_store.get_best_strategies(regime=regime, min_sharpe=0.0, k=1)
                if best:
                    meta = best[0].get("metadata", {})
                    gaps[regime] = float(meta.get("sharpe", 0.0))
                else:
                    gaps[regime] = 0.0
            except Exception:
                gaps[regime] = 0.0

        return gaps

    # ── Internal: Research execution ──

    async def _run_research_cycle(self, goal: ResearchGoal) -> dict:
        """Execute one research cycle using the orchestrator."""
        if self._orchestrator is None:
            raise RuntimeError("No orchestrator configured")

        # Build research goal text from the dataclass
        goal_text = (
            f"Auto-research: {goal.motivation}\n"
            f"Regime: {goal.regime}\n"
            f"Strategy hint: {goal.strategy_type_hint}\n"
            f"Priority: {goal.priority_score:.2f}"
        )

        logger.info(
            "Research cycle %d: %s",
            self.state.total_cycles, goal_text[:120],
        )

        # Use the existing research loop
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._orchestrator.run_research_loop(
                goal=goal_text,
                max_iterations=3,
                max_cycles=6,
            ),
        )

        logger.info(
            "Research cycle complete: converged=%s, iterations=%d",
            result.get("converged", False),
            result.get("total_iterations", 0),
        )
        return result

    # ── Internal: Regime detection ──

    async def _detect_regime(self) -> str:
        """Detect current market regime using available data."""
        from config import settings
        from data.fetcher import MarketDataFetcher

        try:
            fetcher = MarketDataFetcher()
            df = fetcher.fetch_ohlcv(settings.SYMBOL, "1h", limit=250)
            if df is not None and len(df) > 200:
                if self._regime_detector is None:
                    from data.regime import MarketRegimeDetector
                    self._regime_detector = MarketRegimeDetector()
                regime = self._regime_detector.classify_regime(df)
                logger.info("Current regime: %s", regime)
                return regime
        except Exception as exc:
            logger.warning("Regime detection failed: %s", exc)

        return "unknown"

    # ── Internal: Event emission ──

    async def _emit(self, event_type: str, payload: dict):
        """Emit event to EventBus if configured."""
        if self._event_bus:
            try:
                await self._event_bus.publish(event_type, payload)
            except Exception:
                pass

    async def _sleep(self):
        """Sleep for the configured interval, checking for pause/shutdown."""
        self.state.next_cycle_eta = datetime.utcnow() + timedelta(
            seconds=self._interval_seconds
        )

        # Check every 5 seconds so pause/resume/shutdown is responsive
        for _ in range(self._interval_seconds // 5):
            if self._shutdown:
                return
            if self.state.is_paused:
                await asyncio.sleep(1)
                continue
            await asyncio.sleep(5)


# Register this class so the new CLI flag can reach it
# (imported lazily from main.py to avoid circular deps)
