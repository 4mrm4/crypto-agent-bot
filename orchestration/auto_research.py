"""
Autonomous research mode — searches web for strategy ideas,
converts to testable implementations, iterates to convergence.
Inspired by Karpathy's autoresearch pattern.
"""
import logging
from typing import List, Dict, Any

from api.event_bus import with_token_tracking
from config import settings
from data.fetcher import MarketDataFetcher
from data.regime import MarketRegimeDetector
from data.sentiment import SentimentFetcher
from orchestration.factory import make_orchestrator

logger = logging.getLogger(__name__)


def run_auto_research(topic: str, max_rounds: int = 5, event_bus=None, loop=None):
    """
    Full autonomous research pipeline:
    1. Search web for strategy ideas on topic
    2. Extract structured concepts
    3. Map concepts to strategy types
    4. Backtest and iterate
    5. Converge on best performer
    """
    logger.info("=== AUTO RESEARCH MODE: %s ===", topic)
    logger.info("This will run up to %d research rounds autonomously.", max_rounds)

    # 1. Detect current market regime
    regime = "unknown"
    try:
        fetcher = MarketDataFetcher()
        df = fetcher.fetch_ohlcv(settings.SYMBOL, "1h", limit=250)
        if df is not None and len(df) > 200:
            detector = MarketRegimeDetector()
            regime = detector.classify_regime(df)
            logger.info("Current regime: %s", regime)
    except Exception as exc:
        logger.warning("Regime detection failed: %s", exc)

    # 2. Get sentiment
    sentiment_score = 0.0
    try:
        sf = SentimentFetcher()
        report = sf.get_full_sentiment_report(settings.SYMBOL.replace("/USDT", ""))
        sentiment_score = report["score"]
        logger.info("Sentiment: %s (score=%.2f)", report["bias"], sentiment_score)
    except Exception as exc:
        logger.warning("Sentiment fetch failed: %s", exc)

    # 3. Build enriched research goal
    goal = (
        f"Auto-research: {topic}\n"
        f"Current market regime: {regime}\n"
        f"Current sentiment score: {sentiment_score:.2f}\n"
        f"Find and test the best strategy for these conditions."
    )

    # 4. Build orchestrator via factory
    orchestrator = make_orchestrator()

    # 5. Optionally wire EventBus for WebSocket streaming
    if event_bus is not None:
        try:
            cb = event_bus.make_callback()
            orchestrator.event_callback = with_token_tracking(cb)
        except Exception as exc:
            logger.warning("Could not wire EventBus to auto_research: %s", exc)
    result = orchestrator.run_research_loop(
        goal=goal,
        max_iterations=max_rounds,
        max_cycles=4,
    )

    # 6. Print final summary
    print("\n" + "="*60)
    print("AUTO RESEARCH COMPLETE")
    print("="*60)
    print(f"Topic: {topic}")
    print(f"Regime: {regime}")
    print(f"Iterations: {result.get('total_iterations', 0)}")
    print(f"Converged: {result.get('converged', False)}")
    best = result.get("best_metrics", {})
    print(f"Best Sharpe: {best.get('sharpe_ratio', 0):.2f}")
    print(f"Best Win Rate: {best.get('win_rate', 0):.0%}")
    print(f"Best Trades: {best.get('total_trades', 0)}")
    print("="*60)

    return result
