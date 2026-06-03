"""Hermes-inspired multi-agent orchestrator with a Kanban task board."""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from orchestration.board import TaskBoard
from orchestration.graph import build_orchestration_graph
from orchestration.research import ResearchIteration, check_convergence
from config import settings
from state.state_broker import StateBroker

logger = logging.getLogger(__name__)


class HermesOrchestrator:
    """Multi-agent orchestrator using a Kanban board and a LangGraph loop."""

    def __init__(self, agents: Dict[str, Any], state_broker: Optional[StateBroker] = None):
        self._agent_capabilities: Dict[str, List[str]] = {
            "analyst": ["analysis", "market_research", "sentiment", "analyse", "market"],
            "strategist": ["strategy", "strategies", "generate", "concept",
                           "parameter", "params", "design"],
            "backtester": ["backtest", "backtesting", "walk_forward", "hyperopt",
                           "optimization", "optimise", "compare", "benchmark",
                           "download", "data"],
            "iteration_tracker": ["iteration", "history", "best_strategy",
                                  "store", "track", "record", "memory", "recall"],
            "risk_manager": ["risk", "assessment", "position_sizing"],
            "curator": ["memory", "context", "history"],
            "researcher": ["research", "web", "paper", "novel", "search", "literature"],
        }
        self.agents = agents
        self.board = TaskBoard(agents, self._agent_capabilities)
        self._graph = build_orchestration_graph()
        self._state_broker = state_broker

    # ------------------------------------------------------------------
    # Public API — outer research loop
    # ------------------------------------------------------------------

    def run_research_loop(
        self,
        goal: str,
        max_iterations: int = 5,
        max_cycles: int = 5,
    ) -> Dict[str, Any]:
        """AutoResearch outer loop: hypothesis -> research -> critique -> repeat.

        Wraps the LangGraph inner loop. Each iteration generates a hypothesis,
        runs research, critiques results, checks convergence, and mutates
        the hypothesis for the next round.
        """
        logger.info("=== Research loop start: %s (max_iterations=%d) ===", goal, max_iterations)

        iterations: List[ResearchIteration] = []
        final_output = None

        for i in range(max_iterations):
            iter_num = i + 1
            logger.info("--- Research iteration %d/%d ---", iter_num, max_iterations)

            # 1. Generate hypothesis for this iteration
            hypothesis = self._generate_hypothesis(goal, iterations, iter_num, max_iterations)
            logger.info("Hypothesis: %.200s", hypothesis)

            # 2. Run the inner LangGraph with hypothesis context
            output = self._run_research_goal(goal, max_cycles=max_cycles, hypothesis=hypothesis, iteration=iter_num)
            final_output = output

            # 3. Extract best metrics from the output
            metrics = self._extract_metrics(output)
            strategy_id = self._extract_strategy_id(output)

            # 4. Critique results
            critique = self._critique_iteration(output, goal, hypothesis)

            # 5. Build iteration record
            verdict = "converged" if check_convergence(metrics) else "discarded"
            iteration = ResearchIteration(
                hypothesis=hypothesis,
                strategy_id=strategy_id,
                metrics=metrics,
                verdict=verdict,
                critique=critique,
                iteration=iter_num,
            )
            iterations.append(iteration)

            # 6. Persist to memory
            self._persist_research_iteration(iteration, goal)
            logger.info(
                "Iteration %d: %s | Sharpe=%.2f WR=%.0f%% DD=%.1f%% "
                "(targets: Sharpe>=0.8 WR>=40%% DD<=15%% trades>=5)",
                iter_num, verdict.upper(),
                metrics.get("sharpe_ratio", 0),
                metrics.get("win_rate", 0) * 100,
                abs(metrics.get("max_drawdown", 0)) * 100,
            )

            # Publish system heartbeat via StateBroker
            if self._state_broker:
                try:
                    self._state_broker.set_system_status({
                        "running": True,
                        "mode": self.execution_mode if hasattr(self, 'execution_mode') else "research",
                        "timestamp": datetime.utcnow().isoformat(),
                    })
                except Exception as exc:
                    logger.warning("StateBroker heartbeat failed: %s", exc)

            # 7. Check convergence
            if verdict == "converged":
                logger.info("Converged at iteration %d!", iter_num)
                break

        return {
            "goal": goal,
            "iterations": [it.to_dict() for it in iterations],
            "total_iterations": len(iterations),
            "converged": iterations[-1].verdict == "converged" if iterations else False,
            "best_metrics": iterations[-1].metrics if iterations else {},
            **(final_output or {}),
        }

    # ------------------------------------------------------------------
    # Inner research run
    # ------------------------------------------------------------------

    def _run_research_goal(
        self,
        goal: str,
        max_cycles: int = 5,
        hypothesis: str = "",
        iteration: int = 1,
    ) -> Dict[str, Any]:
        """Execute a single research lifecycle using the LangGraph state graph."""
        logger.info("=== Research goal: %s ===", goal)
        goal_id = uuid.uuid4().hex[:8]

        # Reset board for this run
        self.board = TaskBoard(self.agents, self._agent_capabilities)

        # Inject memory context if a curator agent is registered
        curator = self.agents.get("curator")
        if curator and hasattr(curator, "inject_context"):
            from backtesting.data_split import DATA_SPLIT
            context = curator.inject_context(
                goal, k=3,
                current_research_window=DATA_SPLIT.research_timerange(),
                contamination_guard=True,
            )
            if context:
                self.board.add_task(
                    description=f"Review past research context for: {goal}\nPast insights:\n{context}",
                    assigned_to="curator",
                )
                logger.info("Injected past insights from memory.")

        # Fetch market sentiment if enabled
        sentiment_report = {}
        if getattr(settings, "ENABLE_SENTIMENT", True):
            try:
                from data.sentiment import SentimentFetcher
                sf = SentimentFetcher()
                symbol = settings.SYMBOL.replace("/USDT", "")
                sentiment_report = sf.get_full_sentiment_report(symbol)
                logger.info(
                    "Sentiment: F&G=%d (%s), bias=%s, score=%.2f",
                    sentiment_report["fear_greed"]["value"],
                    sentiment_report["fear_greed"]["classification"],
                    sentiment_report["bias"],
                    sentiment_report["score"],
                )
            except Exception as exc:
                logger.warning("Sentiment fetch failed: %s", exc)

        # Detect chart patterns on recent data
        pattern_report = {}
        if getattr(settings, "ENABLE_PATTERNS", True):
            try:
                from data.fetcher import MarketDataFetcher
                from data.patterns import PatternDetector
                fetcher = MarketDataFetcher()
                df = fetcher.fetch_ohlcv(settings.SYMBOL, "1h", limit=50)
                if df is not None and len(df) > 20:
                    pd_detector = PatternDetector()
                    pattern_report = pd_detector.get_pattern_report(df)
                    logger.info("Patterns: %s (bias=%s)", pattern_report["active_patterns"], pattern_report["bias"])
            except Exception as exc:
                logger.warning("Pattern detection failed: %s", exc)

        # On-chain data (only if enabled)
        onchain_report = {}
        if getattr(settings, "ENABLE_ONCHAIN", False):
            try:
                from data.onchain import OnChainFetcher
                ocf = OnChainFetcher()
                onchain_report = ocf.get_onchain_report(settings.SYMBOL)
                if onchain_report.get("summary"):
                    logger.info("On-chain: %s", onchain_report["summary"])
            except Exception as exc:
                logger.warning("On-chain fetch failed: %s", exc)

        # Build enriched goal with hypothesis context
        enriched_goal = goal
        if hypothesis:
            enriched_goal = (
                f"{goal}\n\nHypothesis (iteration {iteration}): {hypothesis}"
            )
        if iteration > 1:
            enriched_goal += f"\nResearch iteration {iteration} of outer loop."

        # Run researcher agent (inline, before LangGraph) to generate strategy specs
        research_specs = []
        researcher = self.agents.get("researcher")
        if researcher:
            try:
                prompt = (
                    f"Research novel trading strategies for this goal:\n{enriched_goal}\n\n"
                    f"Search the web for strategy ideas, then use generate_custom_strategy_spec "
                    f"to produce 1-2 testable strategy specs. Focus on the indicators, entry and exit logic."
                )
                researcher_result = researcher.run(prompt)
                research_text = researcher_result.get("output", "")
                if hasattr(researcher, "get_specs"):
                    research_specs = list(researcher.get_specs().values())
                if research_text:
                    logger.info("Researcher produced %d chars of output", len(research_text))
                    # Store research text in memory
                    if curator and hasattr(curator, "store_result"):
                        curator.store_result(
                            goal=goal,
                            output=f"Researcher findings: {research_text[:500]}",
                            metadata={"type": "research_findings", "iteration": iteration},
                        )
            except Exception as exc:
                logger.warning("Researcher agent failed: %s", exc)

        # Create initial tasks — researcher runs inline, then strategist gets enriched task
        self.board.add_task(f"Analyse market conditions for: {enriched_goal}", assigned_to="analyst")

        # Strategist task includes research specs if available
        strategist_task_desc = f"Generate and backtest strategies for: {enriched_goal}"
        if research_specs:
            import json
            specs_text = json.dumps(research_specs, indent=2)[:800]
            strategist_task_desc += f"\n\nUse these research specs as starting hypotheses:\n{specs_text}"
        if sentiment_report:
            strategist_task_desc += (
                f"\n\nCurrent market sentiment: {sentiment_report['bias'].upper()} "
                f"(Fear/Greed={sentiment_report['fear_greed']['value']}, score={sentiment_report['score']:.2f}). "
                f"Consider sentiment when choosing strategy type."
            )
        if pattern_report and pattern_report.get("active_patterns"):
            strategist_task_desc += (
                f"\n\nActive chart patterns: {', '.join(pattern_report['active_patterns'])} "
                f"(bias={pattern_report['bias']}). "
                f"Factor this into entry/exit timing."
            )
        if onchain_report and onchain_report.get("netflow"):
            strategist_task_desc += (
                f"\n\nOn-chain signal: {onchain_report['netflow']['signal'].upper()} "
                f"({onchain_report['large_whale_count']} large whale transactions). "
                f"Factor this into directional bias."
            )
        self.board.add_task(strategist_task_desc, assigned_to="strategist")

        self.board.add_task(
            f"Assess risk for strategies targeting: {enriched_goal}", assigned_to="risk_manager"
        )

        # Run the LangGraph
        initial_state = {
            "goal": enriched_goal,
            "board": self.board,
            "current_task_id": None,
            "current_agent_name": None,
            "cycle": 0,
            "max_cycles": max_cycles,
            "final_output": None,
            "messages": [],
            "curator_context": "",
            "research_specs": research_specs if research_specs else None,
        }

        final_state = self._graph.invoke(initial_state)

        output = final_state.get("final_output", self._build_summary(enriched_goal))

        # Store results in memory for future runs
        if curator:
            for tid, result_text in output.get("results", {}).items():
                if result_text:
                    curator.store_result(goal, str(result_text)[:500], {"task_id": tid})
            summary = output.get("board_summary", "")
            if summary:
                curator.store_result(goal, f"Board summary: {summary}", {"type": "summary"})
            logger.info("Stored goal results in memory.")

        # Persist strategist iteration history into memory
        iteration_tracker = self.agents.get("iteration_tracker")
        if iteration_tracker and hasattr(iteration_tracker, "_iteration_history") and iteration_tracker._iteration_history:
            persisted = 0
            for rec in iteration_tracker._iteration_history:
                rec_dict = rec.to_dict()
                text = (
                    f"Strategy {rec_dict['verdict'].upper()}: "
                    f"type={rec_dict['params'].get('strategy_type', 'sma')}, "
                    f"fast_ma={rec_dict['params'].get('fast_ma')}, "
                    f"slow_ma={rec_dict['params'].get('slow_ma')} | "
                    f"Reason: {rec_dict['reason']} | "
                    f"Sharpe={rec_dict['metrics'].get('sharpe_ratio', 'N/A')}, "
                    f"WR={rec_dict['metrics'].get('win_rate', 'N/A')}"
                )
                metadata = {
                    "goal_id": goal_id,
                    "type": f"{rec_dict['verdict']}_record",
                    "reason": rec_dict["reason"][:120],
                    "research_iteration": iteration,
                }
                if curator and hasattr(curator, "store_result"):
                    curator.store_result(goal=goal, output=text, metadata=metadata)
                    persisted += 1
            if persisted:
                logger.info("Persisted %d iteration records to memory", persisted)

        # Attach sentiment/pattern data to output for Web UI emission
        if isinstance(output, dict):
            if sentiment_report:
                output["sentiment"] = sentiment_report
            if pattern_report:
                output["pattern_report"] = pattern_report

        logger.info("=== Inner research complete ===")
        return output

    # ------------------------------------------------------------------
    # Hypothesis generation
    # ------------------------------------------------------------------

    def _generate_hypothesis(
        self,
        goal: str,
        past_iterations: List[ResearchIteration],
        iter_num: int,
        max_iterations: int,
    ) -> str:
        """Generate or mutate a research hypothesis.

        Iteration 1: structured hypothesis from the goal text.
        Iterations 2+: read previous critique and mutate.
        """
        if not past_iterations:
            # First iteration — extract a hypothesis from the goal
            hypothesis = self._llm_hypothesis(goal)
            return hypothesis

        # Later iterations — mutate based on previous critique
        last = past_iterations[-1]
        critique = last.critique or "No critique available."
        hypothesis = self._mutate_hypothesis(goal, last.hypothesis, critique, iter_num, max_iterations)
        return hypothesis

    def _llm_hypothesis(self, goal: str) -> str:
        """Use the curator LLM to generate a structured hypothesis from the goal."""
        from agents.base import BaseAgent
        from langchain_core.tools import Tool

        temp_agent = BaseAgent(
            name="hypothesis_generator",
            tools=[],
            system_prompt=(
                "You are a quantitative research scientist. Given a trading research goal, "
                "write a structured hypothesis with:\n"
                "- Expected edge: what market inefficiency might exist\n"
                "- Indicators: which indicators could capture that edge\n"
                "- Entry logic: when to enter\n"
                "- Exit logic: when to exit\n"
                "- Timeframe suggestion: what timeframe fits best\n\n"
                "Be specific and testable. Output plain ASCII text only."
            ),
        )
        prompt = (
            f"Write a testable trading hypothesis for this goal:\n{goal}\n\n"
            f"Focus on: what edge do we expect? Which indicators capture it? "
            f"What timeframe suits it best?"
        )
        try:
            result = temp_agent.run(prompt)
            hypothesis = result.get("output", "")
            # Truncate and sanitize
            safe = "".join(c if ord(c) < 128 else "?" for c in hypothesis)
            return safe[:800]
        except Exception as exc:
            logger.warning("Hypothesis generation failed: %s", exc)
            return f"Test {goal} with standard indicators and parameters."

    def _mutate_hypothesis(
        self,
        goal: str,
        previous_hypothesis: str,
        critique: str,
        iter_num: int,
        max_iterations: int,
    ) -> str:
        """Use the curator LLM to mutate the hypothesis based on critique."""
        from agents.base import BaseAgent

        temp_agent = BaseAgent(
            name="hypothesis_mutator",
            tools=[],
            system_prompt=(
                "You are a quantitative research scientist refining a hypothesis. "
                "Given the previous hypothesis and critique, produce a new hypothesis "
                "that addresses the critique while keeping what worked.\n\n"
                "Format: Expected edge, Indicators, Entry logic, Exit logic, Timeframe.\n"
                "Output plain ASCII text only."
            ),
        )
        prompt = (
            f"Goal: {goal}\n\n"
            f"Previous hypothesis:\n{previous_hypothesis[:500]}\n\n"
            f"Critique of last iteration:\n{critique[:500]}\n\n"
            f"This is iteration {iter_num} of {max_iterations}. "
            f"Produce an improved hypothesis."
        )
        try:
            result = temp_agent.run(prompt)
            hypothesis = result.get("output", "")
            safe = "".join(c if ord(c) < 128 else "?" for c in hypothesis)
            return safe[:800]
        except Exception as exc:
            logger.warning("Hypothesis mutation failed: %s", exc)
            return f"Iteration {iter_num}: refine strategy parameters for {goal}."

    # ------------------------------------------------------------------
    # Critique
    # ------------------------------------------------------------------

    def _critique_iteration(self, output: Dict[str, Any], goal: str, hypothesis: str) -> str:
        """Use the analyst or risk manager to critique the iteration results."""
        analyst = self.agents.get("analyst")
        if not analyst:
            return "No analyst agent available for critique."

        results = output.get("results", {})
        strategies = output.get("strategies", [])

        if not strategies and not results:
            return "No results produced this iteration."

        try:
            prompt = (
                f"Critique the following research results for the hypothesis:\n{hypothesis[:400]}\n\n"
                f"Goal: {goal}\n\n"
                f"Results:\n{str(results)[:800]}\n\n"
                f"Provide a structured critique with:\n"
                f"- What worked\n- What failed\n- Suggested direction for next iteration"
            )
            result = analyst.run(prompt)
            critique = result.get("output", "")
            safe = "".join(c if ord(c) < 128 else "?" for c in critique)
            return safe[:800]
        except Exception as exc:
            logger.warning("Critique failed: %s", exc)
            return f"Critique unavailable: {exc}"

    # ------------------------------------------------------------------
    # Metric extraction helpers
    # ------------------------------------------------------------------

    def _extract_metrics(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Extract best available metrics from the research output."""
        metrics = {
            "sharpe_ratio": output.get("sharpe_ratio", 0),
            "win_rate": output.get("win_rate", 0),
            "max_drawdown": output.get("max_drawdown", 0),
            "profit_ratio": output.get("profit_ratio", 0),
            "total_trades": output.get("total_trades", 0),
        }

        if metrics["total_trades"]:
            return metrics

        # Search task results for the iteration tracker's backtest output dict
        iteration_tracker = self.agents.get("iteration_tracker")
        if iteration_tracker and hasattr(iteration_tracker, "_iteration_history") and iteration_tracker._iteration_history:
            # Take the best Sharpe from iteration history
            best = max(
                iteration_tracker._iteration_history,
                key=lambda r: r.metrics.get("sharpe_ratio", -999) if isinstance(r.metrics.get("sharpe_ratio"), (int, float)) else -999
            )
            return {
                "sharpe_ratio": best.metrics.get("sharpe_ratio", 0),
                "win_rate": best.metrics.get("win_rate", 0),
                "max_drawdown": best.metrics.get("max_drawdown", 0),
                "profit_ratio": best.metrics.get("profit_ratio", best.metrics.get("total_profit", 0)),
                "total_trades": best.metrics.get("total_trades", 0),
            }

        return metrics

    def _extract_strategy_id(self, output: Dict[str, Any]) -> str:
        """Extract best strategy ID from output."""
        done_tasks = self.board.get_tasks_by_status("DONE")
        for task in done_tasks:
            if task.result and "[" in str(task.result):
                import re
                m = re.search(r'\[([a-f0-9]{6,16})\]', str(task.result))
                if m:
                    return m.group(1)
        return ""

    def _persist_research_iteration(self, iteration: ResearchIteration, goal: str):
        """Store a research iteration in ChromaDB via the curator."""
        curator = self.agents.get("curator")
        if not curator or not hasattr(curator, "store_result"):
            return
        text = (
            f"Research iteration {iteration.iteration}: {iteration.verdict}\n"
            f"Hypothesis: {iteration.hypothesis[:300]}\n"
            f"Metrics: Sharpe={iteration.metrics.get('sharpe_ratio')}, "
            f"WR={iteration.metrics.get('win_rate')}, "
            f"DD={iteration.metrics.get('max_drawdown')}\n"
            f"Critique: {iteration.critique[:300]}"
        )
        metadata = {
            "type": "research_iteration",
            "iteration": str(iteration.iteration),
            "verdict": iteration.verdict,
            "goal": goal[:120],
        }
        try:
            curator.store_result(goal=goal, output=text, metadata=metadata)
            logger.debug("Persisted research iteration %d to memory", iteration.iteration)
        except Exception as exc:
            logger.warning("Failed to persist research iteration: %s", exc)

    # ------------------------------------------------------------------
    # Autonomous goal runner
    # ------------------------------------------------------------------

    def run_from_autonomous_goal(self, research_goal: "ResearchGoal") -> Dict[str, Any]:
        """
        Like run_research_loop() but takes a ResearchGoal dataclass.
        Automatically constructs the hypothesis from regime + strategy_type_hint.
        Logs the motivation and triggered_by to experiment tracker.
        """
        from orchestration.research import ResearchGoal

        # Build the research text from structured goal
        goal_text = (
            f"Autonomous research for {research_goal.regime} regime. "
            f"Focus strategy type: {research_goal.strategy_type_hint}. "
            f"Trigger: {research_goal.triggered_by}. "
            f"{research_goal.motivation}"
        )

        logger.info(
            "=== Autonomous goal: [%s] %s (triggered_by=%s, priority=%.2f) ===",
            research_goal.regime,
            research_goal.strategy_type_hint,
            research_goal.triggered_by,
            research_goal.priority_score,
        )

        # Run the existing research loop
        result = self.run_research_loop(
            goal=goal_text,
            max_iterations=3,
            max_cycles=4,
        )

        # Attach autonomous metadata to result
        if isinstance(result, dict):
            result["autonomous_goal"] = {
                "regime": research_goal.regime,
                "strategy_type_hint": research_goal.strategy_type_hint,
                "motivation": research_goal.motivation,
                "triggered_by": research_goal.triggered_by,
                "priority_score": research_goal.priority_score,
            }

        return result

    # ------------------------------------------------------------------
    # Compatibility: run_research_goal still works as before
    # ------------------------------------------------------------------

    def _build_summary(self, goal: str) -> Dict[str, Any]:
        """Assemble the final output dict from the board (fallback)."""
        done_tasks = self.board.get_tasks_by_status("DONE")
        return {
            "goal": goal,
            "board_summary": self.board.summary(),
            "task_count": len(self.board.tasks),
            "results": {t.id: t.result for t in done_tasks if t.result},
            "strategies": [
                str(t.result) for t in done_tasks
                if "strategy" in t.description.lower() and t.result
            ],
        }

    def run_research_goal(self, goal: str, max_cycles: int = 5) -> Dict[str, Any]:
        """Legacy single-run research (no outer loop)."""
        return self._run_research_goal(goal, max_cycles=max_cycles)