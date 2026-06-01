"""crypto_agent_bot — Modular crypto trading bot with learning agents.

A multi-agent system inspired by Vibe-Trading and Hermes Agent patterns.
Built with LangChain, Freqtrade, CCXT, ChromaDB and Rich.
"""

import argparse
import logging
import sys

from rich.console import Console

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-20s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")
console = Console()


def main():
    parser = argparse.ArgumentParser(description="crypto_agent_bot")
    parser.add_argument("--demo", action="store_true", help="Run full demonstration")
    parser.add_argument("--ui", action="store_true", help="Start Web UI server only (no demo pipeline)")
    parser.add_argument(
        "--auto-research",
        metavar="TOPIC",
        help="Run autonomous research mode: searches web for strategies on TOPIC, "
             "tests and iterates automatically. Example: --auto-research 'BTC momentum strategies'"
    )
    parser.add_argument(
        "--autonomous",
        action="store_true",
        help="Start fully autonomous mode: self-directs research goals based on market "
             "conditions, runs continuously, monitors strategy decay, and adapts without "
             "human input. Use --ui to also start the web dashboard."
    )
    parser.add_argument(
        "command",
        nargs="*",
        help="Workspace command: new-goal | list-goals | review | run",
    )
    args = parser.parse_args()

    # Auto-download historical data on startup
    from backtesting.setup_data import ensure_data_available
    logger.info("Checking historical data availability...")
    data_ok = ensure_data_available()
    if not data_ok:
        logger.warning(
            "Could not download historical data. "
            "Backtests will use whatever data is locally available."
        )

    if args.autonomous:
        _run_autonomous(ui=args.ui)
        return

    if args.auto_research:
        from orchestration.auto_research import run_auto_research
        run_auto_research(args.auto_research)
        return

    if args.ui:
        _run_ui()
        return

    if args.demo:
        _run_demo()
        return

    if args.command:
        _run_cli_command(args.command)
    else:
        console.print("[bold cyan]crypto_agent_bot[/]")
        console.print("  --demo          Full pipeline demonstration")
        console.print("  --ui            Web UI server only (no pipeline)")
        console.print("  run             Interactive workspace")
        console.print("  new-goal <txt>  Run a research goal")
        console.print("  list-goals      Show past goals")
        console.print("  review <id>     Show goal detail")


def _make_orchestrator():
    from orchestration.factory import make_orchestrator
    return make_orchestrator()


def _run_ui():
    """Start the Web UI server only — no demo pipeline."""
    import subprocess
    import time
    import webbrowser

    console.print("[bold cyan]crypto_agent_bot — Web UI[/]\n")
    console.print("[yellow]Starting Web UI server...[/]")

    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.server:app", "--host", "127.0.0.1", "--port", "8765", "--log-level", "info"],
        stdout=None,
        stderr=None,
    )

    import httpx
    for _ in range(20):
        try:
            r = httpx.get("http://127.0.0.1:8765/api/health", timeout=2)
            if r.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(0.5)

    url = "http://127.0.0.1:8765"
    console.print(f"[green]UI running at {url}[/]")
    webbrowser.open(url)
    console.print("\n[green]Press Ctrl+C to stop the server.[/]")
    try:
        server.wait()
    except KeyboardInterrupt:
        server.terminate()
        server.wait()


def _run_autonomous(ui: bool = False):
    """Start the autonomous research loop, optionally with the web UI."""
    import asyncio

    from agents.analyst import AnalystAgent
    from agents.strategist import StrategistAgent
    from agents.risk_manager import RiskManagerAgent
    from agents.curator import CuratorAgent
    from agents.researcher import ResearcherAgent
    from backtesting.engine import BacktestEngine
    from memory.vector_store import VectorStore
    from orchestration.autonomous_loop import AutonomousResearchLoop
    from orchestration.experiment_tracker import ExperimentTracker
    from orchestration.hermes import HermesOrchestrator
    from data.regime import MarketRegimeDetector
    from config import settings

    console.print("\n" + "=" * 60)
    console.print("[bold cyan]CRYPTO AGENT BOT — AUTONOMOUS MODE[/bold cyan]")
    console.print("[white]Self-directing research. Never waits for human input.[/white]")
    console.print(f"[white]Interval: {settings.AUTONOMOUS_INTERVAL_MINUTES} minutes[/white]")
    console.print("=" * 60 + "\n")

    engine = BacktestEngine()
    agents = {
        "analyst": AnalystAgent(),
        "strategist": StrategistAgent(engine=engine),
        "risk_manager": RiskManagerAgent(),
        "curator": CuratorAgent(),
        "researcher": ResearcherAgent(),
    }
    orchestrator = HermesOrchestrator(agents=agents)
    vector_store = VectorStore()
    experiment_tracker = ExperimentTracker()
    regime_detector = MarketRegimeDetector()

    event_bus = None
    if ui:
        from api.event_bus import EventBus
        event_bus = EventBus()

    loop = AutonomousResearchLoop(
        orchestrator=orchestrator,
        regime_detector=regime_detector,
        experiment_tracker=experiment_tracker,
        vector_store=vector_store,
        interval_minutes=settings.AUTONOMOUS_INTERVAL_MINUTES,
        event_bus=event_bus,
    )

    async def _run_with_ui():
        import uvicorn
        from api.server import app as fastapi_app
        fastapi_app.state.autonomous_loop = loop
        fastapi_app.state.event_bus = event_bus
        fastapi_app.state.vector_store = vector_store
        fastapi_app.state.experiment_tracker = experiment_tracker
        # Loop is started by the server's startup event handler
        config = uvicorn.Config(fastapi_app, host="127.0.0.1", port=8765, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()
        loop.shutdown()

    async def _run_headless():
        await loop.run_forever()

    try:
        if ui:
            asyncio.run(_run_with_ui())
        else:
            asyncio.run(_run_headless())
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutdown requested...[/yellow]")
        loop.shutdown()
        console.print("[green]Autonomous loop stopped.[/green]")


def _run_demo():
    """Start the FastAPI server + UI, then run a demo research goal."""
    import subprocess
    import time
    import webbrowser

    console.print("[bold cyan]crypto_agent_bot — Demo Mode[/]\n")
    console.print("[yellow]Starting Web UI server...[/]")

    # Start uvicorn in background
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.server:app", "--host", "127.0.0.1", "--port", "8765", "--log-level", "warning"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait for server
    import httpx
    for _ in range(20):
        try:
            r = httpx.get("http://127.0.0.1:8765/api/health", timeout=2)
            if r.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(0.5)

    url = "http://127.0.0.1:8765"
    console.print(f"[green]UI running at {url}[/]")
    webbrowser.open(url)

    # Kick off a demo research goal
    console.print("[yellow]Kicking off demo research goal...[/]")
    try:
        resp = httpx.post(f"{url}/api/run", json={
            "goal": "Find a momentum strategy with Sharpe > 1 for BTC/USDT",
            "max_cycles": 3,
            "max_iterations": 1,
        }, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            console.print(f"[green]Run started: {data['run_id']}[/]")
        else:
            console.print(f"[red]API error: {resp.status_code}[/]")
    except Exception as exc:
        console.print(f"[red]Failed to start run: {exc}[/]")

    console.print("\n[green]Demo mode active. Press Ctrl+C to stop.[/]")
    try:
        server.wait()
    except KeyboardInterrupt:
        server.terminate()
        server.wait()


def _run_cli_command(cmd_list):
    """Parse and execute a workspace CLI command."""
    import signal
    from orchestration.graph import request_shutdown

    def _handle_sigint(sig, frame):
        logger.info("Received SIGINT — requesting clean graph shutdown...")
        request_shutdown()

    signal.signal(signal.SIGINT, _handle_sigint)
    from orchestration.hermes import HermesOrchestrator
    from workspace.vibe import VibeWorkspace

    orchestrator = _make_orchestrator()
    workspace = VibeWorkspace(orchestrator=orchestrator)

    action = cmd_list[0]
    rest = " ".join(cmd_list[1:]) if len(cmd_list) > 1 else ""

    if action == "run":
        workspace.run_interactive()
    elif action == "new-goal":
        if not rest:
            console.print("[red]Usage: new-goal <description>[/]")
            return
        workspace.create_goal(rest)
    elif action == "list-goals":
        goals = workspace.list_goals()
        if not goals:
            console.print("[yellow]No goals yet.[/]")
            return
        from rich.table import Table
        table = Table(title="Research Goals")
        table.add_column("ID", style="cyan")
        table.add_column("Description")
        table.add_column("Strategies")
        for g in goals:
            table.add_row(
                g["id"],
                g["description"][:60],
                str(len(g.get("result", {}).get("strategies", []))),
            )
        console.print(table)
    elif action == "review":
        if not rest:
            console.print("[red]Usage: review <goal_id>[/]")
            return
        workspace.review_goal(rest)
    else:
        console.print(f"[red]Unknown command: {action}[/]")


if __name__ == "__main__":
    main()