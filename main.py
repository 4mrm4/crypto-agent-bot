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
    parser.add_argument(
        "command",
        nargs="*",
        help="Workspace command: new-goal | list-goals | review | run",
    )
    args = parser.parse_args()

    if args.demo:
        _run_demo()
        return

    if args.command:
        _run_cli_command(args.command)
    else:
        console.print("[bold cyan]crypto_agent_bot[/]")
        console.print("  --demo          Full pipeline demonstration")
        console.print("  run             Interactive workspace")
        console.print("  new-goal <txt>  Run a research goal")
        console.print("  list-goals      Show past goals")
        console.print("  review <id>     Show goal detail")


def _make_orchestrator():
    from orchestration.factory import make_orchestrator
    return make_orchestrator()


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