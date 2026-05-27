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
    from agents.analyst import AnalystAgent
    from agents.strategist import StrategistAgent
    from agents.risk_manager import RiskManagerAgent
    from agents.curator import CuratorAgent
    from orchestration.hermes import HermesOrchestrator
    return HermesOrchestrator(agents={
        "analyst": AnalystAgent(),
        "strategist": StrategistAgent(),
        "risk_manager": RiskManagerAgent(),
        "curator": CuratorAgent(),
    })


def _run_demo():
    """Full end-to-end demonstration."""
    console.print("[bold cyan]crypto_agent_bot — Demo Mode[/]\n")

    from orchestration.hermes import HermesOrchestrator
    from workspace.vibe import VibeWorkspace

    orchestrator = _make_orchestrator()
    workspace = VibeWorkspace(orchestrator=orchestrator)

    console.print("[yellow]Running research goal: Momentum strategy for BTC/USDT...[/]")
    entry = workspace.create_goal(
        "Find a momentum strategy with Sharpe > 1 for BTC/USDT",
        max_cycles=3,
    )

    console.print("\n[green]Demo complete. See registry for results.[/]")


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