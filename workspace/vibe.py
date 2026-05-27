"""VibeWorkspace — research goal workspace with Rich terminal UI.

Inspired by Vibe-Trading's research workspace pattern. Manages goals,
hypotheses, and agent outputs with a beautiful console interface.
"""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config import settings

logger = logging.getLogger(__name__)

console = Console()

REGISTRY_PATH = Path(settings.WORKSPACE_REGISTRY_PATH)


class VibeWorkspace:
    """CLI-based research workspace for managing goals and agent outputs."""

    def __init__(self, orchestrator=None):
        self._orchestrator = orchestrator
        self._registry = self._load_registry()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_goal(self, description: str, max_cycles: int = 5) -> Dict[str, Any]:
        """Create a new research goal, run agents, and store results."""
        goal_id = uuid.uuid4().hex[:8]

        entry = {
            "id": goal_id,
            "description": description,
            "created_at": datetime.utcnow().isoformat(),
            "status": "running",
            "result": None,
            "strategies": [],
        }
        self._registry[goal_id] = entry
        self._save_registry()

        console.print(
            Panel(
                f"[bold cyan]Goal #{goal_id}[/]\n[white]{description}[/]",
                title="[bold]New Research Goal[/]",
                border_style="blue",
            )
        )

        if self._orchestrator is None:
            entry["status"] = "failed"
            entry["result"] = "No orchestrator configured"
            self._save_registry()
            return entry

        with console.status("[yellow]Agents working...") as status:
            try:
                result = self._orchestrator.run_research_goal(
                    description, max_cycles=max_cycles
                )

                entry["status"] = "completed"
                entry["result"] = result
                entry["strategies"] = result.get("strategies", [])
                self._save_registry()

                self._display_result(entry)
            except Exception as exc:
                logger.exception("Goal failed: %s", exc)
                entry["status"] = "failed"
                entry["result"] = {"error": str(exc)}
                self._save_registry()
                console.print(f"[red]Goal failed: {exc}[/]")

        return entry

    def list_goals(self) -> List[Dict[str, Any]]:
        """Return all registered goals."""
        return list(self._registry.values())

    def review_goal(self, goal_id: str) -> Optional[Dict[str, Any]]:
        """Display full details for a specific goal."""
        entry = self._registry.get(goal_id)
        if not entry:
            console.print(f"[red]Goal #{goal_id} not found.[/]")
            return None

        self._display_result(entry)
        return entry

    def accept_strategy(self, goal_id: str, strategy_index: int = 0) -> Optional[str]:
        """Mark a strategy as approved and export it."""
        entry = self._registry.get(goal_id)
        if not entry:
            console.print(f"[red]Goal #{goal_id} not found.[/]")
            return None

        strategies = entry.get("strategies", [])
        if not strategies or strategy_index >= len(strategies):
            console.print(f"[red]No strategy at index {strategy_index}[/]")
            return None

        strategy_text = strategies[strategy_index]
        export_path = Path(f"./accepted_strategies/{goal_id}_strategy.txt")
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_path.write_text(str(strategy_text), encoding="utf-8")

        console.print(
            Panel(
                f"[green]Strategy accepted and exported to [bold]{export_path}[/][/]",
                title="[bold]Strategy Accepted[/]",
                border_style="green",
            )
        )
        return str(export_path)

    # ------------------------------------------------------------------
    # Interactive prompt
    # ------------------------------------------------------------------

    def run_interactive(self):
        """Drop into an interactive prompt loop."""
        console.print(
            Panel.fit(
                "[bold cyan]Vibe Trading Workspace[/]\n"
                "[white]Commands: [green]new-goal <text>[/], [green]list[/], "
                "[green]review <id>[/], [green]accept <id> <n>[/], "
                "[green]quit[/][/]",
                border_style="blue",
            )
        )

        while True:
            try:
                cmd = console.input("[bold cyan]>> [/]").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not cmd:
                continue
            if cmd == "quit":
                break
            elif cmd == "list":
                self._print_goals_table()
            elif cmd.startswith("new-goal "):
                description = cmd[9:]
                self.create_goal(description)
            elif cmd.startswith("review "):
                goal_id = cmd[7:].strip()
                self.review_goal(goal_id)
            elif cmd.startswith("accept "):
                parts = cmd[7:].strip().split()
                gid = parts[0] if len(parts) > 0 else ""
                idx = int(parts[1]) if len(parts) > 1 else 0
                self.accept_strategy(gid, idx)
            elif cmd == "help":
                console.print(
                    "[green]new-goal <text>[/]  - Start new research\n"
                    "[green]list[/]            - Show all goals\n"
                    "[green]review <id>[/]     - Show goal details\n"
                    "[green]accept <id> <n>[/] - Approve strategy n\n"
                    "[green]quit[/]            - Exit"
                )
            else:
                console.print("[yellow]Unknown command. Type 'help'.[/]")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_registry(self) -> Dict[str, Any]:
        if REGISTRY_PATH.exists():
            with open(REGISTRY_PATH) as f:
                data = json.load(f)
                # Handle migration from old list format
                if isinstance(data, list):
                    logger.info("Migrating registry from list to dict format")
                    migrated = {}
                    for item in data:
                        gid = item.get("id", uuid.uuid4().hex[:8])
                        migrated[gid] = item
                    return migrated
                return data
        return {}

    def _save_registry(self):
        REGISTRY_PATH.write_text(
            json.dumps(self._registry, indent=2, default=str),
            encoding="utf-8",
        )

    def _display_result(self, entry: Dict[str, Any]):
        """Print a nice Rich panel for a goal result."""
        result = entry.get("result") or {}
        strategies = entry.get("strategies", [])

        info = (
            f"[bold]ID:[/] #{entry['id']}\n"
            f"[bold]Goal:[/] {entry['description']}\n"
            f"[bold]Status:[/] {'[green]completed[/]' if entry['status'] == 'completed' else '[red]failed[/]'}\n"
            f"[bold]Created:[/] {entry['created_at']}\n"
        )

        if result:
            board = result.get("board_summary", "N/A")
            task_count = result.get("task_count", 0)
            info += f"\n[bold]Board:[/] {board}\n[bold]Tasks:[/] {task_count}\n"

        if strategies:
            info += f"\n[bold]Strategies found:[/] {len(strategies)}\n"
            for i, s in enumerate(strategies):
                snippet = str(s)[:200].replace("\n", " ")
                info += f"  {i}. {snippet}...\n"

        console.print(Panel(info, title=f"Goal #{entry['id']}", border_style="cyan"))

    def _print_goals_table(self):
        """Render a table of all goals."""
        table = Table(title="Research Goals")
        table.add_column("ID", style="cyan")
        table.add_column("Description", style="white")
        table.add_column("Status", style="green")
        table.add_column("Strategies", style="yellow")

        for entry in self.list_goals():
            table.add_row(
                entry["id"][:8],
                entry["description"][:50],
                entry["status"],
                str(len(entry.get("strategies", []))),
            )

        console.print(table)