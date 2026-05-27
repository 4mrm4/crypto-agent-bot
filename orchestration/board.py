"""TaskBoard — Kanban-style task management for the Hermes orchestrator."""

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Task:
    """A single unit of work tracked on the Kanban board."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    description: str = ""
    assigned_to: str = ""
    status: str = "TODO"  # TODO | IN_PROGRESS | REVIEW | DONE
    result: Any = None
    children: List["Task"] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class TaskBoard:
    """Kanban board with agent references for the LangGraph nodes."""

    def __init__(self, agents: Dict[str, Any], capabilities: Dict[str, List[str]]):
        self.tasks: Dict[str, Task] = {}
        self._agents = agents
        self._agent_capabilities = capabilities

    def add_task(
        self,
        description: str,
        assigned_to: str = "",
        parent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Task:
        task = Task(
            description=description,
            assigned_to=assigned_to,
            metadata=metadata or {},
        )
        self.tasks[task.id] = task
        if parent_id and parent_id in self.tasks:
            self.tasks[parent_id].children.append(task)
        return task

    def transition(self, task_id: str, new_status: str):
        if task_id in self.tasks:
            self.tasks[task_id].status = new_status

    def get_tasks_by_status(self, status: str) -> List[Task]:
        return [t for t in self.tasks.values() if t.status == status]

    def summary(self) -> str:
        counts = {"TODO": 0, "IN_PROGRESS": 0, "REVIEW": 0, "DONE": 0}
        for t in self.tasks.values():
            counts[t.status] = counts.get(t.status, 0) + 1
        return f"[Board] TODO:{counts['TODO']} IP:{counts['IN_PROGRESS']} REVIEW:{counts['REVIEW']} DONE:{counts['DONE']}"