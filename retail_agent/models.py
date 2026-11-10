from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    call_id: str


@dataclass
class PlanStep:
    step_id: int
    description: str
    tool_name: str
    arguments: dict[str, Any]
    status: str = "pending"
    result_summary: str = ""


@dataclass
class TaskPlan:
    objective: str
    steps: list[PlanStep]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentState:
    session_id: str
    user_id: str
    query: str
    plan: TaskPlan
    artifacts: dict[str, str] = field(default_factory=dict)
    observations: list[dict[str, Any]] = field(default_factory=list)

    def compact_context(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_request": self.query,
            "plan": self.plan.to_dict(),
            "artifacts": self.artifacts,
            "recent_observations": self.observations[-4:],
        }

