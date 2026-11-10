from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from .memory import MemoryStore
from .models import AgentState, PlanStep, ToolCall
from .planner import LocalWorkflowPlanner, OpenAIResponsesToolSelector
from .tools import ToolContext, ToolRegistry, build_registry


def _resolve_references(value: Any, artifacts: dict[str, str]) -> Any:
    if isinstance(value, str) and value.startswith("$artifacts."):
        key = value.split(".", 1)[1]
        if key not in artifacts:
            raise ValueError(f"Artifact {key!r} is not available yet")
        return artifacts[key]
    if isinstance(value, dict):
        return {key: _resolve_references(item, artifacts) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_references(item, artifacts) for item in value]
    return value


class AgentRuntime:
    def __init__(
        self,
        *,
        db_path: str | Path,
        output_dir: str | Path,
        registry: ToolRegistry | None = None,
    ) -> None:
        self.memory = MemoryStore(db_path)
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.registry = registry or build_registry()
        self.local_planner = LocalWorkflowPlanner()

    def run(
        self,
        *,
        query: str,
        dataset_path: str | Path,
        user_id: str = "demo-user",
        mode: str = "local",
        max_steps: int = 8,
    ) -> dict[str, Any]:
        if mode not in {"local", "openai"}:
            raise ValueError("mode must be 'local' or 'openai'")
        dataset = str(Path(dataset_path).expanduser().resolve())
        if not Path(dataset).exists():
            raise FileNotFoundError(dataset)
        session_id = f"session-{uuid.uuid4().hex[:10]}"
        plan = self.local_planner.build_plan(query, dataset)
        state = AgentState(
            session_id=session_id,
            user_id=user_id,
            query=query,
            plan=plan,
            artifacts={"original_dataset": dataset},
        )
        self.memory.start_session(session_id, user_id, query)
        self.memory.save_plan(session_id, plan.to_dict())
        selector = OpenAIResponsesToolSelector() if mode == "openai" else None

        try:
            for step_index in range(1, max_steps + 1):
                if mode == "local":
                    call = self.local_planner.next_call(state)
                    if call is None:
                        break
                else:
                    call = selector.next_call(
                        state,
                        self.registry.definitions(),
                        self.memory.recall(user_id),
                    )
                result = self._execute_call(state, call, step_index)
                self._update_plan(state, call.name, result)
                state.observations.append(
                    {"tool": call.name, "summary": result.get("summary", "completed")}
                )
                state.artifacts.update(result.get("artifacts", {}))
                self.memory.save_plan(session_id, state.plan.to_dict())
                self.memory.update_session(
                    session_id,
                    "running",
                    " | ".join(item["summary"] for item in state.observations[-4:]),
                )
                if call.name == "export_report":
                    break
            else:
                raise RuntimeError(f"Agent exceeded max_steps={max_steps}")
            if "html_report" not in state.artifacts:
                raise RuntimeError("Agent stopped before exporting a report")
        except Exception as exc:
            self.memory.update_session(session_id, "failed", str(exc))
            raise

        self.memory.update_session(
            session_id,
            "completed",
            f"Completed {len(state.observations)} tool calls; report exported",
        )
        return {
            "session_id": session_id,
            "status": "completed",
            "mode": mode,
            "artifacts": state.artifacts,
            "tool_calls": self.memory.list_tool_calls(session_id),
        }

    def _execute_call(
        self, state: AgentState, call: ToolCall, step_index: int
    ) -> dict[str, Any]:
        arguments = _resolve_references(call.arguments, state.artifacts)
        context = ToolContext(
            session_id=state.session_id,
            user_id=state.user_id,
            output_dir=self.output_dir,
            memory=self.memory,
            artifacts=state.artifacts,
        )
        started = time.perf_counter()
        try:
            result = self.registry.execute(call.name, arguments, context)
        except Exception as exc:
            duration_ms = int((time.perf_counter() - started) * 1000)
            self.memory.append_trace(
                call_id=call.call_id,
                session_id=state.session_id,
                step_index=step_index,
                tool_name=call.name,
                arguments=arguments,
                result={},
                status="failed",
                error=str(exc),
                duration_ms=duration_ms,
            )
            raise
        duration_ms = int((time.perf_counter() - started) * 1000)
        self.memory.append_trace(
            call_id=call.call_id,
            session_id=state.session_id,
            step_index=step_index,
            tool_name=call.name,
            arguments=arguments,
            result=result,
            status="success",
            duration_ms=duration_ms,
        )
        return result

    @staticmethod
    def _update_plan(state: AgentState, tool_name: str, result: dict[str, Any]) -> None:
        step = next(
            (
                item
                for item in state.plan.steps
                if item.tool_name == tool_name and item.status == "pending"
            ),
            None,
        )
        if step:
            step.status = "completed"
            step.result_summary = result.get("summary", "completed")

