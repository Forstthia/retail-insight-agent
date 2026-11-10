from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from .models import AgentState, PlanStep, TaskPlan, ToolCall


class LocalWorkflowPlanner:
    """Builds a transparent baseline plan that is deterministic and testable."""

    def build_plan(self, query: str, dataset_path: str) -> TaskPlan:
        source = str(Path(dataset_path).expanduser().resolve())
        group_by = "product" if "商品" in query or "product" in query.lower() else "country"
        return TaskPlan(
            objective=query,
            steps=[
                PlanStep(1, "Inspect schema and data quality", "inspect_dataset", {"dataset_path": source}),
                PlanStep(2, "Clean invalid and duplicate rows", "clean_data", {"dataset_path": source}),
                PlanStep(3, "Calculate core business metrics", "calculate_metrics", {"dataset_path": "$artifacts.cleaned_dataset", "group_by": group_by}),
                PlanStep(4, "Compare recent periods and detect declines", "detect_anomalies", {"dataset_path": "$artifacts.cleaned_dataset", "group_by": group_by, "decline_threshold": 0.2, "period_days": 30}),
                PlanStep(5, "Generate monthly trend chart", "plot_chart", {"dataset_path": "$artifacts.cleaned_dataset", "title": "Monthly net revenue trend"}),
                PlanStep(6, "Export the analysis report", "export_report", {"title": "Retail Insight Agent Report"}),
            ],
        )

    def next_call(self, state: AgentState) -> ToolCall | None:
        step = next((item for item in state.plan.steps if item.status == "pending"), None)
        if step is None:
            return None
        return ToolCall(step.tool_name, dict(step.arguments), f"local-{uuid.uuid4().hex[:12]}")


class OpenAIResponsesToolSelector:
    """Lets an OpenAI Responses model choose one function tool per loop."""

    def __init__(self) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.model = os.getenv("OPENAI_MODEL", "gpt-5.6-terra").strip()
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required when --mode openai is used")

    def next_call(
        self,
        state: AgentState,
        tools: list[dict[str, Any]],
        long_term_memory: list[dict[str, Any]],
    ) -> ToolCall:
        instructions = (
            "You are a bounded retail data-analysis tool selector. Choose exactly one function tool. "
            "Follow the suggested plan, use existing artifact paths, never invent columns, and export_report only after analysis and charting. "
            "The application executes tools and validates arguments; you do not calculate metrics yourself."
        )
        body = {
            "model": self.model,
            "instructions": instructions,
            "input": json.dumps(
                {
                    "task_state": state.compact_context(),
                    "relevant_long_term_memory": long_term_memory[:5],
                },
                ensure_ascii=False,
            ),
            "tools": tools,
            "tool_choice": "required",
        }
        request = urllib.request.Request(
            f"{self.base_url}/responses",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI Responses API returned {exc.code}: {detail}") from exc
        for item in payload.get("output", []):
            if item.get("type") == "function_call":
                arguments = item.get("arguments", {})
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                return ToolCall(
                    name=item["name"],
                    arguments=arguments,
                    call_id=item.get("call_id") or f"openai-{uuid.uuid4().hex[:12]}",
                )
        raise RuntimeError("The model response did not contain a function_call output item")

