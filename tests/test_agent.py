from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from retail_agent.memory import MemoryStore
from retail_agent.models import AgentState
from retail_agent.planner import LocalWorkflowPlanner, OpenAIResponsesToolSelector
from retail_agent.runtime import AgentRuntime
from retail_agent.sample_data import generate_sample_data
from retail_agent.tools import ToolContext, ToolRegistry, build_registry


class RetailInsightAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = (
            Path(__file__).resolve().parents[1]
            / ".test-work"
            / self._testMethodName
        )
        self.root.mkdir(parents=True, exist_ok=True)
        self.data_path = generate_sample_data(self.root / "sample.csv")
        self.db_path = self.root / "agent.db"
        self.output_dir = self.root / "outputs"

    def tearDown(self) -> None:
        pass

    def test_local_agent_runs_complete_multi_tool_workflow(self) -> None:
        runtime = AgentRuntime(db_path=self.db_path, output_dir=self.output_dir)
        result = runtime.run(
            query="分析最近销售额下降的原因并生成报告",
            dataset_path=self.data_path,
            user_id="tester",
        )

        sequence = [call["tool_name"] for call in result["tool_calls"]]
        self.assertEqual(
            sequence,
            [
                "inspect_dataset",
                "clean_data",
                "calculate_metrics",
                "detect_anomalies",
                "plot_chart",
                "export_report",
            ],
        )
        self.assertTrue(Path(result["artifacts"]["html_report"]).exists())
        anomaly_result = next(
            call["result"]
            for call in result["tool_calls"]
            if call["tool_name"] == "detect_anomalies"
        )
        self.assertEqual(anomaly_result["anomalies"][0]["group"], "United Kingdom")
        self.assertLess(anomaly_result["anomalies"][0]["change_rate"], -0.6)
        self.assertTrue(
            any(
                "units sold decreased" in driver
                for driver in anomaly_result["anomalies"][0]["drivers"]
            )
        )

    def test_dataset_and_metric_definitions_are_recalled_across_sessions(self) -> None:
        runtime = AgentRuntime(db_path=self.db_path, output_dir=self.output_dir)
        runtime.run(
            query="生成分析报告", dataset_path=self.data_path, user_id="remember-me"
        )
        memories = runtime.memory.recall("remember-me")
        categories = {item["category"] for item in memories}
        self.assertEqual(categories, {"dataset_metadata", "metric_definition"})

    def test_tool_schema_rejects_missing_required_argument(self) -> None:
        registry = build_registry()
        context = ToolContext(
            session_id="test-session",
            user_id="tester",
            output_dir=self.output_dir,
            memory=MemoryStore(self.db_path),
            artifacts={},
        )
        with self.assertRaisesRegex(ValueError, "Missing required arguments"):
            registry.execute("inspect_dataset", {}, context)

    def test_failed_tool_call_is_traced_and_session_is_failed(self) -> None:
        runtime = AgentRuntime(
            db_path=self.db_path,
            output_dir=self.output_dir,
            registry=ToolRegistry(),
        )
        with self.assertRaisesRegex(ValueError, "Unknown tool"):
            runtime.run(query="分析数据", dataset_path=self.data_path)
        with runtime.memory.connect() as connection:
            session = connection.execute(
                "SELECT session_id, status FROM sessions ORDER BY rowid DESC LIMIT 1"
            ).fetchone()
        self.assertEqual(session["status"], "failed")
        trace = runtime.memory.list_tool_calls(session["session_id"])
        self.assertEqual(trace[0]["status"], "failed")
        self.assertIn("Unknown tool", trace[0]["error"])

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "OPENAI_MODEL": "test-model"})
    @patch("urllib.request.urlopen")
    def test_openai_selector_parses_function_call(self, mocked_urlopen) -> None:
        payload = {
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call-123",
                    "name": "inspect_dataset",
                    "arguments": json.dumps({"dataset_path": str(self.data_path)}),
                }
            ]
        }

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(payload).encode("utf-8")

        mocked_urlopen.return_value = FakeResponse()
        planner = LocalWorkflowPlanner()
        state = AgentState(
            session_id="session-test",
            user_id="tester",
            query="分析数据",
            plan=planner.build_plan("分析数据", str(self.data_path)),
            artifacts={"original_dataset": str(self.data_path)},
        )
        call = OpenAIResponsesToolSelector().next_call(
            state, build_registry().definitions(), []
        )
        self.assertEqual(call.name, "inspect_dataset")
        self.assertEqual(call.call_id, "call-123")
        self.assertEqual(call.arguments["dataset_path"], str(self.data_path))


if __name__ == "__main__":
    unittest.main()
