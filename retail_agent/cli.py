from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .memory import MemoryStore
from .runtime import AgentRuntime
from .sample_data import generate_sample_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Observable retail data analysis agent")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Run a complete multi-tool analysis")
    run.add_argument("--query", default="分析最近销售额下降的原因并生成报告")
    run.add_argument("--data", default=str(PROJECT_ROOT / "data" / "sample_retail.csv"))
    run.add_argument("--mode", choices=["local", "openai"], default="local")
    run.add_argument("--user-id", default="demo-user")
    run.add_argument("--max-steps", type=int, default=8)
    run.add_argument("--db", default=str(PROJECT_ROOT / "runtime" / "agent_memory.db"))
    run.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs"))

    trace = subparsers.add_parser("trace", help="Inspect one session's tool-call trace")
    trace.add_argument("session_id")
    trace.add_argument("--db", default=str(PROJECT_ROOT / "runtime" / "agent_memory.db"))

    memory = subparsers.add_parser("memory", help="Inspect long-term memory for a user")
    memory.add_argument("--user-id", default="demo-user")
    memory.add_argument("--db", default=str(PROJECT_ROOT / "runtime" / "agent_memory.db"))

    sample = subparsers.add_parser("generate-sample", help="Regenerate deterministic demo data")
    sample.add_argument("--output", default=str(PROJECT_ROOT / "data" / "sample_retail.csv"))
    return parser


def main() -> None:
    load_env(PROJECT_ROOT / ".env")
    args = build_parser().parse_args()
    if args.command == "generate-sample":
        print(generate_sample_data(args.output))
        return
    if args.command == "trace":
        store = MemoryStore(args.db)
        payload = {
            "session": store.get_session(args.session_id),
            "tool_calls": store.list_tool_calls(args.session_id),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    if args.command == "memory":
        store = MemoryStore(args.db)
        print(json.dumps(store.recall(args.user_id), ensure_ascii=False, indent=2))
        return

    data_path = Path(args.data)
    if not data_path.exists() and data_path == PROJECT_ROOT / "data" / "sample_retail.csv":
        generate_sample_data(data_path)
    runtime = AgentRuntime(db_path=args.db, output_dir=args.output_dir)
    result = runtime.run(
        query=args.query,
        dataset_path=data_path,
        user_id=args.user_id,
        mode=args.mode,
        max_steps=args.max_steps,
    )
    concise = {
        "session_id": result["session_id"],
        "status": result["status"],
        "mode": result["mode"],
        "tool_sequence": [item["tool_name"] for item in result["tool_calls"]],
        "artifacts": result["artifacts"],
    }
    print(json.dumps(concise, ensure_ascii=False, indent=2))

