from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class MemoryStore:
    """SQLite-backed short-term task state, long-term memory, and traces."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    query TEXT NOT NULL,
                    status TEXT NOT NULL,
                    plan_json TEXT NOT NULL DEFAULT '{}',
                    context_summary TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS tool_calls (
                    call_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    step_index INTEGER NOT NULL,
                    tool_name TEXT NOT NULL,
                    arguments_json TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error TEXT NOT NULL DEFAULT '',
                    duration_ms INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                );

                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    memory_key TEXT NOT NULL,
                    value_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(user_id, category, memory_key)
                );
                """
            )

    def start_session(self, session_id: str, user_id: str, query: str) -> None:
        now = utc_now()
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO sessions(session_id, user_id, query, status, created_at, updated_at) "
                "VALUES (?, ?, ?, 'running', ?, ?)",
                (session_id, user_id, query, now, now),
            )

    def save_plan(self, session_id: str, plan: dict[str, Any]) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE sessions SET plan_json = ?, updated_at = ? WHERE session_id = ?",
                (json.dumps(plan, ensure_ascii=False), utc_now(), session_id),
            )

    def update_session(
        self, session_id: str, status: str, context_summary: str = ""
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE sessions SET status = ?, context_summary = ?, updated_at = ? "
                "WHERE session_id = ?",
                (status, context_summary, utc_now(), session_id),
            )

    def append_trace(
        self,
        *,
        call_id: str,
        session_id: str,
        step_index: int,
        tool_name: str,
        arguments: dict[str, Any],
        result: dict[str, Any],
        status: str,
        duration_ms: int,
        error: str = "",
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO tool_calls(
                    call_id, session_id, step_index, tool_name, arguments_json,
                    result_json, status, error, duration_ms, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    call_id,
                    session_id,
                    step_index,
                    tool_name,
                    json.dumps(arguments, ensure_ascii=False),
                    json.dumps(result, ensure_ascii=False),
                    status,
                    error,
                    duration_ms,
                    utc_now(),
                ),
            )

    def remember(
        self, user_id: str, category: str, key: str, value: dict[str, Any]
    ) -> None:
        now = utc_now()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO memories(user_id, category, memory_key, value_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, category, memory_key)
                DO UPDATE SET value_json = excluded.value_json, updated_at = excluded.updated_at
                """,
                (user_id, category, key, json.dumps(value, ensure_ascii=False), now, now),
            )

    def recall(self, user_id: str, category: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT category, memory_key, value_json, updated_at FROM memories WHERE user_id = ?"
        params: list[Any] = [user_id]
        if category:
            query += " AND category = ?"
            params.append(category)
        query += " ORDER BY updated_at DESC"
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            {
                "category": row["category"],
                "key": row["memory_key"],
                "value": json.loads(row["value_json"]),
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["plan"] = json.loads(result.pop("plan_json"))
        return result

    def list_tool_calls(self, session_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM tool_calls WHERE session_id = ? ORDER BY step_index, created_at",
                (session_id,),
            ).fetchall()
        calls = []
        for row in rows:
            item = dict(row)
            item["arguments"] = json.loads(item.pop("arguments_json"))
            item["result"] = json.loads(item.pop("result_json"))
            calls.append(item)
        return calls
