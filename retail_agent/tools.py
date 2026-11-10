from __future__ import annotations

import csv
import html
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from .memory import MemoryStore


@dataclass
class ToolContext:
    session_id: str
    user_id: str
    output_dir: Path
    memory: MemoryStore
    artifacts: dict[str, str]


ToolHandler = Callable[[dict[str, Any], ToolContext], dict[str, Any]]


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, definition: ToolDefinition) -> None:
        if definition.name in self._tools:
            raise ValueError(f"Duplicate tool: {definition.name}")
        self._tools[definition.name] = definition

    def definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in self._tools.values()
        ]

    def execute(
        self, name: str, arguments: dict[str, Any], context: ToolContext
    ) -> dict[str, Any]:
        if name not in self._tools:
            raise ValueError(f"Unknown tool: {name}")
        tool = self._tools[name]
        self._validate(arguments, tool.parameters)
        return tool.handler(arguments, context)

    @staticmethod
    def _validate(arguments: dict[str, Any], schema: dict[str, Any]) -> None:
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        missing = [field for field in required if field not in arguments]
        if missing:
            raise ValueError(f"Missing required arguments: {', '.join(missing)}")
        unknown = set(arguments) - set(properties)
        if unknown and schema.get("additionalProperties") is False:
            raise ValueError(f"Unknown arguments: {', '.join(sorted(unknown))}")
        python_types = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
        }
        for field, value in arguments.items():
            expected_name = properties.get(field, {}).get("type")
            expected = python_types.get(expected_name)
            if expected and not isinstance(value, expected):
                raise ValueError(f"Argument {field!r} must be {expected_name}")
            enum = properties.get(field, {}).get("enum")
            if enum and value not in enum:
                raise ValueError(f"Argument {field!r} must be one of {enum}")


def _read_rows(dataset_path: str) -> list[dict[str, str]]:
    path = Path(dataset_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Dataset does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def inspect_dataset(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    path = Path(arguments["dataset_path"]).expanduser().resolve()
    rows = _read_rows(str(path))
    columns = list(rows[0].keys()) if rows else []
    missing = {column: 0 for column in columns}
    dates: list[date] = []
    for row in rows:
        for column in columns:
            if not str(row.get(column, "")).strip():
                missing[column] += 1
        try:
            dates.append(_parse_date(row.get("date", "")))
        except ValueError:
            pass
    metadata = {
        "dataset_path": str(path),
        "row_count": len(rows),
        "columns": columns,
        "missing_values": missing,
        "date_range": [min(dates).isoformat(), max(dates).isoformat()] if dates else [],
    }
    context.memory.remember(
        context.user_id, "dataset_metadata", path.name, metadata
    )
    return {
        "summary": f"Inspected {len(rows)} rows and {len(columns)} columns",
        "metadata": metadata,
    }


def clean_data(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    source = Path(arguments["dataset_path"]).expanduser().resolve()
    rows = _read_rows(str(source))
    output_name = arguments.get("output_name", "cleaned_retail.csv")
    output = context.output_dir / context.session_id / output_name
    output.parent.mkdir(parents=True, exist_ok=True)
    required = {
        "order_id",
        "date",
        "country",
        "product",
        "customer_id",
        "quantity",
        "unit_price",
        "returned",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Dataset must contain columns: {sorted(required)}")

    cleaned: list[dict[str, str]] = []
    seen_orders: set[str] = set()
    rejected = Counter()
    for row in rows:
        order_id = row["order_id"].strip()
        if not order_id or order_id in seen_orders:
            rejected["duplicate_or_empty_order_id"] += 1
            continue
        try:
            _parse_date(row["date"].strip())
            quantity = int(row["quantity"])
            unit_price = float(row["unit_price"])
        except (ValueError, TypeError):
            rejected["invalid_type"] += 1
            continue
        if quantity <= 0 or unit_price < 0:
            rejected["invalid_value"] += 1
            continue
        if not row["country"].strip() or not row["product"].strip():
            rejected["missing_dimension"] += 1
            continue
        normalized = dict(row)
        normalized["quantity"] = str(quantity)
        normalized["unit_price"] = f"{unit_price:.2f}"
        normalized["returned"] = "1" if str(row["returned"]).lower() in {"1", "true", "yes"} else "0"
        cleaned.append(normalized)
        seen_orders.add(order_id)

    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(cleaned)
    return {
        "summary": f"Cleaned {len(rows)} rows; retained {len(cleaned)}",
        "input_rows": len(rows),
        "output_rows": len(cleaned),
        "rejected": dict(rejected),
        "artifacts": {"cleaned_dataset": str(output.resolve())},
    }


def _row_values(row: dict[str, str]) -> tuple[date, int, float, bool]:
    return (
        _parse_date(row["date"]),
        int(row["quantity"]),
        float(row["unit_price"]),
        row["returned"] == "1",
    )


def _metric_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    order_ids: set[str] = set()
    customers = Counter()
    units = 0
    gross_revenue = 0.0
    net_revenue = 0.0
    returned_orders = 0
    for row in rows:
        _, quantity, price, returned = _row_values(row)
        order_ids.add(row["order_id"])
        customers[row["customer_id"]] += 1
        units += quantity
        gross_revenue += quantity * price
        if returned:
            returned_orders += 1
        else:
            net_revenue += quantity * price
    order_count = len(order_ids)
    repeat_customers = sum(count >= 2 for count in customers.values())
    return {
        "orders": order_count,
        "units": units,
        "gross_revenue": round(gross_revenue, 2),
        "net_revenue": round(net_revenue, 2),
        "average_order_value": round(net_revenue / order_count, 2) if order_count else 0.0,
        "return_rate": round(returned_orders / order_count, 4) if order_count else 0.0,
        "repeat_customer_rate": round(repeat_customers / len(customers), 4) if customers else 0.0,
    }


def calculate_metrics(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    rows = _read_rows(arguments["dataset_path"])
    group_by = arguments.get("group_by", "country")
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    if group_by == "overall":
        groups["overall"] = rows
    else:
        for row in rows:
            key = row["date"][:7] if group_by == "month" else row[group_by]
            groups[key].append(row)
    group_metrics = [
        {"group": key, **_metric_summary(group_rows)}
        for key, group_rows in groups.items()
    ]
    group_metrics.sort(key=lambda item: item["net_revenue"], reverse=True)
    overall = _metric_summary(rows)
    context.memory.remember(
        context.user_id,
        "metric_definition",
        "retail_core_metrics",
        {
            "net_revenue": "quantity * unit_price, excluding returned orders",
            "return_rate": "returned orders / unique orders",
            "repeat_customer_rate": "customers with at least two orders / all customers",
        },
    )
    return {
        "summary": f"Calculated core metrics grouped by {group_by}",
        "group_by": group_by,
        "overall": overall,
        "groups": group_metrics,
    }


def detect_anomalies(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    rows = _read_rows(arguments["dataset_path"])
    group_by = arguments.get("group_by", "country")
    threshold = float(arguments.get("decline_threshold", 0.2))
    period_days = int(arguments.get("period_days", 30))
    parsed_dates = [_parse_date(row["date"]) for row in rows]
    max_date = max(parsed_dates)
    current_start = max_date - timedelta(days=period_days - 1)
    previous_start = current_start - timedelta(days=period_days)
    previous_end = current_start - timedelta(days=1)
    period_rows: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(
        lambda: {"current": [], "previous": []}
    )
    for row in rows:
        row_date, _, _, _ = _row_values(row)
        key = row[group_by]
        if current_start <= row_date <= max_date:
            period_rows[key]["current"].append(row)
        elif previous_start <= row_date <= previous_end:
            period_rows[key]["previous"].append(row)

    comparisons = []
    anomalies = []
    for key, values in period_rows.items():
        previous_metrics = _metric_summary(values["previous"])
        current_metrics = _metric_summary(values["current"])
        previous = previous_metrics["net_revenue"]
        current = current_metrics["net_revenue"]
        change = (current - previous) / previous if previous else math.inf
        units_change = (
            (current_metrics["units"] - previous_metrics["units"])
            / previous_metrics["units"]
            if previous_metrics["units"]
            else math.inf
        )
        previous_price = (
            previous_metrics["gross_revenue"] / previous_metrics["units"]
            if previous_metrics["units"]
            else 0.0
        )
        current_price = (
            current_metrics["gross_revenue"] / current_metrics["units"]
            if current_metrics["units"]
            else 0.0
        )
        price_change = (
            (current_price - previous_price) / previous_price
            if previous_price
            else math.inf
        )
        return_rate_delta = current_metrics["return_rate"] - previous_metrics["return_rate"]
        drivers = []
        if math.isfinite(units_change) and units_change <= -0.1:
            drivers.append(f"units sold decreased {abs(units_change):.1%}")
        if return_rate_delta >= 0.03:
            drivers.append(f"return rate increased {return_rate_delta:.1%} points")
        if math.isfinite(price_change) and price_change <= -0.05:
            drivers.append(f"average unit price decreased {abs(price_change):.1%}")
        if not drivers and math.isfinite(change) and change < 0:
            drivers.append("revenue declined without a single dominant volume, return, or price driver")
        item = {
            "group": key,
            "previous_revenue": round(previous, 2),
            "current_revenue": round(current, 2),
            "change_rate": round(change, 4) if math.isfinite(change) else None,
            "previous_metrics": previous_metrics,
            "current_metrics": current_metrics,
            "drivers": drivers,
        }
        comparisons.append(item)
        if math.isfinite(change) and change <= -threshold:
            anomalies.append(item)
    comparisons.sort(key=lambda item: item["change_rate"] if item["change_rate"] is not None else 999)
    anomalies.sort(key=lambda item: item["change_rate"] or 0)
    return {
        "summary": f"Detected {len(anomalies)} revenue decline anomalies",
        "periods": {
            "previous": [previous_start.isoformat(), previous_end.isoformat()],
            "current": [current_start.isoformat(), max_date.isoformat()],
        },
        "threshold": threshold,
        "comparisons": comparisons,
        "anomalies": anomalies,
    }


def plot_chart(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    rows = _read_rows(arguments["dataset_path"])
    monthly: dict[str, float] = defaultdict(float)
    for row in rows:
        _, quantity, price, returned = _row_values(row)
        if not returned:
            monthly[row["date"][:7]] += quantity * price
    values = sorted(monthly.items())
    width, height = 760, 320
    margin_left, margin_top, plot_width, plot_height = 70, 35, 650, 220
    maximum = max((value for _, value in values), default=1.0)
    points = []
    for index, (_, value) in enumerate(values):
        x = margin_left + (plot_width * index / max(len(values) - 1, 1))
        y = margin_top + plot_height - (value / maximum * plot_height)
        points.append((x, y))
    point_string = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    labels = "".join(
        f'<text x="{x:.1f}" y="280" text-anchor="middle" font-size="12">{html.escape(month)}</text>'
        for (month, _), (x, _) in zip(values, points)
    )
    dots = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#0f766e"><title>{value:.2f}</title></circle>'
        for (_, value), (x, y) in zip(values, points)
    )
    title = html.escape(arguments.get("title", "Monthly net revenue trend"))
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="70" y="22" font-size="17" font-family="Arial" font-weight="bold">{title}</text>
<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#9ca3af"/>
<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#9ca3af"/>
<polyline points="{point_string}" fill="none" stroke="#0f766e" stroke-width="3"/>
{dots}{labels}
<text x="14" y="145" font-size="12" transform="rotate(-90 14 145)">Net revenue</text>
</svg>"""
    output = context.output_dir / context.session_id / "monthly_revenue.svg"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(svg, encoding="utf-8")
    return {
        "summary": f"Generated monthly trend chart with {len(values)} points",
        "monthly_values": [{"month": key, "net_revenue": round(value, 2)} for key, value in values],
        "artifacts": {"trend_chart": str(output.resolve())},
    }


def export_report(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    traces = context.memory.list_tool_calls(context.session_id)
    by_tool = {trace["tool_name"]: trace["result"] for trace in traces if trace["status"] == "success"}
    metrics = by_tool.get("calculate_metrics", {})
    anomaly_data = by_tool.get("detect_anomalies", {})
    title = arguments["title"]
    overall = metrics.get("overall", {})
    anomalies = anomaly_data.get("anomalies", [])
    anomaly_lines = []
    for item in anomalies:
        driver_text = "; ".join(item.get("drivers", [])) or "driver requires further investigation"
        anomaly_lines.append(
            f"- {item['group']}: {item['previous_revenue']:.2f} -> {item['current_revenue']:.2f} "
            f"({item['change_rate']:.1%}); likely drivers: {driver_text}"
        )
    if not anomaly_lines:
        anomaly_lines = ["- No group exceeded the configured decline threshold."]
    markdown = "\n".join(
        [
            f"# {title}",
            "",
            "## Executive summary",
            "",
            f"- Net revenue: {overall.get('net_revenue', 0):.2f}",
            f"- Orders: {overall.get('orders', 0)}",
            f"- Average order value: {overall.get('average_order_value', 0):.2f}",
            f"- Return rate: {overall.get('return_rate', 0):.1%}",
            f"- Repeat customer rate: {overall.get('repeat_customer_rate', 0):.1%}",
            "",
            "## Revenue decline anomalies",
            "",
            *anomaly_lines,
            "",
            "## Method",
            "",
            "The agent inspected and cleaned the dataset, calculated deterministic metrics, compared equal-length periods, generated a chart, and retained every tool call in SQLite.",
        ]
    )
    session_output = context.output_dir / context.session_id
    session_output.mkdir(parents=True, exist_ok=True)
    markdown_path = session_output / "analysis_report.md"
    html_path = session_output / "analysis_report.html"
    markdown_path.write_text(markdown, encoding="utf-8")
    chart_path = context.artifacts.get("trend_chart", "")
    chart_html = ""
    if chart_path and Path(chart_path).exists():
        chart_html = Path(chart_path).read_text(encoding="utf-8")
    anomaly_html = "".join(f"<li>{html.escape(line[2:])}</li>" for line in anomaly_lines)
    report_html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>
body{{font-family:Arial,sans-serif;color:#17202a;margin:0;background:#f5f7f8}}main{{max-width:900px;margin:auto;padding:36px}}
.metrics{{display:grid;grid-template-columns:repeat(5,1fr);gap:10px}}.metric{{background:white;border:1px solid #dfe6e9;padding:14px;border-radius:6px}}
.metric strong{{display:block;font-size:21px;color:#0f766e;margin-top:8px}}section{{background:white;margin-top:16px;padding:20px;border:1px solid #dfe6e9;border-radius:6px}}
@media(max-width:720px){{.metrics{{grid-template-columns:repeat(2,1fr)}}main{{padding:16px}}svg{{width:100%;height:auto}}}}
</style></head><body><main><h1>{html.escape(title)}</h1>
<div class="metrics"><div class="metric">Net revenue<strong>{overall.get('net_revenue', 0):.2f}</strong></div>
<div class="metric">Orders<strong>{overall.get('orders', 0)}</strong></div><div class="metric">AOV<strong>{overall.get('average_order_value', 0):.2f}</strong></div>
<div class="metric">Return rate<strong>{overall.get('return_rate', 0):.1%}</strong></div><div class="metric">Repeat rate<strong>{overall.get('repeat_customer_rate', 0):.1%}</strong></div></div>
<section><h2>Monthly trend</h2>{chart_html}</section><section><h2>Revenue decline anomalies</h2><ul>{anomaly_html}</ul></section>
<section><h2>Method</h2><p>Deterministic calculations are executed by Python tools. Agent planning, tool arguments, results, duration, and failures are retained in SQLite.</p></section>
</main></body></html>"""
    html_path.write_text(report_html, encoding="utf-8")
    return {
        "summary": f"Exported Markdown and HTML reports with {len(anomalies)} anomalies",
        "artifacts": {
            "markdown_report": str(markdown_path.resolve()),
            "html_report": str(html_path.resolve()),
        },
    }


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    common_path = {"type": "string", "description": "Absolute or relative CSV file path"}
    registry.register(ToolDefinition("inspect_dataset", "Inspect CSV schema, quality, and date coverage.", {
        "type": "object", "properties": {"dataset_path": common_path}, "required": ["dataset_path"], "additionalProperties": False
    }, inspect_dataset))
    registry.register(ToolDefinition("clean_data", "Validate retail rows, remove duplicates, and write a cleaned CSV.", {
        "type": "object", "properties": {"dataset_path": common_path, "output_name": {"type": "string"}}, "required": ["dataset_path"], "additionalProperties": False
    }, clean_data))
    registry.register(ToolDefinition("calculate_metrics", "Calculate revenue, orders, AOV, return rate, and repeat customer rate.", {
        "type": "object", "properties": {"dataset_path": common_path, "group_by": {"type": "string", "enum": ["country", "product", "month", "overall"]}}, "required": ["dataset_path"], "additionalProperties": False
    }, calculate_metrics))
    registry.register(ToolDefinition("detect_anomalies", "Compare equal recent periods and find group-level revenue declines.", {
        "type": "object", "properties": {"dataset_path": common_path, "group_by": {"type": "string", "enum": ["country", "product"]}, "decline_threshold": {"type": "number"}, "period_days": {"type": "integer"}}, "required": ["dataset_path"], "additionalProperties": False
    }, detect_anomalies))
    registry.register(ToolDefinition("plot_chart", "Create an SVG monthly net-revenue trend chart.", {
        "type": "object", "properties": {"dataset_path": common_path, "title": {"type": "string"}}, "required": ["dataset_path"], "additionalProperties": False
    }, plot_chart))
    registry.register(ToolDefinition("export_report", "Export the completed analysis as Markdown and HTML.", {
        "type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"], "additionalProperties": False
    }, export_report))
    return registry
