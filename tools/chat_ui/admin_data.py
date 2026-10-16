"""Data for the admin page: request summaries from CloudWatch Logs, threads from DynamoDB.

Testing tool only; reads with the local `dev` AWS profile.
"""

import json
import statistics
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from langchain_core.messages import AIMessage, BaseMessage

ENV = "dev"
LOG_GROUP = f"/aws/lambda/agents-{ENV}-orchestrator"
TABLE = f"agents-{ENV}-conversations"
MEMORY_TABLE = f"agents-{ENV}-memory"
CHECKPOINT_PREFIX = "CHECKPOINT_"  # DynamoDBSaver's partition key prefix for a thread

SUMMARY_QUERY = """fields @timestamp, request_id, user_id, thread_id, route, status, duration_ms,
    model_calls, input_tokens, output_tokens, error
| filter message = "Request summary"
| sort @timestamp desc
| limit 1000"""


TRACE_QUERY = """fields @timestamp, step, node, next, duration_ms, received, produced
| filter message = "Trace step" and request_id = "{request_id}"
| sort step asc
| limit 100"""


@dataclass(frozen=True)
class RequestRow:
    timestamp: str
    request_id: str | None
    user_id: str | None
    thread_id: str | None
    route: str | None
    status: int
    duration_ms: int
    input_tokens: int
    output_tokens: int
    error: str | None


@dataclass(frozen=True)
class Stats:
    requests: int
    errors: int
    p50_ms: int
    p95_ms: int
    input_tokens: int
    output_tokens: int

    @property
    def error_rate(self) -> float:
        return self.errors / self.requests if self.requests else 0.0


@dataclass(frozen=True)
class ThreadRef:
    key: str  # checkpoint thread id, "user_id:thread_id"
    user_id: str
    thread_id: str


@dataclass(frozen=True)
class ChatLine:
    role: str  # "user" or "assistant"
    agent: str | None
    text: str


def seconds(ms: int) -> str:
    """Compact latency for a metric tile, e.g. 1800 -> '1.8 s'."""
    return f"{ms / 1000:.1f} s"


def compact(n: int) -> str:
    """Compact count for a metric tile, e.g. 1245 -> '1.2k'."""
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def _int(value: str | None) -> int:
    return int(float(value)) if value not in (None, "", "null") else 0


def _str(value: str | None) -> str | None:
    return value if value not in (None, "", "null") else None


def parse_insights_rows(results: Iterable[Sequence[dict[str, str]]]) -> list[RequestRow]:
    """Turn CloudWatch Logs Insights result rows into typed rows."""
    rows: list[RequestRow] = []
    for result in results:
        fields = {f["field"]: f.get("value") for f in result}
        rows.append(
            RequestRow(
                timestamp=fields.get("@timestamp") or "",
                request_id=_str(fields.get("request_id")),
                user_id=_str(fields.get("user_id")),
                thread_id=_str(fields.get("thread_id")),
                route=_str(fields.get("route")),
                status=_int(fields.get("status")),
                duration_ms=_int(fields.get("duration_ms")),
                input_tokens=_int(fields.get("input_tokens")),
                output_tokens=_int(fields.get("output_tokens")),
                error=_str(fields.get("error")),
            )
        )
    return rows


def _percentile(values: Sequence[int], pct: int) -> int:
    if not values:
        return 0
    if len(values) == 1:
        return values[0]
    return round(statistics.quantiles(values, n=100, method="inclusive")[pct - 1])


def summarize(rows: Sequence[RequestRow]) -> Stats:
    durations = sorted(r.duration_ms for r in rows)
    return Stats(
        requests=len(rows),
        errors=sum(r.status >= 400 for r in rows),
        p50_ms=_percentile(durations, 50),
        p95_ms=_percentile(durations, 95),
        input_tokens=sum(r.input_tokens for r in rows),
        output_tokens=sum(r.output_tokens for r in rows),
    )


def _run_insights(logs: Any, query: str, hours: int, poll_seconds: float) -> list[Any]:
    """Run a Logs Insights query over the last `hours` and wait for its raw results."""
    end = datetime.now(UTC)
    start = end - timedelta(hours=hours)
    query_id = logs.start_query(
        logGroupName=LOG_GROUP,
        startTime=int(start.timestamp()),
        endTime=int(end.timestamp()),
        queryString=query,
    )["queryId"]
    while True:
        response = logs.get_query_results(queryId=query_id)
        if response["status"] in ("Complete", "Failed", "Cancelled", "Timeout"):
            return response.get("results", [])
        time.sleep(poll_seconds)


def query_requests(logs: Any, hours: int, poll_seconds: float = 1.0) -> list[RequestRow]:
    """Request summaries over the last `hours`, newest first."""
    return parse_insights_rows(_run_insights(logs, SUMMARY_QUERY, hours, poll_seconds))


@dataclass(frozen=True)
class TraceStep:
    step: int
    node: str
    next: str
    duration_ms: int
    received: dict[str, Any]
    produced: dict[str, Any]


def _json(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed: Any = json.loads(value)
    except json.JSONDecodeError:
        return {"raw": value}
    if isinstance(parsed, dict):
        return cast(dict[str, Any], parsed)
    return {"value": parsed}


def parse_trace_rows(results: Iterable[Sequence[dict[str, str]]]) -> list[TraceStep]:
    steps: list[TraceStep] = []
    for result in results:
        fields = {f["field"]: f.get("value") for f in result}
        steps.append(
            TraceStep(
                step=_int(fields.get("step")),
                node=fields.get("node") or "?",
                next=fields.get("next") or "?",
                duration_ms=_int(fields.get("duration_ms")),
                received=_json(fields.get("received")),
                produced=_json(fields.get("produced")),
            )
        )
    return sorted(steps, key=lambda s: s.step)


def query_trace(
    logs: Any, request_id: str, hours: int, poll_seconds: float = 1.0
) -> list[TraceStep]:
    """Every step of one request, in order."""
    safe_id = request_id.replace('"', "")
    query = TRACE_QUERY.format(request_id=safe_id)
    return parse_trace_rows(_run_insights(logs, query, hours, poll_seconds))


# Fixed layout of the graph for the flow diagram; agents are added as found.
_FLOW_ORDER = ["user", "supervisor", "recall", "remember", "end"]


def flow_dot(steps: Sequence[TraceStep], agents: Sequence[str]) -> str:
    """Graphviz DOT of the graph, with the path this request took numbered and highlighted."""
    visited: set[str] = {s.node for s in steps} | {"user"}
    if steps:
        visited.add("end")
    nodes = [*_FLOW_ORDER[:3], *agents, *_FLOW_ORDER[3:]]
    lines = [
        "digraph flow {",
        '  rankdir=LR; bgcolor="transparent";',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11];',
        '  edge [fontname="Helvetica", fontsize=10];',
    ]
    for node in nodes:
        on_path = node in visited
        fill = "#dbeafe" if on_path else "#f3f4f6"
        color = "#2563eb" if on_path else "#d1d5db"
        font = "#111827" if on_path else "#9ca3af"
        lines.append(f'  "{node}" [fillcolor="{fill}", color="{color}", fontcolor="{font}"];')
    if steps:
        lines.append('  "user" -> "supervisor" [label="0", color="#2563eb", penwidth=2];')
    for s in steps:
        target = s.next if s.next != "?" else "end"
        lines.append(
            f'  "{s.node}" -> "{target}" '
            f'[label="{s.step}", color="#2563eb", penwidth=2, fontcolor="#2563eb"];'
        )
    lines.append("}")
    return "\n".join(lines)


def thread_refs_from_keys(partition_keys: Iterable[str]) -> list[ThreadRef]:
    """Unique threads from checkpoint partition keys like 'CHECKPOINT_me:t1'."""
    refs: dict[str, ThreadRef] = {}
    for pk in partition_keys:
        if not pk.startswith(CHECKPOINT_PREFIX):
            continue
        key = pk.removeprefix(CHECKPOINT_PREFIX)
        user_id, _, thread_id = key.partition(":")
        refs[key] = ThreadRef(key=key, user_id=user_id, thread_id=thread_id)
    return sorted(refs.values(), key=lambda r: (r.user_id, r.thread_id))


def list_threads(dynamodb: Any) -> list[ThreadRef]:
    """All threads in the conversations table (a small scan; fine for dev)."""
    keys: list[str] = []
    for page in dynamodb.get_paginator("scan").paginate(
        TableName=TABLE,
        ProjectionExpression="PK",
        FilterExpression="begins_with(PK, :p)",
        ExpressionAttributeValues={":p": {"S": CHECKPOINT_PREFIX}},
    ):
        keys.extend(item["PK"]["S"] for item in page["Items"])
    return thread_refs_from_keys(keys)


def conversation_lines(messages: Sequence[BaseMessage]) -> list[ChatLine]:
    return [
        ChatLine(role="assistant", agent=m.name, text=m.text)
        if isinstance(m, AIMessage)
        else ChatLine(role="user", agent=None, text=m.text)
        for m in messages
    ]


def load_conversation(saver: Any, ref: ThreadRef) -> tuple[list[ChatLine], str | None]:
    """The thread's messages and when it was last updated."""
    snapshot = saver.get_tuple({"configurable": {"thread_id": ref.key, "checkpoint_ns": ""}})
    if snapshot is None:
        return [], None
    checkpoint = snapshot.checkpoint
    messages = checkpoint["channel_values"].get("messages", [])
    return conversation_lines(messages), checkpoint.get("ts")


def users_with_contacts(namespaces: Iterable[tuple[str, ...]]) -> list[str]:
    """User ids from store namespaces like ('users', 'me', 'contacts')."""
    users = {
        ns[1] for ns in namespaces if len(ns) >= 3 and ns[0] == "users" and ns[2] == "contacts"
    }
    return sorted(users)


def list_memory_users(store: Any) -> list[str]:
    return users_with_contacts(store.list_namespaces(prefix=("users",), limit=1000))
