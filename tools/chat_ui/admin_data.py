"""Data for the admin page: request summaries from CloudWatch Logs, threads from DynamoDB.

Testing tool only; reads with the local `dev` AWS profile.
"""

import statistics
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage

ENV = "dev"
LOG_GROUP = f"/aws/lambda/agents-{ENV}-orchestrator"
TABLE = f"agents-{ENV}-conversations"
CHECKPOINT_PREFIX = "CHECKPOINT_"  # DynamoDBSaver's partition key prefix for a thread

SUMMARY_QUERY = """fields @timestamp, user_id, thread_id, agent, status, duration_ms,
    model_calls, input_tokens, output_tokens, error
| filter message = "Request summary"
| sort @timestamp desc
| limit 1000"""


@dataclass(frozen=True)
class RequestRow:
    timestamp: str
    user_id: str | None
    thread_id: str | None
    agent: str | None
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
                user_id=_str(fields.get("user_id")),
                thread_id=_str(fields.get("thread_id")),
                agent=_str(fields.get("agent")),
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


def query_requests(logs: Any, hours: int, poll_seconds: float = 1.0) -> list[RequestRow]:
    """Run the summary query over the last `hours` and wait for the result."""
    end = datetime.now(UTC)
    start = end - timedelta(hours=hours)
    query_id = logs.start_query(
        logGroupName=LOG_GROUP,
        startTime=int(start.timestamp()),
        endTime=int(end.timestamp()),
        queryString=SUMMARY_QUERY,
    )["queryId"]
    while True:
        response = logs.get_query_results(queryId=query_id)
        if response["status"] in ("Complete", "Failed", "Cancelled", "Timeout"):
            return parse_insights_rows(response.get("results", []))
        time.sleep(poll_seconds)


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
