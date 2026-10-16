from langchain_core.messages import AIMessage, HumanMessage
from tools.chat_ui.admin_data import (
    ChatLine,
    RequestRow,
    compact,
    conversation_lines,
    parse_insights_rows,
    seconds,
    summarize,
    thread_refs_from_keys,
)


def _field(name: str, value: str) -> dict[str, str]:
    return {"field": name, "value": value}


def test_parse_insights_rows() -> None:
    results = [
        [
            _field("@timestamp", "2026-10-16 18:40:00.000"),
            _field("user_id", "me"),
            _field("thread_id", "t1"),
            _field("agent", "tone"),
            _field("status", "200"),
            _field("duration_ms", "812"),
            _field("input_tokens", "250"),
            _field("output_tokens", "18"),
        ],
        [
            _field("@timestamp", "2026-10-16 18:41:00.000"),
            _field("status", "400"),
            _field("error", "bad"),
        ],
    ]
    first, second = parse_insights_rows(results)
    assert (first.agent, first.status, first.duration_ms, first.input_tokens) == (
        "tone",
        200,
        812,
        250,
    )
    assert (second.user_id, second.status, second.error, second.duration_ms) == (
        None,
        400,
        "bad",
        0,
    )


def _row(status: int, duration_ms: int, tokens: int = 0) -> RequestRow:
    return RequestRow("ts", "u", "t", "tone", status, duration_ms, tokens, tokens, None)


def test_summarize() -> None:
    stats = summarize([_row(200, 100, 10), _row(200, 200, 10), _row(500, 900)])
    assert (stats.requests, stats.errors, stats.p50_ms) == (3, 1, 200)
    assert (stats.input_tokens, stats.output_tokens) == (20, 20)
    assert round(stats.error_rate, 2) == 0.33


def test_summarize_empty() -> None:
    stats = summarize([])
    assert (stats.requests, stats.p95_ms, stats.error_rate) == (0, 0, 0.0)


def test_thread_refs_from_keys_dedupes_and_splits() -> None:
    keys = ["CHECKPOINT_me:t1", "CHECKPOINT_me:t1", "CHUNK_me:t1##x", "CHECKPOINT_u2:a:b"]
    refs = thread_refs_from_keys(keys)
    assert [(r.user_id, r.thread_id) for r in refs] == [("me", "t1"), ("u2", "a:b")]


def test_conversation_lines() -> None:
    lines = conversation_lines([HumanMessage("hi"), AIMessage("Hello.", name="tone")])
    assert lines == [ChatLine("user", None, "hi"), ChatLine("assistant", "tone", "Hello.")]


def test_compact_formats() -> None:
    assert (seconds(1800), seconds(622)) == ("1.8 s", "0.6 s")
    assert (compact(1245), compact(61)) == ("1.2k", "61")
