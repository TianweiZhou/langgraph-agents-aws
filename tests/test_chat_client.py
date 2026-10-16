from tools.chat_ui.client import ChatRequest, build_body, reply_text


def test_body_has_only_what_was_set() -> None:
    body = build_body(ChatRequest(user_id="u1", thread_id="t1", text="hi"))
    assert body == {"user_id": "u1", "thread_id": "t1", "request": "hi"}


def test_body_carries_agent_and_inputs() -> None:
    req = ChatRequest(
        user_id="u1", thread_id="t1", text="hi", agent="tone", tone="formal", audience="boss"
    )
    assert build_body(req) == {
        "user_id": "u1",
        "thread_id": "t1",
        "request": "hi",
        "agent": "tone",
        "inputs": {"tone": "formal", "audience": "boss"},
    }


def test_reply_text_picks_the_readable_field() -> None:
    assert reply_text({"result": {"rewritten_text": "Hello."}}) == "Hello."
    assert reply_text({"result": {"message": "Hi!"}}) == "Hi!"
    assert reply_text({"error": "Unknown agent 'x'"}) == "Error: Unknown agent 'x'"
