from tools.chat_ui.client import ChatRequest, build_body, reply_text


def test_body_is_just_the_message() -> None:
    body = build_body(ChatRequest(user_id="u1", thread_id="t1", text="hi"))
    assert body == {"user_id": "u1", "thread_id": "t1", "request": "hi"}


def test_reply_text() -> None:
    assert reply_text({"reply": "Hello."}) == "Hello."
    assert reply_text({"error": "Invalid request body"}) == "Error: Invalid request body"


def test_reply_text_shows_note_under_reply() -> None:
    body = {"reply": "Hi Zoe.", "note": "I don't know Zoe yet."}
    assert reply_text(body) == "Hi Zoe.\n\n_I don't know Zoe yet._"
