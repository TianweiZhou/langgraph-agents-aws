"""Local chat page for testing. You talk to the supervisor; it decides which agents to use.

Run from the repo root:

    uv run --group ui streamlit run tools/chat_ui/app.py
"""

import uuid
from typing import Any

import streamlit as st
from client import ChatClient, ChatReply, ChatRequest, DevApiClient, LocalClient

TARGETS = {"Dev API": DevApiClient, "Local": LocalClient}
USER_ID = "me"


@st.cache_resource
def get_client(target: str) -> ChatClient:
    return TARGETS[target]()


def new_thread() -> None:
    st.session_state.thread_id = f"chat-{uuid.uuid4().hex[:8]}"
    st.session_state.messages = []


def caption(reply: ChatReply) -> str:
    path = " → ".join(["supervisor", *reply.route]) if reply.route else "supervisor"
    request = f" · request `{reply.request_id[:8]}`" if reply.request_id else ""
    return f"{path} · HTTP {reply.status} · {reply.seconds:.1f}s{request}"


def show(message: dict[str, Any]) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["text"])
        if message.get("caption"):
            st.caption(message["caption"])


st.set_page_config(page_title="Agents chat", page_icon=":speech_balloon:")
if "thread_id" not in st.session_state:
    new_thread()

with st.sidebar:
    st.header("Settings")
    target = st.radio("Target", list(TARGETS), help="Dev API = deployed stack. Local = this Mac.")
    st.divider()
    st.caption(f"user: `{USER_ID}` · thread: `{st.session_state.thread_id}`")
    st.button("New thread", on_click=new_thread)

st.title("Agents chat")
st.caption(
    'You talk to the supervisor. Try "Sara is my manager, keep it formal but warm", '
    'then "replying to Sara, she asked why the report is late. '
    'my draft: sorry got busy, will send tmrw".'
)

for message in st.session_state.messages:
    show(message)

if text := st.chat_input("Message the supervisor..."):
    user_message = {"role": "user", "text": text}
    st.session_state.messages.append(user_message)
    show(user_message)

    request = ChatRequest(user_id=USER_ID, thread_id=st.session_state.thread_id, text=text)
    with st.spinner("Thinking..."):
        try:
            reply = get_client(target).send(request)
            answer = {"role": "assistant", "text": reply.text, "caption": caption(reply)}
        except Exception as exc:
            answer = {"role": "assistant", "text": f"Client error: {exc}", "caption": target}
    st.session_state.messages.append(answer)
    show(answer)
