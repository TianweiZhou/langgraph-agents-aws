"""Local chat page for testing the agents. Run from the repo root:

uv run --group ui streamlit run tools/chat_ui/app.py
"""

import uuid
from typing import Any

import streamlit as st
from client import ChatClient, ChatReply, ChatRequest, DevApiClient, LocalClient

TARGETS = {"Dev API": DevApiClient, "Local": LocalClient}
AGENTS = {"Auto (router decides)": None, "tone": "tone", "hello": "hello"}


@st.cache_resource
def get_client(target: str) -> ChatClient:
    return TARGETS[target]()


def new_thread() -> None:
    st.session_state.thread_id = f"chat-{uuid.uuid4().hex[:8]}"
    st.session_state.messages = []


def caption(reply: ChatReply) -> str:
    return f"{reply.agent or '-'} · HTTP {reply.status} · {reply.seconds:.1f}s"


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
    agent_label = st.selectbox("Agent", list(AGENTS))
    tone = st.text_input("Tone", value="formal", help="Leave empty to skip.")
    audience = st.text_input("Audience (optional)")
    st.divider()
    st.caption(f"thread_id: `{st.session_state.thread_id}`")
    st.button("New thread", on_click=new_thread)

st.title("Agents chat")
st.caption("Each message goes through the router. Use the sidebar to pick the agent and tone.")

for message in st.session_state.messages:
    show(message)

if text := st.chat_input("Type text to rewrite..."):
    user_message = {"role": "user", "text": text}
    st.session_state.messages.append(user_message)
    show(user_message)

    request = ChatRequest(
        user_id="chat-ui",
        thread_id=st.session_state.thread_id,
        text=text,
        agent=AGENTS[agent_label],
        tone=tone.strip() or None,
        audience=audience.strip() or None,
    )
    with st.spinner("Thinking..."):
        try:
            reply = get_client(target).send(request)
            answer = {"role": "assistant", "text": reply.text, "caption": caption(reply)}
        except Exception as exc:
            answer = {"role": "assistant", "text": f"Client error: {exc}", "caption": target}
    st.session_state.messages.append(answer)
    show(answer)
