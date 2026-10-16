"""Admin page: what happened in dev. Opens from the chat app's sidebar."""

from dataclasses import asdict
from typing import Any

import boto3
import streamlit as st
from admin_data import (
    TABLE,
    RequestRow,
    ThreadRef,
    compact,
    list_threads,
    load_conversation,
    query_requests,
    seconds,
    summarize,
)
from client import DEFAULT_PROFILE, REGION
from langgraph_checkpoint_aws import DynamoDBSaver

RANGES = {"Last hour": 1, "Last 24 hours": 24, "Last 7 days": 168}


@st.cache_resource
def aws() -> Any:
    # boto3 ships without type hints, so the session is typed as Any.
    session: Any = boto3.Session(profile_name=DEFAULT_PROFILE, region_name=REGION)
    return session


@st.cache_resource
def saver() -> DynamoDBSaver:
    return DynamoDBSaver(table_name=TABLE, session=aws())


@st.cache_data(ttl=60, show_spinner="Querying CloudWatch logs...")
def requests(hours: int) -> list[RequestRow]:
    return query_requests(aws().client("logs"), hours)


@st.cache_data(ttl=60, show_spinner="Reading conversations...")
def threads() -> list[ThreadRef]:
    return list_threads(aws().client("dynamodb"))


def table(rows: list[RequestRow]) -> None:
    st.dataframe([asdict(r) for r in rows], hide_index=True, use_container_width=True)


st.set_page_config(page_title="Admin", page_icon=":bar_chart:", layout="wide")
st.title("Admin")

with st.sidebar:
    range_label = st.selectbox("Time range", list(RANGES), index=1)
    if st.button("Refresh"):
        st.cache_data.clear()

rows = requests(RANGES[range_label])
stats = summarize(rows)

top = st.columns(3)
top[0].metric("Requests", stats.requests)
top[1].metric("Error rate", f"{stats.error_rate:.0%}", help=f"{stats.errors} errors")
top[2].metric("p50 latency", seconds(stats.p50_ms))
bottom = st.columns(3)
bottom[0].metric("p95 latency", seconds(stats.p95_ms))
bottom[1].metric("Tokens in", compact(stats.input_tokens), help=f"{stats.input_tokens:,}")
bottom[2].metric("Tokens out", compact(stats.output_tokens), help=f"{stats.output_tokens:,}")

requests_tab, conversations_tab, errors_tab = st.tabs(["Requests", "Conversations", "Errors"])

with requests_tab:
    if rows:
        table(rows)
    else:
        st.info("No requests in this time range.")

with conversations_tab:
    refs = threads()
    if not refs:
        st.info("No conversations saved yet.")
    else:
        chosen = st.selectbox("Thread", refs, format_func=lambda r: f"{r.user_id} / {r.thread_id}")
        lines, updated = load_conversation(saver(), chosen)
        st.caption(f"key `{chosen.key}` · last updated {updated or 'unknown'}")
        for line in lines:
            with st.chat_message(line.role):
                st.markdown(line.text)
                if line.agent:
                    st.caption(line.agent)
        thread_rows = [
            r for r in rows if (r.user_id, r.thread_id) == (chosen.user_id, chosen.thread_id)
        ]
        if thread_rows:
            st.subheader("Requests in this thread")
            table(thread_rows)

with errors_tab:
    errors = [r for r in rows if r.status >= 400]
    if errors:
        table(errors)
    else:
        st.success("No errors in this time range.")
