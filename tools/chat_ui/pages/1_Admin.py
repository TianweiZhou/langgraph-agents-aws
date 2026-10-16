"""Admin page: what happened in dev. Opens from the chat app's sidebar."""

import json
from dataclasses import asdict
from typing import Any

import boto3
import streamlit as st
from admin_data import (
    MEMORY_TABLE,
    TABLE,
    RequestRow,
    ThreadRef,
    TraceStep,
    compact,
    flow_dot,
    list_memory_users,
    list_threads,
    load_conversation,
    query_requests,
    query_trace,
    seconds,
    summarize,
)
from client import DEFAULT_PROFILE, REGION, ensure_src_on_path
from langgraph_checkpoint_aws import DynamoDBSaver, DynamoDBStore

ensure_src_on_path()
from agents.registry import ALL_AGENTS  # noqa: E402  (needs src/ on the path)
from shared.memory import PeopleMemory  # noqa: E402

RANGES = {"Last hour": 1, "Last 24 hours": 24, "Last 7 days": 168}
AGENT_NAMES = [a.name for a in ALL_AGENTS]


@st.cache_resource
def aws() -> Any:
    # boto3 ships without type hints, so the session is typed as Any.
    session: Any = boto3.Session(profile_name=DEFAULT_PROFILE, region_name=REGION)
    return session


@st.cache_resource
def saver() -> DynamoDBSaver:
    return DynamoDBSaver(table_name=TABLE, session=aws())


@st.cache_resource
def people() -> PeopleMemory:
    return PeopleMemory(DynamoDBStore(table_name=MEMORY_TABLE, boto3_session=aws()))


@st.cache_data(ttl=60, show_spinner="Querying CloudWatch logs...")
def requests(hours: int) -> list[RequestRow]:
    return query_requests(aws().client("logs"), hours)


@st.cache_data(ttl=300, show_spinner="Loading the trace...")
def trace(request_id: str, hours: int) -> list[TraceStep]:
    return query_trace(aws().client("logs"), request_id, hours)


@st.cache_data(ttl=60, show_spinner="Reading conversations...")
def threads() -> list[ThreadRef]:
    return list_threads(aws().client("dynamodb"))


def table(rows: list[RequestRow]) -> None:
    st.dataframe([asdict(r) for r in rows], hide_index=True, use_container_width=True)


def request_label(row: RequestRow) -> str:
    when = row.timestamp[11:19]
    status = "" if row.status == 200 else f" · HTTP {row.status}"
    route = row.route or "supervisor"
    return f"{when} · {row.user_id} · {route} · {seconds(row.duration_ms)}{status}"


def step_card(step: TraceStep) -> None:
    with st.container(border=True):
        st.markdown(f"**{step.step}. {step.node} → {step.next}**  ·  {step.duration_ms} ms")
        decision: dict[str, Any] = step.produced.get("decision") or {}
        if step.node == "supervisor" and decision:
            target = f" → **{decision['agent']}**" if decision.get("agent") else ""
            how = "model" if decision.get("model_call", True) else "no model call"
            st.markdown(
                f"Decision: `{decision.get('action')}`{target} ({how})  \n"
                f"_{decision.get('reason', '')}_"
            )
        received, produced = st.columns(2)
        received.caption("Received")
        received.code(json.dumps(step.received, indent=2, ensure_ascii=False), language="json")
        produced.caption("Produced")
        produced.code(json.dumps(step.produced, indent=2, ensure_ascii=False), language="json")


st.set_page_config(page_title="Admin", page_icon=":bar_chart:", layout="wide")
st.title("Admin")

with st.sidebar:
    range_label = st.selectbox("Time range", list(RANGES), index=1)
    hours = RANGES[range_label]
    if st.button("Refresh"):
        st.cache_data.clear()

rows = requests(hours)
stats = summarize(rows)

top = st.columns(3)
top[0].metric("Requests", stats.requests)
top[1].metric("Error rate", f"{stats.error_rate:.0%}", help=f"{stats.errors} errors")
top[2].metric("p50 latency", seconds(stats.p50_ms))
bottom = st.columns(3)
bottom[0].metric("p95 latency", seconds(stats.p95_ms))
bottom[1].metric("Tokens in", compact(stats.input_tokens), help=f"{stats.input_tokens:,}")
bottom[2].metric("Tokens out", compact(stats.output_tokens), help=f"{stats.output_tokens:,}")

flow_tab, requests_tab, conversations_tab, people_tab, errors_tab = st.tabs(
    ["Flow", "Requests", "Conversations", "People", "Errors"]
)

with flow_tab:
    traced = [r for r in rows if r.request_id]
    if not traced:
        st.info("No traced requests in this time range. Send a message from the chat page.")
    else:
        chosen = st.selectbox("Request", traced, format_func=request_label)
        steps = trace(chosen.request_id or "", hours)
        if not steps:
            st.warning("No trace lines found yet (logs can take a few seconds). Press Refresh.")
        else:
            st.graphviz_chart(flow_dot(steps, AGENT_NAMES), use_container_width=True)
            st.caption(
                f"user `{chosen.user_id}` · thread `{chosen.thread_id}` · "
                f"request `{chosen.request_id}` · {len(steps)} steps · "
                f"{compact(chosen.input_tokens)} tokens in"
            )
            for step in steps:
                step_card(step)

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
        thread = st.selectbox("Thread", refs, format_func=lambda r: f"{r.user_id} / {r.thread_id}")
        lines, updated = load_conversation(saver(), thread)
        st.caption(f"key `{thread.key}` · last updated {updated or 'unknown'}")
        for line in lines:
            with st.chat_message(line.role):
                st.markdown(line.text)
                if line.agent:
                    st.caption(line.agent)
        thread_rows = [
            r for r in rows if (r.user_id, r.thread_id) == (thread.user_id, thread.thread_id)
        ]
        if thread_rows:
            st.subheader("Requests in this thread")
            table(thread_rows)

with people_tab:
    users = list_memory_users(people().store)
    if not users:
        st.info("No one remembered yet. Try telling the chat 'Sara is my manager'.")
    else:
        user = st.selectbox("User", users)
        for contact in people().list_contacts(user):
            with st.expander(f"{contact.name} · {contact.relationship or 'relationship unknown'}"):
                st.markdown(f"**How to sound:** {contact.tone_notes or '-'}")
                if contact.facts:
                    st.markdown("**Facts:** " + "; ".join(contact.facts))
                for event in people().recent_events(user, contact.name, limit=10):
                    st.caption(f"{event.at[:16].replace('T', ' ')} · {event.summary}")
                if st.button(f"Forget {contact.name}", key=f"forget-{user}-{contact.name}"):
                    people().forget(user, contact.name)
                    st.rerun()

with errors_tab:
    errors = [r for r in rows if r.status >= 400]
    if errors:
        table(errors)
    else:
        st.success("No errors in this time range.")
