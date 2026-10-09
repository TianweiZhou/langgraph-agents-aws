from orchestrator.state import GraphState


def make_state(request: str = "hi", agent: str | None = None) -> GraphState:
    return {"user_id": "u1", "thread_id": "t1", "request": request, "agent": agent, "result": None}
