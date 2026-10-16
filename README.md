# langgraph-agents-aws

A small multi-agent system on AWS. A LangGraph graph runs in Lambda behind API Gateway,
routes each request to an agent, and calls Amazon Bedrock models. The first real agent
will be a tone adjuster.

## Stack

Python 3.12, LangGraph, Amazon Bedrock (`langchain-aws`), AWS CDK (Python), GitHub Actions.
Tooling: `uv`, `ruff`, `pyright`, `pytest`.

## Local setup

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
```

## Layout

- `src/orchestrator/` - graph, router, typed state
- `src/agents/` - one folder per agent, plus the shared agent contract
- `src/shared/` - config, logging, prompt loading, model factory
- `infra/` - CDK app
- `tests/` - unit tests (Bedrock is mocked)

## How it works

You only talk to the **supervisor** (`src/orchestrator/supervisor.py`). For each message it decides,
in a loop, to send work to an agent, reply itself (greetings, questions, clarifying questions), or
finish. Agents report back to it; one message can use several agents, up to 4 steps.

```
START -> supervisor -> recall -> <agent> -> supervisor -> ... -> remember -> END
```

| Agent | Handles |
| --- | --- |
| `tone` | Rewrites a message or draft in a tone; edits its previous rewrite ("shorter please") |
| `memory` | "Sara is my manager", "forget Mike", "who is Sara?", "who do you know?" |

Request and response:

```json
{"user_id": "u1", "thread_id": "t1", "request": "replying to Sara: sorry, will send tmrw"}
{"thread_id": "t1", "request_id": "...", "reply": "...", "route": ["tone"], "note": null, "result": {...}}
```

`inputs` (e.g. `{"tone": "formal"}`) is accepted as an optional hint. Every graph step writes a
`Trace step` log line (request id, step, node, what it received and produced, next node) that the
Admin page's **Flow** tab draws. `TRACE_CONTENT=false` keeps message text out of traces.

## Model

The Bedrock model is set in `cdk.json` (`bedrockModelId`) and reaches the Lambda as `BEDROCK_MODEL_ID`.
To switch, edit `cdk.json` or override one deploy: `cdk deploy -c bedrockModelId=<id> --profile dev`.
IAM permissions follow the configured id automatically.

## Evals

```bash
AWS_PROFILE=dev BEDROCK_MODEL_ID=us.amazon.nova-2-lite-v1:0 PYTHONPATH=src uv run python -m evals.tone.run
```

Runs `evals/tone/cases.jsonl` against the real model and exits non-zero below an 80% pass rate.

## Chat UI (local testing only)

```bash
uv run --group ui streamlit run tools/chat_ui/app.py
```

Opens http://localhost:8501. You chat with the supervisor; pick the **Target** in the sidebar
(`Dev API` = deployed stack, signed with the `dev` profile; `Local` = runs the handler on your Mac
against real Bedrock). Each reply shows the route it took and its request id.

### Admin page

The chat app has an **Admin** page (sidebar) that reads dev with your `dev` profile:
- **Flow**: pick a request, see the path it took through the graph (numbered), and one card per
  step with what arrived, the supervisor's decision and reason, what it produced, and where it went next.
- **Requests**, **Conversations**, **People** (long-term memory, with Forget), **Errors**, plus
  metrics (count, error rate, p50/p95 latency, tokens).
