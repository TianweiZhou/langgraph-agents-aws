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

## Agents

| Agent | Picked when | Inputs |
| --- | --- | --- |
| `tone` | `agent: "tone"`, or `inputs.tone` is set, or the request says rewrite/rephrase/reword/tone | `request` = text, `inputs.tone`, optional `inputs.audience` |
| `hello` | fallback | `request` |

Example request body:

```json
{"user_id": "u1", "request": "hey send me the report", "inputs": {"tone": "formal", "audience": "my manager"}}
```

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

Opens http://localhost:8501. In the sidebar pick the **Target** (`Dev API` = deployed stack, signed with the
`dev` profile; `Local` = runs the handler on your Mac against real Bedrock), the agent, tone, and audience.
The `thread_id` shown in the sidebar is the one to search for in CloudWatch logs.

### Admin page

The chat app has an **Admin** page (sidebar) that reads dev with your `dev` profile:
request metrics (count, error rate, p50/p95 latency, tokens), a table of recent requests,
saved conversations from DynamoDB, and recent errors. Requests come from the
`Request summary` log line the Lambda writes once per request.
