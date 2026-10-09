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
