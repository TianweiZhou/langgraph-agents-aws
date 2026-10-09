# Agentic System on AWS

A multi-agent system built on AWS with an enterprise-style structure. The first agent is a **tone adjuster** (rewrites text in a requested tone). More agents will be added later, so everything is built as a small agent framework first, with the tone agent as the first plugin.

This is a personal learning and portfolio project. Keep it realistic and clean, but do not over-engineer.

## Decisions (do not change without asking)

- Agent framework: **LangGraph**
- Model provider: **Amazon Bedrock** (via `langchain-aws`)
- Language: **Python 3.12**
- Infra as code: **AWS CDK in Python**
- Compute: **API Gateway -> Lambda** (container image is a fallback if packaging or timeouts become a problem)
- CI/CD: **GitHub Actions**
- Tooling: `uv` (deps), `ruff` (lint and format), `pytest`, `pyright` (types)

If something here seems wrong or a better option exists, say so and ask before switching.

## Architecture

```
Client -> API Gateway -> Lambda (LangGraph app)
                           |- Router node (picks agent by intent)
                           |- Tone agent node (first agent)
                           |- Future agent nodes
                           |-> Amazon Bedrock (models)
                           |-> DynamoDB (LangGraph checkpoints, optional at first)

Shared services: Bedrock Guardrails, versioned prompt files, Secrets Manager, CloudWatch logs + X-Ray traces
CI/CD + IaC: GitHub Actions + CDK deploy the whole stack
```

Design rules:
- The orchestrator is a LangGraph graph. Today the router is a simple intent check. Later it can become an LLM-driven supervisor, so keep the router behind a clean interface.
- Every agent is its own module with its own prompt, schemas, tests, and eval set. Adding agent 2 must not require changes to the tone agent.
- Anything shared (config, logging, model factory, guardrails, prompt loading) lives in `shared/`.

## Repo layout

```
.
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── src/
│   ├── handler.py              # Lambda entrypoint
│   ├── orchestrator/
│   │   ├── graph.py            # builds the LangGraph graph
│   │   ├── router.py           # router node
│   │   └── state.py            # typed graph state
│   ├── agents/
│   │   ├── base.py             # agent contract
│   │   └── tone/
│   │       ├── agent.py
│   │       ├── schemas.py
│   │       └── prompt.md
│   └── shared/
│       ├── config.py
│       ├── logging.py
│       ├── models.py           # Bedrock chat model factory
│       ├── prompts.py          # load_prompt(agent, version)
│       └── guardrails.py
├── infra/                      # CDK app
│   ├── app.py
│   └── stacks/
├── evals/
│   └── tone/
│       ├── cases.jsonl
│       └── run.py
├── tests/
└── .github/workflows/
```

## Agent contract

Every agent exposes:
- `name`: unique string id
- `description`: short text the router can use to decide when to pick it
- typed input and output models (pydantic) in its own `schemas.py`
- a node function that takes the graph state and returns a partial state update

The tone agent: input is the text, the target tone, and an optional audience. Output is the rewritten text. Keep meaning intact, only change tone.

## State and request schema

Include `user_id` and `thread_id` from day one, even though memory is not built yet. This makes adding memory later a new store plus a few read/write calls, not an API change.

Starting point for the graph state (refine as needed):

```python
class GraphState(TypedDict):
    user_id: str
    thread_id: str
    request: str
    agent: str | None  # chosen by the router
    result: dict | None  # agent output
```

## Prompts

- Prompts live in files inside each agent's folder. Never hardcode prompts in Python.
- Load them only through `shared/prompts.py` (`load_prompt`). This keeps it easy to later publish prompts to Bedrock Prompt Management and load a pinned version, without touching agent code.

## Config

- Model IDs, region, and feature flags come from config or environment, not hardcoded.
- AWS region: `us-east-1`
- Local AWS profile name: `dev`
- Bedrock model: `<set me>` (confirm model access is enabled in the account first)

## Conventions

- Full type hints, pyright clean.
- `ruff` for lint and format, no warnings left behind.
- Structured JSON logs with `thread_id` and `agent` on every log line.
- Small focused modules, small functions, clear names.
- Tests for every agent and for the router. Mock Bedrock in unit tests.
- Do not add dependencies without saying why.

## Evals

- Each agent has an eval set in `evals/<agent>/cases.jsonl` (input plus the expected tone or properties).
- `evals/<agent>/run.py` runs the cases through the agent and scores the results.
- Prompt changes should be checked against the evals before merging.

## CI/CD

- On PR: lint, type check, unit tests.
- On merge to main: deploy to dev with CDK.
- Prod deploy needs a manual approval step.
- Use GitHub OIDC to assume an AWS role instead of stored keys (set up when we reach CI/CD).

## How to work with me

- LangGraph, `langchain-aws`, and Bedrock APIs change fast. Check current docs before writing integration code instead of relying on memory.
- Work in small steps. After each step, summarize what changed and what is next.
- Ask before making decisions that are not covered here (new services, new frameworks, big refactors).
- Keep explanations short and plain.

## Out of scope for now

Do not build these yet, but do not block them either:
- Supervisor agent (router will evolve into it)
- Long-term memory (only the `user_id` and `thread_id` plumbing now)
- Step Functions
- Bedrock Agents
- Separate prod account

## First task: skeleton only

Build the skeleton and nothing else:
1. Repo structure, `pyproject.toml`, ruff and pyright config.
2. Base agent contract and typed graph state.
3. Router node (simple stub is fine).
4. A hello-world agent wired through the graph, with a unit test.
5. CDK stack with API Gateway and a Lambda running the graph, deployed to dev.
6. A GitHub Actions workflow that runs lint, types, and tests.

Stop after the hello-world graph works end to end in dev. The real tone agent comes next.
