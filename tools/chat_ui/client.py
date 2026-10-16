"""Sends chat messages to the orchestrator, either the deployed dev API or the local graph.

Testing tool only; never deployed.
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

ROOT = Path(__file__).resolve().parents[2]
REGION = "us-east-1"
STACK_NAME = "AgentsApi-dev"
DEFAULT_PROFILE = "dev"


@dataclass(frozen=True)
class ChatRequest:
    user_id: str
    thread_id: str
    text: str
    agent: str | None = None
    tone: str | None = None
    audience: str | None = None


@dataclass(frozen=True)
class ChatReply:
    status: int
    agent: str | None
    text: str
    seconds: float
    body: dict[str, Any]


class ChatClient(Protocol):
    def send(self, req: ChatRequest) -> ChatReply: ...


def build_body(req: ChatRequest) -> dict[str, Any]:
    """The JSON body the API expects."""
    body: dict[str, Any] = {"user_id": req.user_id, "thread_id": req.thread_id, "request": req.text}
    if req.agent:
        body["agent"] = req.agent
    inputs = {k: v for k, v in (("tone", req.tone), ("audience", req.audience)) if v}
    if inputs:
        body["inputs"] = inputs
    return body


def reply_text(body: dict[str, Any]) -> str:
    """The human-readable part of an API response."""
    if "error" in body:
        return f"Error: {body['error']}"
    result: dict[str, Any] = body.get("result") or {}
    for key in ("rewritten_text", "message"):
        if key in result:
            return str(result[key])
    return json.dumps(result)


def _reply(status: int, body: dict[str, Any], started: float) -> ChatReply:
    return ChatReply(
        status=status,
        agent=body.get("agent"),
        text=reply_text(body),
        seconds=time.perf_counter() - started,
        body=body,
    )


class DevApiClient:
    """Calls the deployed API with requests signed by the `dev` AWS profile."""

    def __init__(self, profile: str | None = None) -> None:
        # boto3 ships without type hints, so the session is typed as Any.
        self._session: Any = boto3.Session(
            profile_name=profile or os.environ.get("AWS_PROFILE", DEFAULT_PROFILE),
            region_name=REGION,
        )
        self.url = self._invoke_url()

    def _invoke_url(self) -> str:
        if url := os.environ.get("CHAT_API_URL"):
            return url
        cfn = self._session.client("cloudformation")
        outputs: list[dict[str, str]] = cfn.describe_stacks(StackName=STACK_NAME)["Stacks"][0][
            "Outputs"
        ]
        return next(o["OutputValue"] for o in outputs if o.get("OutputKey") == "InvokeUrl")

    def send(self, req: ChatRequest) -> ChatReply:
        started = time.perf_counter()
        data = json.dumps(build_body(req)).encode()
        signed = AWSRequest(
            method="POST", url=self.url, data=data, headers={"Content-Type": "application/json"}
        )
        credentials = self._session.get_credentials()
        if credentials is None:
            raise RuntimeError("No AWS credentials for the chat client profile")
        SigV4Auth(credentials, "execute-api", REGION).add_auth(signed)
        http_req = urllib.request.Request(
            self.url, data=data, headers=dict(signed.headers.items()), method="POST"
        )
        try:
            with urllib.request.urlopen(http_req, timeout=35) as resp:
                return _reply(resp.status, json.loads(resp.read()), started)
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            body = json.loads(raw) if raw.startswith(b"{") else {"error": raw.decode()}
            return _reply(exc.code, body, started)


def _cdk_model_id() -> str:
    context = json.loads((ROOT / "cdk.json").read_text(encoding="utf-8"))["context"]
    return str(context["bedrockModelId"])


class LocalClient:
    """Runs the Lambda handler in this process (real Bedrock, local code)."""

    def __init__(self) -> None:
        os.environ.setdefault("AWS_PROFILE", DEFAULT_PROFILE)
        os.environ.setdefault("AWS_REGION", REGION)
        os.environ.setdefault("BEDROCK_MODEL_ID", _cdk_model_id())
        os.environ.setdefault("SUPERVISOR_ENABLED", "true")  # match the deployed stack
        src = str(ROOT / "src")
        if src not in sys.path:
            sys.path.insert(0, src)
        import handler

        self._handler: Any = handler

    def send(self, req: ChatRequest) -> ChatReply:
        started = time.perf_counter()
        resp = self._handler.lambda_handler({"body": json.dumps(build_body(req))}, None)
        return _reply(int(resp["statusCode"]), json.loads(resp["body"]), started)
