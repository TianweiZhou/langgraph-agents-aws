"""Conversation memory: saves graph state per thread.

DynamoDB when CHECKPOINT_TABLE is set (deployed), in-memory otherwise (tests, local runs).
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph_checkpoint_aws import DynamoDBSaver

from shared.config import Settings

CHECKPOINT_TTL_SECONDS = 30 * 24 * 3600  # conversations expire after 30 days


def make_checkpointer(settings: Settings) -> BaseCheckpointSaver[str]:
    if settings.checkpoint_table is None:
        return InMemorySaver()
    return DynamoDBSaver(
        table_name=settings.checkpoint_table,
        region_name=settings.aws_region,
        ttl_seconds=CHECKPOINT_TTL_SECONDS,
        enable_checkpoint_compression=True,
    )


def checkpoint_thread_id(user_id: str, thread_id: str) -> str:
    """Thread key scoped to the user, so one user can never load another's thread."""
    return f"{user_id}:{thread_id}"
