"""Deciding which agent an LLM call belongs to.

A single Claude Code worker runs several agents at once: its own main loop, plus one
subagent per ``Agent`` tool call. They share one trace label and their calls arrive
interleaved.

An agent only appends to its history, so the tool_call ids of its next call begin with
the ids of its current one, and attribution is a prefix test over that sequence. An
agent that has issued no tool call yet is matched on its user messages instead.
"""

from dataclasses import dataclass, field
from typing import Any

from nexus.extensions.tasks.hitl.trace.langfuse.upload_dedup import fingerprint
from nexus.trace.processors.langfuse_tracker import _message_tool_call_ids


def tool_call_ids(messages: list[dict[str, Any]]) -> tuple[str, ...]:
    """Every tool_call id in this history, in the order the model issued them.

    Read from the assistant messages alone; each tool result repeats an id its
    assistant message already carries.
    """
    found: list[str] = []
    for message in messages:
        if isinstance(message, dict) and message.get('role') == 'assistant':
            found.extend(_message_tool_call_ids(message))
    return tuple(found)


def user_fingerprints(messages: list[dict[str, Any]]) -> tuple[str, ...]:
    """A fingerprint of every user message, in order.

    This identifies an agent that has issued no tool call yet.
    """
    return tuple(fingerprint(m.get('content')) for m in messages if isinstance(m, dict) and m.get('role') == 'user')


def has_tool_call(messages: list[dict[str, Any]], outputs: Any) -> bool:
    """Whether this completion is an agent turn rather than a one-shot utility call.

    An agent issues tool calls to do work; a title or a summary answers in prose and
    never does. True when the history already carries a tool_call id, or this response
    issues one.
    """
    if tool_call_ids(messages):
        return True
    if isinstance(outputs, dict):
        for choice in outputs.get('choices') or []:
            message = (isinstance(choice, dict) and (choice.get('message') or choice.get('delta'))) or {}
            if isinstance(message, dict) and message.get('tool_calls'):
                return True
    return False


@dataclass
class LangfuseAgentRecord:
    """One agent's identity together with the Langfuse state recorded against it.

    Several of these share a trace label: one for the worker's main loop and one per
    subagent. ``state`` holds the tracker's step counter and pending generation.
    """

    name: str
    state: dict[str, Any] = field(default_factory=dict)
    tool_call_ids: tuple[str, ...] = ()
    user_fingerprints: tuple[str, ...] = ()

    def extend_identity(self, messages: list[dict[str, Any]]) -> None:
        """Grow the two sequences that identify this agent to match ``messages``.

        Either is replaced only by a longer one.
        """
        ids = tool_call_ids(messages)
        if len(ids) > len(self.tool_call_ids):
            self.tool_call_ids = ids
        users = user_fingerprints(messages)
        if len(users) > len(self.user_fingerprints):
            self.user_fingerprints = users


def record_for(
    messages: list[dict[str, Any]],
    records: list[LangfuseAgentRecord],
) -> LangfuseAgentRecord | None:
    """The agent this call belongs to, or None when it opens a new one.

    The tool_call ids are tried first and the user messages second. A subagent's
    opening call gets None; two subagents a worker starts with the same request open
    identically, and the second is read as the first until one issues a tool call.
    """
    if not records:
        return None
    by_ids = _longest_match(tool_call_ids(messages), records, lambda record: record.tool_call_ids)
    if by_ids is not None:
        return by_ids
    return _longest_match(user_fingerprints(messages), records, lambda record: record.user_fingerprints)


def _longest_match(
    sequence: tuple[str, ...],
    records: list[LangfuseAgentRecord],
    identity: Any,
) -> LangfuseAgentRecord | None:
    """The record whose identity this call's sequence begins with, longest first.

    Several match when one identity is the beginning of another, which is one agent
    recorded at two points in its life.
    """
    if not sequence:
        return None
    matches = [record for record in records if _begins_with(sequence, identity(record))]
    if not matches:
        return None
    return max(matches, key=lambda record: len(identity(record)))


def _begins_with(sequence: tuple[str, ...], opening: tuple[str, ...]) -> bool:
    """Whether the sequence begins with a non-empty opening."""
    return bool(opening) and sequence[: len(opening)] == opening


__all__ = [
    'LangfuseAgentRecord',
    'has_tool_call',
    'record_for',
    'tool_call_ids',
    'user_fingerprints',
]
