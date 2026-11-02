"""Standard T3 Agent trajectory materialized for Harbor verifiers.

Sits beside ``T3AgentResult`` and follows the same rules: one small envelope, written once
on the Host and uploaded to a reserved container path, carrying only what a verifier may
legitimately read.

Why the channel exists. Most rubrics grade deliverables and need nothing beyond the
workspace. A minority grade *process* — whether the Agent consulted a source before
answering, whether it refused an unsafe instruction — and no artifact in the workspace can
answer that. A task that does not declare a process rubric should keep judging files.

What it carries, and what it deliberately does not. The envelope hands over the recorded
conversation and stops there: rendering, truncation and which turns a judge should see are
the verifying task's decisions, and they live in its ``tests/``. Baking a format in here
would mean every task family that wants a different one has to come back and edit the
Bridge, which is exactly the coupling the T3 contract avoids.

Reasoning is the one thing this module does decide, because it is a property of the
contract rather than of any task: a judge that reads the model's private thinking scores
the story rather than the work, and reward then follows whichever trajectory argues best.
Tool calls stay, since "did it look this up" is a fact about the run, not about narration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from nexus.extensions.harbor_bridge.agent_result import extract_messages, visible_text
from nexus.extensions.harbor_bridge.verifier_handoff import (
    upload_to_verifier,
    verifier_env_with_reserved_path,
    verifier_path_for,
    write_for_verifier,
)
from nexus.trace.processors import span_processor_registry
from nexus.trace.processors.chat_tracker import ChatTrackerProcessor


T3_AGENT_TRAJECTORY_SCHEMA_VERSION = 't3-agent-trajectory-v1'
T3_AGENT_TRAJECTORY_ENV = 'T3_AGENT_TRAJECTORY_PATH'
T3_AGENT_TRAJECTORY_FILENAME = 'trajectory.json'
HARBOR_TRAJECTORY_PROCESSOR_KEY = 'harbor_agent_trajectory'
HARBOR_TRAJECTORY_PROCESSOR_NAME = 'harbor_agent_trajectory_tracker'

# Every shape a backend reports thinking in. Stripping is deliberately broad: a field that
# slips through reaches a judge, while one stripped in error costs a verifier nothing.
_STRIPPED_REASONING_FIELDS = (
    'reasoning_content',
    'reasoning_items',
    'thinking_blocks',
    'reasoning',
    'thinking',
    'raw_response',
)

# The same thing again for backends that report thinking as a content block rather than a
# field beside it. Anthropic-format Agents take this branch exclusively, so checking only
# the fields above would leave their reasoning in a trajectory documented as free of it.
_STRIPPED_REASONING_BLOCK_TYPES = ('thinking', 'redacted_thinking', 'reasoning')
_STRIPPED_PROVIDER_REASONING_FIELDS = frozenset((*_STRIPPED_REASONING_FIELDS, 'reasoningContentBlocks'))


def _strip_provider_specific_reasoning(value: Any) -> Any:
    """Remove Responses reasoning duplicated inside provider-specific message fields."""
    if isinstance(value, list):
        return [_strip_provider_specific_reasoning(item) for item in value]
    if not isinstance(value, dict):
        return value
    sanitized = {
        key: _strip_provider_specific_reasoning(item)
        for key, item in value.items()
        if key not in _STRIPPED_PROVIDER_REASONING_FIELDS
    }
    anthropic_content = sanitized.get('anthropic_content')
    if isinstance(anthropic_content, list):
        visible_anthropic_content = [
            item
            for item in anthropic_content
            if not (isinstance(item, dict) and item.get('type') in _STRIPPED_REASONING_BLOCK_TYPES)
        ]
        if visible_anthropic_content:
            sanitized['anthropic_content'] = visible_anthropic_content
        else:
            sanitized.pop('anthropic_content', None)
    response_items = sanitized.get('responses_output_items')
    if isinstance(response_items, list):
        sanitized['responses_output_items'] = [
            item
            for item in response_items
            if not (isinstance(item, dict) and item.get('type') in _STRIPPED_REASONING_BLOCK_TYPES)
        ]
    return sanitized


@span_processor_registry.register(HARBOR_TRAJECTORY_PROCESSOR_NAME)
class HarborAgentTrajectoryTracker(ChatTrackerProcessor):
    """Dedicated ChatTracker used only as a verifier-trajectory fallback."""

    def get_result(self) -> dict[str, Any]:
        """Keep the raw fallback trace out of RunOutput and Batch exports."""
        return {}


class T3AgentTrajectory(BaseModel):
    """The conversation the Agent recorded, with reasoning removed."""

    schema_version: Literal['t3-agent-trajectory-v1'] = T3_AGENT_TRAJECTORY_SCHEMA_VERSION
    mode: str
    status: str
    messages: list[dict[str, Any]] = []


def trajectory_processor_config() -> dict[str, dict[str, Any]]:
    """Return the reserved processor config for tasks requesting Agent trajectory."""
    return {
        HARBOR_TRAJECTORY_PROCESSOR_KEY: {
            'name': HARBOR_TRAJECTORY_PROCESSOR_NAME,
            'args': {'tools_mode': 'branch_last'},
        }
    }


def _last_visible_assistant_text(messages: list[dict[str, Any]]) -> str | None:
    for message in reversed(messages):
        if message.get('role') == 'assistant':
            text = visible_text(message.get('content'))
            if text is not None:
                return text
    return None


def chat_tracker_fallback_record(
    snapshot: Any,
    *,
    final_answer: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Return one unambiguous ChatTracker conversation, its tools, and candidate count."""
    if not isinstance(snapshot, dict):
        return [], [], 0

    candidates: list[tuple[list[dict[str, Any]], list[dict[str, Any]], str | None]] = []
    for trajectory in snapshot.get('trajectories') or []:
        raw_messages = trajectory.get('messages') if isinstance(trajectory, dict) else trajectory
        if not isinstance(raw_messages, list):
            continue
        messages = [message for message in raw_messages if isinstance(message, dict)]
        if messages:
            raw_role = trajectory.get('role') if isinstance(trajectory, dict) else None
            role = str(raw_role).strip() if raw_role is not None else None
            raw_tools = trajectory.get('tools') if isinstance(trajectory, dict) else None
            if raw_tools is None:
                raw_tools = snapshot.get('tools')
            tools = [tool for tool in raw_tools or [] if isinstance(tool, dict)]
            candidates.append((messages, tools, role))

    if len(candidates) == 1:
        messages, tools, _role = candidates[0]
        return messages, tools, 1
    main_candidates = [(messages, tools) for messages, tools, role in candidates if role and role.lower() == 'main']
    if len(main_candidates) == 1:
        messages, tools = main_candidates[0]
        return messages, tools, len(candidates)
    if final_answer is not None:
        answer_candidates = [
            (messages, tools)
            for messages, tools, _role in candidates
            if _last_visible_assistant_text(messages) == final_answer
        ]
        if len(answer_candidates) == 1:
            messages, tools = answer_candidates[0]
            return messages, tools, len(candidates)
    return [], [], len(candidates)


def chat_tracker_fallback_messages(
    snapshot: Any,
    *,
    final_answer: str | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Backward-compatible messages-only view of :func:`chat_tracker_fallback_record`."""
    messages, _tools, candidate_count = chat_tracker_fallback_record(snapshot, final_answer=final_answer)
    return messages, candidate_count


def strip_reasoning(trajectory: Any) -> list[dict[str, Any]]:
    """Return the messages with reasoning removed, everything else verbatim.

    Verbatim matters: a grader may pair tool calls with their results by id, or read fields
    this module has no opinion about, and rewriting them would change what it sees.

    A message left with nothing but the reasoning it carried is dropped rather than kept
    empty. Graders that render the trajectory into a bounded window pay for every message
    they are handed, so an emptied one spends that budget saying nothing.
    """
    if not isinstance(trajectory, list):
        return []
    messages: list[dict[str, Any]] = []
    for message in trajectory:
        if not isinstance(message, dict):
            continue
        kept = {key: value for key, value in message.items() if key not in _STRIPPED_REASONING_FIELDS}
        if 'provider_specific_fields' in kept:
            kept['provider_specific_fields'] = _strip_provider_specific_reasoning(kept['provider_specific_fields'])
        content = kept.get('content')
        if isinstance(content, list):
            kept['content'] = [
                block
                for block in content
                if not (isinstance(block, dict) and block.get('type') in _STRIPPED_REASONING_BLOCK_TYPES)
            ]
            if content and not kept['content']:
                continue
        messages.append(kept)
    return messages


def build_t3_agent_trajectory(
    *,
    mode: str,
    status: str,
    agent_output: Any = None,
    fallback_messages: list[dict[str, Any]] | None = None,
) -> T3AgentTrajectory:
    """Build the verifier-facing trajectory envelope from one Nexus Agent result."""
    return T3AgentTrajectory(
        mode=mode,
        status=status,
        messages=strip_reasoning(extract_messages(agent_output, fallback_messages=fallback_messages)),
    )


def write_t3_agent_trajectory(trial_paths: Any, trajectory: T3AgentTrajectory) -> Path:
    """Persist the trajectory once in the Host-private Harbor trial directory."""
    return write_for_verifier(trial_paths, T3_AGENT_TRAJECTORY_FILENAME, trajectory)


def t3_agent_trajectory_container_path(environment_os: Any) -> str:
    """Return the standard trajectory path for the target Harbor environment OS."""
    return verifier_path_for(environment_os, T3_AGENT_TRAJECTORY_FILENAME)


async def upload_t3_agent_trajectory(
    runtime: Any,
    host_path: Path,
    *,
    environment_os: Any,
) -> str:
    """Upload the Host trajectory directly to a Shared or Separate verifier runtime."""
    return await upload_to_verifier(
        runtime,
        host_path,
        environment_os=environment_os,
        filename=T3_AGENT_TRAJECTORY_FILENAME,
        label='T3 AgentTrajectory',
    )


def verifier_env_with_t3_agent_trajectory(
    verifier_env: dict[str, str],
    *,
    environment_os: Any,
) -> dict[str, str]:
    """Inject the reserved trajectory path without allowing task-level overrides."""
    return verifier_env_with_reserved_path(
        verifier_env,
        environment_os=environment_os,
        filename=T3_AGENT_TRAJECTORY_FILENAME,
        env_name=T3_AGENT_TRAJECTORY_ENV,
    )


__all__ = [
    'HARBOR_TRAJECTORY_PROCESSOR_KEY',
    'HARBOR_TRAJECTORY_PROCESSOR_NAME',
    'T3_AGENT_TRAJECTORY_ENV',
    'T3_AGENT_TRAJECTORY_FILENAME',
    'T3_AGENT_TRAJECTORY_SCHEMA_VERSION',
    'HarborAgentTrajectoryTracker',
    'T3AgentTrajectory',
    'build_t3_agent_trajectory',
    'chat_tracker_fallback_record',
    'chat_tracker_fallback_messages',
    'strip_reasoning',
    't3_agent_trajectory_container_path',
    'upload_t3_agent_trajectory',
    'verifier_env_with_t3_agent_trajectory',
    'trajectory_processor_config',
    'write_t3_agent_trajectory',
]
