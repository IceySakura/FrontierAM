"""Standard T3 Agent result materialized for Harbor verifiers."""

from __future__ import annotations

import json

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, field_validator

from nexus.extensions.harbor_bridge.verifier_handoff import (
    upload_to_verifier,
    verifier_env_with_reserved_path,
    verifier_path_for,
    write_for_verifier,
)


T3_AGENT_RESULT_ENV = 'T3_AGENT_RESULT_PATH'
T3_AGENT_RESULT_FILENAME = 'result.json'
_MAX_FAILURE_MESSAGE_CHARS = 2000

AgentResultMode = Literal['agent', 'oracle', 'nop']
AgentResultStatus = Literal['success', 'timeout', 'error', 'incomplete', 'skipped']
AgentFailureKind = Literal['agent', 'infrastructure']
ClassifiedAgentStatus = Literal['success', 'timeout', 'error', 'incomplete']
AgentTerminationReason = Literal[
    'ENV_DONE',
    'MAX_STEPS',
    'TRUNCATION',
    'OUTPUT_LENGTH_EXCEEDED',
    'FORMAT_ERROR',
    'TASK_TIMEOUT',
    'ENV_ERROR',
    'INITIALIZATION_ERROR',
    'UNHANDLED_ERROR',
    # HyCode scoreable process exits; separate reasons so termination/{reason}/count counts apart.
    'SIGKILL',
    'SIGTERM',
    'EXIT_MISSING',
]


@dataclass(frozen=True)
class AgentExitRule:
    """One canonical interpretation of an Agent's raw exit status."""

    status: ClassifiedAgentStatus
    termination_reason: AgentTerminationReason
    inferred_failure_kind: AgentFailureKind | None = None
    include_failure_message: bool = True


_AGENT_EXIT_RULES: dict[str, AgentExitRule] = {
    # Normal completion.
    'success': AgentExitRule('success', 'ENV_DONE', include_failure_message=False),
    'finished': AgentExitRule('success', 'ENV_DONE', include_failure_message=False),
    'completed': AgentExitRule('success', 'ENV_DONE', include_failure_message=False),
    'submitted': AgentExitRule('success', 'ENV_DONE', include_failure_message=False),
    'submit_eval_success': AgentExitRule('success', 'ENV_DONE', include_failure_message=False),
    'env_done': AgentExitRule('success', 'ENV_DONE', include_failure_message=False),
    'done': AgentExitRule('success', 'ENV_DONE', include_failure_message=False),
    # Normal routing/incomplete exits that still represent a settled Agent loop.
    'handoff': AgentExitRule('incomplete', 'ENV_DONE', include_failure_message=False),
    'pass2pass_exhausted': AgentExitRule('incomplete', 'ENV_DONE'),
    'refused': AgentExitRule('incomplete', 'ENV_DONE'),
    'submit_eval_failed': AgentExitRule('incomplete', 'ENV_DONE'),
    # status must not be 'success': that short-circuits before this table is read.
    'sigkill': AgentExitRule('incomplete', 'SIGKILL'),
    'sigterm': AgentExitRule('incomplete', 'SIGTERM'),
    'exit_missing': AgentExitRule('incomplete', 'EXIT_MISSING'),
    # Budget and formatting exits.
    'context_length_exceeded': AgentExitRule('incomplete', 'TRUNCATION'),
    'response_truncated': AgentExitRule('incomplete', 'TRUNCATION'),
    'truncation': AgentExitRule('incomplete', 'TRUNCATION'),
    'output_length_exceeded': AgentExitRule('incomplete', 'OUTPUT_LENGTH_EXCEEDED'),
    'max_iterations': AgentExitRule('incomplete', 'MAX_STEPS'),
    'max_steps': AgentExitRule('incomplete', 'MAX_STEPS'),
    'awaiting_user': AgentExitRule('incomplete', 'UNHANDLED_ERROR'),
    'no_tool_calls_incomplete': AgentExitRule('incomplete', 'FORMAT_ERROR'),
    'format_error': AgentExitRule('error', 'FORMAT_ERROR'),
    'tool_error': AgentExitRule('error', 'FORMAT_ERROR'),
    # Agent/environment failures.
    'error': AgentExitRule('error', 'UNHANDLED_ERROR'),
    'failed': AgentExitRule('error', 'UNHANDLED_ERROR'),
    'failure': AgentExitRule('error', 'UNHANDLED_ERROR'),
    'unhandled_error': AgentExitRule('error', 'UNHANDLED_ERROR'),
    'env_error': AgentExitRule('error', 'ENV_ERROR'),
    # HyCode exit 1 without a result event: the CLI died before reporting, so it is not a clean finish.
    'general_error': AgentExitRule('error', 'ENV_ERROR'),
    'tool_execution_error': AgentExitRule('error', 'ENV_ERROR'),
    'runtime_disconnected': AgentExitRule('error', 'ENV_ERROR', 'infrastructure'),
    'inference_error': AgentExitRule('error', 'UNHANDLED_ERROR', 'infrastructure'),
    'llm_error': AgentExitRule('error', 'UNHANDLED_ERROR', 'infrastructure'),
    'provider_error': AgentExitRule('error', 'UNHANDLED_ERROR', 'infrastructure'),
    'runtime_error': AgentExitRule('error', 'UNHANDLED_ERROR', 'infrastructure'),
    # Timeouts. Structured initialization failure evidence still takes precedence.
    'timeout': AgentExitRule('timeout', 'TASK_TIMEOUT'),
    'inference_timeout': AgentExitRule('timeout', 'TASK_TIMEOUT'),
    'observation_timeout': AgentExitRule('timeout', 'TASK_TIMEOUT'),
}
_INITIALIZATION_FAILURE_STAGES = frozenset({
    'entrypoint',
    'execution_transport',
    'install',
    'pre_start',
    'proxy',
    'runtime_dependency',
    'runtime_protocol',
})


class T3AgentResult(BaseModel):
    """Small, reasoning-free result envelope exposed to verifier tests."""

    mode: AgentResultMode
    status: AgentResultStatus
    agent_exit_status: str | None = None
    final_answer: str | None = None
    failure_kind: AgentFailureKind | None = None
    failure_stage: str | None = None
    failure_message: str | None = None

    @field_validator('failure_message', mode='before')
    @classmethod
    def _normalize_failure_message(cls, value: Any) -> str | None:
        if value is None:
            return None
        message = str(value).strip()
        if not message:
            return None
        if len(message) <= _MAX_FAILURE_MESSAGE_CHARS:
            return message
        marker = f'[truncated, original length={len(message)}]\n'
        return marker + message[-(_MAX_FAILURE_MESSAGE_CHARS - len(marker)) :]


def agent_termination_reason(result: T3AgentResult) -> AgentTerminationReason:
    """Translate normalized Agent facts into the shared training vocabulary."""
    if result.failure_kind == 'infrastructure' and result.failure_stage in _INITIALIZATION_FAILURE_STAGES:
        return 'INITIALIZATION_ERROR'
    if result.status in {'success', 'skipped'}:
        return 'ENV_DONE'
    if result.status == 'timeout':
        return 'TASK_TIMEOUT'
    rule = _AGENT_EXIT_RULES.get((result.agent_exit_status or '').lower())
    return rule.termination_reason if rule is not None else 'UNHANDLED_ERROR'


def visible_text(content: Any) -> str | None:
    """Extract only user-visible text blocks; never fall back to reasoning.

    Shared with the trajectory envelope: both decide what a verifier may read, and the two
    must not drift into different notions of "visible".
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None

    from nexus.extensions.agents.base.utils import extract_text_from_content

    text = extract_text_from_content(content)
    return text if text else None


def _claude_code_events(output: Any) -> list[dict[str, Any]]:
    """Read Claude Code's object, list, or JSONL output as one list of events.

    Which of the three a run prints depends on its ``--output-format`` and ``--verbose``
    flags, and the readers below care about neither — so the shape is settled once, here,
    rather than by each of them.
    """
    if not isinstance(output, str):
        return []
    events: list[dict[str, Any]] = []
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        for line in output.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
            elif isinstance(event, list):
                events.extend(item for item in event if isinstance(item, dict))
    else:
        if isinstance(payload, dict):
            events.append(payload)
        elif isinstance(payload, list):
            events.extend(item for item in payload if isinstance(item, dict))
    return events


def _claude_code_result_text(output: Any) -> str | None:
    """Extract Claude Code's visible result from object, list, or JSONL output."""
    for event in reversed(_claude_code_events(output)):
        if event.get('type') != 'result':
            continue
        if event.get('is_error') is not False:
            return None
        result = event.get('result')
        return result if isinstance(result, str) else None
    return None


def _claude_code_messages(output: Any) -> list[dict[str, Any]]:
    """Extract the conversation Claude Code recorded, in the block layout it wrote.

    Only ``--verbose`` runs report their turns; without it the CLI prints the closing
    envelope alone and there is no conversation to find. ``system`` and ``result`` events
    describe the run rather than the conversation, so they are left out.

    The messages are not reshaped. A grader reads them as text, so the layout is part of
    what it was written against, and only the task knows which layout that is.
    """
    return [
        event['message']
        for event in _claude_code_events(output)
        if isinstance(event.get('message'), dict) and event['message'].get('role')
    ]


def extract_final_answer(agent_output: Any) -> str | None:
    """Return a normalized final answer without exposing reasoning or trajectories."""
    if agent_output is None:
        return None

    answer = getattr(agent_output, 'final_answer', None)
    if answer is not None:
        return visible_text(answer)

    trajectory = getattr(agent_output, 'main_trajectory', None)
    if trajectory:
        for message in reversed(trajectory):
            if not isinstance(message, dict) or message.get('role') != 'assistant':
                continue
            if 'content' not in message:
                continue
            answer = visible_text(message.get('content'))
            if answer is not None:
                return answer

    return _claude_code_result_text(getattr(agent_output, 'output', None))


def extract_reported_messages(agent_output: Any) -> list[dict[str, Any]]:
    """Return a conversation explicitly exposed by the Agent output."""
    if agent_output is not None:
        for field in ('main_trajectory', 'recorded_messages'):
            messages = getattr(agent_output, field, None)
            if messages:
                return [message for message in messages if isinstance(message, dict)]
    return []


def extract_messages(
    agent_output: Any,
    *,
    fallback_messages: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return the best available conversation for one Agent run.

    Agent-reported messages are authoritative. A caller may supply a ChatTracker fallback
    before Claude Code stdout is parsed — the last resort, and the one to delete once that
    Agent reports for itself.
    """
    reported_messages = extract_reported_messages(agent_output)
    if reported_messages:
        return reported_messages
    if fallback_messages:
        return [message for message in fallback_messages if isinstance(message, dict)]
    return _claude_code_messages(getattr(agent_output, 'output', None))


def extract_agent_exit_status(agent_output: Any) -> str | None:
    """Return the Agent's original exit status, accepting the legacy reason field."""
    if agent_output is None:
        return None
    raw_exit_status = getattr(agent_output, 'exit_status', None)
    if raw_exit_status is None:
        # Older Agent adapters such as HyCode exposed the canonical outcome only
        # as ``termination_reason``. Treat it as the exit status rather than
        # degrading MAX_STEPS / FORMAT_ERROR to UNHANDLED_ERROR.
        raw_exit_status = getattr(agent_output, 'termination_reason', None)
    if raw_exit_status is None:
        return None
    value = getattr(raw_exit_status, 'value', raw_exit_status)
    status = str(value).strip()
    return status or None


def extract_agent_failure_kind(agent_output: Any) -> AgentFailureKind | None:
    """Return explicit or unambiguous infrastructure failure classification."""
    value = getattr(agent_output, 'failure_kind', None) if agent_output is not None else None
    if value in ('agent', 'infrastructure'):
        return value
    rule = _AGENT_EXIT_RULES.get((extract_agent_exit_status(agent_output) or '').lower())
    return rule.inferred_failure_kind if rule is not None else None


def extract_agent_failure_stage(agent_output: Any) -> str | None:
    """Return the Adapter stage that produced a structured failure."""
    value = getattr(agent_output, 'failure_stage', None) if agent_output is not None else None
    if value is None:
        return None
    stage = str(value).strip()
    return stage or None


def classify_agent_output(agent_output: Any) -> tuple[ClassifiedAgentStatus, str | None]:
    """Classify common Nexus Agent output contracts without treating every return as success."""
    if agent_output is None:
        return 'incomplete', 'agent returned no output'

    success = getattr(agent_output, 'success', None)
    exit_code = getattr(agent_output, 'exit_code', None)
    stderr = str(getattr(agent_output, 'stderr', '') or '').strip()
    process_failed = success is False or (isinstance(exit_code, int) and exit_code != 0)

    agent_exit_status = extract_agent_exit_status(agent_output)
    if agent_exit_status is not None:
        exit_status = agent_exit_status.lower()
        rule = _AGENT_EXIT_RULES.get(exit_status)
        if rule is None and 'timeout' in exit_status:
            return 'timeout', f'agent exit status: {exit_status}'
        if rule is not None and (rule.status != 'success' or not process_failed):
            if not rule.include_failure_message:
                return rule.status, None
            if rule.status == 'error':
                return rule.status, stderr or f'agent exit status: {exit_status}'
            return rule.status, f'agent exit status: {exit_status}'
        if rule is None and not process_failed:
            return 'incomplete', f'unknown agent exit status: {agent_exit_status}'

    if process_failed:
        detail = stderr or f'agent process exited with code {exit_code}'
        return 'error', detail
    if success is True or exit_code == 0:
        return 'success', None
    return 'incomplete', 'agent returned no explicit completion status'


def build_t3_agent_result(
    *,
    mode: AgentResultMode,
    agent_output: Any = None,
    status: AgentResultStatus | None = None,
    failure_kind: AgentFailureKind | None = None,
    failure_stage: str | None = None,
    failure_message: str | None = None,
) -> T3AgentResult:
    """Build the verifier-facing envelope, classifying a real Agent output exactly once."""
    classified_message: str | None = None
    if mode == 'agent':
        if status is None:
            status, classified_message = classify_agent_output(agent_output)
        elif agent_output is not None:
            classified_status, classified_message = classify_agent_output(agent_output)
            if status != classified_status:
                raise ValueError(f'Explicit status {status!r} conflicts with Agent output status {classified_status!r}')
    elif status is None:
        raise ValueError(f'status is required for mode={mode!r}')

    if status is None:  # pragma: no cover - all branches above resolve or reject it
        raise AssertionError('Agent result status was not resolved')

    if agent_output is not None and mode != 'agent':
        raise ValueError(f'agent_output is not valid for mode={mode!r}')

    if failure_message is None:
        failure_message = classified_message

    effective_failure_kind = failure_kind or extract_agent_failure_kind(agent_output)
    return T3AgentResult(
        mode=mode,
        status=status,
        agent_exit_status=extract_agent_exit_status(agent_output),
        final_answer=extract_final_answer(agent_output),
        failure_kind=effective_failure_kind,
        failure_stage=failure_stage or extract_agent_failure_stage(agent_output),
        failure_message=failure_message,
    )


def write_t3_agent_result(trial_paths: Any, result: T3AgentResult) -> Path:
    """Persist the result once in the Host-private Harbor trial directory."""
    return write_for_verifier(trial_paths, T3_AGENT_RESULT_FILENAME, result)


def t3_agent_result_container_path(environment_os: Any) -> str:
    """Return the standard result path for the target Harbor environment OS."""
    return verifier_path_for(environment_os, T3_AGENT_RESULT_FILENAME)


async def upload_t3_agent_result(
    runtime: Any,
    host_path: Path,
    *,
    environment_os: Any,
) -> str:
    """Upload the Host result directly to a Shared or Separate verifier runtime."""
    return await upload_to_verifier(
        runtime,
        host_path,
        environment_os=environment_os,
        filename=T3_AGENT_RESULT_FILENAME,
        label='T3 AgentResult',
    )


def verifier_env_with_t3_agent_result(
    verifier_env: dict[str, str],
    *,
    environment_os: Any,
) -> dict[str, str]:
    """Inject the reserved result path without allowing task-level overrides."""
    return verifier_env_with_reserved_path(
        verifier_env,
        environment_os=environment_os,
        filename=T3_AGENT_RESULT_FILENAME,
        env_name=T3_AGENT_RESULT_ENV,
    )


__all__ = [
    'AgentTerminationReason',
    'T3_AGENT_RESULT_ENV',
    'T3_AGENT_RESULT_FILENAME',
    'T3AgentResult',
    'agent_termination_reason',
    'build_t3_agent_result',
    'classify_agent_output',
    'extract_agent_failure_kind',
    'extract_agent_failure_stage',
    'extract_agent_exit_status',
    'extract_final_answer',
    'extract_messages',
    'extract_reported_messages',
    't3_agent_result_container_path',
    'upload_t3_agent_result',
    'verifier_env_with_t3_agent_result',
    'visible_text',
    'write_t3_agent_result',
]
