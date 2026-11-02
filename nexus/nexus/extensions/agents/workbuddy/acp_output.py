"""Parse the JSON Lines that ``wb --json`` writes to stdout.

Each line is one of:
  - ``{sessionId, update}``            an ACP ``session/update`` notification
  - ``{sessionId, method, params}``    a conversation extension method
  - ``{sessionId, channel, data}``     a daemon channel event (docs pane)
  - ``{stopReason, _meta}``            one per turn, in order, last of the turn

Only the turn result carries the outcome, so a zero exit code alone does not mean
the agent succeeded: wb exits 0 whenever the turn completed, including when it
completed by refusing or by hitting its loop limit.
"""

from __future__ import annotations

import json
import logging

from typing import Any, Literal


logger = logging.getLogger(__name__)

WbStatus = Literal['completed', 'partial', 'refused', 'cancelled', 'timeout', 'truncation', 'error']

# resolveTaskOutcome() in the 5.7.0 bundle emits exactly these.
_OUTCOME_STATUS: dict[str, WbStatus] = {
    'SUCCESS': 'completed',
    'PARTIAL_SUCCESS': 'partial',
    'CANCELLED': 'cancelled',
    'REFUSED_OR_BLOCKED': 'refused',
    'PERMISSION_DENIED': 'refused',
    'FAILED_MODEL_REQUEST': 'error',
    'FAILED_TOOL_EXECUTION': 'error',
    'FAILED_AGENT_LOOP': 'error',
    'UNKNOWN': 'error',
}

# A model-request failure is the proxy/provider breaking, not the agent reasoning badly.
_INFRASTRUCTURE_OUTCOMES = frozenset({'FAILED_MODEL_REQUEST'})

_STATUS_STAGE: dict[WbStatus, str] = {
    'partial': 'agent_partial',
    'refused': 'agent_refused',
    'cancelled': 'agent_cancelled',
    'timeout': 'process',
    'truncation': 'context_overflow',
    'error': 'process',
}


def parse_wb_json_lines(stdout: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split ``wb --json`` stdout into session updates and per-turn results.

    Non-JSON lines are skipped: stdout also carries whatever a tool printed.
    """
    updates: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for raw in stdout.splitlines():
        line = raw.strip()
        if not line or line[0] != '{':
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        if 'stopReason' in event:
            results.append(event)
        else:
            updates.append(event)
    return updates, results


def final_turn_result(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the last turn result; one-shot runs have exactly one."""
    return results[-1] if results else None


def assistant_text(updates: list[dict[str, Any]]) -> str:
    """Concatenate the agent's visible reply, skipping thoughts and tool traffic."""
    chunks: list[str] = []
    for event in updates:
        update = event.get('update')
        if not isinstance(update, dict) or update.get('sessionUpdate') != 'agent_message_chunk':
            continue
        content = update.get('content')
        if not isinstance(content, dict) or content.get('type') != 'text':
            continue
        text = content.get('text')
        if isinstance(text, str):
            chunks.append(text)
    return ''.join(chunks).strip()


def _meta_value(meta: Any, *suffixes: str) -> Any:
    """Read a ``_meta`` value, tolerating every spelling wb uses for it.

    readPromptResponseMeta() reads the namespaced, nested and bare forms, and 5.7.0
    serializes the outcome as a bare ``taskOutcome`` rather than ``codebuddy.ai/outcome``
    (confirmed against a real sandbox run), so each name is tried in all three shapes.
    """
    if not isinstance(meta, dict):
        return None
    nested = meta.get('codebuddy.ai')
    for suffix in suffixes:
        value = meta.get(f'codebuddy.ai/{suffix}')
        if value is not None:
            return value
        if isinstance(nested, dict) and nested.get(suffix) is not None:
            return nested[suffix]
        if meta.get(suffix) is not None:
            return meta[suffix]
    return None


def task_outcome(result: dict[str, Any] | None) -> str | None:
    value = _meta_value((result or {}).get('_meta'), 'taskOutcome', 'outcome')
    return value if isinstance(value, str) and value else None


def transcript_path(result: dict[str, Any] | None) -> str | None:
    value = _meta_value((result or {}).get('_meta'), 'transcriptPath')
    return value if isinstance(value, str) and value else None


def conversation_id(result: dict[str, Any] | None) -> str | None:
    value = _meta_value((result or {}).get('_meta'), 'conversationId')
    return value if isinstance(value, str) and value else None


def terminal_error(result: dict[str, Any] | None) -> str | None:
    value = _meta_value((result or {}).get('_meta'), 'terminalError')
    if value is None or value == '':
        return None
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def classify_wb_result(
    result: dict[str, Any] | None,
    exit_code: int,
    *,
    json_output: bool,
) -> tuple[WbStatus, str | None]:
    """Map a turn result onto a status plus a human-readable detail."""
    if exit_code == 130:
        return 'cancelled', 'wb was interrupted by SIGINT'
    if exit_code == 143:
        return 'timeout', 'wb was terminated by SIGTERM'
    if exit_code == 2:
        return 'error', 'wb rejected the command line (usage error)'
    if exit_code < 0:
        return 'timeout', 'wb did not exit before the timeout'

    if not json_output:
        # Without --json there is no turn result to inspect; the exit code is all we have.
        return ('completed', None) if exit_code == 0 else ('error', f'wb exited with code {exit_code}')

    if result is None:
        return 'error', 'wb emitted no turn result line'

    detail_error = terminal_error(result)
    outcome = task_outcome(result)
    stop_reason = result.get('stopReason')
    stop_reason = stop_reason.strip().lower() if isinstance(stop_reason, str) else ''

    if stop_reason == 'max_tokens':
        return 'truncation', detail_error or 'wb stopped at the output token limit'
    if stop_reason == 'max_turn_requests':
        return 'error', detail_error or 'wb hit its per-turn request limit'

    if outcome:
        status = _OUTCOME_STATUS.get(outcome.strip().upper())
        if status is not None:
            detail = detail_error or (None if status == 'completed' else f'wb reported taskOutcome {outcome}')
            return status, detail
        logger.warning('Unrecognized wb taskOutcome %r; treating as an error', outcome)
        return 'error', detail_error or f'wb reported unknown taskOutcome {outcome}'

    if stop_reason == 'refusal':
        return 'refused', detail_error or 'wb refused the request'
    if stop_reason == 'cancelled':
        return 'cancelled', detail_error or 'wb cancelled the turn'
    if detail_error:
        return 'error', detail_error
    if exit_code != 0:
        return 'error', f'wb exited with code {exit_code}'
    return 'completed', None


def failure_fields(
    status: WbStatus, outcome: str | None
) -> tuple[Literal['agent', 'infrastructure'] | None, str | None]:
    """Map a status onto ``InstalledAgentOutput.failure_kind`` / ``failure_stage``."""
    if status == 'completed':
        return None, None
    normalized = (outcome or '').strip().upper()
    kind: Literal['agent', 'infrastructure'] = 'infrastructure' if normalized in _INFRASTRUCTURE_OUTCOMES else 'agent'
    if normalized in _INFRASTRUCTURE_OUTCOMES:
        return kind, 'model_request'
    return kind, _STATUS_STAGE.get(status, 'process')


__all__ = [
    'WbStatus',
    'assistant_text',
    'classify_wb_result',
    'conversation_id',
    'failure_fields',
    'final_turn_result',
    'parse_wb_json_lines',
    'task_outcome',
    'terminal_error',
    'transcript_path',
]
