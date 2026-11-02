"""Translate Hyness execution-result v1 at the Agent boundary.

The engine owns execution facts; Nexus retains its result JSON, 64/65/66 control
codes and inference-side sampling. Never infer transport usage or termination
from error prose. This adapter deliberately does not add a second record channel.
"""

from __future__ import annotations

import json
import math

from copy import deepcopy
from typing import Any

from nexus.extensions.agents.hycode.agent import (
    EXIT_CODE_CONTEXT_LENGTH,
    EXIT_CODE_MAX_TURNS,
    EXIT_CODE_TOOL_ERROR,
    SUBTYPE_CONTEXT_LENGTH,
    SUBTYPE_MAX_TURNS,
    SUBTYPE_TOOL_ERROR,
)


_REASONS = frozenset({
    'completed',
    'output_truncated',
    'host_budget_exhausted',
    'blocking_limit',
    'image_error',
    'aborted_streaming',
    'aborted_tools',
    'hook_stopped',
    'stop_hook_prevented',
    'unknown',
    'generation_limit',
    'max_turns',
    'tool_error',
    'model_error',
})
_COUNTERS = (
    ('input_tokens',),
    ('output_tokens',),
    ('cache_creation_input_tokens',),
    ('cache_read_input_tokens',),
    ('server_tool_use', 'web_search_requests'),
    ('server_tool_use', 'web_fetch_requests'),
    ('cache_creation', 'ephemeral_1h_input_tokens'),
    ('cache_creation', 'ephemeral_5m_input_tokens'),
)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    obj: dict[str, Any] = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError('Duplicate JSON property in Hyness output')
        obj[key] = value
    return obj


def _invalid_constant(value: str) -> Any:
    raise ValueError(f'Invalid JSON number in Hyness output: {value}')


def _validate_json(value: Any, depth: int = 0) -> None:
    if depth > 64:
        raise ValueError('Hyness result nesting exceeds protocol limit')
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_json(key, depth + 1)
            _validate_json(item, depth + 1)
    elif isinstance(value, list):
        for item in value:
            _validate_json(item, depth + 1)
    elif isinstance(value, str) and any(0xD800 <= ord(c) <= 0xDFFF for c in value):
        raise ValueError('Invalid Unicode in Hyness output')
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Non-finite number in Hyness output')


def _counts(usage: Any) -> list[int]:
    if not isinstance(usage, dict):
        raise ValueError('Missing Hyness usage counters')
    counters = []
    for path in _COUNTERS:
        value = usage
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        if type(value) is not int or not 0 <= value <= 2**53 - 1:
            raise ValueError('Invalid Hyness usage counter')
        counters.append(value)
    return counters


def decode_execution(event: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if (
        event.get('protocol') != 'hyness.execution-result'
        or type(event.get('schemaVersion')) is not int
        or event['schemaVersion'] != 1
        or event.get('type') != 'execution-result'
        or not isinstance(event.get('resultId'), str)
        or not event['resultId']
        or 'runId' not in event
        or (event['runId'] is not None and (not isinstance(event['runId'], str) or not event['runId']))
    ):
        raise ValueError('Unsupported Hyness execution-result protocol')
    terminal, result = event.get('termination'), event.get('result')
    if (
        not isinstance(terminal, dict)
        or not isinstance(terminal.get('reason'), str)
        or terminal.get('reason') not in _REASONS
        or not isinstance(result, dict)
        or result.get('type') != 'result'
        or type(result.get('is_error')) is not bool
        or not isinstance(result.get('subtype'), str)
        or not result['subtype']
    ):
        raise ValueError('Invalid Hyness execution result')
    for key in ('used', 'limit', 'turnCount'):
        if key in terminal and (type(terminal[key]) is not int or terminal[key] < 0):
            raise ValueError('Invalid Hyness termination counter')
    for key in ('toolCallId', 'code'):
        if key in terminal and (not isinstance(terminal[key], str) or not terminal[key]):
            raise ValueError('Invalid Hyness termination detail')
    controls = {
        'generation_limit',
        'max_turns',
        'tool_error',
        'output_truncated',
        'host_budget_exhausted',
        'model_error',
    }
    if not result['is_error'] and (result['subtype'] != 'success' or terminal['reason'] in controls | {'unknown'}):
        raise ValueError('Hyness termination contradicts successful result')
    return terminal, result


def encode_nexus_result(event: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    terminal, raw = decode_execution(event)
    breakdown = event.get('usageBreakdown')
    if not isinstance(breakdown, dict):
        raise ValueError('Hyness usageBreakdown is required; use a matching rl-runtime engine build')
    total = _counts(raw.get('usage'))
    streamed = _counts(breakdown.get('streaming'))
    fallback = _counts(breakdown.get('nonStreaming'))
    if any(a != b + c for a, b, c in zip(total, streamed, fallback, strict=True)):
        raise ValueError('Hyness usage breakdown does not match total usage')
    result = {**raw, 'usage': deepcopy(breakdown['streaming'])}
    reason = terminal['reason']
    if reason == 'aborted_tools' and raw['is_error'] and raw['subtype'] == SUBTYPE_MAX_TURNS:
        return EXIT_CODE_MAX_TURNS, result
    if reason in {'max_turns', 'generation_limit'} or (
        reason == 'unknown' and raw['is_error'] and raw['subtype'] == SUBTYPE_MAX_TURNS
    ):
        code, subtype = EXIT_CODE_MAX_TURNS, SUBTYPE_MAX_TURNS
        message = f'Reached maximum number of turns ({terminal.get("limit", raw.get("num_turns"))})'
    elif reason == 'tool_error':
        code, subtype = EXIT_CODE_TOOL_ERROR, SUBTYPE_TOOL_ERROR
        message = f'Stopped on tool error (tool_use_id={terminal.get("toolCallId", "")})'
    elif reason in {'output_truncated', 'host_budget_exhausted'} or (
        reason == 'model_error' and terminal.get('code') == 'max_tokens'
    ):
        code, subtype = EXIT_CODE_CONTEXT_LENGTH, SUBTYPE_CONTEXT_LENGTH
        message = (
            'Context length exceeded: the host refused the call on token budget'
            if reason == 'host_budget_exhausted'
            else 'Output token limit: the model response was cut off at max_tokens'
        )
    else:
        return int(raw['is_error']), result
    return code, {**result, 'subtype': subtype, 'is_error': True, 'errors': [message]}


def convert_execution_output(stdout: str, process_exit_code: int) -> tuple[int, str, dict[str, Any]]:
    """Preserve stream events, convert every envelope, and validate process completion."""
    if type(process_exit_code) is not int or process_exit_code not in (0, 1):
        raise ValueError('Hyness process did not exit with a supported completion status')
    lines: list[str] = []
    ids: set[str] = set()
    last: dict[str, Any] | None = None
    mapped_code = 1
    for line in stdout.splitlines():
        if not line.strip():
            continue
        event = json.loads(line, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
        _validate_json(event)
        if not isinstance(event, dict):
            raise ValueError('Invalid Hyness output event')
        if event.get('type') == 'execution-result' or 'protocol' in event:
            mapped_code, result = encode_nexus_result(event)
            if event['resultId'] in ids:
                raise ValueError('Duplicate Hyness execution result ID')
            ids.add(event['resultId'])
            last = event
            lines.append(json.dumps(result, ensure_ascii=False, separators=(',', ':')))
        elif event.get('type') == 'result':
            raise ValueError('Expected execution-result; legacy SDK output is not accepted')
        else:
            lines.append(line)
    if last is None:
        raise ValueError('Hyness exited without an execution result')
    terminal, result = decode_execution(last)
    cancelled = terminal['reason'] in {'aborted_streaming', 'aborted_tools'}
    if process_exit_code != int(result['is_error']) and not (cancelled and process_exit_code == 0):
        raise ValueError('Hyness process exit contradicts its last result')
    return mapped_code, '\n'.join(lines) + ('\n' if stdout.endswith('\n') else ''), terminal
