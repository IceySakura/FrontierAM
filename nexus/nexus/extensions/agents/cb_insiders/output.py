"""Interpret the IDE CLI's own print result without treating exit zero as completion."""

from __future__ import annotations

import json

from typing import Any


def parse_output(stdout: str, exit_code: int, *, timed_out: bool = False) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            events.append(value)
    result = next((event for event in reversed(events) if event.get('type') == 'result'), None)
    text_parts: list[str] = []
    for event in events:
        if event.get('type') != 'assistant' or not isinstance(event.get('message'), dict):
            continue
        content = event['message'].get('content')
        if isinstance(content, str):
            text_parts.append(content)
        elif isinstance(content, list):
            text_parts.extend(
                block['text']
                for block in content
                if isinstance(block, dict) and block.get('type') == 'text' and isinstance(block.get('text'), str)
            )
    assistant = ''.join(text_parts)
    payload: dict[str, Any] = {
        'status': 'error',
        'success': False,
        'assistant_text': assistant,
        'session_id': (result or {}).get('session_id'),
        'result': result,
        'permission_denials': (result or {}).get('permission_denials') or [],
    }
    if result and isinstance(result.get('result'), str):
        payload['assistant_text'] = result['result']
    is_success = result is not None and result.get('is_error') is False and result.get('subtype') == 'success'
    if timed_out:
        payload.update(
            status='timeout', failure_kind='agent', failure_stage='process', detail='CLI exceeded its timeout'
        )
    elif exit_code != 0:
        payload.update(failure_kind='agent', failure_stage='process', detail=f'CLI exited with code {exit_code}')
    elif not result:
        payload.update(
            failure_kind='infrastructure', failure_stage='runtime_protocol', detail='CLI did not emit a final result'
        )
    elif not is_success:
        errors = result.get('errors')
        if isinstance(errors, list):
            detail = '\n'.join(str(error) for error in errors)
        else:
            detail = '' if errors is None else str(errors)
        detail = detail or f'CLI result subtype: {result.get("subtype")}'
        payload.update(failure_kind='agent', failure_stage='process', detail=detail)
    else:
        payload.update(status='completed', success=True, failure_kind=None, failure_stage=None, detail=None)
    return payload
