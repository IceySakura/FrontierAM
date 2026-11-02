"""Interpret the Kimi Code 2.x prompt-mode JSONL contract."""

from __future__ import annotations

import collections
import json

from typing import Any


def parse_output(stdout: str, exit_code: int, *, timed_out: bool = False) -> dict[str, Any]:
    session_id = None
    version = None
    assistant = []
    tools: collections.Counter[str] = collections.Counter()
    errors = []
    malformed = 0
    tool_results = 0
    retries = 0
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except ValueError:
            malformed += 1
            continue
        if not isinstance(event, dict):
            malformed += 1
            continue
        kind = event.get('type')
        if kind == 'system.version':
            version = event.get('version')
        elif kind == 'session.resume_hint':
            session_id = event.get('session_id')
        elif kind == 'error':
            errors.append(str(event.get('message') or event.get('code') or event))
        elif kind == 'turn.step.retrying':
            retries += 1
        if event.get('role') == 'assistant':
            content = event.get('content')
            if isinstance(content, str) and content:
                assistant.append(content)
            for tool in event.get('tool_calls') or []:
                if isinstance(tool, dict) and isinstance(tool.get('function'), dict):
                    tools[str(tool['function'].get('name') or 'unknown')] += 1
        elif event.get('role') == 'tool':
            tool_results += 1
    # The upstream CLI emits the resume marker after the prompt has completed.
    # Exit zero alone cannot establish that a complete transcript was captured.
    success = exit_code == 0 and not timed_out and bool(session_id) and not errors and malformed == 0
    if timed_out:
        status, kind, stage = 'timeout', 'agent', 'process'
        detail = 'Kimi Code execution timed out'
    elif exit_code != 0 or errors:
        status, kind, stage = 'error', 'agent', 'process'
        detail = '\n'.join(errors) or f'Kimi Code exited with code {exit_code}'
    elif not success:
        status, kind, stage = 'error', 'infrastructure', 'runtime_protocol'
        detail = 'Kimi Code did not emit a complete, valid prompt-mode transcript'
    else:
        status, kind, stage, detail = 'finished', None, None, None
    return {
        'success': success,
        'exit_status': status,
        'failure_kind': kind,
        'failure_stage': stage,
        'detail': detail,
        'assistant_text': '\n'.join(assistant),
        'session_id': session_id,
        'trace': {
            'version': version,
            'tool_calls': dict(tools),
            'tool_results': tool_results,
            'retries': retries,
            'malformed_records': malformed,
            'errors': errors,
        },
    }
