"""Structural checks every example run must pass, whatever its case, plus the few a case asks for.

They catch protocol breaks only; whether the conversation did what the case asks is reviewed by
hand against the checklist in the case's markdown. A case-specific check (``ExampleCase.case_checks``)
only runs when that case is validated, so it never fails a case whose flow does not exercise it.
"""

from __future__ import annotations

import json
import re

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from hylive.prompts import FAILURE_TEMPLATE, NOTICE_TEMPLATE, notice_fact

from nexus.extensions.projects.hy_live.example.cases import CaseCheck


NOTICE_PREFIXES = ('[通知]', '[记录]', '[进展]', '[失败]')
"""Openings of the role=user messages the system writes into the cerebellum; none of them is the user."""
NOTICE_HEADS = {
    '[通知]': NOTICE_TEMPLATE.split('{fact}', 1)[0],
    '[失败]': FAILURE_TEMPLATE.split('{detail}', 1)[0],
}
"""The fixed head of each message the system can actually send.

``[记录]`` is defined but the example runner never sends it. ``[进展]`` is no longer a message;
one that appears fails ``notices_well_formed``.
"""
TRAJECTORY_FILES = ('cerebellum.jsonl', 'midbrain.jsonl', 'dialogue.jsonl')
TERMINAL_TASK_STATUSES = frozenset({'done', 'failed', 'cancelled'})
_TOOL_MARKUP = re.compile(r'[｜|]DSML[｜|]|</?tool_calls?\b')
_LEDGER_ID = re.compile(r'permission_id|input_id|artifact_id|\btask_\d+\b')
_ANSWER_HEAD = notice_fact('answer', {}).split('\n', 1)[0]
"""First line of an answer notice's fact: the backend answered a follow-up question."""
_UNANSWERED_LINE = notice_fact('answer', {}).rsplit('\n', 1)[-1]
"""Result line an answer notice carries when the question never got an answer."""
_COMPLETED_HEAD = notice_fact('completed', {}).split('\n', 1)[0]


def _as_mapping(task_output: Any) -> dict[str, Any]:
    if hasattr(task_output, 'model_dump'):
        task_output = task_output.model_dump(mode='json')
    return task_output if isinstance(task_output, dict) else {}


def _text(message: dict[str, Any]) -> str:
    content = message.get('content')
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return ' '.join(str(part.get('text', '')) for part in content if isinstance(part, dict))
    return ''


def _trajectory_messages(mapping: dict[str, Any], layer: str) -> list[dict[str, Any]]:
    record = mapping.get(f'{layer}_trajectory')
    messages = record.get('messages') if isinstance(record, dict) else None
    return [message for message in messages or [] if isinstance(message, dict)]


def _is_notice(message: dict[str, Any]) -> bool:
    return message.get('role') == 'user' and _text(message).startswith(NOTICE_PREFIXES)


def _tool_name(call: dict[str, Any]) -> str:
    function = call.get('function')
    return str(function.get('name') or '') if isinstance(function, dict) else ''


def _hand_off_results(messages: list[dict[str, Any]]) -> list[Any]:
    """The parsed result of every cerebellum tool call except ``web_search``; a missing result is None."""
    results = {
        str(message.get('tool_call_id') or ''): _text(message) for message in messages if message.get('role') == 'tool'
    }
    parsed: list[Any] = []
    for message in messages:
        if message.get('role') != 'assistant':
            continue
        for call in message.get('tool_calls') or []:
            if not isinstance(call, dict) or _tool_name(call) == 'web_search':
                continue
            try:
                parsed.append(json.loads(results.get(str(call.get('id') or ''), '')))
            except json.JSONDecodeError:
                parsed.append(None)
    return parsed


def _notice_well_formed(message: dict[str, Any]) -> bool:
    text = _text(message)
    head = next((head for prefix, head in NOTICE_HEADS.items() if text.startswith(prefix)), None)
    return head is not None and text.startswith(head) and bool(text[len(head) :].strip())


def _every_notice_answered(messages: list[dict[str, Any]]) -> bool:
    """The cerebellum's next reply after each notice says something to the user."""
    for index, message in enumerate(messages):
        if not _is_notice(message):
            continue
        reply = next((later for later in messages[index + 1 :] if later.get('role') == 'assistant'), None)
        if reply is None or not _text(reply).strip():
            return False
    return True


def _delivered_task_ids(mapping: dict[str, Any]) -> set[str]:
    return {
        str(event.get('task_id'))
        for event in mapping.get('delivery_events') or []
        if isinstance(event, dict) and event.get('kind') == 'task_delivery' and event.get('task_id')
    }


def _trajectory_files_parse(output_dir: Path) -> bool:
    for name in TRAJECTORY_FILES:
        path = output_dir / name
        if not path.is_file():
            return False
        try:
            records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        except json.JSONDecodeError:
            return False
        if not records or not all(isinstance(record, dict) for record in records):
            return False
    return True


def _tool_exchanges(messages: list[dict[str, Any]], name: str) -> list[tuple[dict[str, Any], Any]]:
    """Each call of tool ``name`` with its arguments and parsed result; a missing or non-JSON result is None."""
    results = {
        str(message.get('tool_call_id') or ''): _text(message) for message in messages if message.get('role') == 'tool'
    }
    exchanges: list[tuple[dict[str, Any], Any]] = []
    for message in messages:
        if message.get('role') != 'assistant':
            continue
        for call in message.get('tool_calls') or []:
            if not isinstance(call, dict) or _tool_name(call) != name:
                continue
            arguments = call['function'].get('arguments')
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            try:
                result = json.loads(results.get(str(call.get('id') or ''), ''))
            except json.JSONDecodeError:
                result = None
            exchanges.append((arguments if isinstance(arguments, dict) else {}, result))
    return exchanges


def _delivered_contents(mapping: dict[str, Any]) -> list[str]:
    contents: list[str] = []
    for event in mapping.get('delivery_events') or []:
        payload = event.get('payload') if isinstance(event, dict) else None
        contents.append(str(payload.get('content') or '') if isinstance(payload, dict) else '')
    return contents


def _question_answered_during_run(mapping: dict[str, Any], midbrain: list[dict[str, Any]]) -> bool:
    """The midbrain asked the backend, and its answer reached the cerebellum before any work was reported done.

    A follow-up that waits behind the running turn is answered only after that turn completes, so an
    answer delivered first means it ran beside the turn, on a WorkBuddy ``session/fork``.
    """
    asked = any(
        isinstance(result, dict) and result.get('status') == 'accepted'
        for _, result in _tool_exchanges(midbrain, 'ask_backend')
    )
    contents = _delivered_contents(mapping)
    answers = [index for index, content in enumerate(contents) if _ANSWER_HEAD in content]
    completions = [index for index, content in enumerate(contents) if _COMPLETED_HEAD in content]
    if not asked or not answers:
        return False
    first_answer = answers[0]
    answered = _UNANSWERED_LINE not in contents[first_answer]
    return answered and (not completions or first_answer < completions[0])


def _steer_replaced_running_run(mapping: dict[str, Any], midbrain: list[dict[str, Any]]) -> bool:
    """A ``TaskUpdate`` steered a turn that was still going, and the Task it steered ended done."""
    tasks = {str(task.get('task_id')): task for task in mapping.get('tasks') or [] if isinstance(task, dict)}
    for _, result in _tool_exchanges(midbrain, 'TaskUpdate'):
        if not isinstance(result, dict) or not result.get('steered'):
            continue
        task = tasks.get(str(result.get('task_id') or ''))
        if task is not None and task.get('status') == 'done':
            return True
    return False


def _acp_artifact_presented_and_opened(mapping: dict[str, Any], midbrain: list[dict[str, Any]]) -> bool:
    """Require ACP presentation of index.html and a successful open of the registered primary page.

    Directory scanning registers files but emits no presentation Event. Only presentFiles and
    openResultView produce the presentation marker, so finding a file alone cannot pass this check.
    """
    presented = any(
        event.get('kind') == 'display'
        and str(event.get('detail') or '').startswith('展示产物：')
        and 'index.html' in str(event['detail']).removeprefix('展示产物：').split('、')
        for task in mapping.get('tasks') or []
        if isinstance(task, dict) and task.get('status') == 'done'
        for event in task.get('activity') or []
        if isinstance(event, dict)
    )
    return presented and any(
        isinstance(result, dict)
        and result.get('status') == 'opened'
        and result.get('title') == 'index.html'
        and result.get('artifact_id') == arguments.get('artifact_id')
        and bool(result.get('url'))
        for arguments, result in _tool_exchanges(midbrain, 'present_files')
    )


def _successful_native_web_calls(mapping: dict[str, Any], name: str) -> set[str]:
    """Return queries or URLs whose native tool results contain source content rather than an error."""
    successful: set[str] = set()
    for task in mapping.get('tasks') or []:
        result = task.get('result') or {}
        messages = result.get('trajectory') or []
        outputs = {message.get('tool_call_id'): _text(message) for message in messages if message.get('role') == 'tool'}
        for message in messages:
            for call in message.get('tool_calls') or []:
                if _tool_name(call) != name:
                    continue
                arguments = call['function'].get('arguments') or {}
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                content = outputs.get(call.get('id'), '').strip()
                if len(content) < 100 or content.lower().startswith(('error', 'fetch failed', 'failed to fetch')):
                    continue
                if name == 'WebSearch' and 'http://' not in content and 'https://' not in content:
                    continue
                source = arguments.get('url' if name == 'WebFetch' else 'query')
                if source:
                    successful.add(str(source))
    return successful


_CASE_CHECKS: dict[CaseCheck, Callable[[dict[str, Any], list[dict[str, Any]]], bool]] = {
    'question_answered_during_run': _question_answered_during_run,
    'steer_replaced_running_run': _steer_replaced_running_run,
    'acp_artifact_presented_and_opened': _acp_artifact_presented_and_opened,
    'native_web_search_succeeded': lambda mapping, _: bool(_successful_native_web_calls(mapping, 'WebSearch')),
    'native_web_fetch_succeeded': lambda mapping, _: bool(_successful_native_web_calls(mapping, 'WebFetch')),
    'native_web_fetch_two_sources': lambda mapping, _: len(_successful_native_web_calls(mapping, 'WebFetch')) >= 2,
    'tasks_done': lambda mapping, _: (
        bool(mapping.get('tasks')) and all(task.get('status') == 'done' for task in mapping['tasks'])
    ),
}


def validate_run(
    task_output: Any, *, output_dir: Path | None = None, case_checks: Iterable[CaseCheck] = ()
) -> dict[str, Any]:
    """Check one example run's task output, and the trajectory files the CLI wrote when ``output_dir`` is given.

    Args:
        task_output: The run's ``task_output``, as a model or its JSON dump.
        output_dir: Directory the CLI wrote the trajectory files into; None skips that check.
        case_checks: Checks only the validated case's flow makes true (``ExampleCase.case_checks``).

    Returns:
        ``passed``, an ``error`` naming the failed checks, and every check's result.
    """
    mapping = _as_mapping(task_output)
    messages = _trajectory_messages(mapping, 'cerebellum')
    midbrain = _trajectory_messages(mapping, 'midbrain')
    notices = [message for message in messages if _is_notice(message)]
    assistant_texts = [_text(message) for message in messages if message.get('role') == 'assistant']
    hand_offs = _hand_off_results(messages)
    tasks = [task for task in mapping.get('tasks') or [] if isinstance(task, dict)]
    delivered = _delivered_task_ids(mapping)
    checks = {
        'trajectories_present': bool(messages) and bool(midbrain),
        'async_tools_submitted': bool(hand_offs) and all(result == {'status': 'submitted'} for result in hand_offs),
        'notices_well_formed': all(_notice_well_formed(message) for message in notices),
        'notices_hide_ledger_ids': not any(_LEDGER_ID.search(_text(message)) for message in notices),
        'no_tool_markup_leaked': not any(_TOOL_MARKUP.search(text) for text in assistant_texts),
        'every_notice_answered': _every_notice_answered(messages),
        'tasks_finished': bool(tasks) and all(task.get('status') in TERMINAL_TASK_STATUSES for task in tasks),
        'finished_tasks_delivered': all(
            str(task.get('task_id')) in delivered for task in tasks if task.get('status') in {'done', 'failed'}
        ),
    }
    if output_dir is not None:
        checks['trajectory_files_parse'] = _trajectory_files_parse(output_dir)
    for name in case_checks:
        checks[name] = _CASE_CHECKS[name](mapping, midbrain)
    failed = [name for name, passed in checks.items() if not passed]
    return {
        'passed': not failed,
        'error': f'Failed: {", ".join(failed)}' if failed else None,
        'checks': checks,
    }


__all__ = ['NOTICE_PREFIXES', 'validate_run']
