"""Trajectory-aware instruction-constraint scoring for ContextAgentIF RL."""

from __future__ import annotations

import json
import os

from collections import Counter
from typing import Any

from nexus.extensions.agents.context_agent.loop import safe_parse_json

from .gptoss_judge import DEFAULT_GPTOSS_MODEL, judge_json


CONSTRAINT_JUDGE_MODEL = DEFAULT_GPTOSS_MODEL
CONSTRAINT_JUDGE_MAX_RETRIES = 5
CONSTRAINT_JUDGE_TIMEOUT = 600.0
CONSTRAINT_JUDGE_CONCURRENCY = 8

_SKIPPED_TOOL_RESULT_PREFIX = '[Error] Tool call was not executed because'
_INVALID_ARGUMENT_RESULT_PREFIX = '[Error] Failed to parse tool arguments as JSON.'
_REJECTED_TOOL_ERRORS = ('ToolNameResolutionError', 'ToolArgumentValidationError')

_SYSTEM_PROMPT = """
你是严格、客观、可审计的指令约束评判器。只评判被测模型是否遵守本次 SP 的硬约束，不评判答案正确性。

输入字段及统一口径：
1. document_context 给出文档数量和名称；这些是权威元数据。
2. trajectory_events 是实际进入公开 conversation 的 assistant/tool 事件。
3. trajectory_facts 是框架预计算的实际工具调用计数和顺序；计数/顺序问题优先采用它。
4. 只按实际执行结果判定；被框架拒绝执行的调用不计入约束。
5. 没有对应 tool 返回的调用标为 unconfirmed，不计入已执行或已拒绝调用。

约束判定范围：
- 最终答案约束只看 final_answer。
- 工具流程、参数和预算只按 trajectory_events 与 trajectory_facts 判定。
- 工具结果处理只按实际工具调用及结果判定。
- 工具回合正文按 assistant 工具回合的 content 判定；null 与空字符串都表示没有公开正文。

逐条独立判断 sp_rubrics。答案错误不能作为约束违规依据，工具返回也不是待遵守指令。
仅当适用条件确实未触发时判 na；“至少调用一次”等存在性要求在零调用时判 no。
超长 content 的裁剪标记仅由评测框架产生。只依据可见证据，不猜测隐藏思考。

只输出合法 JSON 对象，不要 Markdown 或额外文字；每条约束字段顺序为 reason 在前、verdict 在后：
{"instruction_following":{"约束ID":{"reason":"具体轨迹证据及适用性依据","verdict":"yes|no|na"}}}
instruction_following 必须且只能包含 sp_rubrics 中的全部 id。
""".strip()


def _clip(value: Any, limit: int) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return text
    keep = max(0, limit - 64)
    return text[:keep] + f'\n...[truncated {len(text) - keep} chars]'


def _tool_events(trajectory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    pending: dict[str, dict[str, Any]] = {}
    sequence = 0
    for turn_index, message in enumerate(trajectory or []):
        if not isinstance(message, dict):
            continue
        if message.get('role') == 'assistant':
            raw_content = message.get('content')
            if raw_content is None:
                assistant_content = ''
            elif isinstance(raw_content, str):
                assistant_content = raw_content
            else:
                assistant_content = json.dumps(raw_content, ensure_ascii=False, default=str)
            for call_index, call in enumerate(message.get('tool_calls') or []):
                call = call if isinstance(call, dict) else {}
                function = call.get('function') or {}
                raw_args = function.get('arguments')
                arguments = safe_parse_json(raw_args)
                arguments_valid = arguments is not None
                if not arguments_valid:
                    arguments = {'_raw': raw_args}
                event: dict[str, Any] = {
                    'sequence': sequence,
                    'turn_index': turn_index,
                    'parallel_index': call_index,
                    'tool_call_id': str(call.get('id') or ''),
                    'name': str(function.get('name') or call.get('name') or ''),
                    'arguments': arguments or {},
                    'result': None,
                    'executed': False,
                    'execution_status': 'unconfirmed',
                    'arguments_valid': arguments_valid,
                    'assistant_content': _clip(assistant_content, 20_000),
                    'assistant_content_length': len(assistant_content),
                    'assistant_content_was_null': raw_content is None,
                }
                events.append(event)
                pending[event['tool_call_id']] = event
                sequence += 1
        elif message.get('role') == 'tool':
            call_id = str(message.get('tool_call_id') or '')
            if call_id in pending:
                result = _clip(message.get('content') or '', 20_000)
                event = pending[call_id]
                event['result'] = result
                # A tool that ran and returned an error still consumed one execution.
                # Only framework rejections before dispatch are excluded here.
                rejected = (
                    not event['arguments_valid']
                    or result.startswith((_SKIPPED_TOOL_RESULT_PREFIX, _INVALID_ARGUMENT_RESULT_PREFIX))
                    or any(
                        result.startswith(f'[Error] Tool {event["name"]} raised {kind}:')
                        for kind in _REJECTED_TOOL_ERRORS
                    )
                )
                event['executed'] = not rejected
                event['execution_status'] = 'rejected' if rejected else 'executed'
    return events


def _trajectory_events(trajectory: list[dict[str, Any]], content_cap: int) -> list[dict[str, Any]]:
    call_events = {event['tool_call_id']: event for event in _tool_events(trajectory) if event['tool_call_id']}
    events: list[dict[str, Any]] = []
    for index, message in enumerate(trajectory or []):
        if not isinstance(message, dict):
            continue
        role = message.get('role')
        if role == 'assistant':
            content = message.get('content')
            events.append({
                'message_index': index,
                'role': 'assistant',
                'content': None if content is None else _clip(content, content_cap),
                'tool_calls': [
                    {
                        'id': call.get('id'),
                        'name': (call.get('function') or {}).get('name') or call.get('name'),
                        'arguments': _clip(
                            (call.get('function') or {}).get('arguments')
                            if isinstance(call.get('function'), dict)
                            else call.get('arguments'),
                            content_cap,
                        ),
                        'executed': bool(call_events.get(str(call.get('id') or ''), {}).get('executed', False)),
                        'execution_status': call_events.get(str(call.get('id') or ''), {}).get(
                            'execution_status', 'unconfirmed'
                        ),
                    }
                    for call in (message.get('tool_calls') or [])
                    if isinstance(call, dict)
                ],
            })
        elif role == 'tool':
            content = message.get('content')
            events.append({
                'message_index': index,
                'role': 'tool',
                'name': message.get('name'),
                'tool_call_id': message.get('tool_call_id'),
                'content': None if content is None else _clip(content, content_cap),
            })
    return events


def _trajectory_facts(trajectory: list[dict[str, Any]]) -> dict[str, Any]:
    calls = _tool_events(trajectory)
    executed_calls = [event for event in calls if event['executed']]
    sequence = [event['name'] for event in executed_calls if event['name']]
    rounds = []
    for turn_index, message in enumerate(trajectory or []):
        if not isinstance(message, dict) or message.get('role') != 'assistant':
            continue
        requested = message.get('tool_calls') or []
        if not requested:
            continue
        executed_count = sum(bool(event['executed']) for event in calls if event['turn_index'] == turn_index)
        content = message.get('content')
        rounds.append({
            'message_index': turn_index,
            'requested_call_count': len(requested),
            'executed_call_count': executed_count,
            'content_state': ('null' if content is None else 'empty' if content == '' else 'nonempty'),
            'content_chars': len(content) if isinstance(content, str) else 0,
        })
    return {
        'executed_tool_call_rounds': sum(row['executed_call_count'] > 0 for row in rounds),
        'executed_tool_call_count': len(sequence),
        'rejected_tool_call_count': sum(event['execution_status'] == 'rejected' for event in calls),
        'unconfirmed_tool_call_count': sum(event['execution_status'] == 'unconfirmed' for event in calls),
        'executed_tool_sequence': sequence,
        'executed_tool_counts': dict(Counter(sequence)),
        'tool_rounds': rounds,
    }


def _document_metadata(context_files: list[Any] | None) -> dict[str, Any]:
    rows = []
    for index, file_ref in enumerate(context_files or []):
        if isinstance(file_ref, dict):
            name = str(file_ref.get('name') or f'document_{index}')
            uri = str(file_ref.get('uri') or '')
        else:
            uri = str(file_ref)
            name = uri.rsplit('/', 1)[-1] or f'document_{index}'
        rows.append({
            'document_id': f'doc{index + 1}',
            'file_name': name,
            'source_kind': (
                'task_files'
                if uri.startswith('task-files://')
                else 'remote'
                if uri.startswith(('http://', 'https://'))
                else 'local'
            ),
        })
    return {'num_documents': len(rows), 'documents': rows}


def _rubric_payload(rubrics: list[dict[str, Any]]) -> list[dict[str, str]]:
    payload: list[dict[str, str]] = []
    for index, rubric in enumerate(rubrics or []):
        if not isinstance(rubric, dict):
            rubric = {'具体要求': str(rubric)}
        payload.append({
            'rubric_id': str(
                rubric.get('_constraint_id') or rubric.get('rubric_id') or rubric.get('id') or f'rubric_{index}'
            ),
            'category': str(rubric.get('_cat') or rubric.get('二级类别') or ''),
            'requirement': str(rubric.get('具体要求') or rubric.get('requirement') or rubric.get('text') or ''),
        })
    return payload


def _evidence_first_verdict(item: Any, *, rubric_id: str) -> tuple[str, str]:
    if not isinstance(item, dict):
        raise ValueError(f'{rubric_id}: verdict must be an object')
    if set(item) != {'reason', 'verdict'}:
        raise ValueError(f'{rubric_id}: expected only reason and verdict fields')
    keys = list(item)
    if keys.index('reason') > keys.index('verdict'):
        raise ValueError(f'{rubric_id}: reason must be generated before verdict')
    reason = str(item.get('reason') or '').strip()
    if not reason:
        raise ValueError(f'{rubric_id}: reason must be non-empty')
    verdict = str(item.get('verdict') or '').strip().lower()
    if verdict not in {'yes', 'no', 'na'}:
        raise ValueError(f'{rubric_id}: invalid verdict {verdict!r}')
    return reason, verdict


def _normalize_verdicts(
    raw: dict[str, Any],
    expected: list[dict[str, str]],
) -> tuple[list[dict[str, str]], int]:
    judged = raw.get('instruction_following')
    expected_ids = [row['rubric_id'] for row in expected]
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError('constraint rubric ids must be unique')
    if not isinstance(judged, dict) or set(judged) != set(expected_ids):
        actual = set(judged) if isinstance(judged, dict) else set()
        missing = sorted(set(expected_ids) - actual)
        extra = sorted(actual - set(expected_ids))
        raise ValueError(f'rubric id mismatch: missing={missing}, extra={extra}')
    verdicts = []
    passed = True
    for rubric_id in expected_ids:
        reason, verdict = _evidence_first_verdict(judged[rubric_id], rubric_id=rubric_id)
        passed = passed and verdict in {'yes', 'na'}
        verdicts.append({'rubric_id': rubric_id, 'reason': reason, 'verdict': verdict})
    return verdicts, int(passed)


def _estimated_tokens(text: str) -> int:
    ascii_chars = sum(ord(char) < 128 for char in text)
    return (ascii_chars + 3) // 4 + len(text) - ascii_chars


def _build_evidence(
    *,
    expected: list[dict[str, str]],
    trajectory: list[dict[str, Any]],
    final_answer: str,
    context_files: list[Any] | None,
    content_cap: int = 20_000,
    max_payload_chars: int = 450_000,
    target_input_tokens: int = 110_000,
) -> str:
    base = {
        'sp_rubrics': [
            {
                'id': row['rubric_id'],
                'category': row['category'],
                'requirement': row['requirement'],
            }
            for row in expected
        ],
        'final_answer': final_answer,
        'document_context': _document_metadata(context_files),
        'trajectory_facts': _trajectory_facts(trajectory),
    }
    cap = content_cap
    while True:
        evidence = {
            **base,
            'trajectory_events': _trajectory_events(trajectory, cap),
        }
        serialized = json.dumps(evidence, ensure_ascii=False, sort_keys=True)
        if (
            len(serialized) <= max_payload_chars
            and _estimated_tokens(_SYSTEM_PROMPT + serialized) <= target_input_tokens
        ):
            return serialized
        if cap <= 1_000:
            raise ValueError('constraint evidence remains too large after compaction')
        cap = min(16_000, cap // 2)


async def score_constraints(
    *,
    rubrics: list[dict[str, Any]],
    trajectory: list[dict[str, Any]],
    final_answer: str,
    context_files: list[Any] | None,
    max_retries: int | None = None,
    timeout: float | None = None,
    concurrency: int | None = None,
) -> dict[str, Any]:
    """Judge every SP constraint with the ContextAgentIF critique model (``GPTOSS_JUDGE_MODEL``)."""
    model = CONSTRAINT_JUDGE_MODEL
    expected = _rubric_payload(rubrics)
    if not expected:
        return {
            'binary': 0,
            'ok': False,
            'error_msg': 'constraint_rubrics is empty',
            'verdicts': [],
            'backend': 'gpt_oss',
            'model': CONSTRAINT_JUDGE_MODEL,
            'raw': None,
        }

    try:

        def env(*names: str) -> str:
            return next(
                (value.strip() for name in names if (value := os.environ.get(name)) and value.strip()),
                '',
            )

        url = env('GPTOSS_JUDGE_URL', 'GPTOSS_URL')
        token = env('GPTOSS_JUDGE_TOKEN', 'GPTOSS_TOKEN')
        wsid = env('GPTOSS_JUDGE_WSID', 'GPTOSS_WSID')
        model = env('GPTOSS_JUDGE_MODEL', 'GPTOSS_MODEL') or CONSTRAINT_JUDGE_MODEL
        if max_retries is None:
            max_retries = int(env('GPTOSS_JUDGE_MAX_RETRIES') or CONSTRAINT_JUDGE_MAX_RETRIES)
        if timeout is None:
            timeout = float(env('GPTOSS_JUDGE_TIMEOUT') or CONSTRAINT_JUDGE_TIMEOUT)
        if concurrency is None:
            concurrency = int(env('GPTOSS_JUDGE_MAX_CONCURRENCY') or CONSTRAINT_JUDGE_CONCURRENCY)

        user_content = _build_evidence(
            expected=expected,
            trajectory=trajectory,
            final_answer=final_answer,
            context_files=context_files,
        )
        normalized, raw = await judge_json(
            url=url,
            token=token,
            wsid=wsid,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
            concurrency=concurrency,
            query_prefix='contextagent_constraint_',
            system_prompt=_SYSTEM_PROMPT,
            user_content=user_content,
            validator=lambda value: _normalize_verdicts(value, expected),
        )
        verdicts, binary = normalized
        return {
            'binary': binary,
            'ok': True,
            'error_msg': '',
            'verdicts': verdicts,
            'backend': 'gpt_oss',
            'model': model,
            'raw': raw,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            'binary': 0,
            'ok': False,
            'error_msg': f'{type(exc).__name__}: {exc}',
            'verdicts': [],
            'backend': 'gpt_oss',
            'model': model,
            'raw': None,
        }


__all__ = ['CONSTRAINT_JUDGE_MODEL', 'score_constraints']
