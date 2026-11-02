"""Rule: repetition_tool_call."""

import json
import re

from .base import Rule
from ..models import Finding
from ..utils import call_signature, text


_MAX_POLL_REPEATS = 3
_RETRIEVAL_STATUS = re.compile(r'<retrieval_status>\s*([^<]+?)\s*</retrieval_status>', re.I)
_POLLING_STATUS = re.compile(
    r'(?<![a-z0-9])(?:'
    r'queued|running|pending|in[_ -]?progress|processing|not[_ -]?ready|'
    r'rate[_ -]?limited|throttled|temporarily[_ -]?unavailable|preparation[_ -]?active'
    r')(?![a-z0-9])|'
    r'生成中|排队中|处理中|尚未完成|等待生成|任务运行中|请稍后|暂不可用',
    re.I,
)
_IMAGE_SUFFIX = re.compile(r'\.(?:avif|bmp|gif|jpe?g|png|svg|tiff?|webp)$', re.I)


def _retrieval_status(result):
    if not isinstance(result, dict):
        return ''
    direct = result.get('retrieval_status')
    if isinstance(direct, str):
        return direct.strip().casefold()
    match = _RETRIEVAL_STATUS.search(text(result))
    return match.group(1).strip().casefold() if match else ''


def _is_blocking_task_output_timeout(events):
    return any(
        re.sub(r'[_-]', '', event.name.casefold()) == 'taskoutput' and _retrieval_status(event.result) == 'timeout'
        for event in events
    )


def _is_polling_action(events):
    return any(
        _retrieval_status(event.result)
        in {
            'queued',
            'running',
            'pending',
            'in_progress',
            'processing',
            'not_ready',
            'rate_limited',
            'throttled',
            'temporarily_unavailable',
            'preparation_active',
        }
        or bool(_POLLING_STATUS.search(text(event.result)))
        for event in events
    )


def _is_image_read(events):
    if not events:
        return False
    for event in events:
        if re.sub(r'[_-]', '', event.name.casefold()) != 'read':
            return False
        arguments = event.arguments if isinstance(event.arguments, dict) else {}
        path = next(
            (arguments.get(key) for key in ('file_path', 'path', 'image_path') if isinstance(arguments.get(key), str)),
            '',
        )
        if not _IMAGE_SUFFIX.search(path.split('?', 1)[0].split('#', 1)[0]):
            return False
    return True


def _same_turn_repetitions(index, events):
    groups = {}
    for event in events:
        if not event.name or event.arguments_error or _is_blocking_task_output_timeout([event]):
            continue
        try:
            signature = json.dumps(
                [event.name, event.arguments], ensure_ascii=False, sort_keys=True, separators=(',', ':')
            )
        except (TypeError, ValueError):
            continue
        groups.setdefault(signature, []).append(event)
    out = []
    for repeated in groups.values():
        if len(repeated) < 2:
            continue
        image_read = _is_image_read(repeated)
        detail = {
            'check': 'same_turn_duplicate_request',
            'duplicate_count': len(repeated),
            'tool_call_indices': [event.tool_call_index for event in repeated],
        }
        if image_read:
            detail['monitor_reason'] = 'repeated_image_read'
        out.append(
            Finding(
                index,
                f'同轮等价工具请求重复 {len(repeated)} 次' + ('，按视觉复查单独监控' if image_read else ''),
                ['tool_calls'],
                [event.call for event in repeated],
                is_issue=not image_read,
                detail=detail,
            )
        )
    return out


def scan(ctx):
    out = []
    by_turn = {}
    for e in ctx.calls:
        by_turn.setdefault(e.message_index, []).append(e)
    previous = None
    polling_repeats = 0
    polling_reported = False
    for i, _ in ctx.assistant_messages():
        events = by_turn.get(i, [])
        out.extend(_same_turn_repetitions(i, events))
        signatures = []
        results = []
        valid = bool(events)
        for e in events:
            sig = call_signature(e.call)
            if sig is None or e.result is None:
                valid = False
                break
            signatures.append(sig)
            results.append(text(e.result))
        action = (signatures, results) if valid else None
        if action is None or _is_blocking_task_output_timeout(events):
            previous = None
            polling_repeats = 0
            polling_reported = False
            continue
        if previous == action and _is_polling_action(events):
            polling_repeats += 1
            if polling_repeats > _MAX_POLL_REPEATS and not polling_reported:
                out.append(
                    Finding(
                        i,
                        f'轮询状态连续重复超过 {_MAX_POLL_REPEATS} 次且中间没有等待',
                        ['tool_calls'],
                        [e.call for e in events],
                    )
                )
                polling_reported = True
        elif previous == action and _is_image_read(events):
            out.append(
                Finding(
                    i,
                    '重复读取同一图像，按视觉复查单独监控',
                    ['tool_calls'],
                    [e.call for e in events],
                    is_issue=False,
                    detail={'monitor_reason': 'repeated_image_read'},
                )
            )
        elif previous == action:
            out.append(Finding(i, '等价工具调用在结果无变化时被重复执行', ['tool_calls'], [e.call for e in events]))
        else:
            polling_repeats = 0
            polling_reported = False
        previous = action
    return out


RULE = Rule('repetition_tool_call', '重复工具调用且结果无变化', scan)
