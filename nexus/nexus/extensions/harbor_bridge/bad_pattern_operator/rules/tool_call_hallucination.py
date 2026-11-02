"""Rule: tool_call_hallucination."""

import re

from .base import Rule
from ..bpo_v20260924._core.messages import pair_tool_calls_with_responses
from ..bpo_v20260924._core.rule_detector import trajectory_view
from ..models import Finding


_DYNAMIC_REGISTRY = 'dynamic_registry_without_complete_snapshot'


def _token(value):
    return re.compile(rf'(?<![\w-]){re.escape(value)}(?![\w-])')


def _seen_before_call(view, call, pairs):
    """Whether the agent could have learned the called name before calling it.

    Sources are user/system messages and earlier tool results. For an
    ``mcp__<server>__<tool>`` name, the bare ``<tool>`` also counts when a
    user/system message names it, or when a result from a tool on the same
    server does, e.g. an ``activate_tools`` result that returns
    ``getSupplierList``.
    """
    name = call.raw_name or call.name
    name_token = _token(name)
    server, _, suffix = name.rpartition('__')
    suffix_token = _token(suffix) if name.startswith('mcp__') and server.count('__') >= 1 and suffix else None
    for message in view.messages:
        if message.turn_index >= call.turn_index:
            break
        if message.role not in ('user', 'system'):
            continue
        content = message.content or ''
        if name_token.search(content) or (suffix_token and suffix_token.search(content)):
            return True
    for prior, response in pairs:
        if response is None or response.turn_index >= call.turn_index:
            continue
        text = response.text or ''
        if name_token.search(text):
            return True
        if suffix_token and (prior.raw_name or prior.name).startswith(f'{server}__') and suffix_token.search(text):
            return True
    return False


def scan(ctx):
    out = []
    view = trajectory_view({'messages': ctx.messages, 'tools': ctx.tools})
    pairs = list(pair_tool_calls_with_responses(view.messages))
    for call, response in pairs:
        name = call.raw_name or call.name
        if not name:
            continue
        availability = view.tool_registry.availability_for(name, call.turn_index, response)
        registry_availability = view.tool_registry.availability_for(name, call.turn_index)
        runtime_unavailable = availability.status == 'unavailable' and availability.source.startswith(
            'runtime_response@'
        )
        classification = registry_availability if runtime_unavailable else availability
        is_undefined = classification.status == 'undefined'
        is_explicitly_inactive = (
            classification.status == 'unavailable' and classification.reason == 'tool_inactive_at_call'
        )
        # A dynamic registry makes absence undecidable from the tool list alone.
        # When the runtime also rejects the name and nothing shown to the agent
        # ever mentioned it, the agent invented the name.
        is_unseen_dynamic = (
            runtime_unavailable
            and classification.status == 'unjudgeable'
            and classification.reason == _DYNAMIC_REGISTRY
            and not _seen_before_call(view, call, pairs)
        )
        if is_undefined or is_explicitly_inactive or is_unseen_dynamic:
            if is_undefined:
                state = '未定义'
            elif is_explicitly_inactive:
                state = '已明确停用'
            else:
                state = '未声明，此前也没有任何消息或工具结果提到它，运行时报告不存在'
            out.append(
                Finding(
                    response.turn_index if response is not None else call.turn_index,
                    f'工具 {name!r} 在调用时{state}',
                    ['tool_calls'],
                    name,
                    tool_call_index=next(
                        (
                            e.tool_call_index
                            for e in ctx.calls
                            if e.message_index == call.turn_index and e.call_id == call.tool_call_id
                        ),
                        None,
                    ),
                    tool_call_id=call.tool_call_id,
                    origin_message_index=call.turn_index,
                    detail={
                        'availability': classification.status,
                        'availability_source': classification.source,
                        'availability_reason': classification.reason,
                        'registry_complete': classification.registry_complete,
                        'runtime_availability': availability.status,
                        'runtime_availability_source': availability.source,
                        'runtime_availability_reason': availability.reason,
                        'unseen_before_call': is_unseen_dynamic,
                    },
                )
            )
    return out


RULE = Rule('tool_call_hallucination', '调用未声明或不存在的工具', scan)
