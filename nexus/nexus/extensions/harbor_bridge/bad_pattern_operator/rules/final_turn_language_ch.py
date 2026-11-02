"""Rule: final_turn_language_ch for the Chinese Code-MT agent.

The rule scores only the final non-empty assistant content. Fenced and inline
code are removed before script units are counted. Each CJK character is one
unit, while a continuous Latin word is one unit.
"""

from __future__ import annotations

import re

from typing import Any

from .base import Rule
from ..bpo_v20260924.zh_ask_foreign_reply.detect_zh_ask_foreign_reply import strip_code, token_counts
from ..models import Finding


PENALTY_BY_MINIMUM_CHINESE_RATIO: tuple[tuple[float, float], ...] = (
    (0.9, 0.0),
    (0.8, 0.1),
    (0.7, 0.2),
    (0.6, 0.3),
    (0.5, 0.4),
    (0.0, 0.5),
)
TRUNCATION_REASONS = {
    'CONTEXT_LENGTH_EXCEEDED',
    'LENGTH',
    'MAX_TOKENS',
    'OUTPUT_LENGTH_EXCEEDED',
    'TRUNCATION',
}
ABNORMAL_TERMINATION_REASONS = {
    'FORMAT_ERROR',
    'MAX_STEPS',
    'TASK_TIMEOUT',
}
HY_TOOL_CALL_BLOCK = re.compile(
    r'<tool_calls(?::[0-9A-Za-z_-]+)?>[\s\S]*?</tool_calls(?::[0-9A-Za-z_-]+)>'
    r'|<tool_call(?::[0-9A-Za-z_-]+)?>[\s\S]*?</tool_call(?::[0-9A-Za-z_-]+)>',
    re.IGNORECASE,
)


def _compute_penalty(chinese_ratio: float) -> float:
    for minimum_ratio, penalty in PENALTY_BY_MINIMUM_CHINESE_RATIO:
        if chinese_ratio >= minimum_ratio:
            return penalty
    raise AssertionError(f'chinese_ratio must be non-negative, got {chinese_ratio}')


def _select_final_assistant_content(ctx: Any) -> tuple[int, str] | None:
    for message_index in range(len(ctx.messages) - 1, -1, -1):
        message = ctx.messages[message_index]
        if not isinstance(message, dict) or message.get('role') != 'assistant':
            continue
        content = ctx.content(message_index)
        if content.strip():
            return message_index, content
    return None


def _strip_nonprose_content(content: str) -> str:
    return strip_code(HY_TOOL_CALL_BLOCK.sub('', content))


def scan(ctx: Any) -> list[Finding]:
    selected = _select_final_assistant_content(ctx)
    if selected is None:
        return []

    message_index, content = selected
    cleaned_content = _strip_nonprose_content(content)
    script_units = token_counts(cleaned_content)
    total_units = sum(script_units.values())
    if total_units == 0:
        return []

    chinese_units = int(script_units.get('cjk', 0))
    chinese_ratio = chinese_units / total_units
    penalty = _compute_penalty(chinese_ratio)
    message = ctx.messages[message_index]
    finish_reason = str(message.get('finish_reason') or '').strip().upper().replace('-', '_')
    is_truncated = ctx.termination_reason in TRUNCATION_REASONS or finish_reason in TRUNCATION_REASONS
    is_abnormal_termination = ctx.termination_reason in ABNORMAL_TERMINATION_REASONS
    is_incomplete = ctx.ignore_incomplete_final_assistant and (is_truncated or is_abnormal_termination)
    is_issue = penalty > 0 and not is_incomplete
    detail = {
        'chinese_ratio': round(chinese_ratio, 6),
        'chinese_units': chinese_units,
        'total_language_units': total_units,
        'script_units': script_units,
        'suggested_penalty': penalty,
        'excluded_content': ['fenced_code', 'inline_code', 'tool_call_arguments'],
    }
    if is_incomplete:
        detail.update({
            'suppressed': True,
            'suppressed_reason': (
                'abnormal_termination_no_user_facing_final'
                if is_abnormal_termination
                else 'truncated_final_assistant_not_user_facing'
            ),
            'termination_reason': ctx.termination_reason,
        })
    return [
        Finding(
            message_index=message_index,
            message=(f'final assistant Chinese ratio is {chinese_ratio:.1%}; suggested penalty is {penalty:.1f}'),
            excerpt=cleaned_content,
            is_issue=is_issue,
            severity='medium' if is_issue else None,
            detail=detail,
        )
    ]


RULE = Rule('final_turn_language_ch', '最终回复中文占比', scan)
