"""Adapter from the self-contained 2026-09-24 BPO detectors to Nexus rules."""

from __future__ import annotations

import re

from typing import Any

from .bpo_v20260924 import get_detector
from .bpo_v20260924._core.rule_detector import trajectory_view
from .models import Finding
from .utils import text


_TRUNCATION_REASONS = {
    'CONTEXT_LENGTH_EXCEEDED',
    'LENGTH',
    'MAX_TOKENS',
    'OUTPUT_LENGTH_EXCEEDED',
    'TRUNCATION',
}
_TOOL_PROTOCOL_DETECTORS = {
    'tool_call_protocol_leak_all_family',
    'tool_call_protocol_leak_hy',
}
_THINK_PROTOCOL_DETECTORS = {
    'think_protocol_leak_all_family',
    'think_protocol_leak_hy',
}
_FINAL_LANGUAGE_DETECTOR = 'zh_ask_foreign_reply'
_HY_TOOL_CALL_ONLY = re.compile(
    r'^\s*'
    r'(?:<tool_calls(?::[0-9A-Za-z_-]+)?>\s*)?'
    r'<tool_call(?::[0-9A-Za-z_-]+)?>[\s\S]*?'
    r'</tool_call(?::[0-9A-Za-z_-]+)>'
    r'(?:\s*</tool_calls(?::[0-9A-Za-z_-]+)>)?'
    r'\s*$',
    re.I,
)


def _last_assistant_index(messages: list[Any]) -> int | None:
    return next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if isinstance(messages[index], dict) and messages[index].get('role') == 'assistant'
        ),
        None,
    )


def _is_truncated_final_assistant_finding(ctx: Any, raw: dict[str, Any]) -> bool:
    if not ctx.ignore_incomplete_final_assistant:
        return False

    turn_index = raw.get('turn_index')
    final_assistant_index = _last_assistant_index(ctx.messages)
    if not isinstance(turn_index, int) or turn_index != final_assistant_index:
        return False

    message = ctx.messages[turn_index]
    finish_reason = str(message.get('finish_reason') or '').strip().upper().replace('-', '_')
    return ctx.termination_reason in _TRUNCATION_REASONS or finish_reason in _TRUNCATION_REASONS


def _suppress_truncated_final_protocol_finding(
    ctx: Any,
    detector_name: str,
    raw: dict[str, Any],
) -> bool:
    if detector_name not in _TOOL_PROTOCOL_DETECTORS | _THINK_PROTOCOL_DETECTORS:
        return False
    # A protocol token in the final generated turn of a truncated trajectory is
    # still useful monitoring evidence, but it is not reliable attribution:
    # generation may have stopped after a balanced token/block or inside any of
    # the three scanned channels before the parser could emit structured calls.
    # Earlier turns remain issues because truncation cannot explain them.
    return _is_truncated_final_assistant_finding(ctx, raw)


def _suppress_final_language_finding(
    ctx: Any,
    detector_name: str,
    raw: dict[str, Any],
) -> str | None:
    if detector_name != _FINAL_LANGUAGE_DETECTOR:
        return None

    turn_index = raw.get('turn_index')
    final_assistant_index = _last_assistant_index(ctx.messages)
    if not isinstance(turn_index, int) or turn_index != final_assistant_index:
        return None

    if _is_truncated_final_assistant_finding(ctx, raw):
        return 'truncated_final_assistant_not_user_facing'

    message = ctx.messages[turn_index]
    body = text(message)
    if message.get('tool_calls') and not body.strip():
        return 'tool_call_only_final_assistant'
    if _HY_TOOL_CALL_ONLY.fullmatch(body):
        return 'tool_call_only_final_assistant'
    return None


def _confidence_label(value: Any) -> str:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 'established'
    if confidence >= 0.95:
        return 'established'
    if confidence >= 0.7:
        return 'probable'
    return 'possible'


def scan_bpo_detector(ctx: Any, detector_name: str, *, exposed_rule_id: str | None = None) -> list[Finding]:
    """Run one vendored BPO detector over the current normalized trajectory."""
    result = get_detector(detector_name).detect_view(trajectory_view({'messages': ctx.messages, 'tools': ctx.tools}))
    findings: list[Finding] = []
    for raw in result.get('findings', []):
        if not isinstance(raw, dict):
            continue
        turn_index = raw.get('turn_index')
        message_index = turn_index if isinstance(turn_index, int) else 0
        quotes = raw.get('quotes') if isinstance(raw.get('quotes'), list) else []
        first_quote = next((quote for quote in quotes if isinstance(quote, dict)), {})
        source_detail = raw.get('detail') if isinstance(raw.get('detail'), dict) else {}
        suppressed_reason = _suppress_final_language_finding(ctx, detector_name, raw)
        if suppressed_reason is None and _suppress_truncated_final_protocol_finding(ctx, detector_name, raw):
            suppressed_reason = 'truncated_final_assistant_protocol_monitor_only'
        suppressed = suppressed_reason is not None
        detail = {
            'source_detector': detector_name,
            'source_rule_id': raw.get('rule_id'),
            'exposed_rule_id': exposed_rule_id or detector_name,
            **source_detail,
        }
        if suppressed:
            detail.update({
                'suppressed': True,
                'suppressed_reason': suppressed_reason,
                'termination_reason': ctx.termination_reason,
            })
        findings.append(
            Finding(
                message_index=message_index,
                message=str(raw.get('description') or detector_name),
                excerpt=first_quote.get('snippet') or first_quote.get('window_excerpt') or '',
                confidence=_confidence_label(raw.get('confidence')),
                is_issue=bool(raw.get('is_issue', True)) and not suppressed,
                severity=str(raw.get('severity')) if raw.get('severity') is not None else None,
                detail=detail,
                quotes=[quote for quote in quotes if isinstance(quote, dict)],
            )
        )
    return findings


__all__ = ['scan_bpo_detector']
