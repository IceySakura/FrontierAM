"""Flag individual assistant thinking turns above a configurable character limit."""

from __future__ import annotations

import re

from typing import Any

from .base import Rule
from ..context import DetectionContext
from ..models import Finding


_LEADING_THINK = re.compile(r'\A\s*<think(?:[:_][0-9a-fA-F]+)?>')
_CLOSE_THINK = re.compile(r'</think(?:[:_][0-9a-fA-F]+)?>')


def thinking_span(message: dict[str, Any], content: str = '') -> tuple[str, list[str], int]:
    """Return thinking, its field path and original character offset."""
    provider = message.get('provider_specific_fields')
    provider = provider if isinstance(provider, dict) else {}
    candidates = (
        (message.get('reasoning_content'), ['reasoning_content']),
        (message.get('_reasoning_content'), ['_reasoning_content']),
        (provider.get('reasoning_content'), ['provider_specific_fields', 'reasoning_content']),
        (provider.get('reasoning'), ['provider_specific_fields', 'reasoning']),
        (message.get('thinking'), ['thinking']),
        (message.get('reasoning'), ['reasoning']),
    )
    for value, path in candidates:
        if isinstance(value, str) and value.strip():
            return value, path, 0
    opened = _LEADING_THINK.match(content)
    if opened is None:
        return '', [], 0
    closed = _CLOSE_THINK.search(content, opened.end())
    return content[opened.end() : closed.start() if closed else len(content)], ['content'], opened.end()


def thinking_text(message: dict[str, Any], content: str = '') -> tuple[str, list[str]]:
    thinking, path, _ = thinking_span(message, content)
    return thinking, path


def scan(ctx: DetectionContext) -> list[Finding]:
    findings = []
    limit = ctx.max_thinking_chars
    for index, message in ctx.assistant_messages():
        thinking, path = thinking_text(message, ctx.content(index))
        chars = len(thinking)
        if chars <= limit:
            continue
        source = 'content_think_block' if path == ['content'] else 'reasoning_content'
        findings.append(
            Finding(
                message_index=index,
                message=f'Assistant thinking has {chars:,} characters, exceeding {limit:,}.',
                path=path,
                excerpt=thinking[:300],
                severity='medium',
                detail={
                    'source': source,
                    'thinking_chars': chars,
                    'max_thinking_chars': limit,
                    'finish_reason': message.get('finish_reason') or '',
                },
                quotes=[
                    {
                        'kind': 'single_turn_thinking_too_long',
                        'turn_index': index,
                        'role': 'assistant',
                        'source': source,
                        'head_excerpt': thinking[:300],
                        'tail_excerpt': thinking[-300:],
                    }
                ],
            )
        )
    return findings


RULE = Rule('single_turn_thinking_too_long', 'Single-turn thinking exceeds character limit', scan)
