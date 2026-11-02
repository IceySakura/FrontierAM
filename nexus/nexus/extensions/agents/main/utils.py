"""Utility helpers for main_agent."""

from __future__ import annotations

import logging

from collections.abc import Callable, Sequence
from typing import Any


logger = logging.getLogger(__name__)

TOOL_RESPONSE_TRUNCATION_MARKER = '\n\n... [tool result truncated; middle omitted] ...\n\n'
_MIN_RETAINED_TOOL_RESPONSE_TOKENS = 2


def maybe_truncate(content: str, truncate_after: int | None) -> str:
    """Truncate *content* keeping head and tail with an ellipsis marker in the middle.

    If *truncate_after* is ``None`` or the content is short enough, return it
    unchanged.
    """
    if truncate_after is None or len(content) <= truncate_after:
        return content
    half = truncate_after // 2
    return content[:half] + '\n\n... [truncated] ...\n\n' + content[-half:]


def minimum_tool_response_token_budget(token_encode_fn: Callable[[str], Sequence[int]]) -> int:
    """Return the minimum budget that fits the marker plus one head and tail token."""
    return len(token_encode_fn(TOOL_RESPONSE_TRUNCATION_MARKER)) + _MIN_RETAINED_TOOL_RESPONSE_TOKENS


def truncate_tool_response_by_tokens(
    content: str,
    max_tokens: int,
    token_encode_fn: Callable[[str], Sequence[int]],
    token_decode_fn: Callable[[Sequence[int]], str],
) -> tuple[str, int, int]:
    """Keep the head and tail of a tool result within a model-token budget."""
    if max_tokens <= 0:
        raise ValueError(f'max_tokens must be positive, got {max_tokens}')

    marker_token_ids = list(token_encode_fn(TOOL_RESPONSE_TRUNCATION_MARKER))
    minimum_budget = len(marker_token_ids) + _MIN_RETAINED_TOOL_RESPONSE_TOKENS
    if max_tokens < minimum_budget:
        raise ValueError(
            f'max_tokens must be at least {minimum_budget} to preserve head/tail around the truncation marker, '
            f'got {max_tokens}'
        )

    original_token_ids = list(token_encode_fn(content))
    original_token_count = len(original_token_ids)
    if original_token_count <= max_tokens:
        return content, original_token_count, original_token_count

    def _candidate(kept_tokens: int) -> str:
        head_tokens = (kept_tokens + 1) // 2
        tail_tokens = kept_tokens // 2
        token_ids = original_token_ids[:head_tokens] + marker_token_ids
        if tail_tokens:
            token_ids += original_token_ids[-tail_tokens:]
        return token_decode_fn(token_ids)

    kept_tokens = max_tokens - len(marker_token_ids)
    candidate = _candidate(kept_tokens)
    candidate_token_count = len(token_encode_fn(candidate))
    # Decoding and re-encoding concatenated token slices is normally stable,
    # but trim retained tokens if tokenizer normalization changes the count.
    while candidate_token_count > max_tokens and kept_tokens > _MIN_RETAINED_TOOL_RESPONSE_TOKENS:
        kept_tokens = max(
            _MIN_RETAINED_TOOL_RESPONSE_TOKENS,
            kept_tokens - (candidate_token_count - max_tokens),
        )
        candidate = _candidate(kept_tokens)
        candidate_token_count = len(token_encode_fn(candidate))

    if candidate_token_count > max_tokens:
        raise ValueError(
            'Unable to preserve head/tail and the truncation marker within max_tokens after tokenizer normalization'
        )
    return candidate, original_token_count, candidate_token_count


def extract_tool_output_text(output: Any) -> str:
    """Extract a plain-text representation from ``ToolResponse.output``.

    ``output`` is typically ``list[dict]`` where each item has a ``type`` key.
    Text parts are concatenated; non-text parts are represented by a
    placeholder.
    """
    if isinstance(output, str):
        return output
    if not isinstance(output, list):
        return str(output)

    parts: list[str] = []
    for item in output:
        if isinstance(item, dict):
            if item.get('type') == 'text':
                parts.append(item.get('text', ''))
            elif item.get('type') == 'image_url':
                parts.append('[image]')
            else:
                parts.append(str(item))
        else:
            parts.append(str(item))
    return '\n'.join(parts)


def extract_tool_output_content(output: Any) -> str | list[dict[str, Any]]:
    """Extract history content from ``ToolResponse.output``.

    Returns plain text for text-only outputs and preserves structured content
    parts when a tool returns multimodal data such as ``image_url``.
    """
    if isinstance(output, str):
        return output
    if not isinstance(output, list):
        return str(output)
    if not output:
        return ''

    if all(isinstance(item, dict) and item.get('type') == 'text' for item in output):
        return extract_tool_output_text(output)

    if any(isinstance(item, dict) and item.get('type') != 'text' for item in output):
        return output

    return extract_tool_output_text(output)


# ---------------------------------------------------------------------------
# Debug logging for conversation messages
# ---------------------------------------------------------------------------

_LOG_LIMIT = 200


def fmt_msg(msg: dict[str, Any]) -> str:
    """Format a conversation message compactly for debug logging.

    Preserves message structure (role, tool calls, tool_call_id) but
    truncates long content and replaces images with ``[image]``.
    """
    role = msg.get('role', '?')

    # Content
    content = msg.get('content')
    if content is None:
        content_str = ''
    elif isinstance(content, str):
        content_str = content[:_LOG_LIMIT] + ('...' if len(content) > _LOG_LIMIT else '')
    elif isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                ptype = part.get('type', '?')
                if ptype == 'text':
                    text = part.get('text', '')
                    parts.append(text[:_LOG_LIMIT] + ('...' if len(text) > _LOG_LIMIT else ''))
                elif ptype in ('image_url', 'image'):
                    parts.append('[image]')
                else:
                    parts.append(f'[{ptype}]')
            else:
                parts.append(str(part)[:80])
        content_str = ' | '.join(parts)
    else:
        content_str = str(content)[:_LOG_LIMIT]

    # Reasoning content (assistant message)
    reasoning = msg.get('reasoning_content')
    rc_str = ''
    if reasoning:
        if isinstance(reasoning, str):
            reasoning_str = reasoning[:_LOG_LIMIT] + ('...' if len(reasoning) > _LOG_LIMIT else '')
        else:
            reasoning_str = str(reasoning)[:_LOG_LIMIT]
        rc_str = f' reasoning={reasoning_str}'

    # Tool calls (assistant message)
    tool_calls = msg.get('tool_calls')
    tc_str = ''
    if tool_calls:
        tc_parts: list[str] = []
        for tc in tool_calls:
            if isinstance(tc, dict):
                fn = tc.get('function', {})
                name = fn.get('name', '?')
                args = fn.get('arguments', '')
                if len(args) > 120:
                    args = args[:120] + '...'
                tc_parts.append(f'{name}({args})')
            else:
                tc_parts.append(str(tc)[:80])
        tc_str = ' tools=[' + ', '.join(tc_parts) + ']'

    # Tool response id
    tc_id = msg.get('tool_call_id', '')
    id_str = f' id={tc_id}' if tc_id else ''

    return f'[{role}]{id_str}{tc_str} {content_str}{rc_str}'


def log_messages(msgs: list[dict[str, Any]], prefix: str = '') -> None:
    """Log a list of messages at DEBUG level."""
    for msg in msgs:
        logger.debug('%s%s', prefix, fmt_msg(msg))
