"""Output quality monitors for terminal/terminus tasks.

Pure-function detectors that inspect agent messages and write detection
results into extra_fields. These only FLAG — they never change the score.

Controlled by ``enable_output_monitoring`` in task_config (default False).
"""

from __future__ import annotations

import json
import os
import re

from typing import Any

from nexus.utils.reasoning_repetition import detect_let_me_loop


def _strip_code_spans(text: str) -> str:
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    return re.sub(r'`[^`]*`', '', text)


# ---------------------------------------------------------------------------
# Codeswitch detection
# ---------------------------------------------------------------------------


def detect_codeswitch(messages: list[dict] | None) -> dict[str, Any]:
    """Detect natural-language code switching (Chinese+English mix) in assistant messages."""
    text_parts: list[str] = []
    for message in messages or []:
        if not isinstance(message, dict) or message.get('role') != 'assistant':
            continue
        content = message.get('content')
        if isinstance(content, str) and content.strip():
            text_parts.append(content)

    assistant_text = _strip_code_spans('\n'.join(text_parts))
    cjk_chars = sum(1 for ch in assistant_text if '一' <= ch <= '鿿')
    latin_chars = sum(1 for ch in assistant_text if ('a' <= ch <= 'z') or ('A' <= ch <= 'Z'))
    cjk_min = int(os.getenv('NEXUS_CODESWITCH_CJK_MIN_CHARS', '20'))
    latin_min = int(os.getenv('NEXUS_CODESWITCH_LATIN_MIN_CHARS', '100'))
    is_codeswitch = cjk_chars >= cjk_min and latin_chars >= latin_min
    return {
        'is_codeswitch': is_codeswitch,
        'assistant_cjk_chars': cjk_chars,
        'assistant_latin_chars': latin_chars,
        'cjk_min_chars': cjk_min,
        'latin_min_chars': latin_min,
    }


# ---------------------------------------------------------------------------
# Output format bad pattern detection (F6_7 / F7_2 / F7_3)
# ---------------------------------------------------------------------------

_TOOL_CALL_LITERAL_TOKEN_RE = re.compile(
    r'<\s*/?\s*(?:tool_calls?|function_call|invoke|tool_sep|arg_key|arg_value)(?::[A-Za-z0-9_-]+)?\s*>',
    re.IGNORECASE,
)
_THINK_WITH_SUFFIX_RE = re.compile(r'<\s*/?\s*think:[A-Za-z0-9_-]+\s*>', re.IGNORECASE)
_BARE_THINK_TAG_RE = re.compile(r'<\s*/?\s*think\s*>', re.IGNORECASE)
_BARE_THINK_OPEN_RE = re.compile(r'<\s*think\s*>', re.IGNORECASE)
_BARE_THINK_CLOSE_RE = re.compile(r'<\s*/\s*think\s*>', re.IGNORECASE)


def _snippet_around(text: str, marker: str, radius: int = 160) -> str:
    """Return a short snippet around ``marker`` in ``text`` with newlines escaped."""
    if not text or not marker:
        return ''
    idx = text.find(marker)
    if idx < 0:
        return text[: radius * 2].replace('\n', '\\n')
    left = max(0, idx - radius)
    right = min(len(text), idx + len(marker) + radius)
    prefix = '...' if left > 0 else ''
    suffix = '...' if right < len(text) else ''
    return (prefix + text[left:right] + suffix).replace('\n', '\\n')


def detect_output_format_bad_patterns(messages: list[dict] | None) -> dict[str, Any]:
    """Detect think/tool-call protocol leakage in assistant messages.

    F7_2 detects tool-call protocol text in ``reasoning_content``; F7_3 detects it in
    non-code ``content``. Both only count when the turn carries no parsed ``tool_calls``.
    """
    f6_7_hits: list[dict[str, Any]] = []
    f7_2_hits: list[dict[str, Any]] = []
    f7_3_hits: list[dict[str, Any]] = []

    for turn_index, message in enumerate(messages or []):
        if not isinstance(message, dict) or message.get('role') != 'assistant':
            continue

        content = message.get('content') if isinstance(message.get('content'), str) else ''
        stripped_content = _strip_code_spans(content)
        reasoning_content = (
            message.get('reasoning_content') if isinstance(message.get('reasoning_content'), str) else ''
        )
        tool_calls = message.get('tool_calls')
        has_tool_calls = isinstance(tool_calls, list) and bool(tool_calls)

        if not has_tool_calls:
            tool_match = _TOOL_CALL_LITERAL_TOKEN_RE.search(reasoning_content)
            if tool_match:
                marker = tool_match.group(0)
                f7_2_hits.append({
                    'turn_index': turn_index,
                    'marker': marker,
                    'reasoning_chars': len(reasoning_content),
                    'content_chars': len(content),
                    'snippet': _snippet_around(reasoning_content, marker),
                })

            content_tool_match = _TOOL_CALL_LITERAL_TOKEN_RE.search(stripped_content)
            if content_tool_match:
                marker = content_tool_match.group(0)
                f7_3_hits.append({
                    'turn_index': turn_index,
                    'marker': marker,
                    'token_count': len(_TOOL_CALL_LITERAL_TOKEN_RE.findall(stripped_content)),
                    'reasoning_chars': len(reasoning_content),
                    'content_chars': len(content),
                    'snippet': _snippet_around(stripped_content, marker),
                })

        if not content:
            continue

        think_suffix_match = _THINK_WITH_SUFFIX_RE.search(content)
        if think_suffix_match:
            marker = think_suffix_match.group(0)
            f6_7_hits.append({
                'turn_index': turn_index,
                'kind': 'hash_think_tag',
                'marker': marker,
                'content_chars': len(content),
                'reasoning_chars': len(reasoning_content),
                'snippet': _snippet_around(content, marker),
            })
            continue

        if not _BARE_THINK_TAG_RE.search(content):
            continue
        if reasoning_content.strip():
            continue
        if _TOOL_CALL_LITERAL_TOKEN_RE.search(content):
            continue
        n_open = len(_BARE_THINK_OPEN_RE.findall(stripped_content))
        n_close = len(_BARE_THINK_CLOSE_RE.findall(stripped_content))
        n_total = n_open + n_close
        if n_total and n_open != n_close and n_total <= 6:
            marker_match = _BARE_THINK_TAG_RE.search(stripped_content)
            marker = marker_match.group(0) if marker_match else '<think>'
            f6_7_hits.append({
                'turn_index': turn_index,
                'kind': 'bare_unbalanced',
                'marker': marker,
                'n_open': n_open,
                'n_close': n_close,
                'content_chars': len(content),
                'reasoning_chars': len(reasoning_content),
                'snippet': _snippet_around(content, marker),
            })

    return {
        'is_bad_pattern': bool(f6_7_hits or f7_2_hits or f7_3_hits),
        'F6_7_think_tag_in_content': bool(f6_7_hits),
        'F7_2_reasoning_collapsed_tool_calls': bool(f7_2_hits),
        'F7_3_tool_call_leak_in_content': bool(f7_3_hits),
        'F6_7_count': len(f6_7_hits),
        'F7_2_count': len(f7_2_hits),
        'F7_3_count': len(f7_3_hits),
        'hits': {
            'F6_7_think_tag_in_content': f6_7_hits[:5],
            'F7_2_reasoning_collapsed_tool_calls': f7_2_hits[:5],
            'F7_3_tool_call_leak_in_content': f7_3_hits[:5],
        },
    }


# ---------------------------------------------------------------------------
# Terminus parse error detection (F5.2)
#
# Terminus agent outputs go directly in the assistant `content` field (no
# tool_calls) and the scaffold parses them. Two formats exist:
#   - JSON: {"analysis": ..., "plan": ..., "commands": [...], "task_complete": ...}
#   - XML:  <response><analysis>...</analysis><plan>...</plan><commands>...</commands></response>
#
# Both formats are detected here. The output key is kept as
# ``terminus_json_parse_error`` for backward compatibility with hytuner's
# penalty reading code (the name is historical — it now covers XML too).
# ---------------------------------------------------------------------------

_F5_2_MIN_LEN = 30
_F5_2_JSON_KV_RE = re.compile(r"""["']\s*[A-Za-z_][\w\-\.]*\s*["']\s*:""")
_F5_2_NON_JSON_CONTENT_PREFIX_RE = re.compile(
    r'^\s*\[\s*('
    r'Recent\s+Summary'
    r'|CONTEXT\s+COMPACTION'
    r'|ASSISTANT\s*\]'
    r'|TOOL[_ ]CALL'
    r'|TOOL[_ ]RESULT'
    r'|Expand\s+for\s+details'
    r'|Response\s+truncated'
    r'|System\s+reminder'
    r'|System\s+Notification'
    r'|reply_to_current'
    r'|HEARTBEAT'
    r')',
    re.IGNORECASE,
)
_F5_2_PARSE_ERROR_MARKER = 'Previous response had parsing errors'

_XML_RESPONSE_OPEN_RE = re.compile(r'<\s*response\s*>', re.IGNORECASE)
_XML_RESPONSE_CLOSE_RE = re.compile(r'<\s*/\s*response\s*>', re.IGNORECASE)
_XML_REQUIRED_TAGS = ('analysis', 'plan', 'commands')


def _f5_2_looks_like_json(text: str) -> bool:
    """Heuristic gate: is text trying to be a JSON literal?"""
    stripped = text.strip()
    if not stripped:
        return False
    first, last = stripped[0], stripped[-1]
    if not ((first == '{' and last == '}') or (first == '[' and last == ']')):
        return False
    if _F5_2_NON_JSON_CONTENT_PREFIX_RE.match(stripped):
        return False
    return bool(_F5_2_JSON_KV_RE.search(stripped[:4096]))


def _f5_2_content_is_bad_json(content: str) -> bool:
    """Return True when content passes the JSON heuristic gate but strict parsing fails."""
    stripped = content.strip()
    if len(stripped) < _F5_2_MIN_LEN:
        return False
    if not _f5_2_looks_like_json(stripped):
        return False
    try:
        json.loads(stripped)
    except Exception:
        return True
    return False


def _f5_2_content_is_bad_xml(content: str) -> bool:
    """Return True when content looks like a terminus XML response but is malformed."""
    stripped = content.strip()
    if len(stripped) < _F5_2_MIN_LEN:
        return False
    if not _XML_RESPONSE_OPEN_RE.search(stripped):
        return False
    if not _XML_RESPONSE_CLOSE_RE.search(stripped):
        return True
    lower = stripped.lower()
    return any(f'<{tag}>' not in lower and f'<{tag} ' not in lower for tag in _XML_REQUIRED_TAGS)


def detect_terminus_parse_error(messages: list[dict] | None) -> dict[str, Any]:
    """Detect F5.2 — terminus style parse error (JSON and XML formats).

    Output keys keep the ``terminus_json_*`` / ``is_terminus_json_parse_error``
    naming for backward compatibility with hytuner's penalty reading code,
    even though XML is now covered as well.
    """
    indirect_hit = False
    direct_hits: list[dict[str, Any]] = []

    for turn_index, message in enumerate(messages or []):
        if not isinstance(message, dict):
            continue
        role = message.get('role')
        content = message.get('content') if isinstance(message.get('content'), str) else ''
        if role == 'user':
            if _F5_2_PARSE_ERROR_MARKER in content:
                indirect_hit = True
        elif role == 'assistant' and content:
            bad_json = _f5_2_content_is_bad_json(content)
            bad_xml = _f5_2_content_is_bad_xml(content) if not bad_json else False
            if bad_json or bad_xml:
                direct_hits.append({
                    'turn_index': turn_index,
                    'kind': 'json' if bad_json else 'xml',
                    'content_chars': len(content),
                })

    direct_hit = bool(direct_hits)
    return {
        'is_terminus_json_parse_error': indirect_hit or direct_hit,
        'indirect_scaffold_feedback': indirect_hit,
        'direct_content_as_json': direct_hit,
        'direct_hit_count': len(direct_hits),
        'hits': direct_hits[:5],
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_terminal_monitors(messages: list[dict] | None) -> dict[str, Any]:
    """Run all monitors relevant to terminal/terminus tasks."""
    return {
        'codeswitch': detect_codeswitch(messages),
        'output_format_bad_pattern': detect_output_format_bad_patterns(messages),
        'terminus_json_parse_error': detect_terminus_parse_error(messages),
        'let_me_loop': detect_let_me_loop(messages),
    }
