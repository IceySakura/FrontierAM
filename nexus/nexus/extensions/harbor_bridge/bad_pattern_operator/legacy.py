"""Legacy SWE-compatible detector families used by deferred trainer penalties.

The detector functions are intentionally kept behavior-compatible with the
existing SWE/code_mt monitors.  They only emit observations; HyTuner applies
penalties after reward filtering.
"""

from __future__ import annotations

import json
import os
import re

from typing import Any


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
# str_replace old==new detection
# ---------------------------------------------------------------------------

_STR_REPLACE_TOOL_NAME = 'str_replace'
_STR_REPLACE_EDITOR_TOOL_NAME = 'str_replace_editor'
_EDIT_TOOL_NAME = 'edit'
_STR_REPLACE_ARG_PAIRS = (
    ('old', 'new'),
    ('old_str', 'new_str'),
    ('old_string', 'new_string'),
    ('old_content', 'new_content'),
)
_STR_REPLACE_NAME_KEYS = ('name', 'tool_name', 'function_name', 'tool_call_name')
_STR_REPLACE_ARGUMENT_KEYS = ('arguments', 'args', 'tool_args', 'parameters', 'input', 'kwargs')
_HUNYUAN_TAG_SUFFIX = r'(?::[0-9A-Za-z_]+)?'
_HUNYUAN_ARG_RE = re.compile(
    rf'<arg_key{_HUNYUAN_TAG_SUFFIX}>\s*(?P<key>.*?)\s*</arg_key{_HUNYUAN_TAG_SUFFIX}>'
    rf'\s*<arg_value{_HUNYUAN_TAG_SUFFIX}>(?P<value>.*?)</arg_value{_HUNYUAN_TAG_SUFFIX}>',
    re.DOTALL | re.IGNORECASE,
)
_QWEN_FUNCTION_RE = re.compile(r'<function\s*=\s*(?P<name>[^>]+)>(?P<body>.*?)</function>', re.DOTALL)
_QWEN_PARAMETER_RE = re.compile(r'<parameter\s*=\s*(?P<key>[^>]+)>(?P<value>.*?)</parameter>', re.DOTALL)
_XML_TOOL_CALL_RE = re.compile(
    rf'<tool_call{_HUNYUAN_TAG_SUFFIX}>(?P<body>.*?)</tool_call{_HUNYUAN_TAG_SUFFIX}>',
    re.DOTALL | re.IGNORECASE,
)
_HUNYUAN_TOOL_NAME_END_RE = re.compile(rf'<(?:tool_sep|arg_key){_HUNYUAN_TAG_SUFFIX}>', re.IGNORECASE)


def _maybe_json_loads(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in '[{':
        return value
    try:
        return json.loads(stripped)
    except (TypeError, ValueError):
        return value


def _merge_argument_dict(target: dict[str, Any], value: Any) -> None:
    value = _maybe_json_loads(value)
    if isinstance(value, dict):
        target.update(value)


def _is_named_str_replace_call(tool_name: str | None, args: dict[str, Any]) -> bool:
    if not isinstance(tool_name, str):
        return False
    basename = tool_name.rsplit('/', 1)[-1].strip().casefold()
    if basename in {_STR_REPLACE_TOOL_NAME, _EDIT_TOOL_NAME}:
        return True
    if basename != _STR_REPLACE_EDITOR_TOOL_NAME:
        return False
    command = args.get('command')
    return isinstance(command, str) and command.strip().casefold() in {
        _STR_REPLACE_TOOL_NAME,
        _EDIT_TOOL_NAME,
    }


def _extract_named_str_replace_args(item: dict[str, Any]) -> dict[str, Any] | None:
    tool_name = None
    for key in _STR_REPLACE_NAME_KEYS:
        value = item.get(key)
        if isinstance(value, str):
            tool_name = value
            break

    args: dict[str, Any] = {}
    for old_key, new_key in _STR_REPLACE_ARG_PAIRS:
        if old_key in item:
            args[old_key] = item[old_key]
        if new_key in item:
            args[new_key] = item[new_key]
    if 'command' in item:
        args['command'] = item['command']
    for key in _STR_REPLACE_ARGUMENT_KEYS:
        if key in item:
            _merge_argument_dict(args, item.get(key))
    if not _is_named_str_replace_call(tool_name, args):
        return None
    return args


def _extract_xml_str_replace_args(text: str) -> list[dict[str, Any]]:
    if not re.search(r'\b(?:str_replace(?:_editor)?|edit)\b', text, re.IGNORECASE):
        return []

    calls: list[dict[str, Any]] = []
    for match in _XML_TOOL_CALL_RE.finditer(text):
        body = match.group('body').strip()
        parsed_body = _maybe_json_loads(body)
        if isinstance(parsed_body, dict):
            args = _extract_named_str_replace_args(parsed_body)
            if args is not None:
                calls.append(args)
            continue

        qwen_match = _QWEN_FUNCTION_RE.search(body)
        if qwen_match:
            args = {
                m.group('key').strip(): m.group('value').strip()
                for m in _QWEN_PARAMETER_RE.finditer(qwen_match.group('body'))
            }
            if _is_named_str_replace_call(qwen_match.group('name').strip(), args):
                calls.append(args)
                continue

        name_end = _HUNYUAN_TOOL_NAME_END_RE.search(body)
        tool_name_lines = body[: name_end.start() if name_end else len(body)].strip().splitlines()
        if not tool_name_lines:
            continue
        tool_name = tool_name_lines[0]
        args = {m.group('key').strip(): m.group('value') for m in _HUNYUAN_ARG_RE.finditer(body)}
        if not _is_named_str_replace_call(tool_name, args):
            continue
        calls.append(args)
    return calls


def _iter_str_replace_calls(value: Any) -> list[dict[str, Any]]:
    value = _maybe_json_loads(value)
    if isinstance(value, dict):
        calls: list[dict[str, Any]] = []
        args = _extract_named_str_replace_args(value)
        if args is not None:
            calls.append(args)
        for child in value.values():
            calls.extend(_iter_str_replace_calls(child))
        return calls
    if isinstance(value, list | tuple):
        calls = []
        for child in value:
            calls.extend(_iter_str_replace_calls(child))
        return calls
    if isinstance(value, str):
        return _extract_xml_str_replace_args(value)
    return []


def _has_old_eq_new(args: dict[str, Any]) -> bool:
    for old_key, new_key in _STR_REPLACE_ARG_PAIRS:
        if old_key in args and new_key in args:
            old_value = args[old_key]
            new_value = args[new_key]
            return old_value is not None and new_value is not None and old_value == new_value
    return False


def _get_tool_stats_str_replace_count(extra_fields: dict[str, Any]) -> int:
    tool_stats = extra_fields.get('tool_stats')
    if not isinstance(tool_stats, dict):
        return 0
    counts = tool_stats.get(_STR_REPLACE_TOOL_NAME)
    if not isinstance(counts, dict):
        return 0
    try:
        return max(0, int(counts.get('total', 0)))
    except (TypeError, ValueError):
        return 0


def _select_str_replace_scan_source(extra_fields: dict[str, Any]) -> Any:
    for key in ('agent_timeline', 'tool_calls', 'tool_call_records', 'tool_history', 'messages'):
        value = extra_fields.get(key)
        if value:
            return value
    return extra_fields


def detect_str_replace_old_eq_new(extra_fields: dict[str, Any]) -> dict[str, Any]:
    """Detect str_replace/Edit calls where the old and new text are identical.

    ``str_replace_old_eq_new`` is a legacy settlement family consumed by the
    trainer after reward filtering. Keep that contract while accepting the
    ``Edit`` name and ``old_string``/``new_string`` arguments used by HyCode.
    """
    source = _select_str_replace_scan_source(extra_fields)
    calls = _iter_str_replace_calls(source)
    str_replace_count = max(len(calls), _get_tool_stats_str_replace_count(extra_fields))
    old_eq_new_count = sum(1 for args in calls if _has_old_eq_new(args))
    return {
        'str_replace_count': str_replace_count,
        'old_eq_new_count': old_eq_new_count,
    }


def _merge_code_edit_bad_patterns(
    result: dict[str, Any],
    messages: list[dict] | None,
) -> dict[str, Any]:
    """Fold modern Edit failures into the legacy deferred penalty bucket.

    The trainer currently settles only the historical
    ``str_replace_old_eq_new_count`` column after reward filtering. Code-agent
    Edit failures must therefore be represented by that compatibility hit
    count, while explicit breakdown fields preserve the real reason.
    """
    from .detector import detect_tool_call_bad_patterns

    detection = detect_tool_call_bad_patterns(
        messages,
        enabled_rules={
            'edit_noop_call',
            'edit_call_failure',
            'edit_call_never_recovered',
        },
    )
    counts = detection.get('issue_counts')
    counts = counts if isinstance(counts, dict) else {}
    legacy_noop_count = int(result.get('old_eq_new_count', 0) or 0)
    modern_noop_count = int(counts.get('edit_noop_call', 0) or 0)
    edit_failure_count = int(counts.get('edit_call_failure', 0) or 0)
    edit_never_recovered_count = int(counts.get('edit_call_never_recovered', 0) or 0)
    noop_count = max(legacy_noop_count, modern_noop_count)
    penalty_count = noop_count + edit_failure_count + edit_never_recovered_count
    return {
        **result,
        # Compatibility field consumed by the existing trainer-side deferred
        # str_replace penalty. It is now the aggregate code-edit violation count.
        'old_eq_new_count': penalty_count,
        'noop_edit_count': noop_count,
        'edit_call_failure_count': edit_failure_count,
        'edit_call_never_recovered_count': edit_never_recovered_count,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_swe_monitors(
    messages: list[dict] | None,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run all monitors relevant to SWE tasks.

    Args:
        messages: Chat history (assistant messages contain tool_calls).
        extra_fields: If available, used as scan source for str_replace detection
            (agent_timeline, tool_stats, etc.). Falls back to messages.
    """
    scan_source = extra_fields if extra_fields else {'messages': messages or []}
    return {
        'codeswitch': detect_codeswitch(messages),
        'output_format_bad_pattern': detect_output_format_bad_patterns(messages),
        'str_replace_old_eq_new': detect_str_replace_old_eq_new(scan_source),
    }


def run_legacy_deferred_monitors(
    messages: list[dict] | None,
    *,
    enabled_families: set[str],
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run only the legacy families configured for trainer-side settlement."""
    results: dict[str, Any] = {}
    if 'codeswitch' in enabled_families:
        results['codeswitch'] = detect_codeswitch(messages)
    if 'output_format_bad_pattern' in enabled_families:
        results['output_format_bad_pattern'] = detect_output_format_bad_patterns(messages)
    if 'str_replace_old_eq_new' in enabled_families:
        scan_source = extra_fields if extra_fields else {'messages': messages or []}
        results['str_replace_old_eq_new'] = _merge_code_edit_bad_patterns(
            detect_str_replace_old_eq_new(scan_source),
            messages,
        )
    return results


def legacy_deferred_metrics(results: dict[str, Any]) -> dict[str, float]:
    """Expose legacy detections under the unified bad-pattern metric namespace."""
    metrics: dict[str, float] = {}

    codeswitch = results.get('codeswitch')
    if isinstance(codeswitch, dict):
        hit = bool(codeswitch.get('is_codeswitch'))
        metrics['tool_call_bad_pattern/legacy_codeswitch/hit'] = 1.0 if hit else 0.0
        metrics['tool_call_bad_pattern/legacy_codeswitch/count'] = 1.0 if hit else 0.0

    str_replace = results.get('str_replace_old_eq_new')
    if isinstance(str_replace, dict):
        count = float(str_replace.get('old_eq_new_count', 0) or 0)
        metrics['tool_call_bad_pattern/legacy_str_replace_old_eq_new/hit'] = 1.0 if count > 0 else 0.0
        metrics['tool_call_bad_pattern/legacy_str_replace_old_eq_new/count'] = count
        for reason in ('noop_edit', 'edit_call_failure', 'edit_call_never_recovered'):
            reason_count = float(str_replace.get(f'{reason}_count', 0) or 0)
            metrics[f'tool_call_bad_pattern/legacy_str_replace_old_eq_new/{reason}/hit'] = (
                1.0 if reason_count > 0 else 0.0
            )
            metrics[f'tool_call_bad_pattern/legacy_str_replace_old_eq_new/{reason}/count'] = reason_count

    output_format = results.get('output_format_bad_pattern')
    if isinstance(output_format, dict):
        counts = {
            'F6_7_think_tag_in_content': float(output_format.get('F6_7_count', 0) or 0),
            'F7_2_reasoning_collapsed_tool_calls': float(output_format.get('F7_2_count', 0) or 0),
            'F7_3_tool_call_leak_in_content': float(output_format.get('F7_3_count', 0) or 0),
        }
        total = sum(counts.values())
        metrics['tool_call_bad_pattern/legacy_output_format_bad_pattern/hit'] = 1.0 if total > 0 else 0.0
        metrics['tool_call_bad_pattern/legacy_output_format_bad_pattern/count'] = total
        for family, count in counts.items():
            metrics[f'tool_call_bad_pattern/legacy_output_format_bad_pattern/{family}/hit'] = 1.0 if count > 0 else 0.0
            metrics[f'tool_call_bad_pattern/legacy_output_format_bad_pattern/{family}/count'] = count

    return metrics


__all__ = [
    'detect_codeswitch',
    'detect_output_format_bad_patterns',
    'detect_str_replace_old_eq_new',
    'legacy_deferred_metrics',
    'run_legacy_deferred_monitors',
    'run_swe_monitors',
]
