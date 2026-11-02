"""Pure helpers used by multiple bad-pattern rules."""

from __future__ import annotations

import json
import re

from collections.abc import Iterable
from typing import Any


def text(message: dict[str, Any], field: str = 'content') -> str:
    value = message.get(field)
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get('text'), str):
                parts.append(item['text'])
        return '\n'.join(parts)
    return ''


def function(call: dict[str, Any]) -> dict[str, Any] | None:
    nested = call.get('function')
    if isinstance(nested, dict):
        return nested
    if isinstance(call.get('name'), str):  # HyX compact form
        return call
    return None


def call_arguments(call: dict[str, Any]) -> Any:
    fn = function(call)
    return fn.get('arguments') if fn else None


def parse_arguments(raw: Any) -> tuple[dict[str, Any] | None, str | None]:
    if isinstance(raw, dict):
        return raw, None
    if raw in (None, ''):
        return {}, None
    if not isinstance(raw, str):
        return None, f'arguments 必须是 JSON object 或 JSON string，实际为 {type(raw).__name__}'
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        return None, f'arguments 不是合法 JSON：{exc}'
    if not isinstance(value, dict):
        return None, f'arguments 解析结果不是 object，而是 {json_type(value)}'
    if '_unparseable_arguments' in value:
        return None, 'arguments 含有 invalid-JSON 恢复标记 _unparseable_arguments'
    return value, None


def tool_index(tools: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    result = {}
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        raw_function = tool.get('function')
        fn: dict[str, Any] = raw_function if isinstance(raw_function, dict) else tool
        name = fn.get('name')
        if isinstance(name, str) and name:
            result[name] = fn
    return result


def json_type(value: Any) -> str:
    if value is None:
        return 'null'
    if isinstance(value, bool):
        return 'boolean'
    if isinstance(value, int):
        return 'integer'
    if isinstance(value, float):
        return 'number'
    if isinstance(value, str):
        return 'string'
    if isinstance(value, list):
        return 'array'
    if isinstance(value, dict):
        return 'object'
    return type(value).__name__


def matches_type(value: Any, expected: str) -> bool:
    return {
        'string': isinstance(value, str),
        'number': isinstance(value, (int, float)) and not isinstance(value, bool),
        'integer': isinstance(value, int) and not isinstance(value, bool),
        'boolean': isinstance(value, bool),
        'array': isinstance(value, list),
        'object': isinstance(value, dict),
        'null': value is None,
    }.get(expected, True)


def validate_value(value: Any, schema: dict[str, Any], path: str = '', depth: int = 0) -> list[str]:
    if depth > 8:
        return []
    raw_type = schema.get('type')
    expected = [raw_type] if isinstance(raw_type, str) else [x for x in raw_type or [] if isinstance(x, str)]
    if expected and not any(matches_type(value, kind) for kind in expected):
        return [f'参数 {path} 类型错误：期望 {expected}，实际 {json_type(value)}']
    if isinstance(schema.get('enum'), list) and value not in schema['enum']:
        return [f'参数 {path} 不在 enum {schema["enum"]!r} 中']
    errors: list[str] = []
    if isinstance(value, dict):
        required = schema.get('required')
        if isinstance(required, list):
            errors.extend(
                f'缺少必填参数 {path + "." if path else ""}{key}'
                for key in required
                if isinstance(key, str) and key not in value
            )
        properties = schema.get('properties')
        if isinstance(properties, dict):
            for key, child in value.items():
                child_schema = properties.get(key)
                if isinstance(child_schema, dict):
                    errors.extend(validate_value(child, child_schema, f'{path}.{key}' if path else key, depth + 1))
            if schema.get('additionalProperties') is False:
                errors.extend(
                    f'存在未声明参数 {path + "." if path else ""}{key}' for key in value if key not in properties
                )
    elif isinstance(value, list) and isinstance(schema.get('items'), dict):
        for index, child in enumerate(value):
            errors.extend(validate_value(child, schema['items'], f'{path}[{index}]', depth + 1))
    return errors


def walk_strings(value: Any, path: list[Any]) -> Iterable[tuple[list[Any], str]]:
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, dict):
        for key, item in value.items():
            if key not in {'id', 'name'}:
                yield from walk_strings(item, path + [key])
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from walk_strings(item, path + [index])


def excerpt(value: Any, limit: int = 300) -> str:
    rendered = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return rendered if len(rendered) <= limit else rendered[:limit] + '…'


def language(value: str) -> str:
    cleaned = re.sub(r'```.*?```|https?://\S+', '', value, flags=re.S)
    zh = len(re.findall(r'[\u3400-\u9fff]', cleaned))
    en = len(re.findall(r'\b[A-Za-z]{2,}\b', cleaned))
    if zh >= 8 and zh >= en * 0.25:
        return 'zh'
    if en >= 8 and en >= zh * 2:
        return 'en'
    return 'unknown'


def is_error_result(message: dict[str, Any] | None) -> bool:
    if not isinstance(message, dict):
        return False
    body = text(message).strip()
    status = str(message.get('status') or message.get('tool_result_status') or '').casefold()
    inferred_outcome = str(message.get('inferred_outcome') or '').casefold()
    exit_code = message.get('exit_code')
    return (
        bool(message.get('is_error'))
        or inferred_outcome == 'failure'
        or (isinstance(exit_code, int) and not isinstance(exit_code, bool) and exit_code != 0)
        or status in {'error', 'failed', 'failure'}
        or bool(re.match(r'^<tool_use_error(?:\s[^>]*)?>', body, re.I))
        or bool(re.search(r'^(?:error|tool error|exception|traceback)\b', body, re.I))
    )


def call_signature(call: dict[str, Any]) -> str | None:
    try:
        return json.dumps(
            {k: v for k, v in call.items() if k != 'id'}, ensure_ascii=False, sort_keys=True, separators=(',', ':')
        )
    except (TypeError, ValueError):
        return None
