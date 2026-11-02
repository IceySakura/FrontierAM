"""Tool-schema normalization and merging for bad-pattern validation."""

from __future__ import annotations

from typing import Any


def normalize_tool_schema(tool: Any) -> dict[str, Any] | None:
    """Normalize OpenAI, Anthropic, and MCP tool schemas to one function shape."""
    if not isinstance(tool, dict):
        return None
    raw_function = tool.get('function')
    function = raw_function if isinstance(raw_function, dict) else tool
    name = function.get('name')
    if not isinstance(name, str) or not name.strip():
        return None
    description = function.get('description')
    parameters = function.get('parameters')
    if not isinstance(parameters, dict):
        parameters = function.get('input_schema')
    return {
        'type': 'function',
        'function': {
            'name': name.strip(),
            **({'description': description} if isinstance(description, str) else {}),
            'parameters': parameters if isinstance(parameters, dict) else {},
        },
    }


def merge_tool_schemas(*sources: Any) -> list[dict[str, Any]]:
    """Merge schema sources by advertised tool name; later sources are authoritative."""
    by_name: dict[str, dict[str, Any]] = {}
    for source in sources:
        if not isinstance(source, (list, tuple)):
            continue
        for raw_tool in source:
            schema = normalize_tool_schema(raw_tool)
            if schema is not None:
                by_name[schema['function']['name']] = schema
    return list(by_name.values())


__all__ = ['merge_tool_schemas', 'normalize_tool_schema']
