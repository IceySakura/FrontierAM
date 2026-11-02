from __future__ import annotations

import json

# Import tools to register them.
import nexus.extensions.agents.swe_agent.tools  # noqa: F401

from nexus.core import ToolExecutor, tool_registry


def test_swe_agent_tool_schema_preserves_str_replace_parameter_order():
    executor = ToolExecutor([
        tool_registry.get('swe_agent/bash'),
        tool_registry.get('swe_agent/str_replace_editor'),
        tool_registry.get('swe_agent/review_submit'),
    ])

    schemas = executor.get_tool_schemas()
    str_replace_schema = next(schema for schema in schemas if schema['function']['name'] == 'str_replace_editor')
    properties = str_replace_schema['function']['parameters']['properties']

    assert list(properties) == ['command', 'path', 'file_text', 'old_str', 'new_str', 'insert_line', 'view_range']


def test_json_loads_preserves_swe_agent_argument_order_for_logging_roundtrip():
    raw_arguments = '{"command":"str_replace","path":"/tmp/a.py","old_str":"old","new_str":"new"}'
    parsed = json.loads(raw_arguments)

    assert list(parsed) == ['command', 'path', 'old_str', 'new_str']
    assert list(json.loads(json.dumps(parsed, ensure_ascii=False))) == ['command', 'path', 'old_str', 'new_str']
