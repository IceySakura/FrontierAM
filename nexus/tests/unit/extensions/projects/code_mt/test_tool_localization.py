from __future__ import annotations

import copy
import re

import pytest

import nexus.extensions.projects.code_mt.agent.tools as _tools  # noqa: F401

from nexus.core import ToolExecutor, tool_registry
from nexus.extensions.projects.code_mt.agent import agent as agent_module
from nexus.extensions.projects.code_mt.agent.models import CodeMTAgentInput
from nexus.extensions.projects.code_mt.agent.tool_localization import (
    CHINESE_TOOL_OVERRIDES,
    apply_chinese_tool_descriptions,
)


def _without_descriptions(value):
    if isinstance(value, dict):
        return {key: _without_descriptions(item) for key, item in value.items() if key != 'description'}
    if isinstance(value, list):
        return [_without_descriptions(item) for item in value]
    return value


@pytest.mark.parametrize('tool_name', sorted(CHINESE_TOOL_OVERRIDES))
def test_chinese_tool_descriptions_preserve_tool_protocol(tool_name: str) -> None:
    tool = tool_registry.get(tool_name)
    executor = ToolExecutor([tool], allowed_tool_names=[tool_name])
    original_schema = copy.deepcopy(executor.get_tool_schemas()[0])

    apply_chinese_tool_descriptions(executor, [tool_name])

    localized_schema = executor.get_tool_schemas()[0]
    assert _without_descriptions(localized_schema) == _without_descriptions(original_schema)
    assert re.search(r'[\u3400-\u9fff]', localized_schema['function']['description'])
    for property_schema in localized_schema['function']['parameters'].get('properties', {}).values():
        assert re.search(r'[\u3400-\u9fff]', property_schema['description'])


@pytest.mark.asyncio
async def test_ch_code_mt_enables_chinese_tool_descriptions(monkeypatch) -> None:
    captured_input = None
    expected_output = object()

    async def fake_run(agent_input: CodeMTAgentInput):
        nonlocal captured_input
        captured_input = agent_input
        return expected_output

    monkeypatch.setattr(agent_module, '_run_code_mt_agent', fake_run)

    original_input = CodeMTAgentInput(user_prompt='任务', working_dir='/testbed')
    result = await agent_module.ch_code_mt_agent(original_input)

    assert result is expected_output
    assert captured_input is not original_input
    assert captured_input.tool_description_language == 'zh-CN'
    assert original_input.tool_description_language == 'default'
