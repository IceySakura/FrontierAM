"""注册到 Nexus 工具注册表中的 aidata 本地专用工具。"""

from __future__ import annotations

import importlib.util

from pathlib import Path
from typing import Any

from pydantic import Field

from nexus.core import ToolContext, ToolResponse, ToolType, tool_registry
from nexus.core.types import RobustBaseModel


class AskUserInput(RobustBaseModel):
    question: str = Field(description='The clarification question to ask the user.')


@tool_registry.register(
    'base/ask_user',
    meta={
        'name': 'ask_user',
        'description': 'Ask the simulated user a clarification question and return their deterministic response.',
        'tool_type': ToolType.LOCAL,
    },
)
async def ask_user_tool(params: AskUserInput, context: ToolContext) -> ToolResponse:
    cfg = dict(context.tool_configs.get('base/ask_user') or {})
    script_path = cfg.get('script_path')
    if not isinstance(script_path, str) or not script_path:
        return ToolResponse(output=[{'type': 'text', 'text': 'ask_user is not configured for this task.'}])

    try:
        response = _call_ask_user_script(Path(script_path), params.question)
    except Exception as exc:  # noqa: BLE001 - 将确定性的工具失败暴露给 agent。
        response = f'ask_user failed: {exc}'
    return ToolResponse(output=[{'type': 'text', 'text': str(response)}])


def _call_ask_user_script(script_path: Path, question: str) -> Any:
    spec = importlib.util.spec_from_file_location('_aidata_case_ask_user', script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot import ask_user script: {script_path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, 'ask_user', None)
    if not callable(fn):
        raise RuntimeError(f'{script_path} does not define callable ask_user')
    return fn(question)
