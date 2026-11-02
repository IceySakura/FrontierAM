"""Navi agent tools — 标准 nexus function-calling 工具定义。

提供 4 个 tool，对应 navi 的 4 种动作类型：
- execute_code: 在 WAA VM 上执行 Python 代码（通过 computer.* API）
- wait: 等待 UI 更新
- done: 标记任务完成
- fail: 标记任务不可行
"""

from __future__ import annotations

import asyncio
import logging

from pydantic import BaseModel, Field

from nexus.core import ToolContext, ToolResponse, ToolType, require_context, tool_registry
from nexus.runtime.providers.ags.waa import require_raw_waa_runtime


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Input models
# ---------------------------------------------------------------------------


class ExecuteCodeInput(BaseModel):
    """execute_code tool 的输入参数。"""

    code: str = Field(description='Python code to execute on the VM using the computer.* API.')
    memory: str | None = Field(
        default=None, description='Memory note to store for future steps. Set to null if no update needed.'
    )


class WaitInput(BaseModel):
    """wait tool 的输入参数。"""

    duration: float = Field(default=0.5, description='Seconds to wait before the next observation.')
    memory: str | None = Field(
        default=None, description='Memory note to store for future steps. Set to null if no update needed.'
    )


class DoneInput(BaseModel):
    """done tool 的输入参数。"""

    summary: str = Field(description='Brief summary of what was accomplished.')


class FailInput(BaseModel):
    """fail tool 的输入参数。"""

    reason: str = Field(description='Reason why the task cannot be completed.')


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


@tool_registry.register(
    'navi/execute_code',
    meta={
        'name': 'execute_code',
        'description': (
            'Execute Python code on the Windows VM using the computer.* API. '
            'Use this to interact with the GUI: mouse clicks, keyboard input, clipboard, etc. '
            'The code runs in a persistent Python environment with the `computer` object available.'
        ),
        'tool_type': ToolType.LOCAL,
    },
)
async def execute_code_tool(params: ExecuteCodeInput, context: ToolContext) -> ToolResponse:
    """在 WAA VM 上执行 Python 代码块。"""
    ctx = require_context()
    runtime = require_raw_waa_runtime(await ctx.get_runtime())

    result = await runtime.execute_windows(params.code)

    # 构建输出文本
    output_text = 'Code executed successfully.'
    if isinstance(result, dict):
        if result.get('output'):
            output_text = f'Output:\n{result["output"]}'
        elif result.get('error'):
            output_text = f'Error:\n{result["error"]}'

    # context_updates: 更新 memory（如果提供了新值）
    updates: dict = {}
    if params.memory is not None:
        updates['memory'] = params.memory

    return ToolResponse(
        output=[{'type': 'text', 'text': output_text}],
        context_updates=updates,
    )


@tool_registry.register(
    'navi/wait',
    meta={
        'name': 'wait',
        'description': (
            'Wait for a specified duration before taking the next observation. '
            'Use this when you need to wait for an animation, page load, or UI transition to complete.'
        ),
        'tool_type': ToolType.LOCAL,
    },
)
async def wait_tool(params: WaitInput, context: ToolContext) -> ToolResponse:
    """等待指定时间后继续。"""
    await asyncio.sleep(params.duration)

    updates: dict = {}
    if params.memory is not None:
        updates['memory'] = params.memory

    logger.debug(f'wait_tool: waited {params.duration}s')

    return ToolResponse(
        output=[{'type': 'text', 'text': f'Waited {params.duration}s.'}],
        context_updates=updates,
    )


@tool_registry.register(
    'navi/done',
    meta={
        'name': 'done',
        'description': (
            'Signal that the task has been completed successfully. '
            'Call this only after verifying that the objective has been achieved.'
        ),
        'tool_type': ToolType.LOCAL,
    },
)
async def done_tool(params: DoneInput, context: ToolContext) -> ToolResponse:
    """标记任务已完成。"""
    return ToolResponse(
        output=[{'type': 'text', 'text': params.summary}],
        context_updates={'exit_status': 'done'},
    )


@tool_registry.register(
    'navi/fail',
    meta={
        'name': 'fail',
        'description': (
            'Signal that the task cannot be completed (infeasible). '
            'Call this when you determine the objective is impossible to achieve.'
        ),
        'tool_type': ToolType.LOCAL,
    },
)
async def fail_tool(params: FailInput, context: ToolContext) -> ToolResponse:
    """标记任务不可行。"""
    return ToolResponse(
        output=[{'type': 'text', 'text': params.reason}],
        context_updates={'exit_status': 'fail'},
    )
