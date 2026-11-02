"""``ExecuteCode`` tool for the white-box CC search agent.

A thin wrapper around
:func:`nexus.extensions.tools.code_interpreter.jupyter_code_interpreter.jupyter_code_interpreter_tool`
that exposes a stateful, Jupyter-kernel-backed Python REPL to the agent
under the LLM-facing name ``ExecuteCode``.

The public contract (``meta['name']``, ``meta['description']`` and the
input schema) is a near one-to-one port of Cursor's ``mcp__ide__executeCode``
built-in tool. Two minimal adaptations:

- ``"current notebook file"`` → ``"persistent Jupyter kernel"`` (the
  kernel here lives in the runtime sandbox, not in an IDE notebook).
- An optional ``timeout`` field is added on top of the original ``code``
  parameter so callers can bound long-running cells (mirrors the prior
  ``python_repl`` tool).
"""

from typing import Any

from pydantic import Field, model_validator

from nexus.core import ToolContext, ToolResponse, ToolType, tool_registry
from nexus.extensions.tools.code_interpreter.jupyter_code_interpreter import (
    JupyterCodeInterpreterInput,
    jupyter_code_interpreter_tool,
)


EXECUTE_CODE_TOOL_REGISTRY_NAME = 'cc_search_agent_whitebox/execute_code'
EXECUTE_CODE_TOOL_NAME = 'ExecuteCode'

EXECUTE_CODE_DESCRIPTION = (
    'Execute python code in the persistent Jupyter kernel.\n\n'
    'All code will be executed in the persistent Jupyter kernel.\n\n'
    'Avoid declaring variables or modifying the state of the kernel unless the user '
    'explicitly asks for it.\n\n'
    'Any code executed will persist across calls to this tool, unless the kernel '
    'has been restarted.'
)


class ExecuteCodeInput(JupyterCodeInterpreterInput):
    code: str = Field(description='The code to be executed on the kernel.')
    timeout: float | None = Field(
        default=60,
        description='Execution timeout in seconds.',
    )

    @model_validator(mode='before')
    @classmethod
    def _normalize_missing_required_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            normalized = dict(data)
            if not isinstance(normalized.get('code'), str):
                normalized['code'] = ''
            return normalized
        return data


@tool_registry.register(
    EXECUTE_CODE_TOOL_REGISTRY_NAME,
    meta={
        'name': EXECUTE_CODE_TOOL_NAME,
        'description': EXECUTE_CODE_DESCRIPTION,
        'tool_type': ToolType.RUNTIME,
    },
)
async def execute_code_tool(params: ExecuteCodeInput, context: ToolContext) -> ToolResponse:
    if not params.code.strip():
        return ToolResponse(
            output=[{'type': 'text', 'text': 'Error: Missing required parameter "code"'}],
            meta={'success': False},
        )
    return await jupyter_code_interpreter_tool(params, context)


__all__ = [
    'EXECUTE_CODE_DESCRIPTION',
    'EXECUTE_CODE_TOOL_NAME',
    'EXECUTE_CODE_TOOL_REGISTRY_NAME',
    'ExecuteCodeInput',
    'execute_code_tool',
]
