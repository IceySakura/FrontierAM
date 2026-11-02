"""
Execute Python tool — thin wrapper over jupyter_code_interpreter.
Shares the same persistent Jupyter kernel pool.

Configuration:
    tool_configs:
      jupyter_code_interpreter:
        python_path: /path/to/python  # Optional: specify which Python to use
"""

from pydantic import Field

from nexus.core import ToolContext, ToolResponse, ToolType, tool_registry
from nexus.extensions.tools.code_interpreter.jupyter_code_interpreter import (
    JupyterCodeInterpreterInput,
    jupyter_code_interpreter_tool,
)


class ExecutePythonInput(JupyterCodeInterpreterInput):
    code: str = Field(description='The python code to execute for numerical calculations or data analysis.')


@tool_registry.register(
    'execute_python',
    meta={
        'name': 'execute_python',
        'description': (
            'Use this tool to execute Python code for numerical calculations or data analysis. '
            'The code runs in a stateful Jupyter notebook environment, so imports, variables, and functions persist across calls. '
            'The tool responds with the execution output or a timeout error.'
        ),
        'tool_type': ToolType.RUNTIME,
    },
)
async def execute_python_tool(params: ExecutePythonInput, context: ToolContext) -> ToolResponse:
    return await jupyter_code_interpreter_tool(params, context)
