"""
local-python_execute 工具：在容器内通过 uv run 执行 Python 代码。

对标 old 框架的 python_interpretor.py，通过 runtime.run_command() 在沙箱容器内执行，
agent 可以用它来跑任意 Python 脚本（如 GCP SDK 操作等）。
"""

import base64
import logging
import os
import shlex
import uuid

from pydantic import Field

from nexus.core import ToolContext, ToolResponse, ToolType, require_context, tool_registry
from nexus.core.types import RobustBaseModel


logger = logging.getLogger(__name__)

PYTHON_EXECUTE_WORKSPACE_KEY = '_python_execute_workspace'
# Optional interpreter pinned by the task (e.g. Toolathlon's prebuilt venv). Falls back to `uv run`.
PYTHON_EXECUTE_RUNTIME_PYTHON_KEY = '_python_execute_runtime_python'
# Extra environment variables injected into the executed script (e.g. cloud SDK credentials).
PYTHON_EXECUTE_EXTRA_ENV_KEY = '_python_execute_extra_env'


class PythonExecuteInput(RobustBaseModel):
    code: str = Field(description='Python code to execute (can be directly pasted into a .py file)')
    filename: str | None = Field(
        default=None,
        description='Filename for the Python file (including .py extension). If not provided, a random UUID will be used.',
    )
    timeout: int = Field(
        default=30,
        description='Maximum execution time in seconds. Cannot exceed 120 seconds. Default is 30 seconds.',
    )


@tool_registry.register(
    'local-python_execute',
    meta={
        'name': 'local-python_execute',
        'description': (
            'Execute Python code directly under the agent workspace via `uv run`. '
            'The code is written to a .py file and executed in the container environment. '
            'Returns stdout, stderr, return code, and execution time. '
            'Use this when you need to run Python scripts, e.g. to interact with cloud SDKs, '
            'process data, or perform calculations.'
        ),
        'tool_type': ToolType.LOCAL,
    },
)
async def python_execute_tool(params: PythonExecuteInput, context: ToolContext) -> ToolResponse:
    # Get agent_workspace from tool_state
    agent_workspace = context.tool_state.get(PYTHON_EXECUTE_WORKSPACE_KEY)
    if not agent_workspace:
        return ToolResponse(
            output=[{'type': 'text', 'text': 'Error: agent workspace not configured for python_execute.'}],
            meta={'success': False, 'exit_code': 1},
        )

    # Get runtime
    ctx = require_context()
    runtime = await ctx.get_runtime()

    # Prepare filename — sanitise to prevent directory traversal
    filename = params.filename or f'{uuid.uuid4()}.py'
    filename = os.path.basename(filename)
    if not filename or os.sep in filename or (os.altsep and os.altsep in filename):
        filename = f'{uuid.uuid4()}.py'
    if not filename.endswith('.py'):
        filename += '.py'

    # Clamp timeout
    timeout = min(max(params.timeout, 1), 120)

    runtime_python = context.tool_state.get(PYTHON_EXECUTE_RUNTIME_PYTHON_KEY)
    extra_env = dict(context.tool_state.get(PYTHON_EXECUTE_EXTRA_ENV_KEY) or {})

    # Create .python_tmp dir and write the code file via shell commands
    tmp_dir = f'{agent_workspace}/.python_tmp'
    script_path = f'{tmp_dir}/{filename}'
    quoted_tmp_dir = shlex.quote(tmp_dir)
    quoted_script_path = shlex.quote(script_path)
    # Use base64 to safely transfer arbitrary code content
    code_b64 = base64.b64encode(params.code.encode()).decode()

    # Use stdin pipe to transfer base64 payload, avoiding ARG_MAX limits for large code
    setup_cmd = f'mkdir -p {quoted_tmp_dir} && base64 -d > {quoted_script_path}'
    setup_result = await runtime.run_command(command=setup_cmd, timeout=30, stdin=code_b64)
    if setup_result.return_code != 0:
        return ToolResponse(
            output=[
                {
                    'type': 'text',
                    'text': f'Error writing code file:\nstdout: {setup_result.stdout or ""}\nstderr: {setup_result.stderr or ""}',
                }
            ],
            meta={'success': False, 'exit_code': setup_result.return_code or 1},
        )

    # Execute with the task-pinned interpreter when provided, otherwise via `uv run`
    if runtime_python:
        exec_cmd = f'{shlex.quote(str(runtime_python))} {shlex.quote(f"./.python_tmp/{filename}")}'
    else:
        exec_cmd = f'uv run --directory {shlex.quote(agent_workspace)} {shlex.quote(f"./.python_tmp/{filename}")}'
    result = await runtime.run_command(
        command=exec_cmd,
        cwd=agent_workspace,
        extra_env=extra_env,
        timeout=float(timeout),
    )

    # Build output
    output_parts: list[str] = []
    if result.stdout:
        output_parts.append('=== STDOUT ===')
        output_parts.append(result.stdout.rstrip())
    if result.stderr:
        output_parts.append('=== STDERR ===')
        output_parts.append(result.stderr.rstrip())

    output_parts.append('=== EXECUTION INFO ===')
    output_parts.append(f'Return code: {result.return_code}')
    if result.execution_time is not None:
        output_parts.append(f'Execution time: {result.execution_time:.3f} seconds')
    output_parts.append(f'Timeout limit: {timeout} seconds')

    if not result.stdout and not result.stderr:
        output_parts.insert(0, 'No console output produced.')

    is_success = result.return_code == 0
    text = '\n'.join(output_parts)

    return ToolResponse(
        output=[{'type': 'text', 'text': text}],
        meta={
            'success': is_success,
            'exit_code': result.return_code or 0,
            'stdout': result.stdout or '',
            'stderr': result.stderr or '',
        },
    )
