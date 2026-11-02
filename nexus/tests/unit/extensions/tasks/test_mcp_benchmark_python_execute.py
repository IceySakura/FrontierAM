from __future__ import annotations

import pytest

from nexus.core import ToolContext
from nexus.extensions.tasks.mcp_benchmark.tools import python_execute as python_execute_module
from nexus.runtime.executors.models.command import CommandRunStatus, RunCommandResponse


class _FakeRuntime:
    def __init__(self) -> None:
        self.commands: list[dict[str, object]] = []
        self._responses = [
            RunCommandResponse(status=CommandRunStatus.Finished, return_code=0),
            RunCommandResponse(status=CommandRunStatus.Finished, return_code=0, stdout='ok\n', execution_time=0.5),
        ]

    async def run_command(self, command: str, cwd: str | None = None, timeout: float = 0, **kwargs) -> RunCommandResponse:
        self.commands.append({'command': command, 'cwd': cwd, 'timeout': timeout, 'kwargs': kwargs})
        return self._responses[len(self.commands) - 1]


class _FakeExecutionContext:
    def __init__(self, runtime: _FakeRuntime) -> None:
        self._runtime = runtime

    async def get_runtime(self) -> _FakeRuntime:
        return self._runtime


@pytest.mark.asyncio
async def test_local_python_execute_uses_toolathlon_runtime_python(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _FakeRuntime()
    monkeypatch.setattr(python_execute_module, 'require_context', lambda: _FakeExecutionContext(runtime))

    response = await python_execute_module.python_execute_tool(
        python_execute_module.PythonExecuteInput(code='import openpyxl\nprint("ok")', filename='verify.py'),
        ToolContext(
            tool_state={
                python_execute_module.PYTHON_EXECUTE_WORKSPACE_KEY: '/workspace/case/agent_workspace',
                python_execute_module.PYTHON_EXECUTE_RUNTIME_PYTHON_KEY: '/opt/mcp-benchmarks/datasets/toolathlon/.venv/bin/python',
                python_execute_module.PYTHON_EXECUTE_EXTRA_ENV_KEY: {'GOOGLE_APPLICATION_CREDENTIALS': '/tmp/creds.json'},
            }
        ),
    )

    assert response.meta['success'] is True
    assert runtime.commands[1]['command'] == '/opt/mcp-benchmarks/datasets/toolathlon/.venv/bin/python ./.python_tmp/verify.py'
    assert runtime.commands[1]['cwd'] == '/workspace/case/agent_workspace'
    assert runtime.commands[1]['kwargs']['extra_env'] == {'GOOGLE_APPLICATION_CREDENTIALS': '/tmp/creds.json'}


@pytest.mark.asyncio
async def test_local_python_execute_falls_back_to_uv_run(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _FakeRuntime()
    monkeypatch.setattr(python_execute_module, 'require_context', lambda: _FakeExecutionContext(runtime))

    response = await python_execute_module.python_execute_tool(
        python_execute_module.PythonExecuteInput(code='print("fallback")', filename='fallback.py'),
        ToolContext(
            tool_state={
                python_execute_module.PYTHON_EXECUTE_WORKSPACE_KEY: '/workspace/case/agent_workspace',
            }
        ),
    )

    assert response.meta['success'] is True
    assert runtime.commands[1]['command'] == 'uv run --directory /workspace/case/agent_workspace ./.python_tmp/fallback.py'
    assert runtime.commands[1]['cwd'] == '/workspace/case/agent_workspace'
    assert runtime.commands[1]['kwargs']['extra_env'] == {}
