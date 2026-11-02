import asyncio
import re
import subprocess
import sys

import pytest

from nexus.core import ToolContext
from nexus.extensions.tools.code_interpreter import jupyter_code_interpreter as jupyter_module
from nexus.extensions.tools.code_interpreter.jupyter_code_interpreter import (
    JupyterCodeInterpreterInput,
    jupyter_code_interpreter_tool,
)
from nexus.extensions.tools.code_interpreter.python_code_interpreter import (
    PythonCodeInterpreterInput,
    python_code_interpreter_tool,
    safe_python_code_interpreter_tool,
)


@pytest.fixture(autouse=True)
async def cleanup_jupyter_runners():
    # Keep tests isolated and avoid leaking cached kernels across test cases.
    await jupyter_module._clear_all_kernel_runners()
    yield
    await jupyter_module._clear_all_kernel_runners()


class TestJupyterCodeInterpreter:
    async def test_empty_code_returns_error(self):
        ctx = ToolContext(tool_configs={})
        resp = await jupyter_code_interpreter_tool(JupyterCodeInterpreterInput(code='   '), ctx)
        assert resp.meta.get('success') is False
        assert 'Empty code' in resp.output[0]['text']

    async def test_executes_code(self):
        ctx = ToolContext(tool_configs={})
        resp = await jupyter_code_interpreter_tool(JupyterCodeInterpreterInput(code='print(1+1)'), ctx)
        assert resp.meta.get('success') is True
        assert 'Successfully executed' in resp.output[0]['text']
        assert '2' in resp.output[0]['text']

    async def test_persists_state_across_calls(self):
        ctx = ToolContext(tool_configs={})
        await jupyter_code_interpreter_tool(JupyterCodeInterpreterInput(code='x = 10'), ctx)
        resp = await jupyter_code_interpreter_tool(JupyterCodeInterpreterInput(code='print(x + 5)'), ctx)
        assert resp.meta.get('success') is True
        assert '15' in resp.output[0]['text']

    async def test_extracts_python_code_block(self):
        ctx = ToolContext(tool_configs={})
        code = '```python\nprint(3)\n```'
        resp = await jupyter_code_interpreter_tool(JupyterCodeInterpreterInput(code=code), ctx)
        assert resp.meta.get('success') is True
        assert re.search(r'\b3\b', resp.output[0]['text'])

    async def test_reports_stdout_and_stderr(self):
        ctx = ToolContext(tool_configs={})
        code = "import sys\nprint('out')\nprint('err', file=sys.stderr)"
        resp = await jupyter_code_interpreter_tool(JupyterCodeInterpreterInput(code=code), ctx)
        assert resp.meta.get('success') is True
        text = resp.output[0]['text']
        assert '--- stdout ---' in text
        assert '--- stderr ---' in text
        assert 'out' in text
        assert 'err' in text

    async def test_reports_exceptions(self):
        ctx = ToolContext(tool_configs={})
        resp = await jupyter_code_interpreter_tool(JupyterCodeInterpreterInput(code="raise ValueError('boom')"), ctx)
        assert resp.meta.get('success') is False
        assert 'Error executing code' in resp.output[0]['text']


class TestPythonCodeInterpreter:
    async def test_empty_code_returns_error(self):
        ctx = ToolContext(tool_configs={})
        resp = await python_code_interpreter_tool(PythonCodeInterpreterInput(code='   '), ctx)
        assert resp.meta.get('success') is False
        assert 'Empty code' in resp.output[0]['text']

    async def test_executes_code(self):
        ctx = ToolContext(tool_configs={})
        resp = await python_code_interpreter_tool(PythonCodeInterpreterInput(code='print(1+1)'), ctx)
        assert resp.meta.get('success') is True
        assert 'Successfully executed' in resp.output[0]['text']
        assert '2' in resp.output[0]['text']

    async def test_persists_state_when_timeout_none(self):
        ctx = ToolContext(tool_configs={})
        await python_code_interpreter_tool(PythonCodeInterpreterInput(code='x = 7', timeout=None), ctx)
        resp = await python_code_interpreter_tool(PythonCodeInterpreterInput(code='print(x + 1)', timeout=None), ctx)
        assert resp.meta.get('success') is True
        assert '8' in resp.output[0]['text']


class TestSafePythonCodeInterpreter:
    async def test_empty_code_returns_error(self):
        ctx = ToolContext(tool_configs={})
        resp = await safe_python_code_interpreter_tool(PythonCodeInterpreterInput(code='   '), ctx)
        assert resp.meta.get('success') is False
        assert resp.output, 'expected non-empty output'
        assert 'text' in resp.output[0], 'expected text key in output'
        assert 'Empty code' in resp.output[0]['text']

    async def test_executes_code(self):
        ctx = ToolContext(tool_configs={})
        resp = await safe_python_code_interpreter_tool(PythonCodeInterpreterInput(code='print(1+1)'), ctx)
        assert resp.meta.get('success') is True
        assert resp.output, 'expected non-empty output'
        assert 'text' in resp.output[0], 'expected text key in output'
        # The safe interpreter returns the bare stdout payload (no "Successfully executed" banner).
        assert resp.output[0]['text'] == 'Stdout: 2\n'
        assert resp.meta.get('stdout') == 'Stdout: 2\n'

    async def test_reports_missing_stdout(self):
        ctx = ToolContext(tool_configs={})
        resp = await safe_python_code_interpreter_tool(PythonCodeInterpreterInput(code='1 + 1'), ctx)
        assert resp.meta.get('success') is True
        assert resp.output, 'expected non-empty output'
        assert 'text' in resp.output[0], 'expected text key in output'
        assert 'No Stdout is generated' in resp.output[0]['text']

    async def test_rejects_unsafe_import(self):
        ctx = ToolContext(tool_configs={})
        resp = await safe_python_code_interpreter_tool(PythonCodeInterpreterInput(code='import os'), ctx)
        assert resp.meta.get('success') is False
        assert resp.meta.get('exit_code') == 1
        assert 'not allowed' in str(resp.meta.get('stderr', ''))
        assert resp.output, 'expected non-empty output'
        assert 'text' in resp.output[0], 'expected text key in output'
        assert 'not allowed' in resp.output[0]['text']


class TestJupyterCodeInterpreterInternals:
    async def test_clear_kernel_runners_removes_current_loop_cache(self):
        loop = asyncio.get_running_loop()
        per_loop_runners = jupyter_module._KERNELS.setdefault(loop, {})
        per_loop_runners[None] = jupyter_module._JupyterKernelRunner(None)

        assert loop in jupyter_module._KERNELS
        await jupyter_module._clear_kernel_runners()
        assert loop not in jupyter_module._KERNELS

    async def test_clear_all_kernel_runners_clears_all_loops(self):
        loop = asyncio.get_running_loop()
        per_loop_runners = jupyter_module._KERNELS.setdefault(loop, {})
        per_loop_runners[None] = jupyter_module._JupyterKernelRunner(None)

        await jupyter_module._clear_all_kernel_runners()
        assert len(jupyter_module._KERNELS) == 0

    async def test_config_timeout_is_used(self, monkeypatch: pytest.MonkeyPatch):
        class _FakeResult:
            status = 'ok'
            stdout = ''
            stderr = ''
            display = []
            error = []

        class _RecordingRunner:
            def __init__(self):
                self.timeout = None

            async def run(self, code: str, timeout: float | None = None):
                self.timeout = timeout
                return _FakeResult()

        runner = _RecordingRunner()
        monkeypatch.setattr(jupyter_module, '_get_kernel_runner', lambda python_path=None: runner)
        ctx = ToolContext(tool_configs={'jupyter_code_interpreter': {'timeout': 12}})
        await jupyter_code_interpreter_tool(JupyterCodeInterpreterInput(code='print(1)'), ctx)
        assert runner.timeout == 12

    async def test_default_timeout_is_used_without_config(self, monkeypatch: pytest.MonkeyPatch):
        class _FakeResult:
            status = 'ok'
            stdout = ''
            stderr = ''
            display = []
            error = []

        class _RecordingRunner:
            def __init__(self):
                self.timeout = None

            async def run(self, code: str, timeout: float | None = None):
                self.timeout = timeout
                return _FakeResult()

        runner = _RecordingRunner()
        monkeypatch.setattr(jupyter_module, '_get_kernel_runner', lambda python_path=None: runner)
        ctx = ToolContext(tool_configs={})
        await jupyter_code_interpreter_tool(JupyterCodeInterpreterInput(code='print(1)'), ctx)
        assert runner.timeout == 30

    async def test_runner_exception_propagates(self, monkeypatch: pytest.MonkeyPatch):
        class _FailingRunner:
            async def run(self, code: str, timeout: float | None = None):
                raise RuntimeError('runner failed')

        monkeypatch.setattr(jupyter_module, '_get_kernel_runner', lambda python_path=None: _FailingRunner())
        ctx = ToolContext(tool_configs={})
        with pytest.raises(RuntimeError, match='runner failed'):
            await jupyter_code_interpreter_tool(JupyterCodeInterpreterInput(code='print(1)'), ctx)

    async def test_runner_serializes_concurrent_runs(self, monkeypatch: pytest.MonkeyPatch):
        class _FakeResult:
            status = 'ok'
            stdout = ''
            stderr = ''
            display = []
            error = []

        class _FakeKernel:
            def __init__(self, python_path: str | None):
                self.started = False
                self.inflight = 0
                self.max_inflight = 0

            async def start(self):
                self.started = True

            async def execute(self, code: str, timeout: float | None = None):
                self.inflight += 1
                self.max_inflight = max(self.max_inflight, self.inflight)
                await asyncio.sleep(0.05)
                self.inflight -= 1
                return _FakeResult()

            async def close(self):
                self.started = False

        monkeypatch.setattr(jupyter_module, 'AsyncDirectPythonKernel', _FakeKernel)
        runner = jupyter_module._JupyterKernelRunner(None)

        await asyncio.gather(
            runner.run('print(1)'),
            runner.run('print(2)'),
        )

        assert isinstance(runner._kernel, _FakeKernel)
        assert runner._kernel.max_inflight == 1


def test_jupyter_executor_imports_without_jupyter_client():
    # Subprocess: the executor registry is process-global, so re-importing in-process duplicates entries.
    probe = "import sys; sys.modules['jupyter_client'] = None; import nexus.runtime.executors.jupyter"
    result = subprocess.run([sys.executable, '-c', probe], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
