import pytest

from nexus.core.tool import ToolContext
from nexus.runtime.executors.command import CommandRunStatus
from nexus.runtime.providers.local import LocalRuntimeProvider


@pytest.mark.asyncio
async def test_local_runtime_provider_runs_hello_world() -> None:
    provider = LocalRuntimeProvider()

    runtime = await provider.acquire('local-test')
    result = await runtime.run_command('echo Hello world')

    assert result.status == CommandRunStatus.Finished
    assert result.return_code == 0
    assert (result.stdout or '').strip() == 'Hello world'

    tool_result = await runtime.call_tool(
        import_path='nexus.extensions.tools.code_interpreter.python_code_interpreter::python_code_interpreter_tool',
        params={'code': "print('Hello world')", 'timeout': 10},
        context=ToolContext(),
    )
    assert tool_result.meta.get('success') is True
    assert any('Hello world' in str(part.get('text', '')) for part in tool_result.output if isinstance(part, dict))

    await provider.release('local-test')


@pytest.mark.asyncio
async def test_local_runtime_provider_runs_safe_python_code_interpreter_tool() -> None:
    provider = LocalRuntimeProvider()
    runtime = await provider.acquire('local-test-safe')

    safe_import_path = (
        'nexus.extensions.tools.code_interpreter.python_code_interpreter::safe_python_code_interpreter_tool'
    )
    safe_tool_result = await runtime.call_tool(
        import_path=safe_import_path,
        params={'code': "print('safe hello')", 'timeout': 10},
        context=ToolContext(),
    )
    assert safe_tool_result.meta.get('success') is True
    assert any('safe hello' in str(part.get('text', '')) for part in safe_tool_result.output if isinstance(part, dict))

    blocked_tool_result = await runtime.call_tool(
        import_path=safe_import_path,
        params={'code': 'import os', 'timeout': 10},
        context=ToolContext(),
    )
    # Blocked code is reported as a failed tool call, with the interpreter message in `stderr`.
    assert blocked_tool_result.meta.get('success') is False
    assert 'not allowed' in str(blocked_tool_result.meta.get('stderr', ''))

    await provider.release('local-test-safe')
