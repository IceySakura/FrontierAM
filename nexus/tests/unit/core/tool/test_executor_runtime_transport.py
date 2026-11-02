"""Transport-error classification at the executor's RUNTIME dispatch boundary.

A dead sandbox transport surfaces as raw OS connection errors, but those same
exception types can come from a tool's own business logic (a LOCAL tool
talking to an external service, an MCP tool whose backend reset the
connection). The executor is the only dispatch point that knows which branch
a call took, so it must wrap the sandbox communication errors of its RUNTIME
branch into ``RuntimeTransportError`` and leave LOCAL/MCP errors untouched --
and, within the RUNTIME branch, leave untouched any exception that arrived
*inside* the InvokeResponse (unpickled by ``unpack_response``): a complete
response proves the transport was alive, so that exception is the sandbox
tool's own error, whatever its type.
"""

from __future__ import annotations

import uuid

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from pydantic import BaseModel

from nexus.core.context import ExecutionContext
from nexus.core.tool import ToolContext, ToolExecutor, ToolMeta, ToolResponse, ToolType
from nexus.core.tool.registry import tool_registry
from nexus.runtime.client.base import RuntimeTransportError


class _Input(BaseModel):
    value: str


def _make_tool(name: str, tool_type: ToolType):
    """A tool whose function raises a connection error of its own."""

    async def run(params: _Input, _context: ToolContext) -> ToolResponse:
        raise ConnectionResetError(f'{name} own socket died')

    run.__tool_input_model__ = _Input
    run.__tool_meta__ = ToolMeta(
        name=name,
        description='Raise a connection error',
        tool_type=tool_type,
        input_model=_Input,
        func=run,
    )
    run.nexus_meta = {'type': 'tool', 'name': name}
    return run


def _register_runtime_tool(name: str) -> str:
    """Give *name* a lazy registry entry so _call_runtime finds an import path.

    The entry is never imported: the mocked runtime receives the import path
    and raises before anything is loaded.
    """
    tool_registry.add(name, f'tests.unit.core.tool._missing_{name}::run')
    return name


@pytest.mark.asyncio
async def test_runtime_tool_os_error_is_wrapped(monkeypatch):
    """A connection reset on the sandbox call becomes RuntimeTransportError."""
    name = _register_runtime_tool(f'runtime_probe_{uuid.uuid4().hex[:8]}')
    runtime = SimpleNamespace(call_tool=AsyncMock(side_effect=ConnectionResetError('connection reset by peer')))
    monkeypatch.setattr(ExecutionContext, 'get_runtime', AsyncMock(return_value=runtime))

    executor = ToolExecutor([_make_tool(name, ToolType.RUNTIME)])
    context = ExecutionContext(openai_clients={})
    async with context:
        with pytest.raises(RuntimeTransportError) as exc_info:
            await executor.call(name, {'value': 'x'})

    assert isinstance(exc_info.value.__cause__, ConnectionResetError)
    assert name in str(exc_info.value)
    runtime.call_tool.assert_awaited_once()


@pytest.mark.asyncio
async def test_runtime_tool_broken_pipe_is_wrapped(monkeypatch):
    """BrokenPipeError / ConnectionAbortedError wrap identically."""
    for os_error in (BrokenPipeError('pipe broken'), ConnectionAbortedError('aborted')):
        name = _register_runtime_tool(f'runtime_probe_{uuid.uuid4().hex[:8]}')
        runtime = SimpleNamespace(call_tool=AsyncMock(side_effect=os_error))
        monkeypatch.setattr(ExecutionContext, 'get_runtime', AsyncMock(return_value=runtime))

        executor = ToolExecutor([_make_tool(name, ToolType.RUNTIME)])
        context = ExecutionContext(openai_clients={})
        async with context:
            with pytest.raises(RuntimeTransportError) as exc_info:
                await executor.call(name, {'value': 'x'})

        assert isinstance(exc_info.value.__cause__, type(os_error))


@pytest.mark.asyncio
async def test_remote_exception_inside_response_is_not_wrapped(monkeypatch):
    """A sandbox tool's own connection error arrived inside the InvokeResponse.

    ``runtime.call_tool`` replays what RemoteRuntime does end to end: the
    tool's ConnectionResetError was pickled into ``InvokeResponse.exception``
    on the sandbox, and ``unpack_response`` re-raises it on the host. A
    complete response came back, so the transport was alive -- the error is
    the tool's own business error and must stay raw (recoverable via
    tool_error_continue), not become a fatal RuntimeTransportError.
    """
    from nexus.runtime.models import invoke_response_from_exception, unpack_response

    response = invoke_response_from_exception(ConnectionResetError('tool inside sandbox lost its own socket'))
    name = _register_runtime_tool(f'runtime_probe_{uuid.uuid4().hex[:8]}')

    async def _call_tool_like_remote_runtime(*_args, **_kwargs):
        # RemoteRuntime._resolve_response -> unpack_response: the remote
        # exception is unpickled and re-raised (marked as remote).
        return unpack_response(response)

    runtime = SimpleNamespace(call_tool=AsyncMock(side_effect=_call_tool_like_remote_runtime))
    monkeypatch.setattr(ExecutionContext, 'get_runtime', AsyncMock(return_value=runtime))

    executor = ToolExecutor([_make_tool(name, ToolType.RUNTIME)])
    context = ExecutionContext(openai_clients={})
    async with context:
        with pytest.raises(ConnectionResetError, match='tool inside sandbox lost its own socket'):
            await executor.call(name, {'value': 'x'})

    runtime.call_tool.assert_awaited_once()


@pytest.mark.asyncio
async def test_local_runtime_os_error_is_wrapped_for_now(monkeypatch):
    """Known limitation, accepted in MR review: core does not special-case
    in-process runtimes (it must not import LocalRuntime), so a local tool's
    own ConnectionResetError -- raised straight out of ``LocalRuntime._invoke``,
    never pickled into an InvokeResponse, so it carries no remote-exception
    marker -- currently classifies as a transport failure too. If a local
    provider ever needs tool_error_continue for its tools' own socket errors,
    the exemption belongs behind a runtime-side abstraction, not an isinstance
    of a concrete class in core.
    """
    from nexus.runtime.client.local import LocalRuntime

    class _RaisingLocalRuntime(LocalRuntime):
        async def _invoke(self, request):  # noqa: ARG002
            raise ConnectionResetError('local tool lost its own external socket')

    name = _register_runtime_tool(f'runtime_probe_{uuid.uuid4().hex[:8]}')
    runtime = _RaisingLocalRuntime()
    monkeypatch.setattr(ExecutionContext, 'get_runtime', AsyncMock(return_value=runtime))

    executor = ToolExecutor([_make_tool(name, ToolType.RUNTIME)])
    context = ExecutionContext(openai_clients={})
    async with context:
        with pytest.raises(RuntimeTransportError) as exc_info:
            await executor.call(name, {'value': 'x'})

    assert isinstance(exc_info.value.__cause__, ConnectionResetError)


@pytest.mark.asyncio
async def test_transport_os_error_outside_a_response_still_wrapped(monkeypatch):
    """The exclusion is narrow: an OS error raised by the transport itself
    (no response ever arrived) still becomes RuntimeTransportError."""
    name = _register_runtime_tool(f'runtime_probe_{uuid.uuid4().hex[:8]}')
    runtime = SimpleNamespace(call_tool=AsyncMock(side_effect=BrokenPipeError('http layer died mid-request')))
    monkeypatch.setattr(ExecutionContext, 'get_runtime', AsyncMock(return_value=runtime))

    executor = ToolExecutor([_make_tool(name, ToolType.RUNTIME)])
    context = ExecutionContext(openai_clients={})
    async with context:
        with pytest.raises(RuntimeTransportError) as exc_info:
            await executor.call(name, {'value': 'x'})

    assert isinstance(exc_info.value.__cause__, BrokenPipeError)


@pytest.mark.asyncio
async def test_local_tool_os_error_is_not_wrapped():
    """A LOCAL tool raising ConnectionResetError keeps the raw OS error.

    This is the reviewer's scenario: a local tool whose own access to an
    external service is reset must not be misread as a dead sandbox transport.
    """
    executor = ToolExecutor([_make_tool('local_probe', ToolType.LOCAL)])
    context = ExecutionContext(openai_clients={})
    async with context:
        with pytest.raises(ConnectionResetError, match='local_probe own socket died'):
            await executor.call('local_probe', {'value': 'x'})


@pytest.mark.asyncio
async def test_mcp_tool_os_error_is_not_wrapped():
    """An MCP tool whose backend resets the connection keeps the raw OS error."""
    manager = SimpleNamespace(
        tools={
            'mcp_probe': SimpleNamespace(
                full_name='mcp_probe',
                tool_name='probe',
                description='',
                input_schema={},
                server_name='srv',
            )
        },
        call_tool=AsyncMock(side_effect=ConnectionResetError('mcp backend died')),
        close=AsyncMock(),
    )
    executor = ToolExecutor([])
    context = ExecutionContext(openai_clients={})
    async with context:
        context._mcp_manager = manager
        with pytest.raises(ConnectionResetError, match='mcp backend died'):
            await executor.call('mcp_probe', {'value': 'x'})
