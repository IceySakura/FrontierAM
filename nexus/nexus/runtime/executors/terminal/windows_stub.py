"""Import-safe Windows stubs for the POSIX terminal session executors."""

from typing import Any

from nexus.core import executor_registry
from nexus.runtime.executors.models.terminal import (
    CapturePaneRequest,
    CapturePaneResponse,
    CommandInfo,
    CreateTerminalSessionRequest,
    CreateTerminalSessionResponse,
    DestroyTerminalSessionRequest,
    DestroyTerminalSessionResponse,
    QueryCommandStatusRequest,
    RunCommandInSessionRequest,
    SendKeysToSessionRequest,
    SendKeysToSessionResponse,
    StartCommandInSessionRequest,
    TerminateSessionProcessesRequest,
    TerminateSessionProcessesResponse,
)


_ERROR = 'Terminal sessions are not implemented on Windows'
TERMINAL_SESSIONS: dict[str, Any] = {}


class TerminalSession:
    """Placeholder preserving the public import surface on Windows."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise NotImplementedError(_ERROR)


@executor_registry.register('create_terminal_session')
async def create_terminal_session(_request: CreateTerminalSessionRequest) -> CreateTerminalSessionResponse:
    raise NotImplementedError(_ERROR)


@executor_registry.register('destroy_terminal_session')
async def destroy_terminal_session(_request: DestroyTerminalSessionRequest) -> DestroyTerminalSessionResponse:
    raise NotImplementedError(_ERROR)


@executor_registry.register('run_command_in_terminal_session')
async def run_command_in_session(_request: RunCommandInSessionRequest) -> CommandInfo:
    raise NotImplementedError(_ERROR)


@executor_registry.register('start_command_in_terminal_session')
async def start_command_in_session(_request: StartCommandInSessionRequest) -> CommandInfo:
    raise NotImplementedError(_ERROR)


@executor_registry.register('query_terminal_command_status')
async def query_command_status(_request: QueryCommandStatusRequest) -> CommandInfo:
    raise NotImplementedError(_ERROR)


@executor_registry.register('send_keys_to_terminal_session')
async def send_keys_to_session(_request: SendKeysToSessionRequest) -> SendKeysToSessionResponse:
    raise NotImplementedError(_ERROR)


@executor_registry.register('capture_terminal_pane')
async def capture_pane(_request: CapturePaneRequest) -> CapturePaneResponse:
    raise NotImplementedError(_ERROR)


@executor_registry.register('terminate_terminal_session_processes')
async def terminate_session_processes(
    _request: TerminateSessionProcessesRequest,
) -> TerminateSessionProcessesResponse:
    raise NotImplementedError(_ERROR)


__all__ = [
    'TERMINAL_SESSIONS',
    'TerminalSession',
    'capture_pane',
    'create_terminal_session',
    'destroy_terminal_session',
    'query_command_status',
    'run_command_in_session',
    'send_keys_to_session',
    'start_command_in_session',
    'terminate_session_processes',
]
