"""Runtime driver for single turns and native Codex goals."""

from __future__ import annotations

import asyncio
import contextlib
import time

from typing import Any

from nexus.runtime.executors.command import active_command_timeout
from nexus.utils.shell import get_os_env_without_ld_library_path

from .app_server_goal_session import AppServerGoalSession
from .app_server_protocol import AppServerClient
from .app_server_turn_session import AppServerTurnSession
from .events import CodexEventStream, reduce_app_server_events


async def _accept_approval(_params: dict[str, Any]) -> dict[str, str]:
    return {'decision': 'accept'}


async def _grant_permissions(params: dict[str, Any]) -> dict[str, Any]:
    permissions = {key: value for key, value in params['permissions'].items() if value is not None}
    return {'permissions': permissions, 'scope': 'turn'}


async def _skip_user_input(_params: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {'answers': {}}


async def _decline_elicitation(_params: dict[str, Any]) -> dict[str, Any]:
    return {'action': 'decline'}


async def _reject_dynamic_tool(_params: dict[str, Any]) -> dict[str, Any]:
    return {
        'success': False,
        'contentItems': [{'type': 'inputText', 'text': 'Nexus has no dynamic tool handler'}],
    }


def _register_noninteractive_request_handlers(client: AppServerClient) -> None:
    """Install the default handlers used by non-interactive Nexus runs."""
    client.register_request_handler('item/commandExecution/requestApproval', _accept_approval)
    client.register_request_handler('item/fileChange/requestApproval', _accept_approval)
    client.register_request_handler('item/permissions/requestApproval', _grant_permissions)
    client.register_request_handler('item/tool/requestUserInput', _skip_user_input)
    client.register_request_handler('mcpServer/elicitation/request', _decline_elicitation)
    client.register_request_handler('item/tool/call', _reject_dynamic_tool)


async def run_codex_app_server(
    *,
    launch_argv: list[str],
    working_dir: str,
    env: dict[str, str],
    task: str,
    timeout: float,
    timeout_control_id: str | None = None,
    model: str | None,
    extra_cli_args: list[str] | None,
    developer_instructions: str | None,
    goal_mode: bool = False,
    goal_token_budget: int | None = None,
) -> dict[str, Any]:
    """Run one app-server thread as a single turn or native goal."""
    command = [*launch_argv, 'app-server']
    if extra_cli_args:
        command.extend(extra_cli_args)
    if goal_mode:
        command.extend(['-c', 'features.goals=true'])
    command.extend(['--listen', 'stdio://'])

    process_env = {**get_os_env_without_ld_library_path(), **env}
    start_time = time.time()
    client = None
    session = None
    timed_out = False
    failure_kind = failure_stage = None
    try:
        client = await AppServerClient.start(command, cwd=working_dir, env=process_env)
        session = (
            AppServerGoalSession(client, token_budget=goal_token_budget) if goal_mode else AppServerTurnSession(client)
        )
        _register_noninteractive_request_handlers(client)
        async with active_command_timeout(timeout, timeout_control_id):
            async with session:
                await session.start_thread(
                    working_dir=working_dir,
                    model=model,
                    developer_instructions=developer_instructions,
                )
                result = await session.run(task)
        exit_code = 0 if result.success else 1
        error = result.error
        if exit_code != 0:
            failure_kind, failure_stage = 'agent', 'process'
    except (asyncio.TimeoutError, TimeoutError):
        if session is not None:
            with contextlib.suppress(Exception):
                await session.interrupt()
        timed_out, exit_code, error = True, -1, 'Execution timed out'
        failure_kind, failure_stage = 'agent', 'process'
    except Exception as exc:
        exit_code, error = -1, f'Codex app-server protocol error: {exc}'
        failure_kind = 'infrastructure'
        failure_stage = 'runtime_executor' if client is None else 'runtime_protocol'
    finally:
        if client is not None:
            await client.close()

    event_stream = client.event_stream if client is not None else CodexEventStream(source='app_server')
    reduction = reduce_app_server_events(
        event_stream.events,
        thread_id=session.thread_id if session is not None else None,
        turn_id=session.turn_id if session is not None else None,
    )
    stderr = '\n'.join(line for line in [error, *(client.stderr_lines if client is not None else [])] if line)
    return {
        'success': exit_code == 0,
        'output': reduction.final_message,
        'stderr': stderr,
        'exit_code': exit_code,
        'execution_time': time.time() - start_time,
        'timed_out': timed_out,
        **({'failure_kind': failure_kind, 'failure_stage': failure_stage} if failure_kind is not None else {}),
        'json_events_stdout': '',
        'event_stream': event_stream.model_dump(mode='python'),
    }


__all__ = ['run_codex_app_server']
