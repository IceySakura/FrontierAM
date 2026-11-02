from argparse import Namespace
from unittest.mock import AsyncMock, patch

import aiohttp

from nexus.runtime.executors.models.command import CommandRunStatus, RunCommandResponse


def _args(command: str = 'ls') -> Namespace:
    return Namespace(
        base_url='http://10.0.0.1:32000',
        exec_command=command,
        cwd=None,
        timeout=180,
    )


def test_exec_prints_output_and_closes_all_http_clients(capsys) -> None:
    response = RunCommandResponse(
        status=CommandRunStatus.Finished,
        return_code=0,
        stdout='hello\n',
        stderr='',
    )

    with (
        patch('nexus.runtime.client.remote.RemoteRuntime.run_command', new_callable=AsyncMock) as run_command,
        patch('nexus.utils.async_http.AsyncHTTP.close_all', new_callable=AsyncMock) as close_all,
    ):
        run_command.return_value = response

        from nexus.cli.commands.sandbox import sandbox_exec

        exit_code = sandbox_exec(_args('echo hello'))

    assert exit_code == 0
    assert capsys.readouterr().out == 'hello\n'
    close_all.assert_awaited_once()


def test_exec_closes_all_http_clients_after_request_error(capsys) -> None:
    with (
        patch('nexus.runtime.client.remote.RemoteRuntime.run_command', new_callable=AsyncMock) as run_command,
        patch('nexus.utils.async_http.AsyncHTTP.close_all', new_callable=AsyncMock) as close_all,
    ):
        run_command.side_effect = aiohttp.ClientError('connection refused')

        from nexus.cli.commands.sandbox import sandbox_exec

        exit_code = sandbox_exec(_args())

    assert exit_code == 1
    assert 'cannot reach sandbox' in capsys.readouterr().err
    close_all.assert_awaited_once()
