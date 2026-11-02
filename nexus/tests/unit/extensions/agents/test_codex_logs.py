from __future__ import annotations

import asyncio
import os
import shlex
import shutil
import sys

from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from nexus.extensions.agents.codex import runtime_setup
from nexus.extensions.agents.installed_common.execution import AgentExecutionTransport, PreparedExecution
from nexus.runtime.executors.models.command import CommandRunStatus, RunCommandResponse


_CLI = """import os, pathlib, sys
if sys.argv[1:] == ['--version']:
    print('codex-cli 0.158.0')
    sys.exit(0)
home = pathlib.Path(os.environ['CODEX_HOME'])
if '--ephemeral' not in sys.argv:
    sessions = home / 'sessions' / '2026' / '09' / '24'
    sessions.mkdir(parents=True, exist_ok=True)
    (sessions / 'rollout-test.jsonl').write_text('{"type":"session_meta"}\\n')
(home / 'auth.json').write_text('do-not-export')
(home / 'last_message.txt').write_text('done', encoding='utf-8')
os.write(1, b'{"type":"turn.started"}\\n')
os.write(2, b'cli warning\\n')
os.write(1, b'{"type":"turn.completed"}\\n')
sys.exit(int(os.environ.get('TEST_EXIT_CODE', '0')))
"""
_LOG = '{"type":"turn.started"}\ncli warning\n{"type":"turn.completed"}\n'


@pytest.mark.parametrize('shell_environment', ['inherited', 'startup-script', 'xtrace'])
@pytest.mark.parametrize('host_locale', ['C', 'nexus_invalid_locale'], ids=['c-locale', 'missing-locale'])
@pytest.mark.parametrize('install_source', ['url', 'preinstalled'])
@pytest.mark.parametrize('export_logs', [False, True])
@pytest.mark.parametrize('compose', [False, True])
@pytest.mark.parametrize('exit_code', [0, 7])
async def test_codex_exports_native_files_without_changing_exit_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    export_logs: bool,
    compose: bool,
    exit_code: int,
    install_source: str,
    host_locale: str,
    shell_environment: str,
) -> None:
    monkeypatch.setenv('LC_ALL', host_locale)
    if shell_environment == 'startup-script':
        startup_script = tmp_path / 'shell-startup.sh'
        startup_script.write_text('printf "inherited shell startup warning\\n" >&2\n', encoding='utf-8')
        monkeypatch.setenv('BASH_ENV', str(startup_script))
    elif shell_environment == 'xtrace':
        monkeypatch.setenv('SHELLOPTS', 'xtrace')
        monkeypatch.setenv('PS4', 'inherited shell trace: ')
    cli = tmp_path / 'fake codex'
    cli.write_text(f'#!{sys.executable}\n{_CLI}', encoding='utf-8')
    cli.chmod(0o755)
    captured: dict[str, Any] = {}

    async def install(*_args: Any, **_kwargs: Any) -> Path:
        return cli

    class Proxy:
        async def start(self) -> None:
            pass

        async def stop(self) -> None:
            captured['stopped'] = True

    async def run_command(request: Any) -> RunCommandResponse:
        captured['request'] = request
        # Test the CLI's streams, not inherited CI shell startup scripts or tracing options.
        env = {**os.environ, **request.extra_env, 'LC_ALL': 'C', 'BASH_ENV': '/dev/null'}
        for name in ('ENV', 'SHELLOPTS', 'BASHOPTS', 'BASH_XTRACEFD', 'PS4'):
            env.pop(name, None)
        process = await asyncio.create_subprocess_exec(
            '/bin/bash',
            '--noprofile',
            '--norc',
            '-c',
            request.command,
            cwd=request.cwd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        return RunCommandResponse(
            status=CommandRunStatus.Finished,
            return_code=process.returncode,
            stdout=stdout.decode(),
            stderr=stderr.decode(),
            execution_time=0.1,
        )

    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setattr('nexus.extensions.agents.codex.installer.install_codex_from_url', install)
    monkeypatch.setattr('nexus.runtime.context.require_runtime_control', lambda: SimpleNamespace(sio=object()))
    monkeypatch.setattr('nexus.runtime.executors.command.run_command', run_command)
    monkeypatch.setattr('nexus.runtime.server.services.responses_proxy.ResponsesProxy', lambda **_kw: Proxy())
    monkeypatch.setattr(runtime_setup, 'get_current_span_id', lambda: 'test-budget')
    monkeypatch.setattr(runtime_setup, 'is_windows', lambda: False)

    target = None
    if compose:
        from nexus.extensions.agents.installed_common.compose import ComposeExecutionTransport

        target = {
            'kind': 'compose',
            'project_name': 'test',
            'compose_files': ['/tmp/compose.yaml'],
            'service': 'main',
            'transport_host_dir': '/tmp/nxa/test',
            'transport_container_dir': '/nexus-agent',
            'socket_bridge_host_path': '/tmp/nxa/test/bin/bridge',
            'socket_bridge_container_path': '/nexus-agent/bin/bridge',
        }
        monkeypatch.setattr(AgentExecutionTransport, 'validate', lambda self: None)

        async def initialize(self):
            self._identity = {
                'home': str(tmp_path),
                'user': 'test',
                'uid': 1000,
                'gid': 1000,
                'path': os.environ.get('PATH', ''),
            }

        monkeypatch.setattr(AgentExecutionTransport, 'initialize', initialize)

        async def file(self, operation, path, **kwargs):
            from nexus.extensions.agents.installed_common.target_files import operate

            return operate({'operation': operation, 'path': str(path), **kwargs})

        async def copy_directory_to_local(self, source, destination):
            if source.is_dir():
                shutil.copytree(source, destination, dirs_exist_ok=True)

        monkeypatch.setattr(AgentExecutionTransport, 'file', file)
        monkeypatch.setattr(AgentExecutionTransport, 'copy_directory_to_local', copy_directory_to_local)

        @asynccontextmanager
        async def prepare(self, command, *, working_dir, env, **_kw):
            captured['compose_service'] = self.target.service
            yield PreparedExecution(command=command, cwd=working_dir, env=env)

        monkeypatch.setattr(ComposeExecutionTransport, 'prepare', prepare)

        async def exec_as_agent(self, command, **kwargs):
            assert command == [str(cli), '--version']
            return await run_command(SimpleNamespace(command=shlex.join(command), cwd='/', extra_env={}))

        monkeypatch.setattr(AgentExecutionTransport, 'exec_as_agent', exec_as_agent)

    logs_dir = tmp_path / "logs with 'quote'"
    if install_source == 'preinstalled':
        monkeypatch.setattr('nexus.extensions.agents.codex.installer.install_codex_cli', install)
        monkeypatch.setattr(
            runtime_setup, 'find_nexus_runtime_binary', Mock(side_effect=AssertionError('must not need private Node'))
        )
        monkeypatch.setattr(
            runtime_setup, 'resolve_npm_bin_entry', Mock(side_effect=AssertionError('must not need private npm entry'))
        )
    result = await runtime_setup.setup_and_run_codex(
        task='test log export',
        working_dir=str(tmp_path),
        binary_url='https://example.invalid/codex' if install_source == 'url' else None,
        execution_target=target,
        extra_env={'TEST_EXIT_CODE': str(exit_code)},
        logs_dir=str(logs_dir) if export_logs else None,
    )

    assert result['exit_code'] == exit_code
    assert result['success'] is (exit_code == 0)
    assert result['output'] == 'done'
    assert captured['stopped'] is True
    # Never ephemeral, even with nothing to export: multi_agent_version=v2 rebuilds a subagent's
    # parent context from the persistent thread store, and without it spawning dies with
    # `collab spawn failed: no thread with id: <root thread id>`.
    assert '--ephemeral' not in shlex.split(captured['request'].command)
    if compose:
        assert captured['compose_service'] == 'main'
    if export_logs:
        assert (logs_dir / 'codex.txt').read_text() == _LOG
        assert (logs_dir / 'sessions/2026/09/24/rollout-test.jsonl').read_text() == '{"type":"session_meta"}\n'
        assert not (logs_dir / 'auth.json').exists()
        assert not (logs_dir / 'config.toml').exists()
        assert result['json_events_stdout'] == _LOG
    else:
        assert not logs_dir.exists()
        assert result['stderr'] == 'cli warning\n'


@pytest.mark.parametrize('copy_fails', [False, True])
@pytest.mark.parametrize('cancelled', [False, True])
async def test_codex_exports_partial_sessions_on_timeout_or_cancellation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cancelled: bool, copy_fails: bool
) -> None:
    cli = tmp_path / 'codex'
    cli.touch()
    stopped = []

    async def install(*_args: Any, **_kwargs: Any) -> Path:
        return cli

    class Proxy:
        async def start(self) -> None:
            pass

        async def stop(self) -> None:
            stopped.append(True)

    async def run_command(request: Any) -> RunCommandResponse:
        sessions = Path(request.extra_env['CODEX_HOME']) / 'sessions/2026/09/24'
        sessions.mkdir(parents=True)
        (sessions / 'rollout-partial.jsonl').write_text('partial')
        if cancelled:
            raise asyncio.CancelledError
        return RunCommandResponse(status=CommandRunStatus.TimeLimitExceeded, stdout='partial', stderr='timeout')

    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setattr('nexus.extensions.agents.codex.installer.install_codex_from_url', install)
    monkeypatch.setattr('nexus.runtime.context.require_runtime_control', lambda: SimpleNamespace(sio=object()))
    monkeypatch.setattr('nexus.runtime.executors.command.run_command', run_command)
    monkeypatch.setattr('nexus.runtime.server.services.responses_proxy.ResponsesProxy', lambda **_kw: Proxy())
    monkeypatch.setattr(runtime_setup, 'get_current_span_id', lambda: 'test-budget')
    monkeypatch.setattr(runtime_setup, 'is_windows', lambda: False)
    logs_dir = tmp_path / 'logs'
    if copy_fails:
        monkeypatch.setattr(shutil, 'copytree', Mock(side_effect=OSError('copy failed')))
    invocation = runtime_setup.setup_and_run_codex(
        task='test timeout',
        working_dir=str(tmp_path),
        version='0.114.0',
        binary_url='https://example.invalid/codex',
        logs_dir=str(logs_dir),
    )
    if cancelled:
        with pytest.raises(asyncio.CancelledError):
            await invocation
    else:
        result = await invocation
        assert result['timed_out'] is True
        assert result['success'] is False
    if not copy_fails:
        assert (logs_dir / 'sessions/2026/09/24/rollout-partial.jsonl').read_text() == 'partial'
    assert stopped == [True]


async def test_codex_rejects_ephemeral_with_log_export(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(runtime_setup, 'is_windows', lambda: False)
    with pytest.raises(ValueError, match='remove --ephemeral'):
        await runtime_setup.setup_and_run_codex(task='test', logs_dir=str(tmp_path), extra_cli_args=['--ephemeral'])


async def test_codex_rejects_app_server_with_log_export(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='app_server mode does not support logs_dir'):
        await runtime_setup.setup_and_run_codex(task='test', mode='app_server', logs_dir=str(tmp_path))
