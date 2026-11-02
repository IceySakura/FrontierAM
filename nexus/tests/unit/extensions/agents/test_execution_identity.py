from __future__ import annotations

import json
import os
import subprocess

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from nexus.extensions.agents.installed_common import compose
from nexus.extensions.agents.installed_common.execution import (
    AgentExecutionTransport,
    ComposeExecutionTarget,
    DirectExecutionTarget,
    ExecutionTransportError,
    LegacyAgentExecutionTransport,
    resolve_agent_execution_target,
)
from nexus.runtime.context import execution_user, get_execution_user
from nexus.runtime.executors.models.command import CommandRunStatus, RunCommandResponse


@pytest.fixture
def target():
    return ComposeExecutionTarget(
        project_name='identity-test',
        compose_files=['/task/compose.json'],
        service='main',
        transport_host_dir='/tmp/nx-identity',
        transport_container_dir='/transport',
        socket_bridge_host_path='/tmp/nx-identity/bin/bridge',
        socket_bridge_container_path='/transport/bin/bridge',
        user='alice',
        compose_env={'COMPOSE_VALUE': 'from-controller'},
    )


@pytest.mark.parametrize('user', ['alice', 1000, '1000', 'root', None])
def test_direct_target_roundtrips(user):
    target = DirectExecutionTarget(user=user)
    assert resolve_agent_execution_target(target.model_dump(mode='json')) == target


def test_direct_requires_management_privileges(monkeypatch):
    monkeypatch.setattr(os, 'geteuid', lambda: 1000)
    with pytest.raises(ExecutionTransportError, match='root Worker'):
        AgentExecutionTransport(DirectExecutionTarget(user='alice')).validate()


async def test_compose_account_and_root_commands_are_explicit(monkeypatch, target):
    transport = AgentExecutionTransport(target)
    monkeypatch.setattr(transport, 'validate', lambda: None)
    calls = []

    async def run(request):
        calls.append(request)
        if request.stdin:
            assert json.loads(request.stdin) == {
                'operation': 'prepare_home',
                'path': '/transport/agent-1000',
            }
            return RunCommandResponse(
                status=CommandRunStatus.Finished,
                return_code=0,
                stdout=json.dumps('/transport/agent-1000/home-test'),
            )
        assert get_execution_user() is None
        return RunCommandResponse(
            status=CommandRunStatus.Finished,
            return_code=0,
            stdout=json.dumps({
                'uid': 1000,
                'gid': 1000,
                'home': '/home/alice',
                'user': 'alice',
                'path': '/usr/bin:/bin',
            }),
        )

    monkeypatch.setattr('nexus.runtime.executors.command.run_command', run)
    await transport.initialize()
    assert '--user alice' in calls[0].command
    assert transport.home_dir(invocation_id='ignored', default='/root') == Path('/transport/agent-1000/home-test')
    assert transport.install_dir(agent_name='codex', default='/root/install') == '/transport/agent-1000/install/codex'
    assert '--user alice' in calls[1].command
    assert len(calls) == 2
    await transport.exec_as_agent(['npm', 'install'], env={'HOME': '/root'})
    request = calls[-1]
    assert '--user alice' in request.command
    assert 'HOME=/transport/agent-1000/home-test' in request.command
    assert request.extra_env['HOME'] == os.environ['HOME']
    assert request.extra_env['BASH_ENV'] == '/dev/null'
    assert request.extra_env['COMPOSE_VALUE'] == 'from-controller'
    await transport.exec_as_root(['id', '-u'])
    assert '--user root' in calls[-1].command
    assert 'COMPOSE_VALUE' in calls[-1].extra_env


@pytest.mark.parametrize('initialized', [False, True])
async def test_compose_identity_preparation_keeps_target_home_binary_and_restores_config(
    monkeypatch, target, initialized
):
    transport = AgentExecutionTransport(target)
    identity = {'uid': 1000, 'gid': 1000, 'home': '/home/alice', 'user': 'alice', 'path': '/usr/bin:/bin'}
    monkeypatch.setattr(transport, 'validate', lambda: None)
    home = '/transport/agent-1000/home-test'
    probe = AsyncMock(
        side_effect=[
            RunCommandResponse(status=CommandRunStatus.Finished, return_code=0, stdout=json.dumps(identity)),
            RunCommandResponse(status=CommandRunStatus.Finished, return_code=0, stdout=json.dumps(home)),
        ]
    )
    monkeypatch.setattr(transport, '_execute', probe)
    if initialized:
        await transport.initialize()
    content = {'/home/alice/config.toml': 'base_url = "http://127.0.0.1:8123/v1"'}

    async def read(path):
        return content.get(str(path))

    async def write(path, value, **kwargs):
        content[str(path)] = value

    class Bridge:
        def __init__(self, *args):
            pass

        async def __aenter__(self):
            return 'http://127.0.0.1:9123'

        async def __aexit__(self, *args):
            pass

    monkeypatch.setattr(compose, 'ProxyBridgeManager', Bridge)
    monkeypatch.setattr(compose.ComposeExecutionTransport, '_terminate_agent', AsyncMock())
    monkeypatch.setattr(transport, 'read_text', read)
    monkeypatch.setattr(transport, 'write_text', write)
    async with transport.prepare(
        ['/home/alice/.local/bin/claude', '--print'],
        working_dir='/workspace',
        invocation_id='run',
        env={'NEXUS_AGENT_PROXY_URL': 'http://127.0.0.1:8123', 'HOME': '/root', 'BASH_ENV': '/home/alice/toolchain.sh'},
        proxy_files=['/home/alice/config.toml'],
    ) as prepared:
        assert prepared.command[prepared.command.index('--user') + 1] == 'alice'
        assert '/home/alice/.local/bin/claude' in prepared.command
        assert f'HOME={home}' in prepared.command
        assert 'BASH_ENV=/home/alice/toolchain.sh' in prepared.command
        assert prepared.env['BASH_ENV'] == '/dev/null'
        assert prepared.env['HOME'] == os.environ['HOME']
        assert prepared.env['NEXUS_AGENT_PROXY_URL'] == 'http://127.0.0.1:9123'
        assert '9123/v1' in content['/home/alice/config.toml']
    assert '8123/v1' in content['/home/alice/config.toml']
    assert content[f'{home}/.gitconfig'] == '[safe]\n\tdirectory = "/workspace"\n'
    assert probe.await_count == 2


async def test_compose_session_collection_uses_target_not_worker_filesystem(monkeypatch, target, tmp_path):
    transport = AgentExecutionTransport(target)
    monkeypatch.setattr(transport, 'file', AsyncMock(return_value=True))
    run = AsyncMock(return_value=RunCommandResponse(status=CommandRunStatus.Finished, return_code=0))
    monkeypatch.setattr('nexus.runtime.executors.command.run_command', run)
    await transport.copy_directory_to_local(Path('/home/alice/codex/sessions'), tmp_path / 'sessions')
    command = run.call_args.args[0].command
    assert ' cp main:/home/alice/codex/sessions/.' in command
    assert ' exec ' not in command


@pytest.mark.parametrize('kind', ['direct', 'compose'])
def test_account_paths_require_initialization_instead_of_falling_back(target, kind):
    transport = AgentExecutionTransport(target if kind == 'compose' else DirectExecutionTarget(user='alice'))
    with pytest.raises(ExecutionTransportError, match='Initialize'):
        transport.install_dir(agent_name='claude-code', default='/root/install')
    with pytest.raises(ExecutionTransportError, match='Initialize'):
        transport.home_dir(invocation_id='run', default='/root')


async def test_legacy_paths_do_not_change_when_initialized(target):
    transport = LegacyAgentExecutionTransport(target)
    install = transport.install_dir(agent_name='hycode', default='/root/install')
    home = transport.home_dir(invocation_id='run', default='/root')
    await transport.initialize()
    assert transport.install_dir(agent_name='hycode', default='/root/install') == install
    assert transport.home_dir(invocation_id='run', default='/root') == home
    assert not transport.has_identity


async def test_local_command_keeps_ambient_execution_user(monkeypatch):
    async def run(request):
        assert get_execution_user() == 'existing-user'
        assert 'BASH_ENV' not in request.extra_env
        return RunCommandResponse(status=CommandRunStatus.Finished, return_code=0)

    monkeypatch.setattr('nexus.runtime.executors.command.run_command', run)
    with execution_user('existing-user'):
        await AgentExecutionTransport().exec_as_agent(['npm', 'install'])


@pytest.mark.parametrize('bash_env', [None, '/home/alice/toolchain.sh'])
def test_compose_bash_env_is_scoped_to_agent_or_management(target, bash_env):
    transport = AgentExecutionTransport(target)
    env = {} if bash_env is None else {'BASH_ENV': bash_env}
    prepared = transport._prepare_command(['bash', '-c', 'true'], env=env)
    assert prepared.env['BASH_ENV'] == '/dev/null'
    if bash_env is None:
        assert not any(arg.startswith('BASH_ENV=') for arg in prepared.command)
    else:
        assert f'BASH_ENV={bash_env}' in prepared.command
    root = transport._prepare_command(['bash', '-c', 'true'], env=env, root=True)
    assert 'BASH_ENV=/dev/null' in root.command
    assert root.env['BASH_ENV'] == '/dev/null'


async def test_compose_git_safe_directory_preserves_config_and_is_idempotent(monkeypatch, target, tmp_path):
    home = tmp_path / 'home'
    home.mkdir()
    repo = tmp_path / 'repo with spaces'
    subprocess.run(['git', 'init', str(repo)], check=True, capture_output=True)
    config = home / '.gitconfig'
    original = '[user]\n\tname = Existing User\n[safe]\n\tdirectory = /existing\n'
    config.write_text(original)
    env = {
        **os.environ,
        'HOME': str(home),
        'GIT_CONFIG_GLOBAL': str(config),
        'GIT_CONFIG_NOSYSTEM': '1',
        'GIT_TEST_ASSUME_DIFFERENT_OWNER': '1',
        # Inheriting os.environ means a localized developer shell makes git translate the message
        # below. C also wins over LANGUAGE, which gettext ignores outside C/POSIX.
        'LC_ALL': 'C',
    }
    before = subprocess.run(['git', '-C', str(repo), 'status', '--porcelain'], env=env, capture_output=True)
    assert before.returncode != 0
    assert b'dubious ownership' in before.stderr
    transport = AgentExecutionTransport(target)
    transport._identity = {'uid': 1000, 'gid': 1000, 'home': str(home), 'user': 'alice', 'path': '/usr/bin:/bin'}

    async def read(path):
        return Path(path).read_text() if Path(path).exists() else None

    async def write(path, content, **kwargs):
        Path(path).write_text(content)

    class Bridge:
        def __init__(self, *args):
            pass

        async def __aenter__(self):
            return 'http://127.0.0.1:9123'

        async def __aexit__(self, *args):
            pass

    monkeypatch.setattr(transport, 'read_text', read)
    monkeypatch.setattr(transport, 'write_text', write)
    monkeypatch.setattr(compose, 'ProxyBridgeManager', Bridge)
    monkeypatch.setattr(compose.ComposeExecutionTransport, '_terminate_agent', AsyncMock())
    for _ in range(2):
        async with transport.prepare(
            ['claude'],
            working_dir=str(repo),
            invocation_id='git',
            env={'NEXUS_AGENT_PROXY_URL': 'http://127.0.0.1:8123'},
        ):
            after = subprocess.run(['git', '-C', str(repo), 'status', '--porcelain'], env=env, capture_output=True)
            assert after.returncode == 0, after.stderr
    assert config.read_text().startswith(original)
    assert config.read_text().count(str(repo)) == 1
