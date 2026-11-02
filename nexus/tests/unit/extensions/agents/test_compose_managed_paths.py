from __future__ import annotations

import json
import os

from pathlib import Path

import pytest

from nexus.extensions.agents.installed_common import target_files
from nexus.extensions.agents.installed_common.execution import (
    AgentExecutionTransport,
    ComposeExecutionTarget,
    DirectExecutionTarget,
)
from nexus.runtime.executors.models.command import CommandRunStatus, RunCommandResponse


pytestmark = pytest.mark.skipif(not hasattr(os, 'geteuid'), reason='POSIX execution identity')


@pytest.fixture
def target(tmp_path):
    return ComposeExecutionTarget(
        project_name='managed-paths',
        compose_files=['/task/compose.json'],
        service='main',
        transport_host_dir='/tmp/nx-paths',
        transport_container_dir=str(tmp_path),
        socket_bridge_host_path='/tmp/nx-paths/bridge',
        socket_bridge_container_path=str(tmp_path / 'bridge'),
        user='nobody',
    )


@pytest.mark.parametrize('account_home', ['/nonexistent', '/root', ''])
async def test_compose_prepares_home_as_agent_without_using_account_home(monkeypatch, target, tmp_path, account_home):
    identity = {
        'uid': os.geteuid(),
        'gid': os.getegid(),
        'user': 'nobody',
        'home': account_home,
        'path': '/usr/bin:/bin',
    }
    requests = []

    async def execute(command, **kwargs):
        assert not kwargs.get('root')
        assert kwargs['env'] == {}
        if kwargs.get('stdin'):
            request = json.loads(kwargs['stdin'])
            requests.append(request)
            result = target_files.operate(request)
        else:
            result = identity
        return RunCommandResponse(status=CommandRunStatus.Finished, return_code=0, stdout=json.dumps(result))

    def transport():
        result = AgentExecutionTransport(target)
        monkeypatch.setattr(result, 'validate', lambda: None)
        monkeypatch.setattr(result, '_execute', execute)
        return result

    first = transport()
    await first.initialize()
    root = tmp_path / f'agent-{identity["uid"]}'
    home = first.home_dir(invocation_id='run', default='/root')
    assert home.parent == root
    assert home.name.startswith('home-')
    assert home.is_dir()
    assert root.stat().st_mode & 0o777 == 0o700
    assert home.stat().st_mode & 0o777 == 0o700
    assert home.stat().st_uid == identity['uid']
    assert first.agent_env({'HOME': '/root'})['HOME'] == str(home)
    install = first.install_dir(agent_name='claude-code', default='/tmp/nexus/agents/claude-code')
    assert install == str(root / 'install/claude-code')
    assert not Path(install).exists()
    await first.initialize()
    assert first.home_dir(invocation_id='install', default='/root') == home
    assert len(requests) == 1

    second = transport()
    await second.initialize()
    assert second.home_dir(invocation_id='run', default='/root') != home
    assert second.install_dir(agent_name='claude-code', default='/unused') == install


def test_compose_install_cache_is_scoped_by_uid(target):
    paths = []
    for uid in (1000, 1001):
        transport = AgentExecutionTransport(target)
        transport._identity = {'uid': uid, 'home': '/unused'}
        paths.append(transport.install_dir(agent_name='claude-code', default='/unused'))
    assert paths[0] != paths[1]
    assert paths == [f'{target.transport_container_dir}/agent-{uid}/install/claude-code' for uid in (1000, 1001)]


def test_local_and_direct_paths_keep_existing_behavior():
    local = AgentExecutionTransport()
    assert local.install_dir(agent_name='claude-code', default='/custom/install') == '/custom/install'
    assert local.home_dir(invocation_id='run', default='/custom/home') == Path('/custom/home')
    assert local.agent_env({'HOME': '/custom/home'}) == {'HOME': '/custom/home'}
    direct = AgentExecutionTransport(DirectExecutionTarget(user='alice'))
    direct._identity = {'uid': 1000, 'home': '/home/alice', 'user': 'alice', 'path': '/usr/bin'}
    assert direct.install_dir(agent_name='claude-code', default='/unused') == (
        '/home/alice/.local/share/nexus/agents/claude-code'
    )
    assert direct.home_dir(invocation_id='run', default='/root') == Path('/home/alice')


@pytest.mark.parametrize('kind', ['symlink', 'public', 'other-owner'])
def test_managed_home_rejects_untrusted_user_directory(tmp_path, monkeypatch, kind):
    directory = tmp_path / 'agent-1000'
    if kind == 'symlink':
        destination = tmp_path / 'elsewhere'
        destination.mkdir(mode=0o700)
        directory.symlink_to(destination, target_is_directory=True)
    else:
        directory.mkdir(mode=0o700)
        if kind == 'public':
            directory.chmod(0o777)
        else:
            monkeypatch.setattr(target_files.os, 'geteuid', lambda: directory.stat().st_uid + 1)
    with pytest.raises(PermissionError, match='private and owned'):
        target_files.operate({'operation': 'prepare_home', 'path': str(directory)})
