"""Reuse installed Codex only when it matches the default or explicitly requested version."""

from __future__ import annotations

import asyncio
import json

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from nexus.extensions.agents.codex import installer
from nexus.extensions.agents.installed_common.execution import AgentExecutionTransport
from nexus.extensions.agents.installed_common.target_files import operate
from nexus.runtime.executors.models.command import CommandRunStatus, RunCommandResponse


_VERSION_KWARGS = [{}, {'version': None}, {'version': '0.114.0'}, {'version': '0.147.0'}]


@pytest.fixture(params=['local', 'direct', 'compose'])
def context(request, monkeypatch, tmp_path):
    target = None
    if request.param == 'direct':
        target = {'kind': 'direct', 'user': 'alice'}
    elif request.param == 'compose':
        target = {
            'kind': 'compose',
            'project_name': 'preinstalled',
            'compose_files': ['/tmp/compose.yaml'],
            'service': 'main',
            'transport_host_dir': '/tmp/nx-preinstalled',
            'transport_container_dir': '/transport',
            'socket_bridge_host_path': '/tmp/nx-preinstalled/bridge',
            'socket_bridge_container_path': '/transport/bridge',
            'user': 1101,
        }
    transport = AgentExecutionTransport(target)
    transport._identity = {'uid': 1101, 'gid': 1101, 'user': 'alice', 'home': str(tmp_path), 'path': '/usr/bin:/bin'}
    initialize = AsyncMock()
    find_binary = AsyncMock(side_effect=lambda name: '/usr/local/bin/' + name)
    execute = AsyncMock()

    async def file(operation, path, **kwargs):
        return operate({'operation': operation, 'path': str(path), **kwargs})

    monkeypatch.setattr(transport, 'initialize', initialize)
    monkeypatch.setattr(transport, 'find_binary', find_binary)
    monkeypatch.setattr(transport, 'exec_as_agent', execute)
    monkeypatch.setattr(transport, 'file', file)
    return transport, find_binary, execute


@pytest.mark.parametrize('version', ['0.114.0', '0.147.0', '0.131.0-alpha.15'])
async def test_matching_preinstalled_codex_skips_npm_and_private_install(context, tmp_path, version):
    transport, find_binary, execute = context
    execute.return_value = RunCommandResponse(
        status=CommandRunStatus.Finished, return_code=0, stdout=f'codex-cli {version}\n'
    )
    directory = tmp_path / 'absent-install'
    binary = await installer.install_codex_cli(str(directory), version=version, transport=transport)
    assert binary == Path('/usr/local/bin/codex')
    find_binary.assert_awaited_once_with('codex')
    execute.assert_awaited_once_with(['/usr/local/bin/codex', '--version'], timeout=30)
    assert not directory.exists()


@pytest.mark.parametrize('version_kwargs', _VERSION_KWARGS)
@pytest.mark.parametrize('installed_version', ['0.114.0', '0.147.0', '0.158.0', '0.131.0-alpha.15'])
async def test_preinstalled_codex_must_match_default_or_explicit_version(
    context, tmp_path, version_kwargs, installed_version
):
    transport, find_binary, execute = context
    expected_version = version_kwargs.get('version') or '0.114.0'
    directory = tmp_path / 'absent-install'
    managed_binary = directory / 'node_modules/.bin/codex'

    async def run(argv, **kwargs):
        if argv[-1] == '--version':
            return RunCommandResponse(
                status=CommandRunStatus.Finished, return_code=0, stdout=f'codex-cli {installed_version}\n'
            )
        managed_binary.parent.mkdir(parents=True)
        managed_binary.touch()
        return RunCommandResponse(status=CommandRunStatus.Finished, return_code=0)

    execute.side_effect = run
    binary = await installer.install_codex_cli(str(directory), transport=transport, **version_kwargs)
    if installed_version == expected_version:
        assert binary == Path('/usr/local/bin/codex')
        find_binary.assert_awaited_once_with('codex')
        execute.assert_awaited_once_with(['/usr/local/bin/codex', '--version'], timeout=30)
        assert not directory.exists()
    else:
        assert binary == managed_binary
        assert [call.args[0] for call in find_binary.await_args_list] == ['codex', 'npm']
        assert execute.await_count == 2
        assert execute.await_args.args[0][-1] == f'@openai/codex@{expected_version}'


@pytest.mark.parametrize('version_kwargs', _VERSION_KWARGS)
@pytest.mark.parametrize('installed_version', ['0.114.0', '0.147.0', '0.158.0'])
async def test_managed_codex_cache_must_match_default_or_explicit_version(
    context, tmp_path, version_kwargs, installed_version
):
    transport, find_binary, execute = context
    expected_version = version_kwargs.get('version') or '0.114.0'
    binary = tmp_path / 'node_modules/.bin/codex'
    binary.parent.mkdir(parents=True)
    binary.touch()
    metadata = tmp_path / 'node_modules/@openai/codex/package.json'
    metadata.parent.mkdir(parents=True)
    metadata.write_text(json.dumps({'version': installed_version}))
    find_binary.side_effect = lambda name: name if name == 'codex' else '/usr/local/bin/npm'
    execute.return_value = RunCommandResponse(status=CommandRunStatus.Finished, return_code=0)

    assert await installer.install_codex_cli(str(tmp_path), transport=transport, **version_kwargs) == binary
    if installed_version == expected_version:
        find_binary.assert_not_awaited()
        execute.assert_not_awaited()
    else:
        assert [call.args[0] for call in find_binary.await_args_list] == ['codex', 'npm']
        execute.assert_awaited_once()
        assert execute.await_args.args[0][-1] == f'@openai/codex@{expected_version}'


@pytest.mark.parametrize('version_kwargs', _VERSION_KWARGS)
async def test_missing_codex_installs_default_or_explicit_version(context, tmp_path, version_kwargs):
    transport, find_binary, execute = context
    find_binary.side_effect = lambda name: name if name == 'codex' else '/usr/local/bin/npm'
    binary = tmp_path / 'node_modules/.bin/codex'

    async def install(argv, **kwargs):
        binary.parent.mkdir(parents=True)
        binary.touch()
        return RunCommandResponse(status=CommandRunStatus.Finished, return_code=0)

    execute.side_effect = install
    assert await installer.install_codex_cli(str(tmp_path), transport=transport, **version_kwargs) == binary
    execute.assert_awaited_once()
    assert execute.await_args.args[0][-1] == f'@openai/codex@{version_kwargs.get("version") or "0.114.0"}'


@pytest.mark.parametrize(
    ('status', 'return_code', 'stdout'),
    [
        (CommandRunStatus.Finished, 0, 'codex-cli 0.113.0\n'),
        (CommandRunStatus.Finished, 0, 'codex-cli 0.114.01\n'),
        (CommandRunStatus.Finished, 0, 'codex-cli 0.114.0-alpha.1\n'),
        (CommandRunStatus.Finished, 0, 'unrecognized output'),
        (CommandRunStatus.Finished, 0, ''),
        (CommandRunStatus.Finished, 1, 'codex-cli 0.114.0\n'),
        (CommandRunStatus.Finished, None, 'codex-cli 0.114.0\n'),
        (CommandRunStatus.TimeLimitExceeded, None, 'codex-cli 0.114.0\n'),
        (CommandRunStatus.Error, None, 'codex-cli 0.114.0\n'),
    ],
)
async def test_unusable_preinstalled_codex_falls_back_to_pinned_install(
    context, tmp_path, status, return_code, stdout
):
    transport, find_binary, execute = context
    binary = tmp_path / 'node_modules/.bin/codex'

    async def run(argv, **kwargs):
        if argv[-1] == '--version':
            return RunCommandResponse(status=status, return_code=return_code, stdout=stdout)
        binary.parent.mkdir(parents=True)
        binary.touch()
        return RunCommandResponse(status=CommandRunStatus.Finished, return_code=0)

    execute.side_effect = run
    assert await installer.install_codex_cli(str(tmp_path), version='0.114.0', transport=transport) == binary
    assert [call.args[0] for call in find_binary.await_args_list] == ['codex', 'npm']
    assert execute.await_count == 2
    assert execute.await_args.args[0][-1] == '@openai/codex@0.114.0'


async def test_latest_keeps_install_policy_instead_of_reusing_preinstalled(context, tmp_path):
    transport, find_binary, execute = context
    binary = tmp_path / 'node_modules/.bin/codex'
    binary.parent.mkdir(parents=True)
    binary.touch()
    execute.return_value = RunCommandResponse(status=CommandRunStatus.Finished, return_code=0)
    assert await installer.install_codex_cli(str(tmp_path), version='latest', transport=transport) == binary
    find_binary.assert_awaited_once_with('npm')
    execute.assert_awaited_once()
    assert execute.await_args.args[0][-1] == '@openai/codex'


async def test_preinstalled_probe_does_not_swallow_cancellation(context, tmp_path):
    transport, find_binary, execute = context
    execute.side_effect = asyncio.CancelledError
    with pytest.raises(asyncio.CancelledError):
        await installer.install_codex_cli(str(tmp_path), transport=transport)
    find_binary.assert_awaited_once_with('codex')
    execute.assert_awaited_once()
