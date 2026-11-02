"""Claude Code version constraints apply to the actual Agent environment."""

from __future__ import annotations

import asyncio
import os

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nexus.extensions.agents.claude_code import agent, install_support, installer, runtime_setup
from nexus.extensions.agents.installed_common.execution import AgentExecutionTransport
from nexus.extensions.agents.installed_common.target_files import operate
from nexus.runtime.executors.models.command import CommandRunStatus, RunCommandResponse


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
    monkeypatch.setattr(transport, 'initialize', AsyncMock())
    find = AsyncMock(side_effect=lambda name: '/usr/local/bin/' + name)
    execute = AsyncMock()
    monkeypatch.setattr(transport, 'find_binary', find)
    monkeypatch.setattr(transport, 'exec_as_agent', execute)

    async def file(operation, path, **kwargs):
        return operate({'operation': operation, 'path': str(path), **kwargs})

    async def operation(transport, operation, **kwargs):
        return install_support.operate({'operation': operation, **kwargs})

    monkeypatch.setattr(transport, 'file', file)
    monkeypatch.setattr(installer, '_installation_operation', operation)
    monkeypatch.setattr(install_support, '_detect_claude_native_package', lambda: None)
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    return transport, find, execute


@pytest.mark.parametrize('version', ['2.1.112', '2.1.283'])
@pytest.mark.parametrize('pinned', [False, True])
async def test_preinstalled_claude_reused_without_npm_or_install_directory(context, tmp_path, version, pinned):
    transport, find, execute = context
    execute.return_value = RunCommandResponse(
        status=CommandRunStatus.Finished, return_code=0, stdout=f'{version} (Claude Code)\n'
    )
    directory = tmp_path / 'unused'
    binary = await installer.install_claude_code_via_npm(
        str(directory), version=version if pinned else None, transport=transport
    )
    assert binary == Path('/usr/local/bin/claude')
    find.assert_awaited_once_with('claude')
    execute.assert_awaited_once_with(['/usr/local/bin/claude', '--version'], timeout=30)
    assert not directory.exists()


async def test_native_install_location_uses_agent_home_when_not_on_path(context, tmp_path):
    transport, find, execute = context
    find.side_effect = lambda name: name
    binary = tmp_path / '.local/bin/claude'
    binary.parent.mkdir(parents=True)
    binary.touch()
    execute.return_value = RunCommandResponse(
        status=CommandRunStatus.Finished, return_code=0, stdout='2.1.283 (Claude Code)\n'
    )
    assert await installer.install_claude_code_via_npm(str(tmp_path / 'unused'), transport=transport) == binary
    execute.assert_awaited_once_with([str(binary), '--version'], timeout=30)


@pytest.mark.parametrize('requested', [None, '2.1.283', 'latest'])
async def test_missing_claude_installs_default_or_requested_release(context, tmp_path, requested):
    transport, find, execute = context
    find.side_effect = lambda name: name if name == 'claude' else '/usr/bin/npm'
    directory = tmp_path / 'install'
    binary = directory / 'node_modules/@anthropic-ai/claude-code-linux-x64/claude'

    async def install(argv, **kwargs):
        binary.parent.mkdir(parents=True)
        binary.write_bytes(b'fixture-native')
        metadata = directory / 'node_modules/@anthropic-ai/claude-code/package.json'
        metadata.parent.mkdir(parents=True)
        metadata.write_text('{"version":"2.1.283"}')
        return RunCommandResponse(status=CommandRunStatus.Finished, return_code=0)

    execute.side_effect = install
    result = await installer.install_claude_code_via_npm(str(directory), version=requested, transport=transport)
    assert result.resolve() == binary
    execute.assert_awaited_once()
    expected = '@anthropic-ai/claude-code@2.1.283' if requested == '2.1.283' else '@anthropic-ai/claude-code'
    assert execute.await_args.args[0][-1] == expected
    assert installer.DEFAULT_CLAUDE_CODE_VERSION == 'latest'


@pytest.mark.parametrize(
    ('status', 'code', 'stdout'),
    [
        (CommandRunStatus.Finished, 0, '2.1.112 (Claude Code)'),
        (CommandRunStatus.Finished, 0, '2.1.2830 (Claude Code)'),
        (CommandRunStatus.Finished, 0, '2.1.283-beta.1 (Claude Code)'),
        (CommandRunStatus.Finished, 0, 'unexpected'),
        (CommandRunStatus.Finished, 1, '2.1.283 (Claude Code)'),
        (CommandRunStatus.TimeLimitExceeded, None, '2.1.283 (Claude Code)'),
    ],
)
async def test_mismatch_or_unusable_cli_falls_back_to_explicit_install(context, tmp_path, status, code, stdout):
    transport, find, execute = context
    execute.side_effect = [
        RunCommandResponse(status=status, return_code=code, stdout=stdout),
        RunCommandResponse(status=CommandRunStatus.Finished, return_code=1, stderr='install failed'),
    ]
    with pytest.raises(RuntimeError, match='install failed'):
        await installer.install_claude_code_via_npm(str(tmp_path), version='2.1.283', transport=transport)
    assert [call.args[0] for call in find.await_args_list] == ['claude', 'npm']
    assert execute.await_args.args[0][-1] == '@anthropic-ai/claude-code@2.1.283'


async def test_explicit_latest_does_not_probe_preinstalled_claude(context, tmp_path):
    transport, find, execute = context
    execute.return_value = RunCommandResponse(status=CommandRunStatus.Finished, return_code=1, stderr='install failed')
    with pytest.raises(RuntimeError, match='install failed'):
        await installer.install_claude_code_via_npm(str(tmp_path), version='latest', transport=transport)
    find.assert_awaited_once_with('npm')
    execute.assert_awaited_once()


async def test_probe_cancellation_propagates(context, tmp_path):
    transport, _, execute = context
    execute.side_effect = asyncio.CancelledError
    with pytest.raises(asyncio.CancelledError):
        await installer.install_claude_code_via_npm(str(tmp_path), transport=transport)
    execute.assert_awaited_once()


@pytest.mark.parametrize('version', [None, '2.1.112', 'latest'])
async def test_agent_passes_version_constraint_without_coercion(monkeypatch, version):
    call = AsyncMock(return_value={'success': True, 'output': 'done', 'stderr': '', 'exit_code': 0, 'execution_time': 1})
    monkeypatch.setattr(agent, 'call_runtime_agent', call)
    await agent.run_claude_code(agent.ClaudeCodeInput(user_prompt='test', working_dir='/workspace', version=version))
    assert call.await_args.kwargs['version'] == version
    assert agent.ClaudeCodeInput.model_fields['version'].default is None
    assert agent.ClaudeCodeNativeBridgeInput.model_fields['version'].default == 'latest'
    assert agent.ClaudeCodeInteractiveInput.model_fields['version'].default == 'latest'


@pytest.mark.parametrize('from_url', [False, True])
async def test_runtime_launch_and_mcp_use_selected_entry(monkeypatch, tmp_path, from_url):
    binary = tmp_path / 'system-bin/claude'
    binary.parent.mkdir()
    binary.touch()
    selected = AsyncMock(return_value=binary)
    rejected = AsyncMock(side_effect=AssertionError('wrong installation source'))
    monkeypatch.setattr(installer, 'install_claude_code_via_npm', rejected if from_url else selected)
    monkeypatch.setattr(installer, 'install_claude_code_from_url', selected if from_url else rejected)
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setattr('nexus.runtime.context.require_runtime_control', lambda: SimpleNamespace(sio=object()))
    proxy = SimpleNamespace(start=AsyncMock(), stop=AsyncMock())
    monkeypatch.setattr('nexus.runtime.server.services.anthropic_proxy.AnthropicProxy', lambda **kwargs: proxy)
    register = AsyncMock()
    monkeypatch.setattr('nexus.extensions.agents.claude_code.mcp.register_mcp_servers', register)
    run = AsyncMock(return_value=(0, 'done', ''))
    monkeypatch.setattr(runtime_setup, '_run_claude_streaming', run)
    result = await runtime_setup.setup_and_run_claude_code(
        'test', working_dir=str(tmp_path), binary_url='https://example.invalid/claude' if from_url else None
    )
    assert result['success']
    rejected.assert_not_awaited()
    assert run.await_args.kwargs['cmd'][0] == str(binary)
    assert run.await_args.kwargs['env']['PATH'].split(os.pathsep)[0] == str(binary.parent)
    assert register.await_args.args[0] == binary
    proxy.stop.assert_awaited_once()
