import hashlib
import inspect
import os

from pathlib import Path

import pytest

from nexus.core.agent import AgentRunInput, AgentRunOutput
from nexus.extensions.agents.claude_code import agent as claude_agent
from nexus.extensions.agents.claude_code.agent import ClaudeCodeInput
from nexus.extensions.agents.claude_code.runtime_setup import _resolve_config_path, setup_and_run_claude_code
from nexus.extensions.agents.codex.agent import CodexInput
from nexus.extensions.agents.codex.runtime_setup import setup_and_run_codex
from nexus.extensions.agents.hycode.agent import HyCodeInput
from nexus.extensions.agents.hycode.runtime_setup import setup_and_run_hycode
from nexus.extensions.agents.installed_common import compose as compose_transport
from nexus.extensions.agents.installed_common.execution import (
    AgentExecutionTransport,
    ComposeExecutionTarget,
    LegacyAgentExecutionTransport,
    agent_execution_target,
    execution_target_shared_path,
    get_agent_execution_target,
)
from nexus.runner import single as single_runner
from nexus.runner.single import RunInput


def _target(tmp_path: Path | None = None) -> ComposeExecutionTarget:
    host_dir = '/tmp/nxa/123456789abc'
    if tmp_path is not None:
        short_dir = Path('/tmp') / f'nxt-{hashlib.sha1(str(tmp_path).encode()).hexdigest()[:12]}'
        short_dir.symlink_to(tmp_path, target_is_directory=True)
        host_dir = str(short_dir)
    bridge = Path(host_dir) / 'bin' / 'nexus-socket-bridge'
    if tmp_path is not None:
        bridge.parent.mkdir(parents=True, exist_ok=True)
        bridge.write_text('#!/bin/sh\n')
        bridge.chmod(0o755)
    return ComposeExecutionTarget(
        project_name='task-run-1',
        project_directory='/harbor/environment',
        compose_files=['/harbor/compose/base.yaml', '/harbor/environment/docker-compose.yaml'],
        service='main',
        transport_host_dir=host_dir,
        transport_container_dir='/opt/nexus-agent',
        socket_bridge_host_path=str(bridge),
        socket_bridge_container_path='/opt/nexus-agent/bin/nexus-socket-bridge',
    )


@pytest.mark.asyncio
async def test_local_execution_transport_is_transparent() -> None:
    transport = AgentExecutionTransport()
    command = ['/nix/runtime/nix-env/bin/claude', '--print']
    env = {'HOME': '/root', 'CUSTOM_AGENT_TOKEN': 'secret'}

    async with transport.prepare(
        command,
        working_dir='/workspace',
        invocation_id='invocation-1',
        env=env,
    ) as prepared:
        assert prepared.command is command
        assert prepared.cwd == '/workspace'
        assert prepared.env is env


@pytest.mark.asyncio
async def test_compose_execution_transport_builds_main_process(monkeypatch, tmp_path: Path) -> None:
    target = _target(tmp_path)
    target = target.model_copy(update={'user': 'agent'})
    transport = LegacyAgentExecutionTransport(target)
    install_dir = Path(target.transport_host_dir) / 'install' / 'claude-code'
    install_dir.mkdir(parents=True)
    launcher = install_dir / 'claude'
    launcher.write_text('')
    config_path = Path(target.transport_host_dir) / 'config.toml'
    config_path.write_text('base_url = "http://127.0.0.1:8123/v1"\n')
    checked_calls: list[tuple[list[str], str]] = []

    async def fake_checked(self, argv, *, operation, timeout=30):
        checked_calls.append((argv, operation))
        return '/usr/local/bin:/usr/bin\n' if operation == 'Resolve target PATH' else ''

    async def fake_terminate(self, _pid_file):
        return None

    class FakeProxyBridgeManager:
        def __init__(self, _target, _invocation_id, proxy_url):
            assert proxy_url == 'http://127.0.0.1:8123'

        async def __aenter__(self):
            return 'http://127.0.0.1:9123'

        async def __aexit__(self, _exc_type, _exc, _traceback):
            return None

    monkeypatch.setattr(compose_transport.ComposeExecutionTransport, '_checked', fake_checked)
    monkeypatch.setattr(compose_transport.ComposeExecutionTransport, '_terminate_agent', fake_terminate)
    monkeypatch.setattr(compose_transport, 'ProxyBridgeManager', FakeProxyBridgeManager)

    command = [str(launcher), '--print', '__NEXUS_AGENT_PROXY_URL__/v1']
    env = {
        'HOME': str(Path(target.transport_host_dir) / 'home' / 'invocation-1'),
        'NEXUS_AGENT_PROXY_URL': 'http://127.0.0.1:8123',
        'CUSTOM_AGENT_TOKEN': 'secret',
    }
    async with transport.prepare(
        command,
        working_dir='/workspace',
        invocation_id='invocation-1',
        env=env,
        proxy_files=[config_path],
    ) as prepared:
        assert prepared.command[:7] == [
            'docker',
            'compose',
            '--project-name',
            'task-run-1',
            '--project-directory',
            '/harbor/environment',
            '--file',
        ]
        assert prepared.command.count('--file') == 2
        assert prepared.command[prepared.command.index('--workdir') + 1] == '/workspace'
        assert prepared.command[prepared.command.index('--user') + 1] == 'agent'
        assert prepared.command[prepared.command.index('--env') + 1] == ('NEXUS_AGENT_PROXY_URL=http://127.0.0.1:9123')
        assert 'CUSTOM_AGENT_TOKEN' in prepared.command
        assert '/opt/nexus-agent/install/claude-code/claude' in prepared.command
        assert 'http://127.0.0.1:9123/v1' in prepared.command
        assert prepared.cwd == '/'
        assert prepared.env['CUSTOM_AGENT_TOKEN'] == 'secret'
        assert 'http://127.0.0.1:9123/v1' in config_path.read_text()
        git_config = Path(target.transport_host_dir) / 'home' / 'invocation-1' / '.gitconfig'
        assert git_config.read_text() == '[safe]\n\tdirectory = "/workspace"\n'
        assert git_config.stat().st_mode & 0o777 == 0o644
        assert {operation for _argv, operation in checked_calls} == {
            'Resolve target PATH',
            'Validate working directory',
            'Validate Agent executable',
            'Validate socket bridge',
        }
        for argv, _operation in checked_calls:
            assert argv[argv.index('--user') + 1] == 'agent'

    assert 'http://127.0.0.1:8123/v1' in config_path.read_text()


@pytest.mark.asyncio
async def test_compose_execution_transport_terminates_as_agent_user(monkeypatch, tmp_path: Path) -> None:
    target = _target(tmp_path).model_copy(update={'user': 1000})
    pid_file = tmp_path / 'agent.pid'
    pid_file.write_text('123')
    calls: list[list[str]] = []

    async def fake_run_capture(argv, **_kwargs):
        calls.append(argv)
        return 0, '', ''

    monkeypatch.setattr(compose_transport, '_run_capture', fake_run_capture)

    await compose_transport.ComposeExecutionTransport(target)._terminate_agent(pid_file)

    assert len(calls) == 1
    assert calls[0][calls[0].index('--user') + 1] == '1000'
    assert calls[0][calls[0].index('main') + 1] == '/nix/runtime/nix-env/bin/bash'


def test_legacy_transport_manages_shared_install_home_and_files(tmp_path: Path) -> None:
    target = _target(tmp_path)
    transport = LegacyAgentExecutionTransport(target)
    transport.validate()
    home = transport.home_dir(invocation_id='run-1', default=tmp_path)

    assert home == Path(target.transport_host_dir) / 'home' / 'run-1'
    assert (
        _resolve_config_path('~/.claude/settings.json', home=str(home))
        == (home / '.claude' / 'settings.json').resolve()
    )
    assert transport.install_dir(agent_name='claude-code', default='/tmp/default') == (
        f'{target.transport_host_dir}/install/claude-code'
    )
    assert execution_target_shared_path(target, 'strict-mcp.json', default='/tmp/strict.json') == (
        Path(target.transport_host_dir) / 'strict-mcp.json'
    )
    assert os.access(target.socket_bridge_host_path, os.X_OK)


def test_execution_transport_preserves_failure_boundary() -> None:
    transport = AgentExecutionTransport(_target())

    assert transport.classify_failure(125, 'Cannot connect to the Docker daemon') == (
        'infrastructure',
        'execution_transport',
    )
    assert transport.classify_failure(1, 'Agent command failed') == ('agent', 'process')
    assert transport.classify_failure(0, '') == (None, None)


def test_execution_target_context_is_scoped() -> None:
    target = _target()

    assert get_agent_execution_target() is None
    with agent_execution_target(target):
        assert get_agent_execution_target() is target
    assert get_agent_execution_target() is None


def test_installed_agents_share_runtime_execution_target_contract() -> None:
    assert 'execution_target' in inspect.signature(setup_and_run_claude_code).parameters
    assert 'execution_target' in inspect.signature(setup_and_run_codex).parameters
    assert 'execution_target' in inspect.signature(setup_and_run_hycode).parameters
    assert 'execution_target' not in ClaudeCodeInput.model_fields
    assert 'execution_target' not in CodexInput.model_fields
    assert 'execution_target' not in HyCodeInput.model_fields


@pytest.mark.asyncio
async def test_claude_agent_input_stays_independent_from_execution_target(monkeypatch) -> None:
    captured = {}

    async def fake_call_runtime_agent(_target, /, **kwargs):
        captured.update(kwargs)
        return {
            'success': True,
            'output': 'done',
            'stderr': '',
            'exit_code': 0,
            'execution_time': 1,
        }

    monkeypatch.setattr(claude_agent, 'call_runtime_agent', fake_call_runtime_agent)

    with agent_execution_target(_target()):
        await claude_agent.run_claude_code(
            ClaudeCodeInput(
                user_prompt='repair the service',
                working_dir='/workspace',
            )
        )

    assert 'execution_target' not in ClaudeCodeInput.model_fields
    assert 'execution_target' not in captured
    assert 'compose_target' not in captured


def test_runner_propagates_agent_metadata(monkeypatch) -> None:
    class UnsupportedInput(AgentRunInput):
        user_prompt: str
        working_dir: str

    async def unsupported_agent(_input: UnsupportedInput) -> AgentRunOutput:
        return AgentRunOutput()

    unsupported_agent.nexus_meta = {'name': 'unsupported', 'type': 'agent', 'meta': {}}
    monkeypatch.setattr(single_runner.agent_registry, 'get', lambda _name: unsupported_agent)
    run_agent, _ = single_runner._build_run_agent_factory(
        RunInput(task_name='noop', agent_name='unsupported', agent_config={}),
        task_needs_agent=True,
    )

    assert run_agent is not None
    assert run_agent.nexus_meta == unsupported_agent.nexus_meta
