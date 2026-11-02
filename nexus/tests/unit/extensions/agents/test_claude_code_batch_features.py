import json
import shlex

from pathlib import Path
from types import SimpleNamespace

import pytest

from nexus.extensions.agents.claude_code import agent as claude_agent
from nexus.extensions.agents.claude_code.agent import ClaudeCodeInput, _apply_batch_features
from nexus.extensions.agents.claude_code.batch_features import apply_batch_features
from nexus.extensions.agents.claude_code.mcp import mcp_config_servers_from_specs, register_mcp_servers


BASE = {
    'extra_cli_args': ['--output-format', 'json'],
    'config_file_patches': [{'path': '~/.claude/settings.json', 'content': {'existing': True}}],
    'pre_start_scripts': [{'interpreter': 'bash', 'code': 'echo hi'}],
}


def test_all_features_off_is_a_noop():
    args, patches, scripts = apply_batch_features(**BASE)

    assert args == BASE['extra_cli_args']
    assert patches == BASE['config_file_patches']
    assert scripts == BASE['pre_start_scripts']


def test_inputs_are_not_mutated():
    args_in = list(BASE['extra_cli_args'])
    patches_in = list(BASE['config_file_patches'])
    scripts_in = list(BASE['pre_start_scripts'])

    apply_batch_features(
        extra_cli_args=args_in,
        config_file_patches=patches_in,
        pre_start_scripts=scripts_in,
        mcp_config_servers={'srv': {'type': 'http', 'url': 'http://localhost:9900/mcp'}},
    )

    assert args_in == BASE['extra_cli_args']
    assert len(patches_in) == 1
    assert len(scripts_in) == 1


def test_mcp_cli_mode_passes_inline_json_and_allowlist():
    servers = {'canyon': {'type': 'http', 'url': 'http://localhost:9900/mcp'}}
    args, patches, _ = apply_batch_features(**BASE, mcp_config_servers=servers)

    assert '--mcp-config' in args
    assert (
        args[args.index('--mcp-config') + 1]
        == '{"mcpServers": {"canyon": {"type": "http", "url": "http://localhost:9900/mcp"}}}'
    )
    assert '--strict-mcp-config' in args
    assert args[args.index('--allowedTools') + 1] == 'mcp__canyon__*'
    # cli mode must not persist anything into the sandbox home.
    assert patches == BASE['config_file_patches']


def test_mcp_file_mode_writes_user_scope_config():
    servers = {'word': {'type': 'stdio', 'command': 'uv', 'args': ['run', 'word_mcp_server']}}
    args, patches, _ = apply_batch_features(**BASE, mcp_config_servers=servers, mcp_config_mode='file')

    assert '--mcp-config' not in args
    assert patches[-1] == {'path': '~/.claude/mcp.json', 'content': {'mcpServers': servers}}
    # The allowlist is still needed without --dangerously-skip-permissions.
    assert args[args.index('--allowedTools') + 1] == 'mcp__word__*'


def test_mcp_both_mode_does_cli_and_file():
    servers = {'a': {'type': 'http', 'url': 'http://h/mcp'}, 'b': {'type': 'http', 'url': 'http://h2/mcp'}}
    args, patches, _ = apply_batch_features(**BASE, mcp_config_servers=servers, mcp_config_mode='both')

    assert '--mcp-config' in args
    assert patches[-1]['path'] == '~/.claude/mcp.json'
    assert args[args.index('--allowedTools') + 1] == 'mcp__a__*,mcp__b__*'


def test_existing_allowlist_is_not_overridden():
    args, _, _ = apply_batch_features(
        extra_cli_args=['--allowedTools', 'Bash'],
        config_file_patches=[],
        pre_start_scripts=[],
        mcp_config_servers={'a': {'type': 'http', 'url': 'http://h/mcp'}},
    )

    assert args.count('--allowedTools') == 1
    assert args[args.index('--allowedTools') + 1] == 'Bash'


@pytest.mark.asyncio
async def test_host_preparation_honors_custom_paths_and_permissions(monkeypatch):
    from unittest.mock import AsyncMock

    upload = AsyncMock()
    monkeypatch.setattr(claude_agent, '_upload_runtime_text', upload)
    scripts = []
    hook_path = '/tmp/task with spaces/hooks.py'
    await claude_agent._prepare_unattended_interaction_hooks(scripts, remote_path=hook_path, file_mode=0o644)
    assert upload.call_args.args[0] == hook_path
    assert upload.call_args.kwargs == {'mode': 0o644}
    assert scripts[-1]['code'].endswith(f'exec "$py" \'{hook_path}\'')
    assert 'chmod' not in scripts[-1]['code']

    args = []
    mcp_path = '/tmp/task with spaces/mcp.json'
    await claude_agent._prepare_strict_mcp_file(
        ClaudeCodeInput(user_prompt='test', working_dir='/workspace'),
        args,
        remote_path=mcp_path,
        file_mode=0o640,
    )
    assert upload.call_args.args[:2] == (mcp_path, '{"mcpServers":{}}')
    assert upload.call_args.kwargs == {'mode': 0o640}
    assert args == ['--mcp-config', mcp_path, '--strict-mcp-config']


@pytest.mark.asyncio
async def test_unattended_hooks_upload_once_and_run_after_caller_scripts(monkeypatch):
    from nexus import core as core_module

    uploads = []
    commands = []
    captured = {}

    class FakeRuntime:
        async def upload_file(self, files, overwrite=False):
            uploads.append((files, overwrite))
            return SimpleNamespace(
                results=[SimpleNamespace(path=item.path, success=True, message='') for item in files]
            )

        async def run_command(self, **kwargs):
            commands.append(kwargs)
            return SimpleNamespace(return_code=0, stdout='', stderr='')

    class FakeContext:
        async def get_runtime(self):
            return FakeRuntime()

    async def fake_call_runtime_agent(_target, /, **kwargs):
        captured.update(kwargs)
        return {
            'success': True,
            'output': 'done',
            'stderr': '',
            'exit_code': 0,
            'execution_time': 1,
        }

    monkeypatch.setattr(core_module, 'require_context', lambda: FakeContext())
    monkeypatch.setattr(claude_agent, 'call_runtime_agent', fake_call_runtime_agent)

    await claude_agent.run_claude_code(
        ClaudeCodeInput(
            user_prompt='test',
            working_dir='/workspace',
            pre_start_scripts=[{'interpreter': 'bash', 'code': 'echo caller'}],
            enable_unattended_interaction_hooks=True,
        )
    )

    assert len(uploads) == 1
    ([uploaded], overwrite) = uploads[0]
    assert overwrite is True
    assert uploaded.path == '/tmp/nexus_claude_unattended_hooks.py'
    assert 'def install_unattended_interaction_hooks' in uploaded.content
    assert commands == [{'command': 'chmod 0600 -- /tmp/nexus_claude_unattended_hooks.py', 'timeout': 30}]
    assert captured['pre_start_scripts'][0]['code'] == 'echo caller'
    installer_command = captured['pre_start_scripts'][-1]['code']
    assert 'chmod' not in installer_command
    assert 'command -v python3' in installer_command
    assert '/nix/runtime/nexus/.venv-*/bin/python' in installer_command
    assert installer_command.endswith('exec "$py" /tmp/nexus_claude_unattended_hooks.py')


@pytest.mark.asyncio
async def test_strict_file_mcp_is_converted_and_uploaded_on_host(monkeypatch):
    monkeypatch.setattr('nexus.extensions.agents.claude_code.mcp.is_windows', lambda: False)
    from nexus import core as core_module
    from nexus.extensions.agents.installed_common.execution import ComposeExecutionTarget, agent_execution_target

    uploads = []
    commands = []
    captured = {}

    class FakeRuntime:
        async def upload_file(self, files, overwrite=False):
            uploads.append((files, overwrite))
            return SimpleNamespace(
                results=[SimpleNamespace(path=item.path, success=True, message='') for item in files]
            )

        async def run_command(self, **kwargs):
            commands.append(kwargs)
            return SimpleNamespace(return_code=0, stdout='', stderr='')

    class FakeContext:
        async def get_runtime(self):
            return FakeRuntime()

    async def fake_call_runtime_agent(_target, /, **kwargs):
        captured.update(kwargs)
        return {
            'success': True,
            'output': 'done',
            'stderr': '',
            'exit_code': 0,
            'execution_time': 1,
        }

    monkeypatch.setattr(core_module, 'require_context', lambda: FakeContext())
    monkeypatch.setattr(claude_agent, 'call_runtime_agent', fake_call_runtime_agent)

    execution_target = ComposeExecutionTarget(
        project_name='test-project',
        project_directory='/nexus/compose/task',
        compose_files=['/nexus/compose/task/docker-compose.yaml'],
        service='agent',
        transport_host_dir='/tmp/nexus-agent/abc',
        transport_container_dir='/opt/nexus-agent',
        socket_bridge_host_path='/tmp/nexus-agent/abc/bin/nexus-socket-bridge',
        socket_bridge_container_path='/opt/nexus-agent/bin/nexus-socket-bridge',
    )
    with agent_execution_target(execution_target):
        await claude_agent.run_claude_code(
            ClaudeCodeInput(
                user_prompt='test',
                working_dir='/workspace',
                mcp_servers=[
                    {
                        'name': 'task-tools',
                        'client': {
                            'transport': 'stdio',
                            'command': 'python',
                            'args': ['server.py'],
                            'cwd': '/workspace/server',
                            'env': {'TOKEN': 'literal'},
                        },
                    },
                ],
                mcp_servers_mode='strict_file',
                mcp_config_servers={'baseline': {'type': 'http', 'url': 'http://localhost/mcp'}},
            )
        )

    assert len(uploads) == 1
    ([uploaded], overwrite) = uploads[0]
    assert overwrite is True
    assert uploaded.path == '/tmp/nexus-agent/abc/strict-mcp.json'
    assert captured['uploaded_files'] == [uploaded.path]
    assert commands == [{'command': 'chmod 0600 -- /tmp/nexus-agent/abc/strict-mcp.json', 'timeout': 30}]
    config = json.loads(uploaded.content)
    assert set(config['mcpServers']) == {'task-tools', 'baseline'}
    assert config['mcpServers']['task-tools']['command'] == '/bin/sh'
    assert config['mcpServers']['task-tools']['env'] == {'TOKEN': 'literal'}
    assert captured['mcp_servers'] == []
    assert captured['extra_cli_args'][-3:] == [
        '--mcp-config',
        '/tmp/nexus-agent/abc/strict-mcp.json',
        '--strict-mcp-config',
    ]
    assert captured['pre_start_scripts'] == []


@pytest.mark.asyncio
async def test_strict_file_mcp_rejects_duplicate_sources(monkeypatch):
    from nexus import core as core_module

    class FakeContext:
        async def get_runtime(self):
            raise AssertionError('duplicate validation must happen before upload')

    monkeypatch.setattr(core_module, 'require_context', lambda: FakeContext())

    with pytest.raises(ValueError, match='Duplicate strict MCP server names'):
        await claude_agent.run_claude_code(
            ClaudeCodeInput(
                user_prompt='test',
                working_dir='/workspace',
                mcp_servers=[{'name': 'same', 'command': 'python'}],
                mcp_servers_mode='strict_file',
                mcp_config_servers={'same': {'type': 'http', 'url': 'http://localhost/mcp'}},
            )
        )


def test_empty_servers_skip_mcp_wiring_entirely():
    args, patches, scripts = apply_batch_features(**BASE, mcp_config_servers={})

    assert args == BASE['extra_cli_args']
    assert patches == BASE['config_file_patches']
    assert scripts == BASE['pre_start_scripts']


def test_invalid_mode_raises():
    with pytest.raises(ValueError, match='mcp_config_mode'):
        apply_batch_features(**BASE, mcp_config_servers={'a': {'type': 'http', 'url': 'u'}}, mcp_config_mode='A')


def test_shared_mcp_specs_convert_to_claude_native_config(monkeypatch):
    monkeypatch.setattr('nexus.extensions.agents.claude_code.mcp.is_windows', lambda: False)
    servers = [
        {
            'name': 'api',
            'client': {
                'transport': 'streamable_http',
                'url': 'http://localhost:9900/mcp',
                'headers': {'Authorization': 'Bearer t'},
            },
        },
        {
            'name': 'events',
            'client': {'transport': 'sse', 'url': 'http://localhost:9901/events'},
        },
        {
            'name': 'local',
            'client': {
                'transport': 'stdio',
                'command': 'uv',
                'args': ['run', 'server'],
                'env': {'K': 'v'},
                'cwd': '/workspace/server',
                'timeout_s': 60,
            },
        },
    ]

    assert mcp_config_servers_from_specs(servers) == {
        'api': {
            'type': 'http',
            'url': 'http://localhost:9900/mcp',
            'headers': {'Authorization': 'Bearer t'},
        },
        'events': {'type': 'sse', 'url': 'http://localhost:9901/events'},
        'local': {
            'type': 'stdio',
            'command': '/bin/sh',
            'args': [
                '-c',
                'cd -- "$1" && shift && exec "$@"',
                'nexus-mcp-cwd',
                '/workspace/server',
                'uv',
                'run',
                'server',
            ],
            'env': {'K': 'v'},
            'timeout': 60000,
        },
    }


def test_claude_mcp_rejects_unsupported_prefix_and_duplicate_names():
    prefixed = [
        {
            'name': 'local',
            'prefix': 'custom_',
            'client': {'transport': 'stdio', 'command': 'python'},
        }
    ]
    duplicate = [
        {'name': 'same', 'client': {'transport': 'stdio', 'command': 'python'}},
        {'name': 'same', 'client': {'transport': 'stdio', 'command': 'node'}},
    ]

    with pytest.raises(ValueError, match='prefix'):
        mcp_config_servers_from_specs(prefixed)
    with pytest.raises(ValueError, match='Duplicate'):
        mcp_config_servers_from_specs(duplicate)


def test_legacy_mcp_servers_do_not_enable_strict_inline_config():
    args, patches, scripts = _apply_batch_features(
        ClaudeCodeInput(
            user_prompt='test',
            working_dir='/workspace',
            mcp_servers=[{'name': 'legacy', 'command': 'python', 'args': ['server.py']}],
        )
    )

    assert '--mcp-config' not in args
    assert '--strict-mcp-config' not in args
    assert patches == []
    assert scripts == []


def test_explicit_mcp_config_servers_remain_strict():
    args, _, _ = _apply_batch_features(
        ClaudeCodeInput(
            user_prompt='test',
            working_dir='/workspace',
            mcp_config_servers={'strict': {'type': 'http', 'url': 'http://localhost/mcp'}},
        )
    )

    assert '--mcp-config' in args
    assert '--strict-mcp-config' in args


@pytest.mark.asyncio
async def test_runtime_registration_adds_legacy_and_standard_servers(monkeypatch):
    monkeypatch.setattr('nexus.extensions.agents.claude_code.mcp.is_windows', lambda: False)
    from nexus.runtime.executors import command as command_module

    requests = []
    parsed = {}
    config_paths = []

    async def fake_run_command(request):
        requests.append(request)
        argv = shlex.split(request.command)
        config_arg = argv[4]
        assert config_arg.startswith('$(cat -- ')
        config_path = Path(shlex.split(config_arg[2:-1])[-1])
        config_paths.append(config_path)
        parsed[argv[3]] = json.loads(config_path.read_text())
        assert 'Bearer token' not in request.command
        return SimpleNamespace(return_code=0, stdout='', stderr='')

    monkeypatch.setattr(command_module, 'run_command', fake_run_command)

    await register_mcp_servers(
        '/opt/claude',
        working_dir='/workspace',
        claude_env={'TOKEN': 'host'},
        mcp_servers=[
            {'name': 'legacy', 'command': 'python', 'args': ['server.py']},
            {
                'name': 'api',
                'client': {
                    'transport': 'streamable_http',
                    'url': 'http://localhost/mcp',
                    'headers': {'Authorization': 'Bearer token'},
                    'timeout_s': 30,
                },
            },
            {'name': 'events', 'client': {'transport': 'sse', 'url': 'http://localhost/events'}},
        ],
    )

    assert len(requests) == 3
    for request in requests:
        argv = shlex.split(request.command)
        assert argv[:3] == ['/opt/claude', 'mcp', 'add-json']
        assert argv[4].startswith('$(cat -- ')
        assert request.cwd == '/workspace'
        assert request.extra_env == {'TOKEN': 'host'}

    assert parsed == {
        'legacy': {'type': 'stdio', 'command': 'python', 'args': ['server.py']},
        'api': {
            'type': 'http',
            'url': 'http://localhost/mcp',
            'headers': {'Authorization': 'Bearer token'},
            'timeout': 30000,
        },
        'events': {'type': 'sse', 'url': 'http://localhost/events'},
    }
    assert all(not path.exists() for path in config_paths)


@pytest.mark.asyncio
async def test_runtime_registration_fails_closed_and_removes_temp_config(monkeypatch):
    monkeypatch.setattr('nexus.extensions.agents.claude_code.mcp.is_windows', lambda: False)
    from nexus.runtime.executors import command as command_module

    config_paths = []

    async def fake_run_command(request):
        argv = shlex.split(request.command)
        config_path = Path(shlex.split(argv[4][2:-1])[-1])
        config_paths.append(config_path)
        assert config_path.exists()
        return SimpleNamespace(return_code=1, stdout='', stderr='invalid server')

    monkeypatch.setattr(command_module, 'run_command', fake_run_command)

    with pytest.raises(RuntimeError, match='invalid server'):
        await register_mcp_servers(
            '/opt/claude',
            working_dir='/workspace',
            claude_env={},
            mcp_servers=[{'name': 'broken', 'command': 'python', 'args': ['server.py']}],
        )

    assert all(not path.exists() for path in config_paths)


@pytest.mark.asyncio
async def test_single_agent_keeps_mcp_servers_on_additive_runtime_path(monkeypatch):
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
    server = {'name': 'legacy', 'command': 'python', 'args': ['server.py']}

    await claude_agent.run_claude_code(
        ClaudeCodeInput(
            user_prompt='test',
            working_dir='/workspace',
            mcp_servers=[server],
            no_thinking_request_args={'model': 'no-think'},
            no_thinking_request_args_mode='inherit',
        )
    )

    assert captured['mcp_servers'] == [server]
    assert captured['no_thinking_request_args_mode'] == 'inherit'
    assert '--mcp-config' not in captured['extra_cli_args']
    assert '--strict-mcp-config' not in captured['extra_cli_args']


@pytest.mark.asyncio
async def test_single_agent_passes_task_skills_dir_to_runtime(monkeypatch):
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

    await claude_agent.run_claude_code(
        ClaudeCodeInput(
            user_prompt='test',
            working_dir='/workspace',
            skills_dir='/skills',
        )
    )

    assert captured['skills_dir'] == '/skills'


@pytest.mark.asyncio
async def test_single_agent_preserves_infrastructure_failure_metadata(monkeypatch):
    async def fake_call_runtime_agent(_target, /, **_kwargs):
        return {
            'success': False,
            'output': '',
            'stderr': 'proxy unavailable',
            'exit_code': 1,
            'execution_time': 1,
            'failure_kind': 'infrastructure',
            'failure_stage': 'proxy',
        }

    monkeypatch.setattr(claude_agent, 'call_runtime_agent', fake_call_runtime_agent)

    output = await claude_agent.run_claude_code(ClaudeCodeInput(user_prompt='test', working_dir='/workspace'))

    assert output.failure_kind == 'infrastructure'
    assert output.failure_stage == 'proxy'


@pytest.mark.asyncio
async def test_single_agent_fails_closed_for_unclassified_runtime_failure(monkeypatch):
    async def fake_call_runtime_agent(_target, /, **_kwargs):
        return {
            'success': False,
            'output': '',
            'stderr': 'legacy runtime setup failed',
            'exit_code': -1,
            'execution_time': 1,
        }

    monkeypatch.setattr(claude_agent, 'call_runtime_agent', fake_call_runtime_agent)

    output = await claude_agent.run_claude_code(ClaudeCodeInput(user_prompt='test', working_dir='/workspace'))

    assert output.failure_kind == 'infrastructure'
    assert output.failure_stage == 'runtime_protocol'
    assert 'update the Nexus Runtime image' in output.stderr


@pytest.mark.asyncio
async def test_single_agent_preserves_classified_process_failure(monkeypatch):
    async def fake_call_runtime_agent(_target, /, **_kwargs):
        return {
            'success': False,
            'output': '',
            'stderr': 'claude exited',
            'exit_code': 1,
            'execution_time': 1,
            'failure_kind': 'agent',
            'failure_stage': 'process',
        }

    monkeypatch.setattr(claude_agent, 'call_runtime_agent', fake_call_runtime_agent)

    output = await claude_agent.run_claude_code(ClaudeCodeInput(user_prompt='test', working_dir='/workspace'))

    assert output.failure_kind == 'agent'
    assert output.failure_stage == 'process'
