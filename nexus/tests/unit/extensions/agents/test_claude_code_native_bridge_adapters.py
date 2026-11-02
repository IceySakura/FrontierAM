from __future__ import annotations

import asyncio
import base64
import contextlib
import re

from typing import Any

import pytest

from nexus.runtime.bridge.adapters.base import BridgeLaunchResult
from nexus.runtime.providers.ags.osworld import AGSOSWorldRuntime, AGSRawOSWorldRuntime
from nexus.runtime.providers.devcloud.client import DevCloudRuntime


class FakeRuntime:
    pass


class FakeOSWorldRuntime(AGSOSWorldRuntime):
    def __init__(self) -> None:
        pass


class FakeRawOSWorldRuntime(AGSRawOSWorldRuntime):
    def __init__(self, nexus_runtime: FakeOSWorldRuntime) -> None:
        self.nexus_runtime = nexus_runtime

    def to_nexus(self) -> FakeOSWorldRuntime:
        return self.nexus_runtime


@pytest.fixture
def launch() -> BridgeLaunchResult:
    return BridgeLaunchResult(
        base_url='http://bridge.example:8765',
        local_base_url='http://127.0.0.1:8765',
    )


@pytest.mark.asyncio
async def test_agent_runtime_delegates_install_and_run_to_target_adapter(
    monkeypatch: pytest.MonkeyPatch,
    launch: BridgeLaunchResult,
) -> None:
    from nexus.extensions.agents.claude_code.native_bridge import agent_runtime
    from nexus.extensions.agents.claude_code.native_bridge.adapters.types import ClaudeRunResult

    captured: dict[str, Any] = {}

    class FakeClaudeAdapter:
        async def ensure_claude(self, spec: Any) -> str:
            captured['install_spec'] = spec
            return '/tmp/nexus-claude-code/claude'

        async def register_mcp_servers(self, claude_executable: str, mcp_servers: Any, **kwargs: Any) -> None:
            captured['mcp_registration'] = (claude_executable, mcp_servers, kwargs)

        async def run_claude(self, request: Any) -> ClaudeRunResult:
            captured['run_request'] = request
            return ClaudeRunResult(exit_code=0, stdout='sorted: 1 2 3\n', stderr='')

    def fake_build_adapter(runtime: Any) -> FakeClaudeAdapter:
        captured['runtime'] = runtime
        return FakeClaudeAdapter()

    monkeypatch.setattr(agent_runtime, 'build_claude_native_bridge_adapter', fake_build_adapter)
    runtime = FakeRuntime()

    result = await agent_runtime.setup_and_run_native_bridge_claude_code(
        runtime=runtime,
        launch=launch,
        task='write quicksort',
        working_dir='/tmp/work',
        timeout=123.0,
        version='1.2.3',
        npm_registry='https://registry.example',
        extra_env={'FOO': 'bar'},
        extra_cli_args=['--output-format', 'json'],
        mcp_servers=[{'name': 'hy-computer-use', 'command': 'node', 'args': ['server.mjs']}],
    )

    assert result['success'] is True
    assert result['output'] == 'sorted: 1 2 3\n'
    assert result['stderr'] == ''
    assert result['exit_code'] == 0
    assert captured['runtime'] is runtime
    assert captured['install_spec'].version == '1.2.3'
    assert captured['install_spec'].npm_registry == 'https://registry.example'
    assert captured['run_request'].claude_executable == '/tmp/nexus-claude-code/claude'
    assert captured['run_request'].launch is launch
    assert captured['run_request'].extra_env == {'FOO': 'bar'}
    assert captured['run_request'].extra_cli_args == ['--output-format', 'json']
    assert captured['mcp_registration'][1][0]['name'] == 'hy-computer-use'


@pytest.mark.asyncio
async def test_agent_runtime_reports_empty_install_exception_type(
    monkeypatch: pytest.MonkeyPatch,
    launch: BridgeLaunchResult,
) -> None:
    from nexus.extensions.agents.claude_code.native_bridge import agent_runtime

    class FakeClaudeAdapter:
        async def ensure_claude(self, spec: Any) -> str:
            raise asyncio.TimeoutError()

        async def run_claude(self, request: Any) -> Any:  # pragma: no cover - install fails first
            raise AssertionError('run_claude should not be called')

    monkeypatch.setattr(
        agent_runtime,
        'build_claude_native_bridge_adapter',
        lambda runtime: FakeClaudeAdapter(),
    )

    result = await agent_runtime.setup_and_run_native_bridge_claude_code(
        runtime=FakeRuntime(),
        launch=launch,
        task='write quicksort',
    )

    assert result['success'] is False
    assert result['exit_code'] == -1
    assert result['stderr'] == 'Failed to install Claude Code: TimeoutError'




def test_claude_adapter_factory_builds_osworld_adapter_from_raw_runtime() -> None:
    from nexus.extensions.agents.claude_code.native_bridge.adapters import build_claude_native_bridge_adapter
    from nexus.extensions.agents.claude_code.native_bridge.adapters.osworld import OSWorldClaudeNativeBridgeAdapter

    nexus_runtime = FakeOSWorldRuntime()
    adapter = build_claude_native_bridge_adapter(FakeRawOSWorldRuntime(nexus_runtime))

    assert isinstance(adapter, OSWorldClaudeNativeBridgeAdapter)
    assert adapter.runtime is nexus_runtime


@pytest.mark.asyncio
@pytest.mark.parametrize('with_screenshot', [False, True])
@pytest.mark.parametrize(('rc', 'text'), [(0, 'Completed'), (0, 'TASK_INFEASIBLE'), (1, 'TASK_DONE')])
async def test_osworld_profile_uses_main_execution_contract(monkeypatch, launch, rc, text, with_screenshot):
    import json

    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from nexus.extensions.agents.claude_code.native_bridge.adapters.shell import build_claude_code_env
    from nexus.extensions.agents.claude_code.native_bridge.adapters.types import ClaudeRunResult
    from nexus.extensions.tasks.os_world.execution.claude_code import OSWorldClaudeExecutionProfile

    captured = {}

    class Adapter:
        async def ensure_claude(self, spec):
            captured['install'] = spec
            return '/usr/bin/claude'

        async def run_claude(self, request):
            captured['request'] = request
            return ClaudeRunResult(rc, json.dumps({'type': 'result', 'result': text}), 'diagnostic')

    from nexus.extensions.tasks.os_world.execution import claude_code as task_cc
    monkeypatch.setattr(task_cc, 'OSWorldClaudeNativeBridgeAdapter', lambda runtime: Adapter())
    prompt = 'Task'
    if with_screenshot:
        prompt = [
            {'type': 'text', 'text': 'Task'},
            {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + 'A' * 200000}},
        ]
    profile = OSWorldClaudeExecutionProfile({'guide_text': 'Guide'}, {}, prompt)
    assert profile.bridge_options(None)['api_protocol'] == 'anthropic'
    runtime = FakeOSWorldRuntime()
    runtime.run_command = AsyncMock(return_value=SimpleNamespace(return_code=0, stdout='/home/test-user'))
    runtime.osworld_use_proxy = True
    result = await profile.execute(runtime=runtime, launch=launch, timeout=123)
    request = captured['request']
    assert request.task == 'Guide\nTask\n'
    assert request.working_dir == '/home/test-user'
    runtime.run_command.assert_awaited_once_with(command='printf "%s" "$HOME"', timeout=30.0)
    assert request.timeout == 123
    assert request.extra_env == {}
    assert set(request.__dataclass_fields__) == {
        'claude_executable', 'launch', 'task', 'working_dir', 'timeout', 'extra_env', 'extra_cli_args',
    }
    env = build_claude_code_env(local_bridge_base_url=launch.local_base_url, extra_env=request.extra_env)
    assert env['CLAUDE_ENABLE_STREAM_WATCHDOG'] == '0'
    assert not {'PATH', 'BASH_ENV', 'HTTP_PROXY', 'HTTPS_PROXY'} & env.keys()
    assert captured['install'].ensure_claude is True
    assert result['success'] is (rc == 0)
    assert result['exit_status'] == ('tool_error' if rc else 'infeasible' if text == 'TASK_INFEASIBLE' else 'finished')
    assert result['stderr'] == 'diagnostic'
    assert result['final_answer'] == text


@pytest.mark.parametrize('field', ['stall_bail', 'record_history', 'max_continues', 'poll_interval', 'engineio_debug'])
def test_osworld_rejects_removed_policy_options(field):
    from nexus.extensions.tasks.os_world.execution.claude_code import OSWorldExecutionOptions

    with pytest.raises(ValueError, match='Extra inputs are not permitted'):
        OSWorldExecutionOptions.model_validate({field: 0})


@pytest.mark.asyncio
async def test_osworld_explicit_working_dir_does_not_probe_home(monkeypatch, launch):
    from unittest.mock import AsyncMock

    from nexus.extensions.agents.claude_code.native_bridge import agent_runtime
    from nexus.extensions.tasks.os_world.execution.claude_code import OSWorldClaudeExecutionProfile

    runtime = FakeOSWorldRuntime()
    runtime.run_command = AsyncMock(side_effect=AssertionError('must preserve explicit working_dir'))
    runner = AsyncMock(return_value={'success': True, 'output': '', 'exit_code': 0})
    monkeypatch.setattr(agent_runtime, 'setup_and_run_native_bridge_claude_code', runner)
    profile = OSWorldClaudeExecutionProfile({'guide_text': '', 'working_dir': '/tmp/cc-work'}, {}, 'Task')
    await profile.execute(runtime=runtime, launch=launch, timeout=30)
    assert runner.call_args.kwargs['working_dir'] == '/tmp/cc-work'
    runtime.run_command.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(('return_code', 'stdout'), [(1, '/home/user'), (0, ''), (0, 'relative/home')])
async def test_osworld_invalid_home_fails_before_runner(monkeypatch, launch, return_code, stdout):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from nexus.extensions.agents.claude_code.native_bridge import agent_runtime
    from nexus.extensions.tasks.os_world.execution.claude_code import OSWorldClaudeExecutionProfile

    runtime = FakeOSWorldRuntime()
    runtime.run_command = AsyncMock(return_value=SimpleNamespace(return_code=return_code, stdout=stdout))
    runner = AsyncMock()
    monkeypatch.setattr(agent_runtime, 'setup_and_run_native_bridge_claude_code', runner)
    profile = OSWorldClaudeExecutionProfile({'guide_text': ''}, {}, 'Task')
    with pytest.raises(RuntimeError, match='working directory'):
        await profile.execute(runtime=runtime, launch=launch, timeout=30)
    runner.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('path_version', 'private_version', 'expected'),
    [
        ('2.1.205 (Claude Code)', '', '/usr/bin/claude'),
        ('2.1.204 (Claude Code)', '2.1.205 (Claude Code)', '/tmp/nexus-claude-code/node_modules/.bin/claude'),
        ('', '2.1.205', '/tmp/nexus-claude-code/node_modules/.bin/claude'),
        ('2.1.2050', '', '/installed'),
        ('unexpected output', '', '/installed'),
    ],
)
async def test_osworld_reuses_only_exact_expected_cli_version(monkeypatch, path_version, private_version, expected):
    from unittest.mock import AsyncMock

    from nexus.extensions.agents.claude_code.native_bridge.adapters.types import ClaudeInstallSpec
    from nexus.extensions.tasks.os_world.execution.claude_code import OSWorldClaudeNativeBridgeAdapter

    responses = [(0, '/usr/bin/claude\n', ''), (0 if path_version else 1, path_version, '')]
    if expected != '/usr/bin/claude':
        responses.append((0 if private_version else 1, private_version, ''))
    runtime = FakePreStartRuntime(responses=responses)
    adapter = object.__new__(OSWorldClaudeNativeBridgeAdapter)
    adapter.runtime = runtime
    install = AsyncMock(return_value='/installed')
    monkeypatch.setattr(adapter, '_install_via_npm', install)
    spec = ClaudeInstallSpec(True, None, '2.1.205')
    assert await adapter.ensure_claude(spec) == expected
    if expected == '/installed':
        install.assert_awaited_once_with(spec)
    else:
        install.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('matching', [False, True])
async def test_osworld_version_reuse_respects_disabled_install(monkeypatch, matching):
    from unittest.mock import AsyncMock

    from nexus.extensions.agents.claude_code.native_bridge.adapters.types import ClaudeInstallSpec
    from nexus.extensions.tasks.os_world.execution.claude_code import OSWorldClaudeNativeBridgeAdapter

    adapter = object.__new__(OSWorldClaudeNativeBridgeAdapter)
    adapter.runtime = FakePreStartRuntime(responses=[
        (0, '/usr/bin/claude', ''),
        (0 if matching else 1, '2.1.205 (Claude Code)', ''),
        (1, '', ''),
    ])
    install = AsyncMock()
    monkeypatch.setattr(adapter, '_install_via_npm', install)
    if matching:
        assert await adapter.ensure_claude(ClaudeInstallSpec(False, None, '2.1.205')) == '/usr/bin/claude'
    else:
        with pytest.raises(RuntimeError, match='requires ensure_claude=true'):
            await adapter.ensure_claude(ClaudeInstallSpec(False, None, '2.1.205'))
    install.assert_not_awaited()


@pytest.mark.asyncio
async def test_osworld_latest_keeps_standard_discovery(monkeypatch):
    from unittest.mock import AsyncMock

    from nexus.extensions.agents.claude_code.native_bridge.adapters.types import ClaudeInstallSpec
    from nexus.extensions.tasks.os_world.execution.claude_code import OSWorldClaudeNativeBridgeAdapter

    adapter = object.__new__(OSWorldClaudeNativeBridgeAdapter)
    adapter.runtime = FakePreStartRuntime(responses=[(0, '/usr/bin/claude', '')])
    install = AsyncMock()
    monkeypatch.setattr(adapter, '_install_via_npm', install)
    assert await adapter.ensure_claude(ClaudeInstallSpec(True, None, 'latest')) == '/usr/bin/claude'
    assert adapter.runtime.commands == ['command -v claude']
    install.assert_not_awaited()


@pytest.mark.asyncio
async def test_osworld_binary_url_is_authoritative_over_matching_install(monkeypatch):
    from unittest.mock import AsyncMock

    from nexus.extensions.agents.claude_code.native_bridge.adapters.types import ClaudeInstallSpec
    from nexus.extensions.tasks.os_world.execution.claude_code import OSWorldClaudeNativeBridgeAdapter

    adapter = object.__new__(OSWorldClaudeNativeBridgeAdapter)
    adapter.runtime = FakePreStartRuntime(responses=[(0, '/usr/bin/claude', ''), (0, '2.1.285 (Claude Code)', '')])
    install = AsyncMock(return_value='/tmp/nexus-claude-code/package/claude')
    monkeypatch.setattr(adapter, '_install_from_binary_url', install)
    spec = ClaudeInstallSpec(True, None, '2.1.285', binary_url='https://cos.example.com/claude.tgz')
    assert await adapter.ensure_claude(spec) == '/tmp/nexus-claude-code/package/claude'
    install.assert_awaited_once_with(spec)
    assert adapter.runtime.commands == []


def test_claude_adapter_factory_builds_devcloud_macos_adapter() -> None:
    from nexus.extensions.agents.claude_code.native_bridge.adapters import build_claude_native_bridge_adapter
    from nexus.extensions.agents.claude_code.native_bridge.adapters.devcloud_macos import (
        DevCloudMacOSClaudeNativeBridgeAdapter,
    )

    runtime = DevCloudRuntime(base_url='http://127.0.0.1:5000')
    adapter = build_claude_native_bridge_adapter(runtime)

    assert isinstance(adapter, DevCloudMacOSClaudeNativeBridgeAdapter)
    assert adapter.runtime is runtime


class FakePreStartRuntime:
    def __init__(self, *, return_code: int = 0, responses: list[tuple[int, str, str]] | None = None) -> None:
        self.return_code = return_code
        self.responses = list(responses or [])
        self.commands: list[str] = []

    async def run_command(self, command: str, *, timeout: float = 180, **kwargs):
        from nexus.runtime.executors.models.command import RunCommandResponse

        self.commands.append(command)
        if self.responses:
            return_code, stdout, stderr = self.responses.pop(0)
            return RunCommandResponse(status='Finished', return_code=return_code, stdout=stdout, stderr=stderr)
        return RunCommandResponse(status='Finished', return_code=self.return_code, stdout='', stderr='boom')


@pytest.mark.asyncio
async def test_posix_adapter_installs_pinned_version_via_npm() -> None:
    from nexus.extensions.agents.claude_code.native_bridge.adapters.devcloud_macos import (
        DevCloudMacOSClaudeNativeBridgeAdapter,
    )
    from nexus.extensions.agents.claude_code.native_bridge.adapters.types import ClaudeInstallSpec

    rt = FakePreStartRuntime(
        responses=[
            (0, '', ''),  # npm install
        ]
    )
    adapter = DevCloudMacOSClaudeNativeBridgeAdapter.__new__(DevCloudMacOSClaudeNativeBridgeAdapter)
    adapter.runtime = rt

    executable = await adapter.ensure_claude(
        ClaudeInstallSpec(ensure_claude=True, npm_registry='https://registry.npmjs.org', version='2.1.204')
    )

    assert executable == '/tmp/nexus-claude-code/node_modules/.bin/claude'
    # main installs an explicitly pinned version directly.
    assert len(rt.commands) == 1
    assert '@anthropic-ai/claude-code@2.1.204' in rt.commands[-1]
    assert 'https://registry.npmjs.org' in rt.commands[-1]


class LocalShellRuntime:
    def __init__(self) -> None:
        self.commands: list[str] = []

    async def run_command(self, command: str, *, timeout: float = 180, **kwargs):
        import subprocess

        from nexus.runtime.executors.models.command import RunCommandResponse

        self.commands.append(command)
        completed = subprocess.run(['bash', '-c', command], capture_output=True, text=True, timeout=timeout)
        return RunCommandResponse(
            status='Finished', return_code=completed.returncode, stdout=completed.stdout, stderr=completed.stderr
        )


@contextlib.contextmanager
def _serve_directory(directory):
    import functools
    import http.server
    import threading

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}'
    finally:
        server.shutdown()
        server.server_close()


def _write_native_package(path, version: str) -> str:
    import hashlib
    import io
    import tarfile

    script = f'#!/bin/sh\necho "{version} (Claude Code)"\n'.encode()
    with tarfile.open(path, 'w:gz') as archive:
        info = tarfile.TarInfo('package/claude')
        info.size = len(script)
        info.mode = 0o755
        archive.addfile(info, io.BytesIO(script))
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.asyncio
@pytest.mark.parametrize('archived', [True, False])
async def test_posix_adapter_installs_binary_url_without_npm(tmp_path, monkeypatch, archived) -> None:
    import hashlib

    from nexus.extensions.agents.claude_code.native_bridge.adapters import posix
    from nexus.extensions.agents.claude_code.native_bridge.adapters.types import ClaudeInstallSpec

    install_dir = tmp_path / 'nexus-claude-code'
    executable = install_dir / ('package/claude' if archived else 'claude')
    monkeypatch.setattr(posix, 'DEFAULT_CLAUDE_INSTALL_DIR', str(install_dir))
    source = tmp_path / ('claude-code-linux-x64.tgz' if archived else 'claude')
    if archived:
        checksum = _write_native_package(source, '2.1.285')
    else:
        source.write_text('#!/bin/sh\necho "2.1.285 (Claude Code)"\n')
        checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    rt = LocalShellRuntime()
    adapter = posix.BasePosixClaudeNativeBridgeAdapter(rt)

    with _serve_directory(tmp_path) as base_url:
        resolved = await adapter.ensure_claude(
            ClaudeInstallSpec(
                ensure_claude=True,
                npm_registry=None,
                version='2.1.285',
                binary_url=f'{base_url}/{source.name}?q-signature=abc',
                binary_checksum=checksum.upper(),
            )
        )

    assert resolved == str(executable)
    assert executable.is_file()
    assert not (install_dir / 'claude.url.tmp').exists()
    assert rt.commands == [rt.commands[0]]
    assert 'npm install' not in rt.commands[0]


@pytest.mark.asyncio
async def test_posix_adapter_rejects_binary_url_checksum_mismatch(tmp_path, monkeypatch) -> None:
    from nexus.extensions.agents.claude_code.native_bridge.adapters import posix
    from nexus.extensions.agents.claude_code.native_bridge.adapters.types import ClaudeInstallSpec

    install_dir = tmp_path / 'nexus-claude-code'
    monkeypatch.setattr(posix, 'DEFAULT_CLAUDE_INSTALL_DIR', str(install_dir))
    _write_native_package(tmp_path / 'claude.tgz', '2.1.285')
    adapter = posix.BasePosixClaudeNativeBridgeAdapter(LocalShellRuntime())

    with _serve_directory(tmp_path) as base_url, pytest.raises(RuntimeError, match='Checksum verification failed'):
        await adapter.ensure_claude(
            ClaudeInstallSpec(
                ensure_claude=True,
                npm_registry=None,
                version='2.1.285',
                binary_url=f'{base_url}/claude.tgz',
                binary_checksum='0' * 64,
            )
        )
    assert not (install_dir / 'package').exists()


@pytest.mark.asyncio
async def test_posix_adapter_rejects_binary_url_version_mismatch(tmp_path, monkeypatch) -> None:
    from nexus.extensions.agents.claude_code.native_bridge.adapters import posix
    from nexus.extensions.agents.claude_code.native_bridge.adapters.types import ClaudeInstallSpec

    monkeypatch.setattr(posix, 'DEFAULT_CLAUDE_INSTALL_DIR', str(tmp_path / 'cc'))
    rt = FakePreStartRuntime(responses=[(0, '2.1.284 (Claude Code)\n', '')])
    adapter = posix.BasePosixClaudeNativeBridgeAdapter(rt)

    with pytest.raises(RuntimeError, match='expected 2.1.285'):
        await adapter.ensure_claude(
            ClaudeInstallSpec(
                ensure_claude=True, npm_registry=None, version='2.1.285', binary_url='https://example.com/claude.tgz'
            )
        )


@pytest.mark.asyncio
async def test_posix_adapter_requires_install_permission_for_pinned_claude() -> None:
    from nexus.extensions.agents.claude_code.native_bridge.adapters.devcloud_macos import (
        DevCloudMacOSClaudeNativeBridgeAdapter,
    )
    from nexus.extensions.agents.claude_code.native_bridge.adapters.types import ClaudeInstallSpec

    rt = FakePreStartRuntime(
        responses=[
            (0, '', ''),
            (0, '2.1.205 (Claude Code)\n', ''),
        ]
    )
    adapter = DevCloudMacOSClaudeNativeBridgeAdapter.__new__(DevCloudMacOSClaudeNativeBridgeAdapter)
    adapter.runtime = rt

    with pytest.raises(RuntimeError, match='pinned Claude Code version requires ensure_claude=true'):
        await adapter.ensure_claude(ClaudeInstallSpec(ensure_claude=False, npm_registry=None, version='2.1.205'))
    assert rt.commands == []


@pytest.mark.asyncio
async def test_run_pre_start_scripts_injects_bridge_env_and_runs_in_order(launch: BridgeLaunchResult) -> None:
    from nexus.extensions.agents.claude_code.native_bridge.adapters.devcloud_macos import (
        DevCloudMacOSClaudeNativeBridgeAdapter,
    )

    rt = FakePreStartRuntime()
    adapter = DevCloudMacOSClaudeNativeBridgeAdapter.__new__(DevCloudMacOSClaudeNativeBridgeAdapter)
    adapter.runtime = rt

    await adapter.run_pre_start_scripts(
        [
            {'interpreter': 'bash', 'code': 'echo one'},
            {'interpreter': 'python', 'code': 'print(2)', 'cwd': '/tmp/custom', 'env': {'K': 'V'}},
        ],
        working_dir='/tmp/work',
        launch=launch,
        extra_env={'EXTRA': '1'},
    )

    assert len(rt.commands) == 2
    # Shares Claude's launch env (points at the local bridge) plus extra/script env.
    assert 'ANTHROPIC_BASE_URL=http://127.0.0.1:8765' in rt.commands[0]
    assert 'EXTRA=1' in rt.commands[0]
    assert 'cd /tmp/work' in rt.commands[0]
    assert 'bash -c' in rt.commands[0]
    assert 'cd /tmp/custom' in rt.commands[1]
    assert 'python3 -c' in rt.commands[1]
    assert 'K=V' in rt.commands[1]


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_posix_adapter_registers_mcp_servers(launch: BridgeLaunchResult) -> None:
    """The `claude mcp add` route stays supported for builds without --mcp-config.

    agent_runtime no longer calls this (it stages mcp-servers.json instead), but the
    adapter method is still the escape hatch for targets whose claude cannot take
    --mcp-config, so keep it covered.
    """
    from nexus.extensions.agents.claude_code.native_bridge.adapters.devcloud_macos import (
        DevCloudMacOSClaudeNativeBridgeAdapter,
    )

    rt = FakePreStartRuntime()
    adapter = DevCloudMacOSClaudeNativeBridgeAdapter.__new__(DevCloudMacOSClaudeNativeBridgeAdapter)
    adapter.runtime = rt

    await adapter.register_mcp_servers(
        '/tmp/claude code/claude',
        [
            {'name': 'files', 'command': 'python3', 'args': ['server script.py', '--root', '/tmp/work']},
            {'name': '', 'command': 'ignored'},
        ],
        working_dir='/tmp/work',
        launch=launch,
        extra_env={'TOKEN': 'secret'},
    )

    assert len(rt.commands) == 2
    assert rt.commands[0] == 'mkdir -p /tmp/work'
    command = rt.commands[1]
    assert "'/tmp/claude code/claude' mcp add --transport stdio files -- python3 'server script.py'" in command
    assert 'ANTHROPIC_BASE_URL=http://127.0.0.1:8765' in command
    assert 'TOKEN=secret' in command


@pytest.mark.asyncio
async def test_run_pre_start_scripts_raises_on_failure_when_fail_on_error(launch: BridgeLaunchResult) -> None:
    from nexus.extensions.agents.claude_code.native_bridge.adapters.devcloud_macos import (
        DevCloudMacOSClaudeNativeBridgeAdapter,
    )

    rt = FakePreStartRuntime(return_code=1)
    adapter = DevCloudMacOSClaudeNativeBridgeAdapter.__new__(DevCloudMacOSClaudeNativeBridgeAdapter)
    adapter.runtime = rt

    with pytest.raises(RuntimeError, match='pre-start .* failed'):
        await adapter.run_pre_start_scripts(
            [{'interpreter': 'bash', 'code': 'exit 1'}],
            working_dir='/tmp/work',
            launch=launch,
        )

    # explicit fail_on_error=False → non-zero tolerated.
    await adapter.run_pre_start_scripts(
        [{'interpreter': 'bash', 'code': 'exit 1', 'fail_on_error': False}],
        working_dir='/tmp/work',
        launch=launch,
    )


class FakeClaudeBgRuntime:
    """Scripts run_command responses for the detached launch + poll flow."""

    def __init__(self, poll_sequence: list[str], *, stdout: str = 'claude out\n', stderr: str = '') -> None:
        self.poll_sequence = list(poll_sequence)
        self.stdout_content = stdout
        self.stderr_content = stderr
        self.commands: list[str] = []
        self.killed = False

    async def run_command(self, command: str, *, timeout: float = 180, **kwargs):
        from nexus.runtime.executors.models.command import RunCommandResponse

        self.commands.append(command)

        def ok(out: str = '') -> RunCommandResponse:
            return RunCommandResponse(status='Finished', return_code=0, stdout=out, stderr='')

        if 'echo LAUNCHED' in command:
            return ok('LAUNCHED\n')
        if 'base64 -d' in command and 'B64EOF' in command:
            encoded = re.search(r"<<'B64EOF'\n([^\n]+)\nB64EOF", command).group(1)
            return ok(f'{len(base64.b64decode(encoded))}\n')
        if 'elif kill -0' in command:  # poll command
            nxt = self.poll_sequence.pop(0) if self.poll_sequence else 'RUNNING'
            return ok(nxt + '\n')
        if 'pkill -P' in command:  # kill command
            self.killed = True
            return ok('')
        if 'kill -TERM -- "-$pid"' in command:  # process-group cleanup command
            self.killed = True
            return ok('')
        if 'children=$(descendants "$pid")' in command:  # recursive process-tree cleanup command
            self.killed = True
            return ok('')
        if command.startswith('cat ') and 'stdout.log' in command:
            return ok(self.stdout_content)
        if command.startswith('cat ') and 'stderr.log' in command:
            return ok(self.stderr_content)
        return ok('')


def _make_run_request(timeout: float = 60.0):
    from nexus.extensions.agents.claude_code.native_bridge.adapters.types import ClaudeRunRequest

    return ClaudeRunRequest(
        claude_executable='/tmp/nexus-claude-code/node_modules/.bin/claude',
        launch=BridgeLaunchResult(base_url='http://bridge:9281', local_base_url='http://127.0.0.1:9281'),
        task='write quicksort',
        working_dir='/tmp/work',
        timeout=timeout,
    )


@pytest.mark.asyncio
async def test_run_claude_launches_detached_and_polls_until_done() -> None:
    from nexus.extensions.agents.claude_code.native_bridge.adapters.devcloud_macos import (
        DevCloudMacOSClaudeNativeBridgeAdapter,
    )

    rt = FakeClaudeBgRuntime(['DONE 0'], stdout='sorted: 1 2 3\n')
    adapter = DevCloudMacOSClaudeNativeBridgeAdapter.__new__(DevCloudMacOSClaudeNativeBridgeAdapter)
    adapter.runtime = rt

    result = await adapter.run_claude(_make_run_request())

    assert result.exit_code == 0
    assert result.stdout == 'sorted: 1 2 3\n'
    # Launch detaches Claude with nohup; cleanup discovers descendants recursively.
    assert any('nohup bash -c' in c and 'echo LAUNCHED' in c for c in rt.commands)
    assert not rt.killed
    assert any(c.startswith('rm -rf ') and 'nexus-claude-run-' in c for c in rt.commands)


@pytest.mark.asyncio
async def test_run_claude_polls_multiple_times(monkeypatch: pytest.MonkeyPatch) -> None:
    from nexus.extensions.agents.claude_code.native_bridge.adapters import posix
    from nexus.extensions.agents.claude_code.native_bridge.adapters.devcloud_macos import (
        DevCloudMacOSClaudeNativeBridgeAdapter,
    )

    monkeypatch.setattr(posix, '_POLL_INTERVAL', 0)
    rt = FakeClaudeBgRuntime(['RUNNING', 'RUNNING', 'DONE 3'])
    adapter = DevCloudMacOSClaudeNativeBridgeAdapter.__new__(DevCloudMacOSClaudeNativeBridgeAdapter)
    adapter.runtime = rt

    result = await adapter.run_claude(_make_run_request())

    assert result.exit_code == 3  # non-zero exit propagates
    assert sum(1 for c in rt.commands if 'elif kill -0' in c) == 3


@pytest.mark.asyncio
async def test_run_claude_times_out_and_kills() -> None:
    from nexus.extensions.agents.claude_code.native_bridge.adapters.devcloud_macos import (
        DevCloudMacOSClaudeNativeBridgeAdapter,
    )

    rt = FakeClaudeBgRuntime(['RUNNING'], stdout='partial\n')
    adapter = DevCloudMacOSClaudeNativeBridgeAdapter.__new__(DevCloudMacOSClaudeNativeBridgeAdapter)
    adapter.runtime = rt
    request = _make_run_request(timeout=0.0)  # deadline hits immediately after first poll

    result = await adapter.run_claude(request)

    assert result.exit_code == -1
    assert 'timed out' in result.stderr
    assert result.stdout == 'partial\n'  # partial output still recovered
    assert rt.killed


@pytest.mark.asyncio
async def test_run_claude_gone_still_kills_possible_descendants() -> None:
    from nexus.extensions.agents.claude_code.native_bridge.adapters.devcloud_macos import (
        DevCloudMacOSClaudeNativeBridgeAdapter,
    )

    rt = FakeClaudeBgRuntime(['GONE'], stdout='partial\n')
    adapter = DevCloudMacOSClaudeNativeBridgeAdapter.__new__(DevCloudMacOSClaudeNativeBridgeAdapter)
    adapter.runtime = rt

    result = await adapter.run_claude(_make_run_request())

    assert result.exit_code == -1
    assert 'exited without writing an exit code' in result.stderr
    assert rt.killed
