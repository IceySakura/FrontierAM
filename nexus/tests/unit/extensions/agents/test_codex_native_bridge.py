from __future__ import annotations

import base64
import json
import tomllib

from types import SimpleNamespace
from typing import Any

import pytest

from nexus.extensions.agents.codex.agent import CodexNativeBridgeInput, run_codex_native_bridge
from nexus.extensions.agents.codex.installer import DEFAULT_CODEX_VERSION
from nexus.extensions.agents.codex.native_bridge.adapters.packages import NATIVE_EXECUTABLE_MARKER
from nexus.extensions.agents.codex.native_bridge.adapters.posix import BasePosixCodexNativeBridgeAdapter
from nexus.extensions.agents.codex.native_bridge.adapters.shell import (
    build_codex_command,
    build_codex_config,
)
from nexus.extensions.agents.codex.native_bridge.adapters.types import CodexInstallSpec, CodexRunRequest, CodexRunResult
from nexus.extensions.agents.codex.native_bridge.agent_runtime import setup_and_run_native_bridge_codex
from nexus.runtime.bridge.adapters.base import BridgeLaunchResult
from nexus.runtime.executors.models.command import CommandRunStatus


def _request(**overrides: Any) -> CodexRunRequest:
    values = {
        'codex_executable': '/tmp/codex',
        'launch': BridgeLaunchResult(
            base_url='http://host.example:8765',
            local_base_url='http://127.0.0.1:8765',
        ),
        'task': 'fix the bug',
        'working_dir': '/workspace',
        'timeout': 600.0,
        'version': '0.143.0',
        'npm_registry': 'https://registry.npmmirror.com',
    }
    values.update(overrides)
    return CodexRunRequest(**values)


@pytest.mark.asyncio
@pytest.mark.parametrize('exit_code', [0, 1])
async def test_terminal_error_survives_without_retaining_raw_json(monkeypatch, exit_code):
    from unittest.mock import AsyncMock

    event = {'type': 'turn.failed', 'error': {'message': 'Unsupported parameter: reasoning_effort'}}
    monkeypatch.setattr(
        'nexus.extensions.agents.codex.agent.call_native_bridge_runtime_agent',
        AsyncMock(
            return_value={
                'success': exit_code == 0,
                'exit_code': exit_code,
                'output': '',
                'stderr': 'Reading additional input from stdin...\n',
                'json_events_stdout': json.dumps(event),
            }
        ),
    )
    monkeypatch.setattr('nexus.extensions.agents.codex.agent.get_current_span_id', lambda: None)
    monkeypatch.setattr('nexus.extensions.agents.codex.agent._emit_codex_timeline_spans', lambda *a, **kw: None)
    output = await run_codex_native_bridge(
        CodexNativeBridgeInput(user_prompt='test', working_dir='/tmp/test', emit_json_trace=True)
    )
    assert output.json_events_stdout is None
    if exit_code:
        assert output.termination_reason == 'CODEX_TURN_FAILED'
        assert output.stderr == 'Reading additional input from stdin...\n'
    else:
        assert output.termination_reason == 'ENV_DONE'
        assert output.stderr == 'Reading additional input from stdin...\n'


def test_codex_native_bridge_defaults_to_pinned_version() -> None:
    assert DEFAULT_CODEX_VERSION == '0.114.0'
    assert CodexNativeBridgeInput(user_prompt='hello', working_dir='/workspace').version == '0.114.0'


def test_codex_native_bridge_config_uses_local_responses_endpoint() -> None:
    config = build_codex_config(
        _request(model='gpt-5.4', auto_compact_token_limit=123456, extra_config_toml='model_context_window = 200000')
    )

    assert 'model = "gpt-5.4"' in config
    assert 'model_provider = "nexus"' in config
    assert 'base_url = "http://127.0.0.1:8765/v1"' in config
    assert 'wire_api = "responses"' in config
    assert 'model_auto_compact_token_limit = 123456' in config
    parsed = tomllib.loads(config)
    assert parsed['model_context_window'] == 200000
    assert 'model_context_window' not in parsed['model_providers']['nexus']


@pytest.mark.parametrize(
    'stale_key_line',
    [
        '"model_instructions_file" = "/tmp/stale.txt"',
        "'model_instructions_file' = '/tmp/stale.txt'",
        'model_instructions_file = """\n/tmp/stale.txt\n"""',
        "model_instructions_file = '''\n/tmp/stale.txt\n'''",
    ],
    ids=('quoted-key', 'single-quoted-key', 'multiline-basic-string', 'multiline-literal-string'),
)
def test_codex_native_bridge_config_replaces_quoted_and_multiline_stale_keys(stale_key_line: str) -> None:
    """Review follow-up: the replacement must also consume quoted keys
    (double- and single-quoted, both legal TOML) and multi-line string values,
    whose continuation lines a line-only pattern would leave behind as stray,
    invalid TOML."""
    config = build_codex_config(
        _request(
            model_instructions_file='/tmp/ppt_sysp_staged.txt',
            extra_config_toml=f'{stale_key_line}\nmodel_context_window = 200000',
        )
    )

    assert 'model_instructions_file = "/tmp/ppt_sysp_staged.txt"' in config
    assert '/tmp/stale.txt' not in config
    assert 'model_context_window = 200000' in config
    parsed = tomllib.loads(config)
    assert parsed['model_instructions_file'] == '/tmp/ppt_sysp_staged.txt'
    assert parsed['model_context_window'] == 200000


def test_codex_native_bridge_config_keeps_example_line_inside_multiline_string() -> None:
    """Review follow-up (round 3): a line-anchored strip cannot tell a root
    key from a continuation line of a multi-line string value. Same-named
    example lines inside a multi-line developer_instructions value must
    survive; only the root-level key is replaced."""
    extra = (
        'model_instructions_file = "/tmp/stale.txt"\n'
        'developer_instructions = """\n'
        'Example: model_instructions_file = "/tmp/example.txt"\n'
        '"""\n'
        'model_context_window = 200000'
    )
    config = build_codex_config(
        _request(
            model_instructions_file='/tmp/ppt_sysp_staged.txt',
            extra_config_toml=extra,
        )
    )

    assert 'model_instructions_file = "/tmp/ppt_sysp_staged.txt"' in config
    assert '/tmp/stale.txt' not in config
    # The example line is string content, not a root key: it must survive.
    assert 'model_instructions_file = "/tmp/example.txt"' in config
    parsed = tomllib.loads(config)
    assert parsed['model_instructions_file'] == '/tmp/ppt_sysp_staged.txt'
    assert parsed['model_context_window'] == 200000
    assert '/tmp/example.txt' in parsed['developer_instructions']


def test_codex_native_bridge_config_adds_mcp_servers() -> None:
    config = build_codex_config(
        _request(
            mcp_servers=[
                {
                    'name': 'everything',
                    'command': 'npx',
                    'args': ['-y', '@modelcontextprotocol/server-everything'],
                    'env': {'NPM_CONFIG_REGISTRY': 'https://registry.npmjs.org/'},
                    'startup_timeout_sec': 300,
                    'enabled_tools': ['echo'],
                    'omit_tools_from': ['deferred'],
                }
            ]
        )
    )

    assert '[mcp_servers."everything"]' in config
    assert 'command = "npx"' in config
    assert 'args = ["-y", "@modelcontextprotocol/server-everything"]' in config
    assert 'env = {"NPM_CONFIG_REGISTRY" = "https://registry.npmjs.org/"}' in config
    assert 'startup_timeout_sec = 300' in config
    assert 'enabled_tools = ["echo"]' in config
    assert 'omit_tools_from = ["deferred"]' in config


def test_codex_native_bridge_command_is_noninteractive_and_keeps_trace() -> None:
    command = build_codex_command(
        _request(developer_instructions='Follow the repository rules.', extra_cli_args=['--color', 'never']),
        output_file='/tmp/last.txt',
    )

    assert command[:4] == [
        '/tmp/codex',
        'exec',
        '--dangerously-bypass-approvals-and-sandbox',
        '--skip-git-repo-check',
    ]
    # Never ephemeral: multi_agent_version=v2 rebuilds a subagent's parent context from the
    # persistent thread store, and without it spawning dies with `collab spawn failed`.
    assert '--ephemeral' not in command
    assert '--json' in command
    assert command[-1] == 'Follow the repository rules.\n\nfix the bug'
    assert command[-4:-1] == ['/workspace', '-o', '/tmp/last.txt']


def test_posix_launch_installs_config_and_preserves_json_stdout() -> None:
    adapter = BasePosixCodexNativeBridgeAdapter(SimpleNamespace())
    paths = SimpleNamespace(
        dir='/tmp/run',
        stdout_log='/tmp/run/stdout.log',
        stderr_log='/tmp/run/stderr.log',
        exit_code='/tmp/run/exit_code',
        pid='/tmp/run/pid',
    )

    command = adapter._build_launch_command(_request(), paths)

    assert '/tmp/run/codex-home/config.toml' in command
    assert 'NEXUS_CODEX_OUTPUT_BASE64=' in command
    assert 'export PATH=/tmp:"$PATH"' in command
    assert 'http://127.0.0.1:8765/v1' not in command  # config is base64 encoded, not shell interpolated


@pytest.mark.asyncio
async def test_posix_run_uses_codex_lifecycle_and_extracts_output() -> None:
    encoded_output = base64.b64encode(b'done').decode()

    class FakeRuntime:
        def __init__(self) -> None:
            self.commands: list[str] = []

        async def run_command(self, *, command: str, timeout: float) -> Any:
            self.commands.append(command)
            if 'nohup bash' in command:
                return SimpleNamespace(
                    return_code=0,
                    stdout='LAUNCHED\n',
                    stderr='',
                    status=CommandRunStatus.Finished,
                )
            if 'echo "DONE ' in command:
                return SimpleNamespace(
                    return_code=0,
                    stdout='DONE 0\n',
                    stderr='',
                    status=CommandRunStatus.Finished,
                )
            if '/stdout.log' in command:
                return SimpleNamespace(
                    return_code=0,
                    stdout=f'{{"type":"turn.completed"}}\nNEXUS_CODEX_OUTPUT_BASE64={encoded_output}\n',
                    stderr='',
                    status=CommandRunStatus.Finished,
                )
            return SimpleNamespace(return_code=0, stdout='', stderr='', status=CommandRunStatus.Finished)

    runtime = FakeRuntime()
    adapter = BasePosixCodexNativeBridgeAdapter(runtime)  # type: ignore[arg-type]

    result = await adapter.run_codex(_request())

    assert result.exit_code == 0
    assert result.stdout == '{"type":"turn.completed"}\n'
    assert result.output == 'done'
    assert '/tmp/nexus-codex-run-' in runtime.commands[0]
    assert all('nexus-claude-run' not in command for command in runtime.commands)


@pytest.mark.asyncio
async def test_posix_install_downloads_binary_url_and_verifies_checksum() -> None:
    class FakeRuntime:
        def __init__(self) -> None:
            self.commands: list[tuple[str, float]] = []

        async def run_command(self, *, command: str, timeout: float) -> Any:
            self.commands.append((command, timeout))
            return SimpleNamespace(return_code=0, stdout='codex-cli 0.143.0', stderr='', status='Finished')

    runtime = FakeRuntime()
    adapter = BasePosixCodexNativeBridgeAdapter(runtime)  # type: ignore[arg-type]
    executable = await adapter.ensure_codex(
        CodexInstallSpec(
            version='0.143.0',
            npm_registry='https://registry.npmmirror.com',
            binary_url='http://cos-internal.example/codex?signature=a&part=1',
            binary_checksum='abc123',
        )
    )

    assert executable == '/tmp/nexus-codex-cli/codex'
    assert len(runtime.commands) == 1
    command, timeout = runtime.commands[0]
    assert timeout == 1800.0
    assert 'curl --fail --location --retry 3 --connect-timeout 30' in command
    assert "'http://cos-internal.example/codex?signature=a&part=1'" in command
    assert 'sha256sum' in command
    assert 'expected_checksum=abc123' in command
    assert 'Checksum verification failed: expected %s, got %s' in command
    assert 'npm install' not in command


@pytest.mark.asyncio
async def test_osworld_install_reuses_matching_codex_without_npm() -> None:
    from nexus.extensions.agents.codex.native_bridge.adapters.osworld import OSWorldCodexNativeBridgeAdapter

    class FakeRuntime:
        def __init__(self) -> None:
            self.commands: list[tuple[str, float]] = []

        async def run_command(self, *, command: str, timeout: float) -> Any:
            self.commands.append((command, timeout))
            return SimpleNamespace(return_code=0, stdout='codex-cli 0.143.0', stderr='', status='Finished')

    runtime = FakeRuntime()
    adapter = object.__new__(OSWorldCodexNativeBridgeAdapter)
    adapter.runtime = runtime  # type: ignore[assignment]

    executable = await adapter.ensure_codex(
        CodexInstallSpec(version='0.143.0', npm_registry='https://registry.npmmirror.com')
    )

    assert 'ensure_codex' not in OSWorldCodexNativeBridgeAdapter.__dict__
    assert executable == '/tmp/nexus-codex-cli/node_modules/.bin/codex'
    assert len(runtime.commands) == 2
    assert runtime.commands[0][0].startswith('test -x ')
    command, timeout = runtime.commands[1]
    assert timeout == 30.0
    assert '--version' in command
    assert 'npm install' not in command
    assert '/setup/upload' not in command


def test_waa_runner_publishes_exit_sentinel_atomically() -> None:
    from nexus.extensions.agents.codex.native_bridge.adapters.waa import WAACodexNativeBridgeAdapter

    adapter = object.__new__(WAACodexNativeBridgeAdapter)
    paths = SimpleNamespace(
        directory=r'C:\Temp\run',
        stdout_log=r'C:\Temp\run\stdout.log',
        stderr_log=r'C:\Temp\run\stderr.log',
        exit_code=r'C:\Temp\run\exit_code',
        pid=r'C:\Temp\run\pid',
    )

    script = adapter._build_runner_script(
        _request(codex_executable=r'C:\Temp\codex.cmd', working_dir=r'C:\Temp\work'),
        paths,
        working_dir=r'C:\Temp\work',
    )

    assert "Set-Content -LiteralPath 'C:\\Temp\\run\\exit_code.tmp'" in script
    assert "Move-Item -LiteralPath 'C:\\Temp\\run\\exit_code.tmp' -Destination 'C:\\Temp\\run\\exit_code'" in script


def test_waa_runner_invokes_a_native_executable_directly() -> None:
    """The npm-free install returns a real .exe, so the node.exe rewrite stays dormant.

    That matters on a node-less image: rewriting to node.exe there would invoke
    an interpreter that does not exist.
    """
    from nexus.extensions.agents.codex.native_bridge.adapters.waa import WAACodexNativeBridgeAdapter

    adapter = object.__new__(WAACodexNativeBridgeAdapter)
    paths = SimpleNamespace(
        directory=r'C:\Temp\run',
        stdout_log=r'C:\Temp\run\stdout.log',
        stderr_log=r'C:\Temp\run\stderr.log',
        exit_code=r'C:\Temp\run\exit_code',
        pid=r'C:\Temp\run\pid',
    )

    script = adapter._build_runner_script(
        _request(
            codex_executable=r'C:\Windows\Temp\nexus-codex-cli\package\vendor\x86_64-pc-windows-msvc\bin\codex.exe',
            working_dir=r'C:\Temp\work',
        ),
        paths,
        working_dir=r'C:\Temp\work',
    )

    assert 'node.exe' not in script
    assert 'codex.js' not in script
    assert "& 'C:\\Windows\\Temp\\nexus-codex-cli\\package\\vendor\\x86_64-pc-windows-msvc\\bin\\codex.exe'" in script


@pytest.mark.asyncio
async def test_waa_run_uses_codex_lifecycle_and_extracts_output() -> None:
    from nexus.extensions.agents.codex.native_bridge.adapters.waa import WAACodexNativeBridgeAdapter

    encoded_output = base64.b64encode(b'done').decode()
    commands: list[str] = []

    class FakeRuntime:
        def __init__(self) -> None:
            self.launches: list[list[str]] = []
            self.files: dict[str, str] = {}

        async def setup_create_folder(self, path: str) -> None:
            return None

        async def setup_create_file(self, path: str, content: str) -> None:
            self.files[path] = content

        async def setup_launch(self, command: list[str], *, shell: bool) -> None:
            assert shell is False
            self.launches.append(command)

    runtime = FakeRuntime()
    adapter = object.__new__(WAACodexNativeBridgeAdapter)
    adapter.runtime = runtime  # type: ignore[assignment]

    async def fake_execute(command: str, *, timeout: float | None = None) -> dict[str, Any]:
        commands.append(command)
        if 'ExpandEnvironmentVariables' in command:
            return {'returncode': 0, 'output': r'C:\work'}
        if 'WriteAllBytes' in command:
            return {'returncode': 0, 'output': ''}
        if 'Write-Output ("DONE "' in command:
            return {'returncode': 0, 'output': 'DONE 0'}
        if 'stdout.log' in command:
            return {
                'returncode': 0,
                'output': f'{{"type":"turn.completed"}}\nNEXUS_CODEX_OUTPUT_BASE64={encoded_output}\n',
            }
        return {'returncode': 0, 'output': ''}

    adapter._execute = fake_execute  # type: ignore[method-assign]

    result = await adapter.run_codex(_request(codex_executable=r'C:\codex.exe', working_dir=r'C:\work'))

    assert result.exit_code == 0
    assert result.stdout == '{"type":"turn.completed"}\n'
    assert result.output == 'done'
    assert runtime.launches
    assert 'nexus-codex-run-' in runtime.launches[0][-1]
    assert all('nexus-claude-run' not in command for command in commands)


@pytest.mark.asyncio
async def test_waa_install_prefers_sandbox_npm() -> None:
    from nexus.extensions.agents.codex.native_bridge.adapters.waa import WAACodexNativeBridgeAdapter

    adapter = object.__new__(WAACodexNativeBridgeAdapter)
    commands: list[str] = []

    async def fake_execute(command: str, *, timeout: float | None = None) -> dict[str, Any]:
        commands.append(command)
        if len(commands) == 1:
            return {'returncode': 0, 'output': r'C:\\Program Files\\nodejs\\npm.cmd'}
        return {'returncode': 0, 'output': 'codex-cli 0.143.0'}

    adapter._execute = fake_execute  # type: ignore[method-assign]
    executable = await adapter.ensure_codex(
        CodexInstallSpec(version='0.143.0', npm_registry='https://registry.npmmirror.com')
    )

    assert executable == r'C:\Windows\Temp\nexus-codex-cli\node_modules\.bin\codex.cmd'
    assert len(commands) == 2
    assert '@openai/codex@0.143.0' in commands[1]
    assert r'C:\\Program Files\\nodejs\\npm.cmd' in commands[1]


@pytest.mark.asyncio
async def test_waa_install_downloads_binary_url_and_verifies_checksum() -> None:
    from nexus.extensions.agents.codex.native_bridge.adapters.waa import WAACodexNativeBridgeAdapter

    adapter = object.__new__(WAACodexNativeBridgeAdapter)
    commands: list[str] = []

    async def fake_execute(command: str, *, timeout: float | None = None) -> dict[str, Any]:
        commands.append(command)
        assert timeout == 1800
        return {'returncode': 0, 'output': 'codex-cli 0.143.0'}

    adapter._execute = fake_execute  # type: ignore[method-assign]
    executable = await adapter.ensure_codex(
        CodexInstallSpec(
            version='0.143.0',
            npm_registry='https://registry.npmmirror.com',
            binary_url='http://cos-internal.example/codex.exe?signature=a&part=1',
            binary_checksum='ABC123',
        )
    )

    assert executable == r'C:\Windows\Temp\nexus-codex-cli\codex.exe'
    assert len(commands) == 1
    assert (
        "Invoke-WebRequest -UseBasicParsing -Uri 'http://cos-internal.example/codex.exe?signature=a&part=1'"
        in commands[0]
    )
    assert 'Get-FileHash' in commands[0]
    assert "$expectedChecksum = 'ABC123'" in commands[0]
    assert 'npm.cmd' not in commands[0]


@pytest.mark.asyncio
async def test_waa_install_raises_when_sandbox_npm_fails() -> None:
    from nexus.extensions.agents.codex.native_bridge.adapters.waa import WAACodexNativeBridgeAdapter

    adapter = object.__new__(WAACodexNativeBridgeAdapter)
    commands: list[str] = []

    async def fake_execute(command: str, *, timeout: float | None = None) -> dict[str, Any]:
        commands.append(command)
        if len(commands) == 1:
            return {'returncode': 0, 'output': r'C:\\Program Files\\nodejs\\npm.cmd'}
        return {'returncode': 1, 'error': 'npm timed out'}

    adapter._execute = fake_execute  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match='Failed to install Codex CLI via sandbox npm: npm timed out'):
        await adapter.ensure_codex(CodexInstallSpec(version='0.143.0', npm_registry='https://registry.npmmirror.com'))

    assert len(commands) == 2


@pytest.mark.asyncio
async def test_posix_install_falls_back_to_native_package_without_npm() -> None:
    """The 167 Gym Linux application images are not guaranteed to carry node."""
    installed = '/tmp/nexus-codex-cli/package/vendor/x86_64-unknown-linux-musl/bin/codex'

    class FakeRuntime:
        def __init__(self) -> None:
            self.commands: list[tuple[str, float]] = []

        async def run_command(self, *, command: str, timeout: float) -> Any:
            self.commands.append((command, timeout))
            if command.startswith(('test -x ', 'command -v codex', 'command -v npm')):
                return SimpleNamespace(return_code=1, stdout='', stderr='', status=CommandRunStatus.Finished)
            return SimpleNamespace(
                return_code=0,
                stdout=(
                    'codex 0.153.4-linux-x64 from https://registry.npmjs.org\n'
                    f'{NATIVE_EXECUTABLE_MARKER}{installed}\ncodex-cli 0.153.4\n'
                ),
                stderr='',
                status=CommandRunStatus.Finished,
            )

    runtime = FakeRuntime()
    adapter = BasePosixCodexNativeBridgeAdapter(runtime)  # type: ignore[arg-type]

    executable = await adapter.ensure_codex(
        CodexInstallSpec(version='latest', npm_registry='http://mirrors.internal/npm/')
    )

    assert executable == installed
    install, timeout = runtime.commands[-1]
    assert timeout == 1800.0
    assert 'npm install' not in install
    assert 'ref="${pinned}-${platform}"' in install
    assert '/@openai%2fcodex/${ref}' in install
    # configured registry first, then the public one as a fallback
    assert install.index('http://mirrors.internal/npm ') < install.index('https://registry.npmjs.org')
    assert 'tar -xzf' in install
    # Codex statically links musl and ships a single linux-<arch> build, so
    # unlike claude-code there is no libc variant to detect.
    assert '-musl' not in install
    assert 'platform="linux-${arch}"' in install


@pytest.mark.asyncio
async def test_posix_native_install_requires_the_reported_executable_path() -> None:
    class FakeRuntime:
        async def run_command(self, *, command: str, timeout: float) -> Any:
            if command.startswith(('test -x ', 'command -v codex', 'command -v npm')):
                return SimpleNamespace(return_code=1, stdout='', stderr='', status=CommandRunStatus.Finished)
            return SimpleNamespace(
                return_code=0, stdout='codex-cli 0.153.4\n', stderr='', status=CommandRunStatus.Finished
            )

    adapter = BasePosixCodexNativeBridgeAdapter(FakeRuntime())  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match='did not report an executable path'):
        await adapter.ensure_codex(CodexInstallSpec(version='latest', npm_registry=None))


@pytest.mark.asyncio
async def test_native_runtime_installs_and_runs_codex(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class FakeAdapter:
        async def ensure_codex(self, spec: Any) -> str:
            captured['spec'] = spec
            return '/tmp/codex'

        async def run_pre_start_scripts(self, scripts: Any, **kwargs: Any) -> None:
            captured['pre_start_scripts'] = scripts
            captured['pre_start_kwargs'] = kwargs

        async def run_codex(self, request: CodexRunRequest) -> CodexRunResult:
            captured['request'] = request
            return CodexRunResult(
                exit_code=0,
                stdout='{"type":"turn.completed"}\n',
                stderr='',
                output='done',
            )

    monkeypatch.setattr(
        'nexus.extensions.agents.codex.native_bridge.agent_runtime.build_codex_native_bridge_adapter',
        lambda runtime: FakeAdapter(),
    )
    result = await setup_and_run_native_bridge_codex(
        runtime=object(),
        launch=_request().launch,
        task='fix it',
        version='0.143.0',
        model='gpt-5.4',
        binary_url='http://cos-internal.example/codex',
        binary_checksum='abc123',
        extra_env={'NPM_CONFIG_REGISTRY': 'https://registry.npmjs.org/'},
        pre_start_scripts=[{'interpreter': 'bash', 'code': 'echo ready'}],
        mcp_servers=[{'name': 'everything', 'command': 'npx', 'args': ['server']}],
    )

    assert captured['spec'].version == '0.143.0'
    assert captured['spec'].binary_url == 'http://cos-internal.example/codex'
    assert captured['spec'].binary_checksum == 'abc123'
    assert captured['request'].model == 'gpt-5.4'
    assert captured['request'].version == '0.143.0'
    assert captured['pre_start_scripts'] == [{'interpreter': 'bash', 'code': 'echo ready'}]
    assert captured['pre_start_kwargs']['working_dir'] == '/workspace'
    assert captured['request'].mcp_servers == [{'name': 'everything', 'command': 'npx', 'args': ['server']}]
    assert captured['request'].extra_env == {'NPM_CONFIG_REGISTRY': 'https://registry.npmjs.org/'}
    assert result['success'] is True
    assert result['output'] == 'done'
    assert result['json_events_stdout'] == '{"type":"turn.completed"}\n'


@pytest.mark.asyncio
async def test_codex_native_bridge_agent_uses_common_bridge_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def fake_call(target: Any, **kwargs: Any) -> dict[str, Any]:
        captured['target'] = target
        captured.update(kwargs)
        return {
            'success': True,
            'output': 'done',
            'stderr': '',
            'exit_code': 0,
            'execution_time': 1.0,
        }

    monkeypatch.setattr('nexus.extensions.agents.codex.agent.call_native_bridge_runtime_agent', fake_call)
    monkeypatch.setattr('nexus.extensions.agents.codex.agent.get_current_span_id', lambda: None)

    output = await run_codex_native_bridge(
        CodexNativeBridgeInput(
            user_prompt='fix it',
            working_dir='/work',
            model='gpt-5.4',
            binary_url='http://cos-internal.example/codex',
            binary_checksum='abc123',
            extra_env={'CUSTOM': 'value'},
            pre_start_scripts=[{'interpreter': 'bash', 'code': 'echo ready'}],
            mcp_servers=[{'name': 'everything', 'command': 'npx'}],
        )
    )

    assert output.success is True
    assert captured['task'] == 'fix it'
    assert captured['api_protocol'] == 'responses'
    assert captured['working_dir'] == '/work'
    assert captured['version'] == '0.114.0'
    assert captured['model'] == 'gpt-5.4'
    assert captured['binary_url'] == 'http://cos-internal.example/codex'
    assert captured['binary_checksum'] == 'abc123'
    assert captured['extra_env'] == {'CUSTOM': 'value'}
    assert captured['pre_start_scripts'][0]['code'] == 'echo ready'
    assert captured['mcp_servers'] == [{'name': 'everything', 'command': 'npx'}]
