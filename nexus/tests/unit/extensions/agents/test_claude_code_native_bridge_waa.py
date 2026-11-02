from __future__ import annotations

import base64

from typing import Any

import pytest

from nexus.extensions.agents.claude_code.native_bridge.adapters.types import (
    ClaudeInstallSpec,
    ClaudeRunRequest,
)
from nexus.extensions.agents.claude_code.native_bridge.adapters.waa import (
    WAAClaudeNativeBridgeAdapter,
    _SessionPaths,
)
from nexus.runtime.bridge.adapters.base import BridgeLaunchResult
from nexus.runtime.providers.ags.waa import AGSRawWindowsAgentArenaRuntime


class FakeWAARuntime(AGSRawWindowsAgentArenaRuntime):
    def __init__(self, responses: list[dict[str, Any]] | None = None) -> None:
        self.responses = list(responses or [])
        self.scripts: list[str] = []
        self.launches: list[tuple[str | list[str], bool]] = []

    async def execute(self, command: str | list[str], shell: bool = False) -> Any:
        assert isinstance(command, list)
        assert command[:4] == ['powershell.exe', '-NoProfile', '-NonInteractive', '-EncodedCommand']
        self.scripts.append(base64.b64decode(command[4]).decode('utf-16-le'))
        if self.responses:
            return self.responses.pop(0)
        return {'returncode': 0, 'output': '', 'error': '', 'status': 'success'}

    async def setup_launch(self, command: str | list[str], shell: bool = True) -> str:
        self.launches.append((command, shell))
        return 'launched'


@pytest.fixture
def launch() -> BridgeLaunchResult:
    return BridgeLaunchResult(
        base_url='https://8765-instance.example.com',
        local_base_url='http://127.0.0.1:8765',
        headers={'X-Access-Token': 'token'},
    )


@pytest.mark.asyncio
async def test_waa_adapter_reuses_preinstalled_claude() -> None:
    runtime = FakeWAARuntime(
        [{'returncode': 0, 'output': 'C:\\Users\\Docker\\AppData\\Roaming\\npm\\claude.cmd\n', 'error': ''}]
    )
    adapter = WAAClaudeNativeBridgeAdapter(runtime)

    executable = await adapter.ensure_claude(
        ClaudeInstallSpec(ensure_claude=True, npm_registry='https://registry.example', version='latest')
    )

    assert executable == r'C:\Users\Docker\AppData\Roaming\npm\claude.cmd'
    assert 'Get-Command claude.cmd' in runtime.scripts[0]


@pytest.mark.asyncio
async def test_waa_adapter_installs_pinned_version_via_npm() -> None:
    runtime = FakeWAARuntime()
    adapter = WAAClaudeNativeBridgeAdapter(runtime)

    executable = await adapter.ensure_claude(
        ClaudeInstallSpec(ensure_claude=True, npm_registry='https://registry.npmjs.org', version='2.1.204')
    )

    assert executable == r'C:\Windows\Temp\nexus-claude-code\node_modules\.bin\claude.cmd'
    assert 'Get-Command claude.cmd' not in runtime.scripts[0]
    assert "'@anthropic-ai/claude-code@2.1.204'" in runtime.scripts[0]
    assert "'https://registry.npmjs.org'" in runtime.scripts[0]


@pytest.mark.asyncio
async def test_waa_adapter_installs_binary_url_without_npm() -> None:
    runtime = FakeWAARuntime([{'returncode': 0, 'output': '2.1.285 (Claude Code)\r\n', 'error': ''}])
    adapter = WAAClaudeNativeBridgeAdapter(runtime)

    executable = await adapter.ensure_claude(
        ClaudeInstallSpec(
            ensure_claude=True,
            npm_registry=None,
            version='2.1.285',
            binary_url='https://cos.example.com/claude-code-win32-x64.tgz?sig=1',
            binary_checksum='A' * 64,
        )
    )

    assert executable == r'C:\Windows\Temp\nexus-claude-code\package\claude.exe'
    assert len(runtime.scripts) == 1
    script = runtime.scripts[0]
    assert 'npm' not in script
    assert "Invoke-WebRequest -UseBasicParsing -Uri 'https://cos.example.com/claude-code-win32-x64.tgz?sig=1'" in script
    assert f"$expectedChecksum = '{'A' * 64}'" in script
    assert r"tar.exe -xzf 'C:\Windows\Temp\nexus-claude-code\claude.url.tmp'" in script
    assert r"& 'C:\Windows\Temp\nexus-claude-code\package\claude.exe' --version" in script


@pytest.mark.asyncio
async def test_waa_adapter_rejects_binary_url_version_mismatch() -> None:
    runtime = FakeWAARuntime([{'returncode': 0, 'output': '2.1.284 (Claude Code)\r\n', 'error': ''}])
    adapter = WAAClaudeNativeBridgeAdapter(runtime)

    with pytest.raises(RuntimeError, match='expected 2.1.285'):
        await adapter.ensure_claude(
            ClaudeInstallSpec(
                ensure_claude=True,
                npm_registry=None,
                version='2.1.285',
                binary_url='https://cos.example.com/claude.exe',
            )
        )


@pytest.mark.asyncio
async def test_waa_pre_start_injects_bridge_environment(launch: BridgeLaunchResult) -> None:
    runtime = FakeWAARuntime(
        [
            {'returncode': 0, 'output': 'C:\\Windows\\Temp\\nexus-work\n', 'error': ''},
            {'returncode': 0, 'output': '', 'error': ''},
        ]
    )
    adapter = WAAClaudeNativeBridgeAdapter(runtime)

    await adapter.run_pre_start_scripts(
        [{'interpreter': 'powershell', 'code': "Set-Content answer.txt 'ok'", 'env': {'K': 'V'}}],
        working_dir=r'%TEMP%\nexus-work',
        launch=launch,
        extra_env={'EXTRA': '1'},
    )

    assert "ExpandEnvironmentVariables('%TEMP%\\nexus-work')" in runtime.scripts[0]
    script = runtime.scripts[1]
    assert "$env:ANTHROPIC_BASE_URL = 'http://127.0.0.1:8765'" in script
    assert "$env:EXTRA = '1'" in script
    assert "$env:K = 'V'" in script
    assert "Set-Location -LiteralPath 'C:\\Windows\\Temp\\nexus-work'" in script


@pytest.mark.asyncio
async def test_waa_registers_mcp_servers(launch: BridgeLaunchResult) -> None:
    runtime = FakeWAARuntime(
        [
            {'returncode': 0, 'output': 'C:\\work\n', 'error': ''},
            {'returncode': 0, 'output': '', 'error': ''},
        ]
    )
    adapter = WAAClaudeNativeBridgeAdapter(runtime)

    await adapter.register_mcp_servers(
        r'C:\Program Files\Claude\claude.cmd',
        [{'name': 'files', 'command': 'python.exe', 'args': ['server script.py', '--root', r'C:\work']}],
        working_dir=r'C:\work',
        launch=launch,
        extra_env={'TOKEN': 'secret'},
    )

    script = runtime.scripts[1]
    assert "& 'C:\\Program Files\\Claude\\claude.cmd' @('mcp', 'add', '--transport', 'stdio', 'files'" in script
    assert "'python.exe', 'server script.py', '--root', 'C:\\work'" in script
    assert "$env:ANTHROPIC_BASE_URL = 'http://127.0.0.1:8765'" in script
    assert "$env:TOKEN = 'secret'" in script


@pytest.mark.asyncio
async def test_waa_run_claude_launches_background_process_and_reads_logs(
    monkeypatch: pytest.MonkeyPatch, launch: BridgeLaunchResult
) -> None:
    from nexus.extensions.agents.claude_code.native_bridge.adapters import waa

    monkeypatch.setattr(waa, '_POLL_INTERVAL', 0)
    runtime = FakeWAARuntime(
        [
            {'returncode': 0, 'output': 'C:\\work\n', 'error': ''},
            {'returncode': 0, 'output': '', 'error': ''},
            {'returncode': 0, 'output': 'STARTING\n', 'error': ''},
            {'returncode': 0, 'output': 'RUNNING\n', 'error': ''},
            {'returncode': 0, 'output': 'DONE 0\n', 'error': ''},
            {'returncode': 0, 'output': 'sorted: 1 2 3\n', 'error': ''},
            {'returncode': 0, 'output': '', 'error': ''},
            {'returncode': 0, 'output': '', 'error': ''},
        ]
    )
    adapter = WAAClaudeNativeBridgeAdapter(runtime)

    result = await adapter.run_claude(
        ClaudeRunRequest(
            claude_executable=r'C:\Users\Docker\AppData\Roaming\npm\claude.cmd',
            launch=launch,
            task='write quicksort',
            working_dir=r'C:\work',
            timeout=30,
        )
    )

    assert result.exit_code == 0
    assert result.stdout == 'sorted: 1 2 3\n'
    assert '[IO.File]::WriteAllBytes' in runtime.scripts[1]
    command, shell = runtime.launches[0]
    assert isinstance(command, list)
    assert command[:4] == ['cmd.exe', '/d', '/s', '/c']
    assert any(part.endswith('runner.cmd') for part in command)
    assert shell is False
    assert any('STARTING' in script and 'DONE ' in script for script in runtime.scripts)


def test_waa_runner_preserves_complete_multiline_prompt(launch: BridgeLaunchResult) -> None:
    adapter = WAAClaudeNativeBridgeAdapter(FakeWAARuntime())
    task = '请按以下顺序操作：\r\n1. 检查文件。\n2. 读取并输出内容。'

    script = adapter._build_runner_script(
        ClaudeRunRequest(
            claude_executable=r'C:\Users\Docker\AppData\Roaming\npm\claude.cmd',
            launch=launch,
            task=task,
            working_dir=r'C:\work',
            timeout=30,
        ),
        paths=_SessionPaths(
            directory=r'C:\session',
            stdout_log=r'C:\session\stdout.log',
            stderr_log=r'C:\session\stderr.log',
            exit_code=r'C:\session\exit_code',
            pid=r'C:\session\pid',
        ),
        working_dir=r'C:\work',
    )

    assert '请按以下顺序操作： 1. 检查文件。 2. 读取并输出内容。' in script
    assert '请按以下顺序操作：\n' not in script
    assert '检查文件。\n' not in script


@pytest.mark.asyncio
async def test_waa_run_claude_gone_still_kills_possible_descendants(
    launch: BridgeLaunchResult,
) -> None:
    runtime = FakeWAARuntime(
        [
            {'returncode': 0, 'output': 'C:\\work\n', 'error': ''},
            {'returncode': 0, 'output': '', 'error': ''},
            {'returncode': 0, 'output': 'GONE\n', 'error': ''},
        ]
    )
    adapter = WAAClaudeNativeBridgeAdapter(runtime)

    result = await adapter.run_claude(
        ClaudeRunRequest(
            claude_executable=r'C:\Users\Docker\AppData\Roaming\npm\claude.cmd',
            launch=launch,
            task='write quicksort',
            working_dir=r'C:\work',
            timeout=30,
        )
    )

    assert result.exit_code == -1
    assert 'exited without writing an exit code' in result.stderr
    assert any('taskkill.exe /PID $processId /T /F' in script for script in runtime.scripts)


def test_waa_claude_adapter_factory() -> None:
    from nexus.extensions.agents.claude_code.native_bridge.adapters import build_claude_native_bridge_adapter

    runtime = FakeWAARuntime()
    adapter = build_claude_native_bridge_adapter(runtime)

    assert isinstance(adapter, WAAClaudeNativeBridgeAdapter)
    assert adapter.runtime is runtime
