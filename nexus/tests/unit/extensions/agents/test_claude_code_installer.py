from unittest.mock import AsyncMock, Mock

import pytest

from nexus.extensions.agents import installed_common
from nexus.extensions.agents.claude_code import install_support, installer
from nexus.extensions.agents.claude_code.installer import _alternate_linux_native_package
from nexus.runtime.executors import command
from nexus.runtime.executors.models.command import CommandRunStatus, RunCommandResponse


def test_alternate_linux_native_package_switches_libc_variant():
    assert (
        _alternate_linux_native_package('@anthropic-ai/claude-code-linux-x64-musl')
        == '@anthropic-ai/claude-code-linux-x64'
    )
    assert (
        _alternate_linux_native_package('@anthropic-ai/claude-code-linux-x64')
        == '@anthropic-ai/claude-code-linux-x64-musl'
    )


def test_alternate_linux_native_package_ignores_non_linux_packages():
    assert _alternate_linux_native_package('@anthropic-ai/claude-code-darwin-arm64') is None


@pytest.mark.parametrize('windows', [True, False])
async def test_install_retry_preserves_platform_command_and_native_package(monkeypatch, tmp_path, windows):
    package = '@anthropic-ai/claude-code-' + ('win32-x64' if windows else 'linux-x64-musl')
    expected_package = package if windows else '@anthropic-ai/claude-code-linux-x64'
    native = tmp_path / 'claude.exe'
    monkeypatch.setattr(installed_common, 'is_windows', lambda: windows)
    monkeypatch.setattr(installer, '_uses_native_subpackages', lambda _: True)
    monkeypatch.setattr(install_support, '_detect_claude_native_package', lambda: package)
    monkeypatch.setattr(install_support, '_resolve_claude_executable', lambda *a, **kw: native)
    monkeypatch.setattr(install_support, 'get_installed_version', lambda _: None)
    monkeypatch.setattr(install_support, '_find_native_package_executable', Mock(side_effect=[None, native]))
    make_executable = Mock(return_value=native)
    monkeypatch.setattr(install_support, '_make_self_executable', make_executable)
    npm = 'C:/Program Files/nodejs/npm.cmd' if windows else '/opt/node bin/npm'
    monkeypatch.setattr(installed_common, 'find_runtime_binary', lambda name: npm if name == 'npm' else None)
    sleep = AsyncMock()
    monkeypatch.setattr(installer.asyncio, 'sleep', sleep)
    run = AsyncMock(
        side_effect=[
            RunCommandResponse(
                status=CommandRunStatus.Finished, return_code=1, stderr='ECONNRESET' if windows else 'EBADPLATFORM'
            ),
            RunCommandResponse(status=CommandRunStatus.Finished, return_code=0),
        ]
    )
    monkeypatch.setattr(command, 'run_command', run)

    assert await installer.install_claude_code_via_npm(str(tmp_path), version='2.1.185') == native
    assert run.await_count == 2
    first, second = [call.args[0] for call in run.call_args_list]
    assert package + '@2.1.185' in first.command
    assert expected_package + '@2.1.185' in second.command
    assert first.command.startswith(installed_common.build_runtime_command([npm]))
    assert '--ignore-scripts' in second.command
    assert make_executable.call_args.kwargs['preferred_package'] == expected_package
    if windows:
        sleep.assert_awaited_once_with(1)
    else:
        assert second.extra_env['NPM_CONFIG_LIBC'] == 'glibc'
