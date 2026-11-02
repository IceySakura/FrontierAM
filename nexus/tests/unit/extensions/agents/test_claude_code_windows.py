"""Minimal Windows support for the installed native Claude Code path."""

import base64
import json
import os
import shlex
import sys

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from nexus.extensions.agents import installed_common
from nexus.extensions.agents.claude_code import install_support, runtime_setup
from nexus.runtime.executors import command
from nexus.runtime.executors.models.command import CommandRunStatus, RunCommandResponse
from nexus.utils.platform import is_windows


def test_windows_command_quotes_executable_and_literal_arguments(monkeypatch):
    monkeypatch.setattr(installed_common, 'is_windows', lambda: True)
    assert installed_common.build_runtime_command([r'C:\Program Files\nodejs\npm.CMD', "a'b", '$value']) == (
        "& 'C:\\Program Files\\nodejs\\npm.CMD' 'a''b' '$value'"
    )


def test_posix_command_retains_shlex_quoting(monkeypatch):
    monkeypatch.setattr(installed_common, 'is_windows', lambda: False)
    argv = ['/path with spaces/npm', 'install', "a'b"]
    assert installed_common.build_runtime_command(argv) == shlex.join(argv)


@pytest.mark.parametrize('separator', [';', ':'])
def test_path_uses_platform_separator(monkeypatch, separator):
    from nexus.utils import shell

    monkeypatch.setattr(os, 'pathsep', separator)
    monkeypatch.setattr(shell, 'get_clean_sandbox_env', lambda: {'PATH': 'inherited'})
    assert installed_common.build_runtime_path('first', 'second') == separator.join(['first', 'second', 'inherited'])


def test_windows_native_executable_does_not_create_symlink(monkeypatch, tmp_path):
    native = tmp_path / 'claude.exe'
    native.write_bytes(b'MZ')
    monkeypatch.setattr(install_support, 'is_windows', lambda: True)
    monkeypatch.setattr(install_support, '_resolve_claude_executable', lambda *a, **kw: native)
    monkeypatch.setattr(Path, 'symlink_to', lambda *a, **kw: pytest.fail('Windows native launch needs no symlink'))
    assert install_support._make_self_executable(str(tmp_path)) == native.resolve()


@pytest.mark.asyncio
async def test_windows_python_pre_start_uses_call_operator(monkeypatch):
    monkeypatch.setattr(installed_common, 'is_windows', lambda: True)
    monkeypatch.setattr(runtime_setup, 'find_runtime_binary', lambda _: r'C:\Program Files\Python\python.exe')
    run = AsyncMock(return_value=RunCommandResponse(status=CommandRunStatus.Finished, return_code=0))
    monkeypatch.setattr(command, 'run_command', run)
    await runtime_setup._run_pre_start_script({'interpreter': 'python', 'code': "print('hello')"}, 'work', {})
    request = run.call_args.args[0]
    assert request.command == "& 'C:\\Program Files\\Python\\python.exe' '-c' 'print(''hello'')'"


@pytest.mark.skipif(not is_windows(), reason='Real Windows PowerShell argv transport')
async def test_windows_mcp_add_json_preserves_unicode_and_quotes(tmp_path):
    from nexus.extensions.agents.claude_code.mcp import register_mcp_servers

    # Python stands in for Claude: its first argument is a script named mcp.
    (tmp_path / 'mcp').write_text(
        "import json, pathlib, sys; pathlib.Path('received.json').write_text(json.dumps(sys.argv[1:]), encoding='utf-8')",
        encoding='utf-8',
    )
    config = {
        'type': 'http',
        'url': 'http://localhost/mcp',
        'headers': {'Authorization': 'Bearer test "quoted" 你好 $value'},
    }
    await register_mcp_servers(
        sys.executable,
        working_dir=str(tmp_path),
        claude_env={},
        mcp_servers=[{'name': "server's name", 'client': {'transport': 'streamable_http', **config}}],
    )
    received = json.loads((tmp_path / 'received.json').read_text(encoding='utf-8'))
    assert received[:2] == ['add-json', "server's name"]
    assert json.loads(received[2]) == config


def test_windows_mcp_cwd_uses_powershell(monkeypatch):
    from nexus.extensions.agents.claude_code import mcp

    monkeypatch.setattr(mcp, 'is_windows', lambda: True)
    monkeypatch.setattr(installed_common, 'is_windows', lambda: True)
    native = mcp.mcp_config_servers_from_specs(
        [{'name': 'local', 'client': {'transport': 'stdio', 'command': 'python', 'cwd': "C:/work's dir"}}]
    )['local']
    decoded = base64.b64decode(native['args'][-1]).decode('utf-16-le')
    assert "Set-Location -LiteralPath 'C:/work''s dir'" in decoded
    assert "& 'python'" in decoded
