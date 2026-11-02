"""The same installer policies must hold on local and target filesystems."""

from __future__ import annotations

import json
import shlex
import sys

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from nexus.extensions.agents import installed_common
from nexus.extensions.agents.claude_code import install_support
from nexus.extensions.agents.claude_code import installer as claude_installer
from nexus.extensions.agents.codex import installer as codex_installer
from nexus.extensions.agents.installed_common.execution import AgentExecutionTransport, DirectExecutionTarget
from nexus.extensions.agents.installed_common.target_files import operate
from nexus.runtime.executors.models.command import CommandRunStatus, RunCommandResponse


class FixtureTargetTransport(AgentExecutionTransport):
    """Run target-side helpers against a temporary filesystem, without npm/network."""

    def __init__(self, home: Path):
        super().__init__(DirectExecutionTarget(user='alice'))
        self._identity = {'uid': 1000, 'gid': 1000, 'user': 'alice', 'home': str(home), 'path': '/usr/bin:/bin'}

    async def file(self, operation, path, **kwargs):
        return operate({'operation': operation, 'path': str(path), **kwargs})

    async def python(self, source, payload, *, timeout=60):
        namespace = {'__name__': 'target_install_support'}
        exec(source, namespace)
        return namespace['operate'](payload)

    async def find_binary(self, name):
        return name if name in {'codex', 'claude'} else '/fixture/bin/' + name


@pytest.fixture(params=['local', 'target'])
def install_context(request, monkeypatch, tmp_path):
    transport = AgentExecutionTransport() if request.param == 'local' else FixtureTargetTransport(tmp_path)
    calls = []
    responses = []
    after_install = [lambda: None]

    async def execute(argv, **kwargs):
        calls.append((argv, kwargs))
        result = responses.pop(0) if responses else RunCommandResponse(status=CommandRunStatus.Finished, return_code=0)
        if result.return_code == 0:
            after_install[0]()
        return result

    async def run(request):
        return await execute(shlex.split(request.command), env=request.extra_env, working_dir=request.cwd)

    if transport.is_local:
        monkeypatch.setattr(
            installed_common,
            'find_runtime_binary',
            lambda name: None if name in {'codex', 'claude'} else '/fixture/bin/' + name,
        )
        monkeypatch.setattr('nexus.runtime.executors.command.run_command', run)
    else:
        monkeypatch.setattr(transport, 'exec_as_agent', execute)
    return transport, calls, responses, after_install


@pytest.mark.parametrize('metadata', ['{"version":', '[]', 'null', '{"version": 123}', None])
async def test_codex_damaged_cache_reinstalls_then_reuses(install_context, tmp_path, metadata):
    transport, calls, _, after_install = install_context
    binary = tmp_path / 'node_modules/.bin/codex'
    binary.parent.mkdir(parents=True)
    binary.touch()
    package_json = tmp_path / 'node_modules/@openai/codex/package.json'
    package_json.parent.mkdir(parents=True)
    if metadata is not None:
        package_json.write_text(metadata)
    after_install[0] = lambda: package_json.write_text('{"version":"0.114.0"}')
    assert await codex_installer.install_codex_cli(str(tmp_path), transport=transport) == binary
    assert len(calls) == 1
    assert calls[0][0][-1] == '@openai/codex@0.114.0'
    assert calls[0][1]['env']['NPM_CONFIG_STRICT_SSL'] == 'false'
    assert await codex_installer.install_codex_cli(str(tmp_path), transport=transport) == binary
    assert len(calls) == 1


async def test_codex_latest_reinstalls_and_checks_binary(install_context, tmp_path):
    transport, calls, _, after_install = install_context
    binary = tmp_path / 'node_modules/.bin/codex'
    binary.parent.mkdir(parents=True)
    after_install[0] = binary.touch
    for _ in range(2):
        assert await codex_installer.install_codex_cli(str(tmp_path), version='latest', transport=transport) == binary
    assert len(calls) == 2
    assert all(argv[-1] == '@openai/codex' for argv, _ in calls)
    after_install[0] = lambda: None
    binary.unlink()
    with pytest.raises(RuntimeError, match='binary is missing'):
        await codex_installer.install_codex_cli(str(tmp_path), transport=transport)


async def test_codex_install_failure_has_same_diagnostics(install_context, tmp_path):
    transport, _, responses, _ = install_context
    responses.append(RunCommandResponse(status=CommandRunStatus.Finished, return_code=1, stderr='npm failed'))
    with pytest.raises(RuntimeError, match='Failed to install Codex CLI: npm failed'):
        await codex_installer.install_codex_cli(str(tmp_path), transport=transport)


def create_legacy_claude(directory, version):
    package = directory / 'node_modules/@anthropic-ai/claude-code'
    package.mkdir(parents=True, exist_ok=True)
    (package / 'package.json').write_text(json.dumps({'version': version}))
    entry = package / 'cli.js'
    entry.write_text('#!/usr/bin/env node\nconsole.log("fixture");\n')
    link = directory / 'node_modules/.bin/claude'
    link.parent.mkdir(parents=True, exist_ok=True)
    link.unlink(missing_ok=True)
    link.symlink_to(entry)


async def test_claude_retries_and_reuses_legacy_install_in_same_filesystem(install_context, monkeypatch, tmp_path):
    transport, calls, responses, after_install = install_context
    monkeypatch.setattr(install_support, 'find_runtime_binary', lambda name: sys.executable)
    if not transport.is_local:
        original_python = transport.python

        async def python(source, payload, **kwargs):
            source = source.replace("node = find_runtime_binary('node')", f'node = {sys.executable!r}')
            return await original_python(source, payload, **kwargs)

        monkeypatch.setattr(transport, 'python', python)
    responses.append(RunCommandResponse(status=CommandRunStatus.Finished, return_code=1, stderr='ECONNRESET'))
    sleep = AsyncMock()
    monkeypatch.setattr(claude_installer.asyncio, 'sleep', sleep)
    after_install[0] = lambda: create_legacy_claude(tmp_path, '2.1.112')
    binary = await claude_installer.install_claude_code_via_npm(str(tmp_path), version='2.1.112', transport=transport)
    assert binary == tmp_path / 'claude'
    assert binary.is_symlink()
    assert len(calls) == 2
    assert all('--ignore-scripts' not in argv and '--force' not in argv for argv, _ in calls)
    sleep.assert_awaited_once_with(1)
    assert (
        await claude_installer.install_claude_code_via_npm(str(tmp_path), version='2.1.112', transport=transport)
        == binary
    )
    assert len(calls) == 2
    assert await claude_installer.install_claude_code_via_npm(str(tmp_path), transport=transport) == binary
    assert len(calls) == 2


async def test_claude_probed_package_is_authoritative_and_requires_exact_binary(install_context, tmp_path):
    transport, calls, responses, _ = install_context
    package = '@anthropic-ai/claude-code-linux-x64-musl'
    responses.append(RunCommandResponse(status=CommandRunStatus.Finished, return_code=1, stderr='EBADPLATFORM'))
    with pytest.raises(RuntimeError, match='EBADPLATFORM'):
        await claude_installer.install_claude_code_via_npm(
            str(tmp_path),
            version='2.1.185',
            native_package=package,
            transport=transport,
        )
    assert len(calls) == 1
    assert '--force' in calls[0][0]
    assert '--ignore-scripts' in calls[0][0]
    assert package + '@2.1.185' in calls[0][0]
    assert calls[0][1]['env']['NPM_CONFIG_LIBC'] == 'musl'
    with pytest.raises(RuntimeError, match='expected native package was not found'):
        await claude_installer.install_claude_code_via_npm(
            str(tmp_path),
            version='2.1.185',
            native_package=package,
            transport=transport,
        )


@pytest.mark.parametrize('version', [None, 'latest', '2.1.185'])
async def test_claude_native_install_uses_shared_cache_policy(install_context, monkeypatch, tmp_path, version):
    transport, calls, _, after_install = install_context
    package = '@anthropic-ai/claude-code-linux-x64'
    monkeypatch.setattr(install_support, '_detect_claude_native_package', lambda: package)
    if not transport.is_local:
        original_python = transport.python

        async def python(source, payload, **kwargs):
            if payload['operation'] == 'detect':
                return package
            return await original_python(source, payload, **kwargs)

        monkeypatch.setattr(transport, 'python', python)

    def install():
        metadata = tmp_path / 'node_modules/@anthropic-ai/claude-code/package.json'
        metadata.parent.mkdir(parents=True, exist_ok=True)
        metadata.write_text('{"version":"2.1.185"}')
        binary = tmp_path / 'node_modules' / package / 'claude'
        binary.parent.mkdir(parents=True, exist_ok=True)
        binary.write_bytes(b'native binary fixture')

    after_install[0] = install
    for _ in range(2):
        binary = await claude_installer.install_claude_code_via_npm(str(tmp_path), version=version, transport=transport)
        assert binary == tmp_path / 'claude'
        assert binary.resolve() == tmp_path / 'node_modules' / package / 'claude'
    assert len(calls) == (2 if version == 'latest' else 1)
    assert ('--force' in calls[0][0]) is (not transport.is_local)
    assert '--ignore-scripts' in calls[0][0]
