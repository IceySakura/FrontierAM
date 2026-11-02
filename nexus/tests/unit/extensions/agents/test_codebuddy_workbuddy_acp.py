from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import subprocess
import tarfile

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from nexus.extensions.agents.codebuddy import workbuddy_acp
from nexus.extensions.agents.codebuddy.workbuddy_acp import (
    WorkBuddyAcpOptions,
    build_workbuddy_acp_argv,
    build_workbuddy_acp_env,
    check_node_version,
    ensure_workbuddy_cli,
    prepare_workbuddy_config_dir,
    upload_workbuddy_package,
)


_SHA = 'a' * 64


def _package_tgz(version: str = '5.11.0', *, with_entry: bool = True) -> bytes:
    """A minimal wb linux-x64 tarball: ``package/bin/wb.js`` and ``package/package.json``."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz') as archive:
        members = {'package/package.json': json.dumps({'name': '@tencent-ai/workbuddy-cli', 'version': version})}
        if with_entry:
            members['package/bin/wb.js'] = '#!/usr/bin/env node\n'
        for name, text in members.items():
            data = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def _split(data: bytes, directory: Path, parts: int = 2) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    size = len(data) // parts + 1
    paths = []
    for index in range(parts):
        path = directory / f'wb.tgz.part{index:03d}'
        path.write_bytes(data[index * size : (index + 1) * size])
        paths.append(str(path))
    return paths


def _fake_node(tmp_path: Path, version: str) -> str:
    node = tmp_path / f'node-{version}'
    node.write_text(f'#!/bin/sh\necho {version}\n')
    node.chmod(0o755)
    return str(node)


@pytest.fixture
def install_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    image_dir = tmp_path / 'image-workbuddy'
    root = tmp_path / 'workbuddy-acp'
    monkeypatch.setattr(workbuddy_acp, 'WORKBUDDY_PREINSTALLED_ROOT', tmp_path / 'nix-workbuddy')
    monkeypatch.setattr(workbuddy_acp, 'WORKBUDDY_DEFAULT_INSTALL_DIR', str(image_dir))
    monkeypatch.setattr(workbuddy_acp, 'WORKBUDDY_ACP_INSTALL_ROOT', root)
    monkeypatch.setattr(workbuddy_acp, '_INSTALL_LOCK', asyncio.Lock())
    return image_dir, root


def test_runtime_options_keep_the_host_path_out_of_the_sandbox() -> None:
    options = WorkBuddyAcpOptions(package_path='/host/wb.tgz', package_sha256=_SHA.upper(), mode='craft')
    assert options.runtime_options(['/tmp/p0', '/tmp/p1']) == {
        'version': '5.11.0',
        'cli_path': None,
        'package_parts': ['/tmp/p0', '/tmp/p1'],
        'package_sha256': _SHA,
        'preset': None,
        'mode': 'craft',
        'effort': None,
        'features': {'ecosystem': True, 'tencentDocsEngine': True},
        'fork_mode': 'quick',
        'permission_mode': 'bypassPermissions',
    }
    computed = 'b' * 64
    assert options.runtime_options(None, package_sha256=computed)['package_sha256'] == computed
    assert options.runtime_options(None)['package_parts'] is None


@pytest.mark.parametrize(
    ('fields', 'expected'), [({'permission_mode': 'fullAccess'}, 'fullAccess'), ({'permission_mode': None}, None)]
)
def test_data_overrides_the_bypass_permissions_default(fields: dict[str, Any], expected: str | None) -> None:
    assert WorkBuddyAcpOptions.model_validate(fields).runtime_options(None)['permission_mode'] == expected


@pytest.mark.parametrize(
    'fields',
    [{'package_sha256': 'not-a-digest'}, {'mode': 'yolo'}, {'permission_mode': 'auto'}, {'unknown': 1}],
)
def test_options_reject_values_wb_would_refuse(fields: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match=next(iter(fields))):
        WorkBuddyAcpOptions(**fields)


def test_argv_selects_the_custom_local_model_and_appends_only_given_defaults() -> None:
    assert build_workbuddy_acp_argv('/node', '/wb.js', model_id='m', version='5.11.0') == [
        '/node',
        '/wb.js',
        '--acp',
        '--model',
        'custom-local:m',
    ]
    cmd = build_workbuddy_acp_argv(
        '/node',
        '/wb.js',
        model_id='custom-local:m',
        version='5.11.0',
        preset='work',
        mode='plan',
        effort='high',
        features={'rollout': True},
        extra_cli_args=['--extra'],
    )
    assert cmd[3:] == [
        '--model',
        'custom-local:m',
        '--preset',
        'work',
        '--mode',
        'plan',
        '--effort',
        'high',
        '--features',
        '{"rollout":true}',
        '--extra',
    ]


def test_env_pins_model_routing_and_keeps_web_search_credentials(caplog: pytest.LogCaptureFixture) -> None:
    base_env = {
        'HOME': '/caller-home',
        'PATH': '/codebuddy/bin',
        'IS_SANDBOX': '1',
        'CODEBUDDY_API_KEY': 'caller-key',
        'NODE_OPTIONS': '--require /preload.js',
        'NO_COLOR': '1',
        'SSL_CERT_FILE': '/custom/ca.pem',
        'CODEBUDDY_INTERNET_ENVIRONMENT': 'ioa',
    }
    with caplog.at_level(logging.WARNING, logger=workbuddy_acp.__name__):
        env = build_workbuddy_acp_env(
            base_env=base_env,
            home='/session-home',
            config_dir='/session-home/workbuddy',
            proxy_base_url='http://127.0.0.1:9/v1',
            api_key='caller-key',
        )
    assert env['HOME'] == '/session-home'
    assert env['CODEBUDDY_BASE_URL'] == 'http://127.0.0.1:9/v1', 'Model traffic must stay on the sandbox proxy'
    assert env['CODEBUDDY_API_KEY'] == 'caller-key', 'Web search authentication is separate from models.json'
    assert env['WORKBUDDY_CONFIG_DIR'] == '/session-home/workbuddy'
    assert env['WORKBUDDY_USER_DATA_DIR'] == '/session-home/workbuddy/app', 'app-config.json is read from here'
    assert env['NODE_OPTIONS'] == '--require /preload.js'
    assert env['SSL_CERT_FILE'] == '/custom/ca.pem', 'A caller CA bundle wins over the default'
    assert 'NODE_EXTRA_CA_CERTS' in env
    assert 'CODEBUDDY_INTERNET_ENVIRONMENT' not in env
    warnings = [record.getMessage() for record in caplog.records]
    assert any('CODEBUDDY_INTERNET_ENVIRONMENT' in message for message in warnings)
    assert not any('Ignoring protected' in message for message in warnings), 'Managed keys are not a caller override'


def test_env_warns_when_a_caller_overrides_a_protected_route(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger=workbuddy_acp.__name__):
        env = build_workbuddy_acp_env(
            base_env={'CODEBUDDY_BASE_URL': 'http://elsewhere/v1'},
            home='/h',
            config_dir='/h/workbuddy',
            proxy_base_url='http://127.0.0.1:9/v1',
            api_key='k',
        )
    assert env['CODEBUDDY_BASE_URL'] == 'http://127.0.0.1:9/v1'
    assert any('CODEBUDDY_BASE_URL' in record.getMessage() for record in caplog.records)


def test_config_dir_names_the_proxy_and_turns_memory_off(tmp_path: Path) -> None:
    config_dir = tmp_path / 'workbuddy'
    (config_dir / 'app').mkdir(parents=True)
    (config_dir / 'app' / 'app-config.json').write_text(json.dumps({'theme': 'dark', 'generateMemoryEnabled': True}))
    prepare_workbuddy_config_dir(
        config_dir, proxy_url='http://127.0.0.1:9/v1/chat/completions', model_id='m', api_key='k'
    )
    model = json.loads((config_dir / 'models.json').read_text())['models'][0]
    assert model['url'] == 'http://127.0.0.1:9/v1/chat/completions'
    assert model['maxOutputTokens'] == 65536
    assert json.loads((config_dir / 'app' / 'app-config.json').read_text()) == {
        'theme': 'dark',
        'generateMemoryEnabled': False,
    }


def test_config_dir_links_image_skills_and_keeps_session_ones(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image_root = tmp_path / 'nix-workbuddy'
    for name in ('pptx', 'xlsx'):
        (image_root / 'skills' / name).mkdir(parents=True)
        (image_root / 'skills' / name / 'SKILL.md').write_text(f'---\nname: {name}\n---\n')
    (image_root / 'skills' / 'notes').mkdir()
    monkeypatch.setattr(workbuddy_acp, 'WORKBUDDY_PREINSTALLED_ROOT', image_root)
    config_dir = tmp_path / 'workbuddy'
    (config_dir / 'skills' / 'xlsx').mkdir(parents=True)

    for _ in range(2):
        prepare_workbuddy_config_dir(
            config_dir, proxy_url='http://127.0.0.1:9/v1/chat/completions', model_id='m', api_key='k'
        )

    skills = config_dir / 'skills'
    assert (skills / 'pptx').resolve() == image_root / 'skills' / 'pptx'
    assert not (skills / 'xlsx').is_symlink(), 'A skill already in the session wins over the image copy'
    assert not (skills / 'notes').exists(), 'A directory without SKILL.md is not a skill'


def test_env_appends_the_image_toolchain_after_the_workspace_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_root = tmp_path / 'nix-workbuddy'
    toolchain = image_root / 'toolchain'
    for directory in ('bin', 'node/node_modules', 'etc/fonts'):
        (toolchain / directory).mkdir(parents=True)
    (toolchain / 'etc' / 'fonts' / 'fonts.conf').write_text('<fontconfig/>')
    monkeypatch.setattr(workbuddy_acp, 'WORKBUDDY_PREINSTALLED_ROOT', image_root)
    home = tmp_path / 'h'

    def env_for(base_env: dict[str, str]) -> dict[str, str]:
        return build_workbuddy_acp_env(
            base_env=base_env,
            home=str(home),
            config_dir=str(home / 'workbuddy'),
            proxy_base_url='http://127.0.0.1:9/v1',
            api_key='k',
        )

    env = env_for({})
    assert env['PATH'].split(':')[-1] == str(toolchain / 'bin'), 'Workspace tools of the same name must win'
    assert env['NODE_PATH'] == str(toolchain / 'node' / 'node_modules')
    assert env['FONTCONFIG_FILE'] == str(toolchain / 'etc' / 'fonts' / 'fonts.conf')

    env = env_for({'NODE_PATH': '/workspace/node_modules', 'FONTCONFIG_FILE': '/etc/fonts/fonts.conf'})
    assert env['NODE_PATH'] == f'/workspace/node_modules:{toolchain / "node" / "node_modules"}'
    assert env['FONTCONFIG_FILE'] == '/etc/fonts/fonts.conf'

    monkeypatch.setattr(workbuddy_acp, 'WORKBUDDY_PREINSTALLED_ROOT', tmp_path / 'missing')
    home = tmp_path / 'other-home'
    env = env_for({})
    assert str(toolchain) not in env['PATH']
    assert 'NODE_PATH' not in env
    assert 'FONTCONFIG_FILE' not in env
    assert not (home / '.profile').exists()


@pytest.mark.parametrize('profile', ['.profile', '.zshrc'])
def test_shell_profile_appends_the_image_toolchain_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, profile: str
) -> None:
    image_root = tmp_path / 'nix-workbuddy'
    toolchain = image_root / 'toolchain'
    for directory in ('bin', 'node/node_modules'):
        (toolchain / directory).mkdir(parents=True)
    monkeypatch.setattr(workbuddy_acp, 'WORKBUDDY_PREINSTALLED_ROOT', image_root)
    home = tmp_path / 'h'
    home.mkdir()
    (home / profile).write_text('export KEEP=1')
    for _ in range(2):
        build_workbuddy_acp_env(
            base_env={},
            home=str(home),
            config_dir=str(home / 'workbuddy'),
            proxy_base_url='http://127.0.0.1:9/v1',
            api_key='k',
        )

    script = f'. ./{profile}; . ./{profile}; echo "$KEEP|$PATH|$NODE_PATH"'
    result = subprocess.run(
        ['sh', '-c', script], cwd=home, env={'PATH': '/usr/bin:/bin'}, capture_output=True, text=True, check=True
    )
    keep, path, node_path = result.stdout.strip().split('|')
    assert keep == '1', 'Existing profile content stays'
    assert path == f'/usr/bin:/bin:{toolchain / "bin"}'
    assert node_path == str(toolchain / 'node' / 'node_modules')


@pytest.mark.parametrize('version', ['v22.16.0', 'v22.19.0', 'v24.1.0'])
async def test_supported_node_versions_pass(tmp_path: Path, version: str) -> None:
    assert await check_node_version(_fake_node(tmp_path, version)), 'The declared Node range must be accepted'


@pytest.mark.parametrize('version', ['v20.11.1', 'v22.15.1', 'v23.4.0'])
async def test_unsupported_node_versions_name_the_requirement(tmp_path: Path, version: str) -> None:
    with pytest.raises(RuntimeError, match=r'\^22\.16\.0 \|\| >=24'):
        await check_node_version(_fake_node(tmp_path, version))


async def test_install_from_parts_verifies_and_unpacks_into_the_versioned_dir(
    tmp_path: Path, install_roots: tuple[Path, Path]
) -> None:
    _, root = install_roots
    data = _package_tgz()
    digest = hashlib.sha256(data).hexdigest()
    parts = _split(data, tmp_path / 'parts')
    options = WorkBuddyAcpOptions(package_sha256=digest).runtime_options(parts)
    wb_js = await ensure_workbuddy_cli(options, node_path=_fake_node(tmp_path, 'v22.19.0'))
    assert wb_js == root / f'5.11.0-{digest[:12]}' / 'package' / 'bin' / 'wb.js'
    assert wb_js.is_file()
    assert sorted(path.name for path in root.iterdir()) == [f'5.11.0-{digest[:12]}'], 'Staging files are removed'
    again = await ensure_workbuddy_cli({**options, 'package_parts': None}, node_path=_fake_node(tmp_path, 'v24.0.0'))
    assert again == wb_js, 'An installed package needs no parts'


async def test_concurrent_sessions_unpack_the_package_once(
    tmp_path: Path, install_roots: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _package_tgz()
    parts = _split(data, tmp_path / 'parts')
    extract = workbuddy_acp._extract_archive
    calls: list[Path] = []

    async def counting_extract(archive: Path, destination: Path) -> None:
        calls.append(destination)
        await extract(archive, destination)

    monkeypatch.setattr(workbuddy_acp, '_extract_archive', counting_extract)
    options = WorkBuddyAcpOptions(package_sha256=hashlib.sha256(data).hexdigest()).runtime_options(parts)
    node = _fake_node(tmp_path, 'v22.19.0')
    paths = await asyncio.gather(*(ensure_workbuddy_cli(options, node_path=node) for _ in range(3)))
    assert len(set(paths)) == 1
    assert len(calls) == 1


async def test_preinstalled_version_in_the_image_wins(tmp_path: Path, install_roots: tuple[Path, Path]) -> None:
    image_dir, _ = install_roots
    with tarfile.open(fileobj=io.BytesIO(_package_tgz()), mode='r:gz') as archive:
        archive.extractall(image_dir, filter='data')
    options = WorkBuddyAcpOptions().runtime_options(['/missing.part000'])
    wb_js = await ensure_workbuddy_cli(options, node_path=_fake_node(tmp_path, 'v22.19.0'))
    assert wb_js == image_dir / 'package' / 'bin' / 'wb.js'


@pytest.mark.parametrize('versioned', [False, True])
async def test_runtime_image_package_runs_without_an_upload(
    tmp_path: Path, install_roots: tuple[Path, Path], versioned: bool
) -> None:
    directory = workbuddy_acp.WORKBUDDY_PREINSTALLED_ROOT
    if versioned:
        directory /= '5.11.0'
    with tarfile.open(fileobj=io.BytesIO(_package_tgz()), mode='r:gz') as archive:
        archive.extractall(directory, filter='data')
    entry = await ensure_workbuddy_cli(
        WorkBuddyAcpOptions().runtime_options(None), node_path=_fake_node(tmp_path, 'v22.19.0')
    )
    assert entry == directory / 'package' / 'bin' / 'wb.js'


async def test_runtime_image_npm_install_runs_without_an_upload(
    tmp_path: Path, install_roots: tuple[Path, Path]
) -> None:
    root = workbuddy_acp.WORKBUDDY_PREINSTALLED_ROOT
    package = root / 'node_modules' / '@tencent-ai' / 'workbuddy-cli'
    (package / 'bin').mkdir(parents=True)
    entry = package / 'bin' / 'wb.js'
    entry.write_text('#!/usr/bin/env node\n')
    (package / 'package.json').write_text(json.dumps({'version': '5.11.0'}))
    (root / 'node_modules' / '.bin').mkdir()
    (root / 'node_modules' / '.bin' / 'wb').symlink_to(entry)
    found = await ensure_workbuddy_cli(
        WorkBuddyAcpOptions().runtime_options(None), node_path=_fake_node(tmp_path, 'v22.19.0')
    )
    assert found == entry


async def test_runtime_image_package_extracted_without_its_wrapper_runs(
    tmp_path: Path, install_roots: tuple[Path, Path]
) -> None:
    root = workbuddy_acp.WORKBUDDY_PREINSTALLED_ROOT
    (root / 'bin').mkdir(parents=True)
    entry = root / 'bin' / 'wb.js'
    entry.write_text('#!/usr/bin/env node\n')
    (root / 'package.json').write_text(json.dumps({'version': '5.11.0'}))
    assert (
        await ensure_workbuddy_cli(
            WorkBuddyAcpOptions().runtime_options(None), node_path=_fake_node(tmp_path, 'v22.19.0')
        )
        == entry
    )


async def test_configured_cli_path_runs_from_another_install_directory(tmp_path: Path) -> None:
    package = tmp_path / 'custom-install' / 'package'
    (package / 'bin').mkdir(parents=True)
    entry = package / 'bin' / 'wb.js'
    entry.write_text('#!/usr/bin/env node\n')
    (package / 'package.json').write_text(json.dumps({'version': '5.11.0'}))
    options = WorkBuddyAcpOptions(cli_path=str(entry)).runtime_options(None)
    assert options['cli_path'] == str(entry)
    assert await ensure_workbuddy_cli(options, node_path=_fake_node(tmp_path, 'v22.19.0')) == entry


async def test_corrupted_upload_is_rejected_and_leaves_no_install(
    tmp_path: Path, install_roots: tuple[Path, Path]
) -> None:
    _, root = install_roots
    parts = _split(_package_tgz(), tmp_path / 'parts')
    options = WorkBuddyAcpOptions(package_sha256=_SHA).runtime_options(parts)
    with pytest.raises(RuntimeError, match='SHA-256'):
        await ensure_workbuddy_cli(options, node_path=_fake_node(tmp_path, 'v22.19.0'))
    assert not any(path.name.startswith('5.11.0') for path in root.iterdir())


async def test_package_of_another_version_is_rejected(tmp_path: Path, install_roots: tuple[Path, Path]) -> None:
    parts = _split(_package_tgz('5.9.0'), tmp_path / 'parts')
    with pytest.raises(RuntimeError, match="version '5.9.0', expected '5.11.0'"):
        await ensure_workbuddy_cli(
            WorkBuddyAcpOptions().runtime_options(parts), node_path=_fake_node(tmp_path, 'v22.19.0')
        )


async def test_tarball_without_wb_js_is_rejected(tmp_path: Path, install_roots: tuple[Path, Path]) -> None:
    parts = _split(_package_tgz(with_entry=False), tmp_path / 'parts')
    with pytest.raises(RuntimeError, match='package/bin/wb.js'):
        await ensure_workbuddy_cli(
            WorkBuddyAcpOptions().runtime_options(parts), node_path=_fake_node(tmp_path, 'v22.19.0')
        )


async def test_missing_package_explains_how_to_provide_one(tmp_path: Path, install_roots: tuple[Path, Path]) -> None:
    with pytest.raises(RuntimeError, match='workbuddy.package_path'):
        await ensure_workbuddy_cli(
            WorkBuddyAcpOptions().runtime_options(None), node_path=_fake_node(tmp_path, 'v22.19.0')
        )


async def test_missing_parts_ask_for_a_fresh_runtime(tmp_path: Path, install_roots: tuple[Path, Path]) -> None:
    options = WorkBuddyAcpOptions().runtime_options([str(tmp_path / 'gone.part000')])
    with pytest.raises(RuntimeError, match='parts are missing'):
        await ensure_workbuddy_cli(options, node_path=_fake_node(tmp_path, 'v22.19.0'))


class _Runtime:
    """Records ``upload_file`` calls; ``fail_first`` makes the first part upload fail once."""

    def __init__(self, *, fail_first: bool = False) -> None:
        self.uploads: list[tuple[str, int]] = []
        self.fail_first = fail_first
        self.release = asyncio.Event()
        self.release.set()

    async def upload_file(self, files: list[Any], *, overwrite: bool) -> Any:
        assert overwrite
        await self.release.wait()
        item = files[0]
        if self.fail_first:
            self.fail_first = False
            return SimpleNamespace(results=[SimpleNamespace(success=False, message='disk full')])
        self.uploads.append((item.path, len(item.content)))
        return SimpleNamespace(results=[SimpleNamespace(success=True, message='')])


@pytest.fixture
def small_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(workbuddy_acp, '_UPLOAD_CHUNK_BYTES', 64)


async def test_upload_slices_the_package_and_runs_once_per_runtime(tmp_path: Path, small_chunks: None) -> None:
    package = tmp_path / 'wb.tgz'
    package.write_bytes(bytes(range(256)) * 2)
    options = WorkBuddyAcpOptions(package_path=str(package))
    runtime = _Runtime()
    runtime.release.clear()
    first = asyncio.create_task(upload_workbuddy_package(runtime, options))
    second = asyncio.create_task(upload_workbuddy_package(runtime, options))
    await asyncio.sleep(0.05)
    runtime.release.set()
    parts = await first
    assert await second == parts
    digest = hashlib.sha256(package.read_bytes()).hexdigest()
    assert parts == [f'/tmp/nexus/workbuddy-acp/wb-5.11.0-{digest[:12]}.tgz.part{index:03d}' for index in range(8)]
    assert len(runtime.uploads) == 8, 'Concurrent sessions share one upload'
    assert await upload_workbuddy_package(runtime, options) == parts
    assert len(runtime.uploads) == 8
    assert len(await upload_workbuddy_package(_Runtime(), options) or []) == 8, 'Another sandbox uploads again'


async def test_failed_upload_is_retried_by_the_next_session(tmp_path: Path, small_chunks: None) -> None:
    package = tmp_path / 'wb.tgz'
    package.write_bytes(b'x' * 100)
    options = WorkBuddyAcpOptions(package_path=str(package))
    runtime = _Runtime(fail_first=True)
    with pytest.raises(RuntimeError, match='disk full'):
        await upload_workbuddy_package(runtime, options)
    assert len(await upload_workbuddy_package(runtime, options) or []) == 2


async def test_upload_rejects_a_package_that_does_not_match_its_pin(tmp_path: Path) -> None:
    package = tmp_path / 'wb.tgz'
    package.write_bytes(b'payload')
    runtime = _Runtime()
    with pytest.raises(RuntimeError, match='package_sha256'):
        await upload_workbuddy_package(runtime, WorkBuddyAcpOptions(package_path=str(package), package_sha256=_SHA))
    assert runtime.uploads == []


async def test_no_host_package_uploads_nothing(tmp_path: Path) -> None:
    runtime = _Runtime()
    assert await upload_workbuddy_package(runtime, WorkBuddyAcpOptions()) is None
    with pytest.raises(FileNotFoundError, match='package_path'):
        await upload_workbuddy_package(runtime, WorkBuddyAcpOptions(package_path=str(tmp_path / 'missing.tgz')))
    assert runtime.uploads == []
