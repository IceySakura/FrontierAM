"""Install the official DeepSeek Harness CLI and ACP bridge."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shlex
import shutil
import sys
import tempfile

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from nexus.extensions.agents.dsh.prebuilt import (
    DEFAULT_DSH_PREBUILT_BASE_URL,
    PrebuiltCompatibilityError,
    PrebuiltIntegrityError,
    PrebuiltUnavailableError,
    RuntimeTarget,
    detect_runtime_target,
    download_and_extract_prebuilt,
    fetch_prebuilt_manifest,
)
from nexus.extensions.agents.installed_common import (
    DEFAULT_AGENT_INSTALL_ROOT,
    DEFAULT_NPM_REGISTRY,
    build_npm_env,
    find_nexus_runtime_binary,
    find_runtime_binary,
    resolve_npm_bin_entry,
    resolve_npm_registry,
)


logger = logging.getLogger(__name__)

DEFAULT_DSH_VERSION = '0.1.0-rc.6'
DEFAULT_DSH_ACP_VERSION = '0.1.0-rc.6'
DEFAULT_INSTALL_DIR = f'{DEFAULT_AGENT_INSTALL_ROOT}/dsh'
DSH_PACKAGE = '@deepseek-ai/dsh'
DSH_ACP_PACKAGE = '@deepseek-ai/dsh-acp'
_INSTALL_LOCK = asyncio.Lock()
_NODE_VERSION_PATTERN = re.compile(r'^v?(\d+)\.(\d+)\.(\d+)')
_EXACT_PACKAGE_VERSION_PATTERN = re.compile(r'^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$')


@dataclass(frozen=True)
class _NodeRuntime:
    path: str
    version: tuple[int, int, int]

    @property
    def major(self) -> int:
        return self.version[0]


def _package_version(install_dir: str | Path, package_name: str) -> str | None:
    package_json = Path(install_dir) / 'node_modules' / package_name / 'package.json'
    if not package_json.exists():
        return None
    try:
        value = json.loads(package_json.read_text()).get('version')
    except (json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, str) else None


def get_installed_versions(install_dir: str | Path) -> tuple[str | None, str | None]:
    """Return the installed DSH CLI and ACP package versions."""
    return (
        _package_version(install_dir, DSH_PACKAGE),
        _package_version(install_dir, DSH_ACP_PACKAGE),
    )


def get_installation_root(entry: Path) -> Path:
    """Return the npm prefix that owns a resolved DSH bin entry."""
    for parent in entry.parents:
        if parent.name == 'node_modules':
            return parent.parent
    raise ValueError(f'DeepSeek Harness entry is not below node_modules: {entry}')


def _parse_node_version(raw: str) -> tuple[int, int, int] | None:
    match = _NODE_VERSION_PATTERN.match(raw.strip())
    if match is None:
        return None
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch)


def _supported_node_version(version: tuple[int, int, int]) -> bool:
    major, minor, _patch = version
    return (major == 22 and minor >= 19) or major >= 24


def _validate_native_build_tools() -> None:
    missing: list[str] = []
    if find_runtime_binary('make') is None:
        missing.append('make')
    if not any(find_runtime_binary(name) for name in ('g++', 'c++')):
        missing.append('a C++ compiler (g++ or c++)')
    if missing:
        raise RuntimeError(
            'DeepSeek Harness npm fallback requires native build tools for node-pty: '
            f'{", ".join(missing)}. Install build-essential (or publish a matching prebuilt artifact).'
        )


def _prepare_install_manifest(install_path: Path, *, version: str, acp_version: str) -> None:
    package = {
        'name': 'nexus-dsh-runtime',
        'private': True,
        'dependencies': {
            DSH_PACKAGE: version,
            DSH_ACP_PACKAGE: acp_version,
        },
    }
    (install_path / 'package.json').write_text(json.dumps(package, indent=2) + '\n')


async def _validate_node_runtime() -> _NodeRuntime:
    from nexus.runtime.executors.command import CommandRunStatus, RunCommandRequest, run_command

    # The Nexus runtime's own Node is checked first. A repo-provided one can be on
    # PATH and far older -- a TypeScript testbed pinned to Node 18 would otherwise
    # fail the check even though the session launches with the runtime's Node 22.
    node = find_nexus_runtime_binary('node')
    if node is None:
        raise RuntimeError('DeepSeek Harness requires Node.js ^22.19.0 or >=24, but node was not found')

    result = await run_command(
        RunCommandRequest(
            command=f'{shlex.quote(node)} --version',
            timeout=30,
            extra_env=build_npm_env(),
        )
    )
    version = _parse_node_version(result.stdout or '')
    if result.status != CommandRunStatus.Finished or result.return_code != 0 or version is None:
        detail = result.stderr or result.stdout or str(result.status)
        raise RuntimeError(f'Failed to determine Node.js version for DeepSeek Harness: {detail}')
    if not _supported_node_version(version):
        rendered = '.'.join(str(part) for part in version)
        raise RuntimeError(f'DeepSeek Harness requires Node.js ^22.19.0 or >=24; runtime node is {rendered}')
    return _NodeRuntime(node, version)


async def _validate_native_dependencies(
    *,
    node: str,
    install_path: Path,
    npm_registry: str | None,
) -> None:
    from nexus.runtime.executors.command import CommandRunStatus, RunCommandRequest, run_command

    node_pty = install_path / 'node_modules' / 'node-pty'
    command = f'{shlex.quote(node)} -e {shlex.quote(f"require({str(node_pty)!r})")}'
    result = await run_command(
        RunCommandRequest(
            command=command,
            cwd=str(install_path),
            timeout=60,
            extra_env=build_npm_env(npm_registry=npm_registry),
        )
    )
    if result.status != CommandRunStatus.Finished or result.return_code != 0:
        detail = result.stderr or result.stdout or str(result.status)
        raise RuntimeError(f'DeepSeek Harness native dependency validation failed (node-pty): {detail}')


def _installation_path(
    root: Path,
    *,
    version: str,
    acp_version: str,
    node_major: int,
    target: RuntimeTarget | None,
) -> Path:
    target_tag = target.tag if target is not None else f'{sys.platform}-unknown-node{node_major}'
    name = f'dsh-{version}--acp-{acp_version}--{target_tag}'
    return root / 'installations' / name


def _validate_install_layout(
    install_path: Path,
    *,
    version: str,
    acp_version: str,
) -> Path:
    entry = resolve_npm_bin_entry(install_path, DSH_PACKAGE, 'dsh')
    if entry is None or not entry.is_file():
        raise RuntimeError('DeepSeek Harness installation is missing the dsh entry point')
    installed = get_installed_versions(install_path)
    expected = (version, acp_version)
    if installed != expected:
        raise RuntimeError(f'DeepSeek Harness installed unexpected versions: {installed!r}, expected {expected!r}')
    return entry


def _publish_install(candidate: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    backup: Path | None = None
    if destination.exists():
        backup = destination.with_name(f'.{destination.name}.old-{uuid4().hex}')
        destination.replace(backup)
    try:
        candidate.replace(destination)
    except BaseException:
        if backup is not None and backup.exists() and not destination.exists():
            backup.replace(destination)
        raise
    if backup is not None:
        shutil.rmtree(backup)


async def _install_prebuilt_candidate(
    *,
    candidate: Path,
    workspace: Path,
    node: _NodeRuntime,
    target: RuntimeTarget,
    base_url: str,
    version: str,
    acp_version: str,
    npm_registry: str | None,
) -> tuple[Path | None, str | None]:
    try:
        manifest = await asyncio.to_thread(
            fetch_prebuilt_manifest,
            base_url,
            dsh_version=version,
            acp_version=acp_version,
            target=target,
        )
    except PrebuiltIntegrityError:
        raise
    except (PrebuiltCompatibilityError, PrebuiltUnavailableError) as exc:
        logger.warning('DeepSeek Harness prebuilt lookup unavailable: %s', exc)
        return None, str(exc)
    if manifest is None:
        detail = f'no prebuilt artifact for DSH {version}, ACP {acp_version}, target {target.tag}'
        logger.info(detail)
        return None, detail
    logger.debug('Installing DeepSeek Harness from prebuilt artifact: %s', manifest.archive_url)
    try:
        await asyncio.to_thread(
            download_and_extract_prebuilt,
            manifest,
            archive_path=workspace / manifest.archive,
            install_path=candidate,
        )
        try:
            entry = _validate_install_layout(candidate, version=version, acp_version=acp_version)
        except RuntimeError as exc:
            raise PrebuiltIntegrityError(f'prebuilt installation layout is invalid: {exc}') from exc
        await _validate_native_dependencies(
            node=node.path,
            install_path=candidate,
            npm_registry=npm_registry,
        )
        return entry, None
    except PrebuiltIntegrityError:
        raise
    except (PrebuiltCompatibilityError, PrebuiltUnavailableError, RuntimeError) as exc:
        logger.warning('DeepSeek Harness prebuilt installation failed; falling back to npm: %s', exc)
        return None, str(exc)


async def _install_npm_candidate(
    *,
    candidate: Path,
    node: _NodeRuntime,
    version: str,
    acp_version: str,
    npm_registry: str | None,
) -> Path:
    from nexus.runtime.executors.command import CommandRunStatus, RunCommandRequest, run_command

    _validate_native_build_tools()
    candidate.mkdir(parents=True, exist_ok=False)
    _prepare_install_manifest(candidate, version=version, acp_version=acp_version)
    logger.info('Installing DeepSeek Harness packages through npm fallback: %s, %s', version, acp_version)
    registry = resolve_npm_registry(npm_registry)
    npm_args = [
        shlex.quote(find_runtime_binary('npm') or 'npm'),
        'install',
        '--prefix',
        shlex.quote(str(candidate)),
        '--no-audit',
        '--no-fund',
        '--registry',
        shlex.quote(registry),
    ]
    command = ' '.join(npm_args)
    result = await run_command(
        RunCommandRequest(
            command=command,
            cwd=str(candidate),
            timeout=1800,
            extra_env=build_npm_env(
                npm_registry=registry,
                extra_env={'npm_config_build_from_source': 'true'},
            ),
        )
    )
    if result.status != CommandRunStatus.Finished or result.return_code != 0:
        detail = result.stderr or result.stdout or str(result.status)
        raise RuntimeError(f'Failed to install DeepSeek Harness through npm fallback: {detail}')
    rebuild_args = [
        shlex.quote(find_runtime_binary('npm') or 'npm'),
        'rebuild',
        'node-pty',
        '--build-from-source=true',
        '--foreground-scripts',
        '--no-audit',
        '--no-fund',
        '--registry',
        shlex.quote(registry),
    ]
    rebuild = await run_command(
        RunCommandRequest(
            command=' '.join(rebuild_args),
            cwd=str(candidate),
            timeout=600,
            extra_env=build_npm_env(
                npm_registry=registry,
                extra_env={'npm_config_build_from_source': 'true'},
            ),
        )
    )
    if rebuild.status != CommandRunStatus.Finished or rebuild.return_code != 0:
        detail = rebuild.stderr or rebuild.stdout or str(rebuild.status)
        raise RuntimeError(f'Failed to build DeepSeek Harness node-pty through npm fallback: {detail}')
    entry = _validate_install_layout(candidate, version=version, acp_version=acp_version)
    await _validate_native_dependencies(node=node.path, install_path=candidate, npm_registry=npm_registry)
    return entry


async def install_dsh(
    install_dir: str = DEFAULT_INSTALL_DIR,
    *,
    version: str = DEFAULT_DSH_VERSION,
    acp_version: str = DEFAULT_DSH_ACP_VERSION,
    npm_registry: str | None = DEFAULT_NPM_REGISTRY,
    prebuilt_base_url: str | None = DEFAULT_DSH_PREBUILT_BASE_URL,
) -> Path:
    """Install exact DSH/ACP versions, preferring a compatible prebuilt npm tree."""
    invalid_versions = [
        value for value in (version, acp_version) if not _EXACT_PACKAGE_VERSION_PATTERN.fullmatch(value)
    ]
    if invalid_versions:
        raise ValueError(f'DeepSeek Harness requires exact package versions, got: {invalid_versions!r}')

    node = await _validate_node_runtime()
    root = Path(install_dir)
    target = detect_runtime_target(node_major=node.major)
    install_path = _installation_path(
        root,
        version=version,
        acp_version=acp_version,
        node_major=node.major,
        target=target,
    )
    install_path.parent.mkdir(parents=True, exist_ok=True)

    async with _INSTALL_LOCK:
        try:
            entry = _validate_install_layout(install_path, version=version, acp_version=acp_version)
            await _validate_native_dependencies(
                node=node.path,
                install_path=install_path,
                npm_registry=npm_registry,
            )
            return entry
        except RuntimeError:
            if install_path.exists():
                logger.warning('Replacing invalid DeepSeek Harness installation at %s', install_path)

        with tempfile.TemporaryDirectory(prefix='.dsh-install-', dir=install_path.parent) as temporary:
            workspace = Path(temporary)
            candidate = workspace / 'candidate'
            prebuilt_failure: str | None = None
            if prebuilt_base_url and target is not None:
                entry, prebuilt_failure = await _install_prebuilt_candidate(
                    candidate=candidate,
                    workspace=workspace,
                    node=node,
                    target=target,
                    base_url=prebuilt_base_url,
                    version=version,
                    acp_version=acp_version,
                    npm_registry=npm_registry,
                )
                if entry is not None:
                    _publish_install(candidate, install_path)
                    logger.debug('DeepSeek Harness prebuilt installation ready at %s', install_path)
                    return _validate_install_layout(install_path, version=version, acp_version=acp_version)
                if candidate.exists():
                    shutil.rmtree(candidate)
            elif prebuilt_base_url:
                prebuilt_failure = 'runtime platform is not supported by the prebuilt artifact repository'

            try:
                await _install_npm_candidate(
                    candidate=candidate,
                    node=node,
                    version=version,
                    acp_version=acp_version,
                    npm_registry=npm_registry,
                )
            except RuntimeError as exc:
                if prebuilt_failure is None:
                    raise
                raise RuntimeError(f'DeepSeek Harness prebuilt path failed ({prebuilt_failure}); {exc}') from exc
            _publish_install(candidate, install_path)
            return _validate_install_layout(install_path, version=version, acp_version=acp_version)


__all__ = [
    'DEFAULT_DSH_ACP_VERSION',
    'DEFAULT_DSH_PREBUILT_BASE_URL',
    'DEFAULT_DSH_VERSION',
    'DEFAULT_INSTALL_DIR',
    'get_installation_root',
    'get_installed_versions',
    'install_dsh',
]
