"""Install the packaged Handy CLI inside a Nexus runtime."""

from __future__ import annotations

import json
import logging
import re
import shlex
import stat

from pathlib import Path

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

NPM_PACKAGE = '@tencent/handy-agent'
BIN_NAME = 'handy'
DEFAULT_HANDY_VERSION = '0.1.14'
DEFAULT_INSTALL_DIR = f'{DEFAULT_AGENT_INSTALL_ROOT}/handy'
MIN_NODE_VERSION = (22, 19, 0)
_SEMVER_RE = re.compile(r'v?(?P<major>\d+)(?:\.(?P<minor>\d+))?(?:\.(?P<patch>\d+))?')


def parse_semver(version: str) -> tuple[int, int, int]:
    """Parse a loose Node-style semantic version string."""
    match = _SEMVER_RE.search(version.strip())
    if not match:
        raise RuntimeError(f'Unable to parse version string: {version}')
    return (
        int(match.group('major')),
        int(match.group('minor') or 0),
        int(match.group('patch') or 0),
    )


def _package_dir(install_dir: str) -> Path:
    return Path(install_dir) / 'node_modules' / '@tencent' / 'handy-agent'


def get_installed_version(install_dir: str) -> str | None:
    """Return the installed Handy package version, if present."""
    package_json = _package_dir(install_dir) / 'package.json'
    if not package_json.exists():
        return None
    try:
        data = json.loads(package_json.read_text())
    except Exception:
        return None
    version = data.get('version')
    return version if isinstance(version, str) else None


async def _run_checked_command(command: str, *, timeout: float, extra_env: dict[str, str]) -> str:
    from nexus.runtime.executors.command import CommandRunStatus, RunCommandRequest, run_command

    result = await run_command(
        RunCommandRequest(
            command=command,
            timeout=timeout,
            extra_env=extra_env,
        )
    )
    if result.status == CommandRunStatus.TimeLimitExceeded:
        raise RuntimeError(f'Command timed out after {timeout}s: {command}')
    if result.status != CommandRunStatus.Finished or (result.return_code or 0) != 0:
        detail = result.stderr or result.stdout or str(result.status)
        raise RuntimeError(f'Command failed: {command}: {detail}')
    return result.stdout or ''


async def ensure_runtime_prerequisites(npm_registry: str | None = DEFAULT_NPM_REGISTRY) -> tuple[str, str]:
    """Require the Node and ripgrep versions needed by the packaged Handy CLI.

    The Nexus runtime's own Node is checked first. A repo-provided one can be on
    PATH and far older -- a TypeScript testbed pinned to Node 18 would otherwise
    fail the check even though the launch PATH puts the runtime's Node 22 first.
    """
    node = find_nexus_runtime_binary('node')
    if not node:
        raise RuntimeError('Handy requires Node >= 22.19.0, but node was not found on PATH')
    rg = find_runtime_binary('rg')
    if not rg:
        raise RuntimeError('Handy requires ripgrep (rg), but rg was not found on PATH')

    registry = resolve_npm_registry(npm_registry)
    node_version = (
        await _run_checked_command(
            f'{shlex.quote(node)} --version',
            timeout=30,
            extra_env=build_npm_env(npm_registry=registry),
        )
    ).strip()
    if parse_semver(node_version) < MIN_NODE_VERSION:
        required = '.'.join(str(part) for part in MIN_NODE_VERSION)
        raise RuntimeError(f'Handy requires Node >= {required}, but runtime has {node_version}')
    return node_version, rg


def _resolve_cli_entry(install_dir: str) -> Path | None:
    entry = resolve_npm_bin_entry(install_dir, NPM_PACKAGE, BIN_NAME)
    return entry if entry and entry.is_file() else None


def _ensure_top_level_symlink(install_dir: str, entry: Path) -> Path:
    target = entry.resolve()
    target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    top_level = Path(install_dir) / BIN_NAME
    if top_level.is_symlink() or top_level.exists():
        top_level.unlink(missing_ok=True)
    top_level.symlink_to(target)
    return top_level


async def install_handy_via_npm(
    install_dir: str = DEFAULT_INSTALL_DIR,
    version: str = DEFAULT_HANDY_VERSION,
    npm_registry: str | None = DEFAULT_NPM_REGISTRY,
) -> Path:
    """Install Handy with npm and return a stable executable path."""
    from nexus.runtime.executors.command import CommandRunStatus, RunCommandRequest, run_command

    node_version, rg = await ensure_runtime_prerequisites(npm_registry)
    logger.debug('Using Node.js %s and ripgrep %s for Handy', node_version, rg)

    install_path = Path(install_dir)
    install_path.mkdir(parents=True, exist_ok=True)

    installed_version = get_installed_version(install_dir)
    existing = _resolve_cli_entry(install_dir)
    if existing is not None and version != 'latest' and installed_version == version:
        top_level = _ensure_top_level_symlink(install_dir, existing)
        logger.debug('Handy already installed at %s (version %s)', top_level, installed_version)
        return top_level

    package_spec = NPM_PACKAGE if version == 'latest' else f'{NPM_PACKAGE}@{version}'
    npm = find_runtime_binary('npm') or 'npm'
    registry = resolve_npm_registry(npm_registry)
    command = (
        f'{shlex.quote(npm)} install '
        f'--prefix {shlex.quote(str(install_path))} --no-audit --no-fund --strict-ssl=false '
        f'--registry {shlex.quote(registry)} {shlex.quote(package_spec)}'
    )
    logger.debug('Installing Handy %s via registry %s', version, registry)
    npm_env = build_npm_env(npm_registry=registry)
    npm_env.update({'NPM_CONFIG_STRICT_SSL': 'false', 'npm_config_strict_ssl': 'false'})
    result = await run_command(
        RunCommandRequest(
            command=command,
            cwd=str(install_path),
            timeout=1800,
            extra_env=npm_env,
        )
    )
    if result.status != CommandRunStatus.Finished or (result.return_code or 0) != 0:
        detail = result.stderr or result.stdout or str(result.status)
        raise RuntimeError(f'Failed to install Handy via npm: {detail}')

    entry = _resolve_cli_entry(install_dir)
    if entry is None:
        raise RuntimeError(f'Handy install succeeded but bin entry {BIN_NAME!r} was not found under {install_path}')
    top_level = _ensure_top_level_symlink(install_dir, entry)
    logger.debug('Handy installed at %s (version %s)', top_level, get_installed_version(install_dir))
    return top_level


__all__ = [
    'BIN_NAME',
    'DEFAULT_HANDY_VERSION',
    'DEFAULT_INSTALL_DIR',
    'MIN_NODE_VERSION',
    'NPM_PACKAGE',
    'ensure_runtime_prerequisites',
    'get_installed_version',
    'install_handy_via_npm',
    'parse_semver',
]
