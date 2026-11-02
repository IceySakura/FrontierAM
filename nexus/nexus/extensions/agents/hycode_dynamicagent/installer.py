"""HyCode DynamicAgent installer.

Installs ``@tencent/hycode-dynamicagent`` via npm inside the runtime sandbox.
The npm package owns the bundled OpenCC binary selection and the Code Arena
harness assets, so Nexus only resolves and launches the package's CLI entry.
"""

from __future__ import annotations

import json
import logging
import shlex
import stat

from pathlib import Path

from nexus.extensions.agents.installed_common import (
    DEFAULT_AGENT_INSTALL_ROOT,
    DEFAULT_NPM_REGISTRY,
    build_npm_env,
    find_runtime_binary,
    resolve_npm_bin_entry,
    resolve_npm_registry,
)


logger = logging.getLogger(__name__)

NPM_PACKAGE = '@tencent/hycode-dynamicagent'
BIN_NAME = 'hycode-dynamicagent'
DEFAULT_HYCODE_DYNAMICAGENT_VERSION = 'latest'
DEFAULT_INSTALL_DIR = f'{DEFAULT_AGENT_INSTALL_ROOT}/hycode-dynamicagent'


def _package_dir(install_dir: str) -> Path:
    return Path(install_dir) / 'node_modules' / '@tencent' / 'hycode-dynamicagent'


def get_installed_version(install_dir: str) -> str | None:
    """Return the installed package version if present."""
    package_json = _package_dir(install_dir) / 'package.json'
    if not package_json.exists():
        return None
    try:
        data = json.loads(package_json.read_text())
    except Exception:
        return None
    version = data.get('version')
    return version if isinstance(version, str) else None


def _make_executable(path: Path) -> None:
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _resolve_cli_entry(install_dir: str) -> Path | None:
    entry = resolve_npm_bin_entry(install_dir, NPM_PACKAGE, BIN_NAME)
    if entry and entry.is_file():
        return entry
    fallback = _package_dir(install_dir) / 'bin' / 'hycode.mjs'
    if fallback.is_file():
        return fallback
    return None


def _ensure_top_level_symlink(install_dir: str, entry: Path) -> Path:
    """Create a stable ``<install_dir>/hycode-dynamicagent`` symlink for PATH resolution."""
    target = entry.resolve()
    _make_executable(target)

    top_level = Path(install_dir) / BIN_NAME
    if top_level.is_symlink() or top_level.exists():
        top_level.unlink(missing_ok=True)
    top_level.symlink_to(target)
    return top_level


async def install_hycode_dynamicagent_via_npm(
    install_dir: str = DEFAULT_INSTALL_DIR,
    version: str = DEFAULT_HYCODE_DYNAMICAGENT_VERSION,
    npm_registry: str | None = DEFAULT_NPM_REGISTRY,
) -> Path:
    """Install HyCode DynamicAgent via npm and return the CLI path."""
    from nexus.runtime.executors.command import CommandRunStatus, RunCommandRequest, run_command

    install_path = Path(install_dir)
    install_path.mkdir(parents=True, exist_ok=True)

    installed_version = get_installed_version(install_dir)
    existing = _resolve_cli_entry(install_dir)
    if existing is not None and version != 'latest' and installed_version == version:
        top_level = _ensure_top_level_symlink(install_dir, existing)
        logger.debug('HyCode DynamicAgent already installed at %s (version %s)', top_level, installed_version)
        return top_level

    package_spec = NPM_PACKAGE if version == 'latest' else f'{NPM_PACKAGE}@{version}'
    npm_executable = find_runtime_binary('npm') or 'npm'
    registry = resolve_npm_registry(npm_registry)
    command = (
        f'{shlex.quote(npm_executable)} install '
        f'--prefix {shlex.quote(str(install_path))} --no-audit --no-fund --strict-ssl=false '
        f'--registry {shlex.quote(registry)} {shlex.quote(package_spec)}'
    )
    logger.debug('Installing HyCode DynamicAgent: %s via registry %s', package_spec, registry)

    result = await run_command(
        RunCommandRequest(
            command=command,
            cwd=str(install_path),
            timeout=1800,
            extra_env=build_npm_env(npm_registry=registry),
        )
    )

    if result.status != CommandRunStatus.Finished or result.return_code != 0:
        stderr = result.stderr or ''
        stdout = result.stdout or ''
        raise RuntimeError(f'Failed to install HyCode DynamicAgent via npm: {stderr or stdout or result.status}')

    entry = _resolve_cli_entry(install_dir)
    if entry is None:
        raise RuntimeError(
            f'HyCode DynamicAgent install succeeded but bin entry {BIN_NAME!r} was not found under '
            f'{install_path}/node_modules/@tencent/hycode-dynamicagent/'
        )

    top_level = _ensure_top_level_symlink(install_dir, entry)
    logger.debug('HyCode DynamicAgent installed at %s (version %s)', top_level, get_installed_version(install_dir))
    return top_level


__all__ = [
    'BIN_NAME',
    'DEFAULT_HYCODE_DYNAMICAGENT_VERSION',
    'DEFAULT_INSTALL_DIR',
    'NPM_PACKAGE',
    'get_installed_version',
    'install_hycode_dynamicagent_via_npm',
]
