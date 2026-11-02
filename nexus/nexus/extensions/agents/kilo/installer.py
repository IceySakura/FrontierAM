"""Kilo Code CLI installer.

Installs ``@kilocode/cli`` via npm inside the runtime.  The package ships a
Node.js launcher (``bin/kilo``) that resolves a platform-specific compiled
binary from ``@kilocode/cli-{platform}-{arch}``.
"""

from __future__ import annotations

import json
import logging
import shlex

from pathlib import Path

from nexus.extensions.agents.installed_common import (
    DEFAULT_NPM_REGISTRY,
    build_npm_env,
    find_runtime_binary,
    resolve_npm_registry,
)


logger = logging.getLogger(__name__)


# Pinned for SWE-bench experiment reproducibility. To upgrade, change this
# constant AND update SCAFFOLD_VERSION_PIN in ceph_configs/swe_generator_nj.py.
DEFAULT_KILO_VERSION = '7.3.0'
NPM_PACKAGE = '@kilocode/cli'


def get_installed_version(install_dir: str) -> str | None:
    """Return the installed Kilo CLI version if present."""
    package_json = Path(install_dir) / 'node_modules' / '@kilocode' / 'cli' / 'package.json'
    if not package_json.exists():
        return None
    try:
        data = json.loads(package_json.read_text())
    except Exception:
        return None
    version = data.get('version')
    return version if isinstance(version, str) else None


def _resolve_kilo_executable(install_dir: str) -> Path | None:
    """Return the Kilo CLI executable path, if present."""
    bin_path = Path(install_dir) / 'node_modules' / '.bin' / 'kilo'
    if bin_path.exists():
        return bin_path
    return None


async def install_kilo_via_npm(
    install_dir: str,
    version: str = DEFAULT_KILO_VERSION,
    npm_registry: str | None = DEFAULT_NPM_REGISTRY,
) -> Path:
    """Install Kilo Code CLI via npm and return the executable path."""
    from nexus.runtime.executors.command import CommandRunStatus, RunCommandRequest, run_command

    install_path = Path(install_dir)
    install_path.mkdir(parents=True, exist_ok=True)

    existing = _resolve_kilo_executable(install_dir)
    installed_version = get_installed_version(install_dir)
    if existing is not None and version != 'latest' and installed_version == version:
        logger.debug(f'Kilo already installed at {existing} (version {installed_version})')
        return existing

    package_spec = NPM_PACKAGE if version == 'latest' else f'{NPM_PACKAGE}@{version}'
    npm_executable = find_runtime_binary('npm') or 'npm'
    registry = resolve_npm_registry(npm_registry)
    command = (
        f'{shlex.quote(npm_executable)} install '
        f'--prefix {shlex.quote(str(install_path))} --no-audit --no-fund --strict-ssl=false '
        f'--registry {shlex.quote(registry)} {shlex.quote(package_spec)}'
    )
    logger.debug(f'Installing Kilo: {package_spec} via registry {registry}')

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
        raise RuntimeError(f'Failed to install Kilo via npm: {stderr or stdout or result.status}')

    kilo_bin = _resolve_kilo_executable(install_dir)
    if kilo_bin is None:
        raise RuntimeError(f'Kilo install succeeded but binary not found under {install_path}/node_modules/.bin/')

    logger.debug(f'Kilo installed at {kilo_bin} (version {get_installed_version(install_dir)})')
    return kilo_bin


__all__ = [
    'DEFAULT_KILO_VERSION',
    'NPM_PACKAGE',
    'get_installed_version',
    'install_kilo_via_npm',
]
