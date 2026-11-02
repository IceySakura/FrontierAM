"""CodeBuddy CLI installer.

Installs ``@tencent-ai/codebuddy-code`` via npm inside the runtime.
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
DEFAULT_CODEBUDDY_VERSION = '2.97.2'
NPM_PACKAGE = '@tencent-ai/codebuddy-code'


def get_installed_version(install_dir: str) -> str | None:
    """Return the installed CodeBuddy version if present."""
    package_json = Path(install_dir) / 'node_modules' / '@tencent-ai' / 'codebuddy-code' / 'package.json'
    if not package_json.exists():
        return None
    try:
        data = json.loads(package_json.read_text())
    except Exception:
        return None
    version = data.get('version')
    return version if isinstance(version, str) else None


def _resolve_codebuddy_executable(install_dir: str) -> Path | None:
    """Return the CodeBuddy executable path, if present."""
    install_path = Path(install_dir)
    bin_path = install_path / 'node_modules' / '.bin' / 'codebuddy'
    if bin_path.exists():
        return bin_path
    return None


async def install_codebuddy_via_npm(
    install_dir: str,
    version: str = DEFAULT_CODEBUDDY_VERSION,
    npm_registry: str | None = DEFAULT_NPM_REGISTRY,
) -> Path:
    """Install CodeBuddy CLI via npm and return the executable path."""
    from nexus.runtime.executors.command import CommandRunStatus, RunCommandRequest, run_command

    install_path = Path(install_dir)
    install_path.mkdir(parents=True, exist_ok=True)

    existing = _resolve_codebuddy_executable(install_dir)
    installed_version = get_installed_version(install_dir)
    if existing is not None and version != 'latest' and installed_version == version:
        logger.debug(f'CodeBuddy already installed at {existing} (version {installed_version})')
        return existing

    package_spec = NPM_PACKAGE if version == 'latest' else f'{NPM_PACKAGE}@{version}'
    npm_executable = find_runtime_binary('npm') or 'npm'
    registry = resolve_npm_registry(npm_registry)
    command = (
        f'{shlex.quote(npm_executable)} install '
        f'--prefix {shlex.quote(str(install_path))} --no-audit --no-fund --strict-ssl=false '
        f'--registry {shlex.quote(registry)} {shlex.quote(package_spec)}'
    )
    logger.debug(f'Installing CodeBuddy: {package_spec} via registry {registry}')

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
        raise RuntimeError(f'Failed to install CodeBuddy via npm: {stderr or stdout or result.status}')

    codebuddy_bin = _resolve_codebuddy_executable(install_dir)
    if codebuddy_bin is None:
        raise RuntimeError(f'CodeBuddy install succeeded but binary not found under {install_path}/node_modules/.bin/')

    logger.debug(f'CodeBuddy installed at {codebuddy_bin} (version {get_installed_version(install_dir)})')
    return codebuddy_bin


__all__ = [
    'DEFAULT_CODEBUDDY_VERSION',
    'NPM_PACKAGE',
    'get_installed_version',
    'install_codebuddy_via_npm',
]
