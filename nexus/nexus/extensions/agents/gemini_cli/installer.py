"""Gemini CLI installer for runtime execution."""

from __future__ import annotations

import logging
import shlex

from dataclasses import dataclass
from pathlib import Path

from nexus.extensions.agents.installed_common import (
    DEFAULT_AGENT_INSTALL_ROOT,
    DEFAULT_NPM_REGISTRY,
    build_npm_env,
    find_runtime_binary,
    resolve_npm_registry,
)


logger = logging.getLogger(__name__)


DEFAULT_GEMINI_CLI_VERSION = '0.33.1'
DEFAULT_INSTALL_ROOT = f'{DEFAULT_AGENT_INSTALL_ROOT}/gemini-cli'


@dataclass(slots=True)
class GeminiCliInstallation:
    """Resolved installation paths for a Gemini CLI runtime install."""

    version: str
    install_dir: Path
    package_dir: Path
    bin_path: Path
    launcher_path: Path


def normalize_version(version: str | None) -> str:
    """Normalize the requested Gemini CLI version."""
    if not version:
        return DEFAULT_GEMINI_CLI_VERSION
    return version.strip()


def _version_dir_name(version: str) -> str:
    return version.replace('/', '-').replace('@', '-')


def get_installation(version: str, install_root: str = DEFAULT_INSTALL_ROOT) -> GeminiCliInstallation:
    """Get resolved installation paths for a Gemini CLI version."""
    normalized_version = normalize_version(version)
    install_dir = Path(install_root) / _version_dir_name(normalized_version)
    package_dir = install_dir / 'node_modules' / '@google' / 'gemini-cli'
    return GeminiCliInstallation(
        version=normalized_version,
        install_dir=install_dir,
        package_dir=package_dir,
        bin_path=install_dir / 'node_modules' / '.bin' / 'gemini',
        launcher_path=install_dir / 'nexus-gemini-launcher.mjs',
    )


async def install_gemini_cli(
    version: str = DEFAULT_GEMINI_CLI_VERSION,
    install_root: str = DEFAULT_INSTALL_ROOT,
    timeout: float = 900,
    npm_registry: str | None = DEFAULT_NPM_REGISTRY,
    extra_env: dict[str, str] | None = None,
) -> GeminiCliInstallation:
    """Install Gemini CLI with npm if it is not already cached."""
    from nexus.runtime.executors.command import CommandRunStatus, RunCommandRequest, run_command

    npm_path = find_runtime_binary('npm')
    node_path = find_runtime_binary('node')
    if not npm_path or not node_path:
        raise RuntimeError('Gemini CLI requires both node and npm in the runtime image')

    installation = get_installation(version=version, install_root=install_root)
    if installation.package_dir.exists() and installation.bin_path.exists():
        logger.debug('Gemini CLI already installed at %s', installation.install_dir)
        return installation

    installation.install_dir.mkdir(parents=True, exist_ok=True)
    registry = resolve_npm_registry(npm_registry)

    package_spec = (
        '@google/gemini-cli' if installation.version == 'latest' else f'@google/gemini-cli@{installation.version}'
    )
    cmd = [
        npm_path,
        'install',
        '--prefix',
        str(installation.install_dir),
        '--no-audit',
        '--no-fund',
        '--loglevel=error',
        package_spec,
    ]
    cmd.extend(['--registry', registry])

    env = build_npm_env(
        npm_registry=registry,
        extra_env={
            'NPM_CONFIG_CACHE': str(installation.install_dir / '.npm-cache'),
            **(extra_env or {}),
        },
    )

    logger.debug(
        'Installing Gemini CLI %s into %s%s',
        installation.version,
        installation.install_dir,
        f' via {registry}' if registry else '',
    )
    result = await run_command(
        RunCommandRequest(
            command=' '.join(shlex.quote(part) for part in cmd),
            cwd=str(installation.install_dir),
            timeout=timeout,
            extra_env=env,
        )
    )

    if result.status == CommandRunStatus.TimeLimitExceeded:
        raise RuntimeError('Timed out while installing Gemini CLI')
    if result.status == CommandRunStatus.Error:
        raise RuntimeError(result.stderr or 'Failed to install Gemini CLI')
    if (result.return_code or 0) != 0:
        raise RuntimeError(result.stderr or result.stdout or 'npm install failed for Gemini CLI')

    if not installation.package_dir.exists() or not installation.bin_path.exists():
        raise RuntimeError(f'Gemini CLI install succeeded but files were not found under {installation.install_dir}')

    logger.debug('Gemini CLI installed at %s', installation.install_dir)
    return installation


__all__ = [
    'DEFAULT_GEMINI_CLI_VERSION',
    'DEFAULT_INSTALL_ROOT',
    'GeminiCliInstallation',
    'get_installation',
    'install_gemini_cli',
    'normalize_version',
]
