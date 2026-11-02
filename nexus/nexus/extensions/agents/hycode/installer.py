"""HyCode installer.

Installs ``@tencent/hycode`` via npm inside the runtime sandbox.
The package bundles two standalone binaries — ``opencc-bin`` (glibc)
and ``opencc-bin-musl`` (musl). The installer auto-detects the sandbox
libc and symlinks the correct variant as the launch binary.
"""

from __future__ import annotations

import json
import logging
import shlex
import stat
import subprocess

from pathlib import Path

from nexus.extensions.agents.installed_common import (
    DEFAULT_AGENT_INSTALL_ROOT,
    DEFAULT_NPM_REGISTRY,
    build_npm_env,
    find_runtime_binary,
    resolve_npm_registry,
)


logger = logging.getLogger(__name__)

NPM_PACKAGE = '@tencent/hycode'
DEFAULT_HYCODE_VERSION = '0.1.14'
DEFAULT_INSTALL_DIR = f'{DEFAULT_AGENT_INSTALL_ROOT}/hycode'

BIN_GLIBC = 'opencc-bin'
BIN_MUSL = 'opencc-bin-musl'


def _package_dir(install_dir: str) -> Path:
    return Path(install_dir) / 'node_modules' / Path(*NPM_PACKAGE.split('/'))


def _is_musl() -> bool:
    """Detect whether the current system uses musl libc."""
    try:
        result = subprocess.run(['ldd', '--version'], capture_output=True, text=True, timeout=5)
        combined = (result.stdout + result.stderr).lower()
        if 'musl' in combined:
            return True
    except Exception:
        pass
    return Path('/lib/ld-musl-x86_64.so.1').exists()


def get_installed_version(install_dir: str) -> str | None:
    """Return the installed HyCode version if present."""
    package_json = _package_dir(install_dir) / 'package.json'
    if not package_json.exists():
        return None
    try:
        data = json.loads(package_json.read_text())
    except Exception:
        return None
    version = data.get('version')
    return version if isinstance(version, str) else None


def _resolve_binary(install_dir: str, *, target_is_musl: bool | None = None) -> Path | None:
    """Find the correct binary (glibc or musl) from the installed package.

    ``target_is_musl`` overrides local detection, for callers installing on one filesystem a
    binary that will be executed on another: under Compose the install happens on the
    Controller while the binary runs inside the Agent's service, and the two need not share a
    libc. Detecting locally picks a build whose loader is missing at run time.
    """
    package_dir = _package_dir(install_dir)
    if not package_dir.is_dir():
        return None

    musl = _is_musl() if target_is_musl is None else target_is_musl
    bin_name = BIN_MUSL if musl else BIN_GLIBC
    candidate = package_dir / bin_name
    if candidate.is_file():
        return candidate

    fallback_name = BIN_GLIBC if musl else BIN_MUSL
    fallback = package_dir / fallback_name
    if fallback.is_file():
        logger.warning('Preferred %s not found, falling back to %s', bin_name, fallback_name)
        return fallback
    return None


def _make_executable(path: Path) -> None:
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _ensure_top_level_symlink(install_dir: str, binary: Path) -> Path:
    """Create a stable ``<install_dir>/opencc-bin`` symlink for PATH resolution."""
    target = binary.resolve()
    _make_executable(target)

    top_level = Path(install_dir) / BIN_GLIBC
    if top_level.is_symlink() or top_level.exists():
        top_level.unlink(missing_ok=True)
    top_level.symlink_to(target)
    return top_level


async def install_hycode_via_npm(
    install_dir: str = DEFAULT_INSTALL_DIR,
    version: str = DEFAULT_HYCODE_VERSION,
    npm_registry: str | None = DEFAULT_NPM_REGISTRY,
    target_is_musl: bool | None = None,
) -> Path:
    """Install HyCode via npm and return the binary path.

    ``target_is_musl`` names the libc of the machine that will run the binary, for callers
    installing somewhere else. Omitted means this machine.
    """
    from nexus.runtime.executors.command import CommandRunStatus, RunCommandRequest, run_command

    install_path = Path(install_dir)
    install_path.mkdir(parents=True, exist_ok=True)

    installed_version = get_installed_version(install_dir)
    existing = _resolve_binary(install_dir, target_is_musl=target_is_musl)
    if existing is not None and version != 'latest' and installed_version == version:
        top_level = _ensure_top_level_symlink(install_dir, existing)
        logger.debug('HyCode already installed at %s (version %s)', top_level, installed_version)
        return top_level

    package_spec = NPM_PACKAGE if version == 'latest' else f'{NPM_PACKAGE}@{version}'
    npm_executable = find_runtime_binary('npm') or 'npm'
    registry = resolve_npm_registry(npm_registry)
    command = (
        f'{shlex.quote(npm_executable)} install '
        f'--prefix {shlex.quote(str(install_path))} --no-audit --no-fund --strict-ssl=false '
        f'--registry {shlex.quote(registry)} {shlex.quote(package_spec)}'
    )
    logger.debug('Installing HyCode: %s via registry %s', package_spec, registry)

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
        raise RuntimeError(f'Failed to install HyCode via npm: {stderr or stdout or result.status}')

    binary = _resolve_binary(install_dir, target_is_musl=target_is_musl)
    if binary is None:
        raise RuntimeError(
            f'HyCode install succeeded but no binary ({BIN_GLIBC}/{BIN_MUSL}) found under {_package_dir(install_dir)}/'
        )

    top_level = _ensure_top_level_symlink(install_dir, binary)
    logger.debug('HyCode installed at %s (version %s)', top_level, get_installed_version(install_dir))
    return top_level


__all__ = [
    'DEFAULT_HYCODE_VERSION',
    'DEFAULT_INSTALL_DIR',
    'NPM_PACKAGE',
    'get_installed_version',
    'install_hycode_via_npm',
]
