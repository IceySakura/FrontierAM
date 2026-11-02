"""OpenClaw CLI installer helpers."""

from __future__ import annotations

import json
import logging
import re
import shlex
import shutil

from pathlib import Path

from nexus.extensions.agents.installed_common import (
    DEFAULT_AGENT_INSTALL_ROOT,
    DEFAULT_NPM_REGISTRY,
    DEFAULT_RUNTIME_BIN_DIR,
    build_npm_env,
    resolve_npm_registry,
)


logger = logging.getLogger(__name__)


DEFAULT_INSTALL_DIR = f'{DEFAULT_AGENT_INSTALL_ROOT}/openclaw'
DEFAULT_PACKAGE_NAME = 'openclaw'
RUNTIME_BIN_DIR = DEFAULT_RUNTIME_BIN_DIR
MIN_NODE_VERSION = (22, 19, 0)
_SEMVER_RE = re.compile(r'v?(?P<major>\d+)(?:\.(?P<minor>\d+))?(?:\.(?P<patch>\d+))?')


def parse_semver(version: str) -> tuple[int, int, int]:
    """Parse a loose semver string like ``v22.16.0``."""
    match = _SEMVER_RE.search(version.strip())
    if not match:
        raise RuntimeError(f'Unable to parse version string: {version}')

    major = int(match.group('major'))
    minor = int(match.group('minor') or 0)
    patch = int(match.group('patch') or 0)
    return major, minor, patch


def build_package_spec(package_name: str, version: str) -> str:
    """Build an npm package spec for install/view commands."""
    return package_name if version == 'latest' else f'{package_name}@{version}'


def _npm_env(npm_registry: str | None = None) -> dict[str, str]:
    env = build_npm_env(npm_registry=npm_registry)
    # Some runtime images rely on host/corporate CA material that is not present
    # in the container. npm may redirect even http registries to https; keep
    # OpenClaw installs usable in those sandboxes.
    env['NPM_CONFIG_STRICT_SSL'] = 'false'
    env['npm_config_strict_ssl'] = 'false'
    return env


async def _run_checked_command(
    command: str,
    *,
    timeout: float,
    cwd: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> str:
    from nexus.runtime.executors.command import CommandRunStatus, RunCommandRequest, run_command

    result = await run_command(
        RunCommandRequest(
            command=command,
            cwd=cwd,
            timeout=timeout,
            extra_env=extra_env or {},
        )
    )

    if result.status == CommandRunStatus.TimeLimitExceeded:
        raise RuntimeError(f'Command timed out after {timeout}s: {command}')
    if result.status != CommandRunStatus.Finished:
        raise RuntimeError(f'Command execution failed: {command}\n{result.stderr or ""}'.strip())
    if (result.return_code or 0) != 0:
        stdout = result.stdout or ''
        stderr = result.stderr or ''
        raise RuntimeError(
            f'Command returned {result.return_code}: {command}\nstdout:\n{stdout}\nstderr:\n{stderr}'.strip()
        )

    return result.stdout or ''


async def ensure_supported_node_version() -> str:
    """Ensure the runtime has a sufficiently new Node.js version."""
    stdout = await _run_checked_command(
        'node --version',
        timeout=30,
        extra_env=_npm_env(),
    )
    node_version = stdout.strip()
    parsed = parse_semver(node_version)
    if parsed < MIN_NODE_VERSION:
        required = '.'.join(str(part) for part in MIN_NODE_VERSION)
        raise RuntimeError(f'OpenClaw requires Node >= {required}, but runtime has {node_version}')
    return node_version


async def get_latest_version(
    package_name: str = DEFAULT_PACKAGE_NAME,
    npm_registry: str | None = DEFAULT_NPM_REGISTRY,
) -> str:
    """Resolve the latest published OpenClaw version from npm."""
    registry = resolve_npm_registry(npm_registry)
    cmd = ['npm', 'view', package_name, 'version', '--json']
    cmd.append(f'--registry={registry}')

    stdout = await _run_checked_command(
        shlex.join(cmd),
        timeout=120,
        extra_env=_npm_env(npm_registry=registry),
    )

    payload = stdout.strip()
    if not payload:
        raise RuntimeError(f'Failed to resolve latest version for {package_name}')

    parsed = json.loads(payload)
    if not isinstance(parsed, str) or not parsed:
        raise RuntimeError(f'Unexpected npm view payload for {package_name}: {payload}')
    return parsed


async def install_openclaw(
    version: str = 'latest',
    install_dir: str = DEFAULT_INSTALL_DIR,
    package_name: str = DEFAULT_PACKAGE_NAME,
    npm_registry: str | None = DEFAULT_NPM_REGISTRY,
) -> Path:
    """Install OpenClaw into a versioned npm prefix and return the CLI path."""
    node_version = await ensure_supported_node_version()
    logger.debug('Using Node.js %s for OpenClaw install', node_version)

    registry = resolve_npm_registry(npm_registry)
    actual_version = version
    if version == 'latest':
        actual_version = await get_latest_version(package_name=package_name, npm_registry=registry)

    install_root = Path(install_dir)
    prefix_path = install_root / actual_version
    version_file = prefix_path / '.version'
    binary_path = prefix_path / 'node_modules' / '.bin' / 'openclaw'

    if binary_path.exists() and version_file.exists() and version_file.read_text().strip() == actual_version:
        logger.debug('Reusing cached OpenClaw install at %s', binary_path)
        return binary_path

    # Check if openclaw is globally installed with the matching version.
    # WildClawBench docker images ship openclaw pre-installed (e.g. /usr/bin/openclaw).
    # If the global version matches, create the expected cache structure instead of
    # running npm install (which may fail when the npm registry is unreachable).
    global_binary = shutil.which('openclaw')
    # shutil.which relies on PATH which may not include standard system dirs
    # inside containerised runtimes – also probe well-known locations.
    if global_binary is None:
        for candidate in ('/usr/bin/openclaw', '/usr/local/bin/openclaw'):
            if Path(candidate).is_file():
                global_binary = candidate
                break
    if global_binary is not None:
        try:
            global_ver_stdout = await _run_checked_command(
                f'{global_binary} --version',
                timeout=30,
                extra_env=_npm_env(npm_registry=registry),
            )
            # Output is like "OpenClaw 2026.3.11 (29dc654)" — extract semver
            global_ver_match = re.search(r'(\d{4}\.\d+\.\d+)', global_ver_stdout)
            if global_ver_match and global_ver_match.group(1) == actual_version:
                logger.debug(
                    'Found globally installed OpenClaw %s at %s, creating cache symlink',
                    actual_version,
                    global_binary,
                )
                prefix_path.mkdir(parents=True, exist_ok=True)
                node_modules_dir = prefix_path / 'node_modules'
                bin_dir = node_modules_dir / '.bin'
                bin_dir.mkdir(parents=True, exist_ok=True)
                symlink_target = bin_dir / 'openclaw'
                if symlink_target.exists() or symlink_target.is_symlink():
                    symlink_target.unlink()
                symlink_target.symlink_to(global_binary)

                # Also symlink the package directory so that
                # resolve_npm_bin_entry() can find package.json and
                # resolve the JS entry point.  The global binary is
                # typically at /usr/bin/openclaw which is a symlink to
                # e.g. /usr/lib/node_modules/openclaw/bin/openclaw.mjs.
                # We resolve the real path and walk up to find the
                # package root (directory containing package.json).
                global_real = Path(global_binary).resolve()
                global_pkg_dir: Path | None = None
                for ancestor in (global_real.parent, global_real.parent.parent, global_real.parent.parent.parent):
                    if (ancestor / 'package.json').exists():
                        global_pkg_dir = ancestor
                        break
                # Also check well-known global npm locations.
                if global_pkg_dir is None:
                    for candidate_dir in (
                        Path('/usr/lib/node_modules/openclaw'),
                        Path('/usr/local/lib/node_modules/openclaw'),
                    ):
                        if (candidate_dir / 'package.json').exists():
                            global_pkg_dir = candidate_dir
                            break

                if global_pkg_dir is not None:
                    local_pkg_link = node_modules_dir / package_name
                    if local_pkg_link.exists() or local_pkg_link.is_symlink():
                        local_pkg_link.unlink()
                    local_pkg_link.symlink_to(global_pkg_dir)
                    logger.debug(
                        'Symlinked global package dir %s -> %s',
                        local_pkg_link,
                        global_pkg_dir,
                    )
                else:
                    logger.warning(
                        'Could not locate global openclaw package directory '
                        '(package.json) from %s; resolve_npm_bin_entry may fail',
                        global_real,
                    )

                version_file.write_text(actual_version)
                logger.debug('Using globally installed OpenClaw %s via symlink at %s', actual_version, symlink_target)
                return symlink_target
        except Exception as exc:
            logger.warning('Global openclaw detection failed, falling back to npm install: %s', exc)

    prefix_path.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(prefix_path / 'node_modules', ignore_errors=True)

    package_spec = build_package_spec(package_name, actual_version)
    cmd = [
        'npm',
        'install',
        '--prefix',
        str(prefix_path),
        '--no-save',
        '--no-package-lock',
        '--omit=dev',
        '--loglevel=error',
        package_spec,
    ]
    cmd.append(f'--registry={registry}')

    logger.debug('Installing %s into %s', package_spec, prefix_path)
    await _run_checked_command(
        shlex.join(cmd),
        timeout=1800,
        extra_env=_npm_env(npm_registry=registry),
    )

    if not binary_path.exists():
        raise RuntimeError(f'OpenClaw binary not found after install: {binary_path}')

    version_file.write_text(actual_version)
    logger.debug('Installed OpenClaw %s at %s', actual_version, binary_path)
    return binary_path


__all__ = [
    'DEFAULT_INSTALL_DIR',
    'DEFAULT_PACKAGE_NAME',
    'MIN_NODE_VERSION',
    'RUNTIME_BIN_DIR',
    'build_package_spec',
    'ensure_supported_node_version',
    'get_latest_version',
    'install_openclaw',
    'parse_semver',
]
