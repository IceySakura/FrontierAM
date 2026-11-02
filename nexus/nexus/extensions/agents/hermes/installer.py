"""Hermes Agent installer helpers."""

from __future__ import annotations

import logging
import re
import shlex
import shutil

from pathlib import Path

from nexus.extensions.agents.installed_common import (
    DEFAULT_AGENT_INSTALL_ROOT,
    build_runtime_path,
    find_nexus_runtime_binary,
)


logger = logging.getLogger(__name__)


DEFAULT_HERMES_VERSION = 'latest'
DEFAULT_INSTALL_ROOT = f'{DEFAULT_AGENT_INSTALL_ROOT}/hermes-agent'
DEFAULT_PACKAGE_NAME = 'hermes-agent'
DEFAULT_GITHUB_ARCHIVE = 'https://github.com/NousResearch/hermes-agent/archive/refs/heads/main.zip'
MIN_PYTHON_VERSION = (3, 11, 0)
_SEMVER_RE = re.compile(r'(?P<major>\d+)\.(?P<minor>\d+)(?:\.(?P<patch>\d+))?')


def _version_dir_name(version: str) -> str:
    return version.replace('/', '-').replace('@', '-').replace('[', '-').replace(']', '-')


def parse_python_version(version: str) -> tuple[int, int, int]:
    """Parse a Python version string."""
    match = _SEMVER_RE.search(version.strip())
    if not match:
        raise RuntimeError(f'Unable to parse Python version string: {version}')

    return (
        int(match.group('major')),
        int(match.group('minor')),
        int(match.group('patch') or 0),
    )


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


async def find_supported_python() -> str:
    """Find a runtime Python executable new enough for Hermes Agent."""
    candidates = [
        find_nexus_runtime_binary('python3.13'),
        find_nexus_runtime_binary('python3.12'),
        find_nexus_runtime_binary('python3.11'),
        find_nexus_runtime_binary('python3'),
        find_nexus_runtime_binary('python'),
    ]
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        snippet = 'import ensurepip, sys; print(sys.version.split()[0])'
        try:
            stdout = await _run_checked_command(
                f'{shlex.quote(candidate)} -c {shlex.quote(snippet)}',
                timeout=30,
                extra_env={'PATH': build_runtime_path()},
            )
            version = stdout.strip()
            if parse_python_version(version) >= MIN_PYTHON_VERSION:
                logger.debug('Using Python %s at %s for Hermes Agent install', version, candidate)
                return candidate
            logger.debug('Skipping Python candidate %s: version %s is too old', candidate, version)
        except Exception as exc:
            logger.debug('Skipping Python candidate %s: %s', candidate, exc)

    required = '.'.join(str(part) for part in MIN_PYTHON_VERSION)
    raise RuntimeError(
        f'Hermes Agent requires Python >= {required} with a working ensurepip, but no supported runtime Python was found'
    )


def _get_installed_version(venv_dir: Path) -> str | None:
    version_file = venv_dir / '.version'
    if not version_file.exists():
        return None
    value = version_file.read_text().strip()
    return value or None


async def _read_package_version(venv_python: Path) -> str:
    snippet = 'from importlib.metadata import version; print(version("hermes-agent"))'
    command = f'{shlex.quote(str(venv_python))} -c {shlex.quote(snippet)}'
    return (
        await _run_checked_command(
            command,
            timeout=30,
            extra_env={'PATH': build_runtime_path(str(venv_python.parent))},
        )
    ).strip()


async def install_hermes_agent(
    version: str = DEFAULT_HERMES_VERSION,
    install_root: str = DEFAULT_INSTALL_ROOT,
    package_name: str = DEFAULT_PACKAGE_NAME,
) -> Path:
    """Install Hermes Agent into a runtime-local virtual environment and return the CLI path."""
    python_path = await find_supported_python()
    normalized_version = version.strip() or DEFAULT_HERMES_VERSION
    install_dir = Path(install_root) / _version_dir_name(normalized_version)
    venv_python = install_dir / 'bin' / 'python'
    hermes_bin = install_dir / 'bin' / 'hermes'
    installed_version = _get_installed_version(install_dir)

    if normalized_version != 'latest' and hermes_bin.exists() and installed_version == normalized_version:
        logger.debug('Reusing cached Hermes Agent install at %s', hermes_bin)
        return hermes_bin

    if normalized_version == 'latest' and hermes_bin.exists():
        logger.debug('Refreshing cached Hermes Agent latest install at %s', install_dir)
    else:
        shutil.rmtree(install_dir, ignore_errors=True)
        install_dir.mkdir(parents=True, exist_ok=True)
        await _run_checked_command(
            f'{shlex.quote(python_path)} -m venv {shlex.quote(str(install_dir))}',
            timeout=300,
            extra_env={'PATH': build_runtime_path()},
        )

    if not venv_python.exists():
        raise RuntimeError(f'Failed to create Hermes virtual environment at {install_dir}')

    pip_env = {
        'PATH': build_runtime_path(str(install_dir / 'bin')),
        'PIP_DISABLE_PIP_VERSION_CHECK': '1',
        'PIP_NO_INPUT': '1',
    }
    await _run_checked_command(
        f'{shlex.quote(str(venv_python))} -m pip install --upgrade pip setuptools wheel',
        timeout=900,
        extra_env=pip_env,
    )

    if normalized_version == 'latest':
        package_spec = f'{package_name}[cli] @ {DEFAULT_GITHUB_ARCHIVE}'
    elif normalized_version.startswith(('git+', 'http://', 'https://')):
        package_spec = f'{package_name}[cli] @ {normalized_version}'
    else:
        package_spec = f'{package_name}[cli]=={normalized_version}'
    await _run_checked_command(
        f'{shlex.quote(str(venv_python))} -m pip install --upgrade {shlex.quote(package_spec)}',
        timeout=1800,
        extra_env=pip_env,
    )

    if not hermes_bin.exists():
        raise RuntimeError(f'Hermes binary not found after install: {hermes_bin}')

    actual_version = await _read_package_version(venv_python)
    (install_dir / '.version').write_text(actual_version)
    logger.debug('Installed Hermes Agent %s at %s', actual_version, hermes_bin)
    return hermes_bin


__all__ = [
    'DEFAULT_GITHUB_ARCHIVE',
    'DEFAULT_HERMES_VERSION',
    'DEFAULT_INSTALL_ROOT',
    'DEFAULT_PACKAGE_NAME',
    'MIN_PYTHON_VERSION',
    'find_supported_python',
    'install_hermes_agent',
    'parse_python_version',
]
