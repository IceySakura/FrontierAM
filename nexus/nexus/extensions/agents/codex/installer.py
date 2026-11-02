"""Codex CLI installer."""

from __future__ import annotations

import hashlib
import json
import logging
import stat
import tempfile

from pathlib import Path

import aiohttp

from nexus.extensions.agents.installed_common import (
    DEFAULT_NPM_REGISTRY,
    build_npm_env,
    resolve_npm_registry,
)
from nexus.extensions.agents.installed_common.execution import AgentExecutionTransport
from nexus.utils.platform import is_windows


logger = logging.getLogger(__name__)


DEFAULT_CODEX_VERSION = '0.114.0'


def _npm_env(npm_registry: str | None = None) -> dict[str, str]:
    env = build_npm_env(npm_registry=npm_registry)
    env['NPM_CONFIG_STRICT_SSL'] = 'false'
    env['npm_config_strict_ssl'] = 'false'
    return env


def _version_from_metadata(metadata: str | None) -> str | None:
    try:
        data = json.loads(metadata) if metadata else None
    except ValueError:
        return None
    version = data.get('version') if isinstance(data, dict) else None
    return version if isinstance(version, str) else None


def get_installed_version(install_dir: str) -> str | None:
    """Return the installed Codex CLI version, treating damaged metadata as absent."""
    package_json = Path(install_dir) / 'node_modules' / '@openai' / 'codex' / 'package.json'
    try:
        return _version_from_metadata(package_json.read_text(encoding='utf-8'))
    except (OSError, UnicodeError):
        return None


async def probe_codex_version(transport: AgentExecutionTransport, executable: str | Path) -> str | None:
    """Read the version from a working CLI in the actual Agent environment."""
    from nexus.runtime.executors.command import CommandRunStatus

    result = await transport.exec_as_agent([str(executable), '--version'], timeout=30)
    if result.status != CommandRunStatus.Finished or result.return_code != 0:
        return None
    output = (result.stdout or '').strip()
    prefix = 'codex-cli '
    return output[len(prefix) :].strip() or None if output.startswith(prefix) else None


async def install_codex_cli(
    install_dir: str,
    version: str | None = DEFAULT_CODEX_VERSION,
    npm_registry: str | None = DEFAULT_NPM_REGISTRY,
    transport: AgentExecutionTransport | None = None,
) -> Path:
    """Install Codex CLI with one cache and npm policy for local and target execution."""
    from nexus.runtime.executors.command import CommandRunStatus

    if version is None:
        version = DEFAULT_CODEX_VERSION
    transport = transport or AgentExecutionTransport()
    await transport.initialize()
    install_path = Path(install_dir)
    windows = transport.is_local and is_windows()
    codex_path = install_path / 'node_modules' / '.bin' / ('codex.cmd' if windows else 'codex')
    try:
        metadata = await transport.read_text(install_path / 'node_modules/@openai/codex/package.json')
    except (OSError, UnicodeError):
        metadata = None
    installed_version = _version_from_metadata(metadata)
    if version != 'latest' and installed_version == version and await transport.file('exists', codex_path):
        logger.debug(f'Codex CLI already installed at {codex_path} (version {installed_version})')
        return codex_path

    if version != 'latest':
        executable = await transport.find_binary('codex')
        if executable != 'codex':
            detected_version = await probe_codex_version(transport, executable)
            if detected_version == version:
                logger.debug('Reusing preinstalled Codex CLI at %s (version %s)', executable, detected_version)
                return Path(executable)

    await transport.mkdir(install_path)
    package_spec = '@openai/codex' if version == 'latest' else f'@openai/codex@{version}'
    npm_executable = await transport.find_binary('npm')
    registry = resolve_npm_registry(npm_registry)
    command = [
        npm_executable,
        'install',
        '--prefix',
        str(install_path),
        '--no-audit',
        '--no-fund',
        '--registry',
        registry,
        package_spec,
    ]
    env = _npm_env(npm_registry=registry)
    if not transport.is_local:
        env['PATH'] = f'{Path(npm_executable).parent}:{transport.agent_env().get("PATH", "")}'
    logger.debug(f'Installing Codex CLI: {package_spec} via registry {registry}')
    result = await transport.exec_as_agent(command, working_dir=str(install_path), timeout=1800, env=env)
    if result.status != CommandRunStatus.Finished or result.return_code != 0:
        stderr = result.stderr or ''
        stdout = result.stdout or ''
        raise RuntimeError(f'Failed to install Codex CLI: {stderr or stdout or result.status}')
    if not await transport.file('exists', codex_path):
        raise RuntimeError(f'Codex CLI install succeeded but binary is missing: {codex_path}')
    logger.debug(f'Codex CLI installed at {codex_path}')
    return codex_path


async def install_codex_from_url(
    url: str,
    install_dir: str,
    checksum: str | None = None,
    transport: AgentExecutionTransport | None = None,
) -> Path:
    """Install a raw binary from a URL (including signed COS URLs)."""
    if transport is not None and not transport.is_local:
        return Path(await transport.file('download', Path(install_dir) / 'codex', url=url, checksum=checksum))
    install_path = Path(install_dir)
    install_path.mkdir(parents=True, exist_ok=True)
    binary_path = install_path / ('codex.exe' if is_windows() else 'codex')
    # Download beside the destination so replacement is atomic, preserving an existing
    # installation on download/checksum failure. Never log a signed URL.
    with tempfile.NamedTemporaryFile(dir=install_path, prefix='codex-', suffix='.tmp', delete=False) as temporary:
        temp_path = Path(temporary.name)
    try:
        digest = hashlib.sha256()
        async with aiohttp.ClientSession(trust_env=True) as session:
            async with session.get(url) as response:
                response.raise_for_status()
                with temp_path.open('wb') as stream:
                    async for chunk in response.content.iter_chunked(64 * 1024):
                        stream.write(chunk)
                        digest.update(chunk)
        if checksum and digest.hexdigest().lower() != checksum.lower():
            raise RuntimeError('Codex binary SHA256 checksum verification failed')
        if not is_windows():
            temp_path.chmod(temp_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        temp_path.replace(binary_path)
    except aiohttp.ClientResponseError as exc:
        raise RuntimeError(f'Codex binary download failed: HTTP {exc.status}') from None
    except aiohttp.ClientError as exc:
        raise RuntimeError(f'Codex binary download failed: {type(exc).__name__}') from None
    finally:
        temp_path.unlink(missing_ok=True)
    logger.debug('Codex binary installed at %s', binary_path)
    return binary_path


__all__ = [
    'DEFAULT_CODEX_VERSION',
    'get_installed_version',
    'install_codex_cli',
    'install_codex_from_url',
]
