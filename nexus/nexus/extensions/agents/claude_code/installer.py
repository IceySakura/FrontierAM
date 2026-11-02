"""Claude Code installer.

Installs the Anthropic ``@anthropic-ai/claude-code`` package via npm inside
the runtime. The default registry points at Tencent's internal mirror; a
custom registry or a direct binary URL can be supplied as overrides.

Upstream publish pattern (confirmed against the npm registry metadata):

* **2.1.0 – 2.1.112** ship as a JS bundle. ``package.json`` declares
  ``bin.claude = "cli.js"`` with a ``#!/usr/bin/env node`` shebang, and the
  only ``optionalDependencies`` are the ``@img/sharp-*`` image libs. Running
  these requires a real ``node`` interpreter on the sandbox ``PATH``.
* **2.1.113 onward** switched to native binaries. ``bin.claude`` now points
  at ``bin/claude.exe`` and the package pulls eight
  ``@anthropic-ai/claude-code-<os>-<arch>[-musl]`` subpackages as optional
  deps (darwin-arm64, darwin-x64, linux-arm64, linux-arm64-musl, linux-x64,
  linux-x64-musl, win32-arm64, win32-x64). A postinstall overwrites
  ``bin/claude.exe`` with the matching native binary; we ignore that hook
  and exec the subpackage's binary directly (see
  :func:`_resolve_claude_executable`), which sidesteps postinstall platform
  misdetection (e.g. glibc Node picking a glibc subpackage on a musl host).
* ``dist-tags`` as of 2026-04: ``latest`` tracks the newest native release,
  but ``stable`` still points at a JS-bundle version — pinning to
  ``stable`` or to a version ``<=2.1.112`` brings back the node dependency.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import stat
import time

from pathlib import Path
from typing import Any

import aiohttp

from nexus.extensions.agents.claude_code import install_support
from nexus.extensions.agents.claude_code.install_support import (
    LINUX_LIBC_PROBE,
    _alternate_linux_native_package,
    _detect_claude_native_package,
    _detect_linux_libc,
    _native_package_npm_env,
    _read_elf_interpreter,
    _resolve_claude_executable,
    _uses_native_subpackages,
    claude_native_package_from_probe,
    get_installed_version,
)
from nexus.extensions.agents.installed_common import (
    DEFAULT_NPM_REGISTRY,
    build_npm_env,
    resolve_npm_registry,
)
from nexus.extensions.agents.installed_common.execution import AgentExecutionTransport


logger = logging.getLogger(__name__)


DEFAULT_CLAUDE_CODE_VERSION = 'latest'
NPM_PACKAGE = '@anthropic-ai/claude-code'


async def _installation_operation(transport: AgentExecutionTransport, operation: str, **kwargs: Any) -> Any:
    request = {'operation': operation, **kwargs}
    if transport.is_local:
        return install_support.operate(request)
    return await transport.python(Path(install_support.__file__).read_text(), request)


async def install_claude_code_via_npm(
    install_dir: str,
    version: str | None = None,
    npm_registry: str | None = DEFAULT_NPM_REGISTRY,
    native_package: str | None = None,
    transport: AgentExecutionTransport | None = None,
) -> Path:
    """Install Claude Code with one policy, inspecting files where the CLI will run.

    An omitted version reuses an installed CLI; otherwise installation uses
    DEFAULT_CLAUDE_CODE_VERSION. Explicit versions and latest remain authoritative.
    ``native_package`` overrides detection when installing for another filesystem.
    """
    from nexus.runtime.executors.command import CommandRunStatus

    transport = transport or AgentExecutionTransport()
    await transport.initialize()
    install_path = Path(install_dir)
    native_package_is_probed = native_package is not None
    local_execution = not native_package_is_probed or not transport.is_local
    cache_version = version
    if cache_version is None:
        cached = await _installation_operation(
            transport,
            'inspect',
            install_dir=install_dir,
            preferred_package=native_package,
            local_execution=local_execution,
        )
        cache_version = cached['version'] or DEFAULT_CLAUDE_CODE_VERSION
    if not _uses_native_subpackages(cache_version):
        native_package = None
        native_package_is_probed = False
        local_execution = True
    elif native_package is None:
        native_package = await _installation_operation(transport, 'detect')
    if not transport.is_local and native_package is not None:
        # Target loader detection is authoritative even if its npm comes from Nix.
        native_package_is_probed = True
    info = await _installation_operation(
        transport,
        'inspect',
        install_dir=install_dir,
        preferred_package=native_package,
        local_execution=local_execution,
    )
    if (
        info['executable'] is not None
        and info['version'] is not None
        and version != 'latest'
        and (version is None or info['version'] == version)
    ):
        executable = await _installation_operation(
            transport,
            'finalize',
            install_dir=install_dir,
            preferred_package=native_package,
            local_execution=local_execution,
        )
        top_level = Path(executable or info['executable'])
        logger.debug(f'Claude Code already installed at {top_level} (version {info["version"]})')
        return top_level

    if version != 'latest' and local_execution:
        executable = await transport.find_binary('claude')
        if executable == 'claude':
            home = transport.home_dir(invocation_id='claude-install', default=Path.home())
            candidate = home / '.local/bin/claude'
            if await transport.file('exists', candidate):
                executable = str(candidate)
        if executable != 'claude':
            result = await transport.exec_as_agent([executable, '--version'], timeout=30)
            match = re.fullmatch(r'(\S+) \(Claude Code\)', (result.stdout or '').strip())
            if (
                result.status == CommandRunStatus.Finished
                and result.return_code == 0
                and match is not None
                and (version is None or match.group(1) == version)
            ):
                logger.debug('Reusing preinstalled Claude Code at %s (version %s)', executable, match.group(1))
                return Path(executable)

    await transport.mkdir(install_path)
    version = DEFAULT_CLAUDE_CODE_VERSION if version is None else version
    # Cache inspection may have selected legacy JS; installation follows the requested/default release.
    if _uses_native_subpackages(version) and native_package is None:
        native_package = await _installation_operation(transport, 'detect')
        native_package_is_probed = native_package is not None and not transport.is_local
    package_spec = NPM_PACKAGE if version == 'latest' else f'{NPM_PACKAGE}@{version}'
    npm_executable = await transport.find_binary('npm')
    registry = resolve_npm_registry(npm_registry)
    selected_native_package = native_package
    result = None
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        package_specs = [package_spec]
        if selected_native_package is not None:
            native_spec = selected_native_package if version == 'latest' else f'{selected_native_package}@{version}'
            package_specs.append(native_spec)
        command = [
            npm_executable,
            'install',
            '--prefix',
            str(install_path),
            '--no-audit',
            '--no-fund',
            '--strict-ssl=false',
            *(['--ignore-scripts'] if selected_native_package is not None else []),
            *(['--force'] if native_package_is_probed else []),
            '--registry',
            registry,
            *package_specs,
        ]
        logger.debug(
            'Installing Claude Code (attempt %s/%s): %s via registry %s',
            attempt,
            max_attempts,
            ', '.join(package_specs),
            registry,
        )
        env = build_npm_env(
            npm_registry=registry,
            extra_env=_native_package_npm_env(selected_native_package) if selected_native_package is not None else None,
        )
        if not transport.is_local:
            env['PATH'] = f'{Path(npm_executable).parent}:{transport.agent_env().get("PATH", "")}'
        result = await transport.exec_as_agent(command, working_dir=str(install_path), timeout=1800, env=env)
        if result.status == CommandRunStatus.Finished and result.return_code == 0:
            native_package = selected_native_package
            break
        output = f'{result.stderr or ""}\n{result.stdout or ""}'
        if 'EBADPLATFORM' in output and selected_native_package is not None and not native_package_is_probed:
            alternate = _alternate_linux_native_package(selected_native_package)
            if alternate is not None and alternate != selected_native_package:
                logger.warning(
                    'npm rejected native package %s; retrying with libc variant %s', selected_native_package, alternate
                )
                selected_native_package = alternate
                continue
        transient_network_error = any(
            marker in output for marker in ('ECONNRESET', 'ETIMEDOUT', 'EAI_AGAIN', 'ECONNREFUSED', 'socket hang up')
        )
        if transient_network_error and attempt < max_attempts:
            logger.warning('Transient npm network error; retrying Claude Code install')
            await asyncio.sleep(2 ** (attempt - 1))
            continue
        break

    assert result is not None
    if result.status != CommandRunStatus.Finished or result.return_code != 0:
        raise RuntimeError(f'Failed to install Claude Code via npm: {result.stderr or result.stdout or result.status}')
    info = await _installation_operation(
        transport,
        'inspect',
        install_dir=install_dir,
        preferred_package=native_package,
        local_execution=local_execution,
    )
    if native_package is not None and info['executable'] is None:
        raise RuntimeError(
            'Claude Code install succeeded but the expected native package was not found or not launchable: '
            f'{native_package}'
        )
    executable = await _installation_operation(
        transport,
        'finalize',
        install_dir=install_dir,
        preferred_package=native_package,
        local_execution=local_execution,
    )
    if executable is None:
        package_hint = f' and expected native package {native_package}' if native_package is not None else ''
        raise RuntimeError(
            'Claude Code install succeeded but no binary was found under '
            f'{install_path}/node_modules/@anthropic-ai/claude-code[-*]{package_hint}'
        )
    logger.debug(f'Claude Code installed at {executable} (version {info["version"]})')
    return Path(executable)


# ---------------------------------------------------------------------------
# Optional: direct binary download (kept for offline / mirror-less setups)
# ---------------------------------------------------------------------------


def _verify_checksum(file_path: Path, expected_checksum: str) -> bool:
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    actual = sha256.hexdigest()
    if actual == expected_checksum:
        return True
    logger.error(f'Checksum mismatch: expected {expected_checksum}, got {actual}')
    return False


def _make_executable(file_path: Path) -> None:
    mode = file_path.stat().st_mode
    file_path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


async def _download_from_url(url: str, dest_path: Path) -> None:
    logger.debug(f'Downloading Claude Code from {url}')
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    connector = aiohttp.TCPConnector(ssl=False)
    async with aiohttp.ClientSession(trust_env=True, connector=connector) as session:
        async with session.get(url) as response:
            response.raise_for_status()
            total_size = response.content_length or 0
            downloaded = 0
            start = time.time()
            last_report = start
            with open(dest_path, 'wb') as f:
                async for chunk in response.content.iter_chunked(8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    now = time.time()
                    if now - last_report >= 10:
                        elapsed = now - start
                        speed = downloaded / elapsed if elapsed > 0 else 0
                        if total_size > 0:
                            pct = (downloaded / total_size) * 100
                            logger.debug(
                                f'Download progress: {pct:.1f}% '
                                f'({downloaded / 1024 / 1024:.2f}/{total_size / 1024 / 1024:.2f} MB), '
                                f'avg speed: {speed / 1024 / 1024:.2f} MB/s'
                            )
                        else:
                            logger.debug(
                                f'Downloaded {downloaded / 1024 / 1024:.2f} MB, '
                                f'avg speed: {speed / 1024 / 1024:.2f} MB/s'
                            )
                        last_report = now
    logger.debug(f'Downloaded Claude Code to {dest_path}')


async def install_claude_code_from_url(
    url: str,
    install_dir: str,
    checksum: str | None = None,
    transport: AgentExecutionTransport | None = None,
) -> Path:
    """Install a prebuilt Claude Code binary from *url*."""
    if transport is not None and not transport.is_local:
        return Path(await transport.file('download', Path(install_dir) / 'claude', url=url, checksum=checksum))
    install_path = Path(install_dir)
    install_path.mkdir(parents=True, exist_ok=True)
    binary_path = install_path / 'claude'
    temp_path = install_path / 'claude.url.tmp'

    await _download_from_url(url, temp_path)

    if checksum and not _verify_checksum(temp_path, checksum):
        temp_path.unlink(missing_ok=True)
        raise RuntimeError('Checksum verification failed')

    temp_path.replace(binary_path)
    _make_executable(binary_path)
    logger.debug(f'Claude Code binary installed at {binary_path}')
    return binary_path


__all__ = [
    'DEFAULT_CLAUDE_CODE_VERSION',
    'NPM_PACKAGE',
    'get_installed_version',
    'install_claude_code_from_url',
    'install_claude_code_via_npm',
    '_detect_claude_native_package',
    '_detect_linux_libc',
    '_read_elf_interpreter',
    '_resolve_claude_executable',
    'LINUX_LIBC_PROBE',
    'claude_native_package_from_probe',
]
