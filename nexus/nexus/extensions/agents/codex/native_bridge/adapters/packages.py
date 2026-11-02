"""Codex install helpers for native bridge targets."""

from __future__ import annotations

from nexus.extensions.agents.installed_common import (
    DEFAULT_NPM_REGISTRY,
    PUBLIC_NPM_REGISTRY,
    npm_install_timeout,
    registry_candidates,
    resolve_npm_registry,
)


NPM_PACKAGE = '@openai/codex'
DEFAULT_CODEX_INSTALL_DIR = '/tmp/nexus-codex-cli'

NATIVE_EXECUTABLE_MARKER = 'nexus-codex-executable='
"""Line the npm-free installer prints so the host learns the binary's path.

Codex vendors its binary under a Rust target triple that only the sandbox can
name (``package/vendor/x86_64-pc-windows-msvc/bin/codex.exe``), and the layout
below that has already moved once -- 0.114.0 shipped ``vendor/<triple>/codex/``
while 0.146.1+ ship ``vendor/<triple>/bin/``. Rather than encode either fact
host-side, the install script locates the entrypoint and reports it back.
"""


def installed_posix_codex_executable() -> str:
    return f'{DEFAULT_CODEX_INSTALL_DIR}/node_modules/.bin/codex'


def package_spec(version: str) -> str:
    return NPM_PACKAGE if version == 'latest' else f'{NPM_PACKAGE}@{version}'


def native_package_ref(version: str) -> str:
    """Registry ref suffix identifying the platform build of *version*.

    Codex publishes its per-platform binaries as *aliased versions of the same
    package* -- ``@openai/codex-win32-x64`` is ``npm:@openai/codex@0.153.4-win32-x64``
    -- so the platform is a version suffix, not a separate package name the way
    Claude Code does it. An empty result means "resolve the platform dist-tag"
    (``win32-x64``, ``linux-x64``, ... all exist), which is how ``latest`` is
    pinned without downloading the 13MB packument.
    """
    return '' if not version or version == 'latest' else version


def parse_native_executable(output: str) -> str:
    """Read the installed Codex path out of an install log."""
    for line in reversed((output or '').splitlines()):
        stripped = line.strip()
        if stripped.startswith(NATIVE_EXECUTABLE_MARKER):
            executable = stripped[len(NATIVE_EXECUTABLE_MARKER) :].strip()
            if executable:
                return executable
    raise RuntimeError(f'Codex install did not report an executable path: {output[-2000:]}')


def npm_registry_arg(npm_registry: str | None = DEFAULT_NPM_REGISTRY) -> str:
    return resolve_npm_registry(npm_registry)


__all__ = [
    'DEFAULT_CODEX_INSTALL_DIR',
    'NATIVE_EXECUTABLE_MARKER',
    'NPM_PACKAGE',
    'PUBLIC_NPM_REGISTRY',
    'installed_posix_codex_executable',
    'native_package_ref',
    'npm_install_timeout',
    'npm_registry_arg',
    'package_spec',
    'parse_native_executable',
    'registry_candidates',
]
