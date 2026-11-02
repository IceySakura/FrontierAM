"""Claude Code install path helpers for native bridge adapters."""

from __future__ import annotations

from nexus.extensions.agents.claude_code.installer import NPM_PACKAGE
from nexus.extensions.agents.installed_common import DEFAULT_NPM_REGISTRY, resolve_npm_registry


DEFAULT_CLAUDE_INSTALL_DIR = '/tmp/nexus-claude-code'


def installed_posix_claude_executable() -> str:
    return f'{DEFAULT_CLAUDE_INSTALL_DIR}/node_modules/.bin/claude'


def package_spec(version: str = 'latest') -> str:
    if not version or version == 'latest':
        return NPM_PACKAGE
    return f'{NPM_PACKAGE}@{version}'


def npm_registry_arg(npm_registry: str | None = DEFAULT_NPM_REGISTRY) -> str:
    return resolve_npm_registry(npm_registry)


def npm_install_timeout() -> float:
    return 1800.0


__all__ = [
    'DEFAULT_CLAUDE_INSTALL_DIR',
    'installed_posix_claude_executable',
    'npm_install_timeout',
    'npm_registry_arg',
    'package_spec',
]
