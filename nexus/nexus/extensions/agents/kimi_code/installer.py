"""Install the TypeScript Kimi Code package into an isolated runtime prefix."""

from __future__ import annotations

import json
import re

from pathlib import Path

from nexus.extensions.agents.installed_common import (
    DEFAULT_AGENT_INSTALL_ROOT,
    DEFAULT_NPM_REGISTRY,
    build_npm_env,
    find_nexus_runtime_binary,
    resolve_npm_registry,
)
from nexus.utils.shell import get_clean_sandbox_env

from .process import run_process


NPM_PACKAGE = '@moonshot-ai/kimi-code'
DEFAULT_VERSION = '2.3.2'
DEFAULT_INSTALL_ROOT = f'{DEFAULT_AGENT_INSTALL_ROOT}/kimi-code'
MIN_NODE_VERSION = (22, 19, 0)


def parse_node_version(value: str) -> tuple[int, int, int]:
    match = re.fullmatch(r'v?(\d+)\.(\d+)\.(\d+)', value.strip())
    if match is None:
        raise ValueError(f'Invalid Node.js version: {value!r}')
    return int(match[1]), int(match[2]), int(match[3])


async def install_kimi_code(
    *,
    version: str = DEFAULT_VERSION,
    npm_registry: str | None = DEFAULT_NPM_REGISTRY,
    install_root: str = DEFAULT_INSTALL_ROOT,
) -> tuple[Path, Path]:
    """Return the selected Node executable and verified package entry point."""
    if not re.fullmatch(r'\d+\.\d+\.\d+(?:-[A-Za-z0-9.-]+)?', version):
        raise ValueError('Kimi Code requires an exact npm version for reproducibility')
    node = find_nexus_runtime_binary('node')
    npm = find_nexus_runtime_binary('npm')
    if not node or not npm:
        raise RuntimeError('Kimi Code requires Node.js >=22.19.0 and npm in the runtime')
    root = Path(install_root) / version
    root.mkdir(parents=True, exist_ok=True)
    registry = resolve_npm_registry(npm_registry)
    env = {**get_clean_sandbox_env(), **build_npm_env(npm_registry=registry)}
    probe = await run_process([node, '--version'], cwd=str(root), env=env, timeout=30)
    if probe.exit_code != 0 or parse_node_version(probe.stdout) < MIN_NODE_VERSION:
        raise RuntimeError(f'Kimi Code requires Node.js >=22.19.0; found {probe.stdout.strip()}')
    package = root / 'node_modules' / NPM_PACKAGE
    entry = package / 'dist/main.mjs'
    metadata_path = package / 'package.json'
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    if not (entry.is_file() and metadata.get('name') == NPM_PACKAGE and metadata.get('version') == version):
        result = await run_process(
            [
                npm,
                'install',
                '--prefix',
                str(root),
                '--no-audit',
                '--no-fund',
                '--registry',
                registry,
                f'{NPM_PACKAGE}@{version}',
            ],
            cwd=str(root),
            env=env,
            timeout=1800,
        )
        if result.exit_code != 0:
            raise RuntimeError(f'Kimi Code npm install failed: {(result.stderr or result.stdout)[-3000:]}')
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        if not (entry.is_file() and metadata.get('name') == NPM_PACKAGE and metadata.get('version') == version):
            raise RuntimeError('Kimi Code installed package does not match the requested version')
    return Path(node), entry
