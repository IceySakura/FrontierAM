"""Install pinned Hyness engine, App and plugin packages in the execution target."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import weakref

from pathlib import Path

from nexus.extensions.agents.installed_common import (
    DEFAULT_AGENT_INSTALL_ROOT,
    build_npm_env,
    find_nexus_runtime_binary,
    resolve_npm_registry,
)
from nexus.extensions.agents.installed_common.execution import AgentExecutionTransport


DEFAULT_INSTALL_ROOT = f'{DEFAULT_AGENT_INSTALL_ROOT}/hyness'
_PINNED_SPEC = re.compile(
    r'(?:@[a-z0-9][a-z0-9._-]*/)?[a-z0-9][a-z0-9._-]*@\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?'
)
_LOCKS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def normalize_package_specs(packages: list[str]) -> list[str]:
    """Require exact npm versions or caller-local tarballs uploaded by the agent."""
    specs = list(dict.fromkeys(spec.strip() for spec in packages))
    if not specs:
        raise ValueError('Specify pinned Hyness packages, including any required App and plugin packages')
    versions: dict[str, str] = {}
    for spec in specs:
        if _PINNED_SPEC.fullmatch(spec):
            name, _, version = spec.rpartition('@')
            if name in versions and versions[name] != version:
                raise ValueError(f'Conflicting versions for {name}: {versions[name]} and {version}')
            versions[name] = version
        elif not (spec.startswith('/') and spec.endswith('.tgz')):
            raise ValueError('Hyness packages must be exact npm versions or absolute runtime .tgz paths')
    return specs


async def install_hyness(
    transport: AgentExecutionTransport,
    packages: list[str],
    npm_registry: str | None,
) -> dict[str, str]:
    specs = normalize_package_specs(packages)
    await transport.initialize()
    if not transport.is_local and any(spec.startswith('/') for spec in specs):
        raise ValueError(
            'Execution targets require registry packages; local tarballs are supported in ordinary sandboxes'
        )
    node = (find_nexus_runtime_binary('node') if transport.is_local else None) or await transport.find_binary('node')
    result = await transport.exec_as_agent([node, '--version'], timeout=30, check=True)
    version = tuple(int(part) for part in result.stdout.strip().removeprefix('v').split('.')[:3])
    if not ((22, 19, 0) <= version < (23, 0, 0) or (23, 11, 0) <= version < (24, 0, 0) or version >= (24, 10, 0)):
        raise RuntimeError('Hyness requires Node ^22.19.0, ^23.11.0 or >=24.10.0')
    digest = hashlib.sha256(json.dumps(specs).encode()).hexdigest()[:20]
    prefix = str(Path(transport.install_dir(agent_name='hyness', default=DEFAULT_INSTALL_ROOT)) / digest)
    marker = str(Path(prefix) / '.nexus-packages.json')
    entry = str(Path(prefix) / 'node_modules/@tencent/hyness/apps/cli/dist/hyness.js')
    loop_locks = _LOCKS.setdefault(asyncio.get_running_loop(), {})
    async with loop_locks.setdefault(prefix, asyncio.Lock()):
        cached = all(_PINNED_SPEC.fullmatch(spec) for spec in specs)
        if not (
            cached and await transport.read_text(marker) == json.dumps(specs) and await transport.file('exists', entry)
        ):
            await transport.mkdir(prefix)
            npm = (find_nexus_runtime_binary('npm') if transport.is_local else None) or await transport.find_binary(
                'npm'
            )
            registry = resolve_npm_registry(npm_registry)
            env = build_npm_env(npm_registry=registry)
            env['PATH'] = f'{Path(node).parent}:{transport.agent_env().get("PATH", env.get("PATH", ""))}'
            await transport.exec_as_agent(
                [
                    npm,
                    'install',
                    '--prefix',
                    prefix,
                    '--ignore-scripts',
                    '--save-exact',
                    '--no-audit',
                    '--no-fund',
                    '--registry',
                    registry,
                    *specs,
                ],
                working_dir=prefix,
                env=env,
                timeout=1800,
                check=True,
            )
            if not await transport.file('exists', entry):
                raise RuntimeError('The Hyness package does not provide its public CLI entry')
            await transport.write_text(marker, json.dumps(specs))
    return {'node': node, 'entry': entry, 'prefix': prefix}


async def resolve_app_config(
    transport: AgentExecutionTransport, installed: dict[str, str], config: str | None
) -> str | None:
    if config is None:
        return None
    relative = Path(config)
    path = relative if relative.is_absolute() else Path(installed['prefix']) / 'node_modules' / relative
    if not await transport.file('exists', path):
        raise FileNotFoundError(f'Hyness App configuration is missing: {path}')
    return str(path)


async def verify_app_engine(transport: AgentExecutionTransport, installed: dict[str, str], app: str) -> None:
    # Public resolution catches nested npm engines even when top-level versions look correct.
    check = (
        "const {createRequire}=require('node:module'),fs=require('node:fs');"
        "const a=createRequire(process.argv[1]).resolve('@tencent/hyness/cli');"
        "const b=createRequire(process.argv[2]).resolve('@tencent/hyness/cli');"
        "if(fs.realpathSync(a)!==fs.realpathSync(b))throw Error('App and CLI resolve different Hyness engines')"
    )
    await transport.exec_as_agent(
        [installed['node'], '-e', check, app, str(Path(installed['prefix']) / 'package.json')],
        timeout=30,
        check=True,
    )
