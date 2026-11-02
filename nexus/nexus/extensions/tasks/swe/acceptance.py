"""Opt-in source verification for isolated HY evaluation acceptance tasks."""

import asyncio
import contextlib
import inspect

from collections.abc import Callable
from types import MethodType
from typing import Any, cast

from nexus.core.context import ExecutionContext
from nexus.runner.hooks.base import RunHook, run_hook_registry
from nexus.runner.single import RunOutput


def source_fingerprint(paths: list[str]) -> dict[str, Any]:
    """Read only Nexus source hashes and an explicit list of public settings."""
    import hashlib
    import importlib.metadata
    import json
    import os

    from pathlib import Path

    import nexus

    package = Path(nexus.__file__).resolve().parent
    distribution = importlib.metadata.distribution('nexus-agent')
    direct_url = json.loads(distribution.read_text('direct_url.json') or '{}')
    hashes = {}
    for path in paths:
        relative = Path(path)
        if relative.is_absolute() or '..' in relative.parts or relative.parts[0] != 'nexus':
            raise ValueError(f'Expected a relative Nexus source path: {path}')
        hashes[path] = hashlib.sha256(package.joinpath(*relative.parts[1:]).read_bytes()).hexdigest()
    return {
        'package_version': distribution.version,
        'source_commit': direct_url.get('vcs_info', {}).get('commit_id'),
        'source_hashes': hashes,
        'environment': {
            key: os.environ.get(key)
            for key in (
                'SWE_AGENT_BASH_BACKEND',
                'NEXUS_CALL_DEADLINE',
                'NEXUS_RETRY_TOTAL_TIMEOUT',
                'NEXUS_LOG_EXEMPT_SIZE',
            )
        },
    }


@run_hook_registry.register('superswe_acceptance')
class SuperSWEAcceptanceHook(RunHook):
    """Fail acceptance when the controller or an acquired worker loads other code."""

    def __init__(
        self, expected_hashes: dict[str, str], controller_environment: dict[str, str], expected_package_version: str
    ) -> None:
        if not expected_hashes:
            raise ValueError('Acceptance requires explicit source hashes')
        if not expected_package_version:
            raise ValueError('Acceptance requires an explicit package version')
        self.expected_hashes = expected_hashes
        self.controller_environment = controller_environment
        self.expected_package_version = expected_package_version
        self.result: dict[str, Any] = {'workers': {}, 'worker_errors': {}, 'release_errors': {}, 'passed': False}
        self._restore_release: Callable[[], None] | None = None

    def _check_hashes(self, fingerprint: dict[str, Any], label: str) -> None:
        if fingerprint['package_version'] != self.expected_package_version:
            raise ValueError(f'{label} loaded an unexpected Nexus package version')
        if fingerprint['source_hashes'] != self.expected_hashes:
            raise ValueError(f'{label} loaded unexpected Nexus source files')

    async def on_context_ready(self, ctx: ExecutionContext) -> None:
        fingerprint = source_fingerprint(list(self.expected_hashes))
        self.result['controller'] = fingerprint
        self._check_hashes(fingerprint, 'Controller')
        actual = fingerprint['environment']
        if any(actual.get(key) != value for key, value in self.controller_environment.items()):
            raise ValueError('Controller environment differs from the acceptance configuration')
        provider = ctx.runtime_provider
        if provider is None:
            raise ValueError('Acceptance requires a runtime provider')
        original_release = provider.release

        async def observed_release(_provider: Any, name: str, *, force: bool = False) -> bool:
            try:
                runtime = ctx._runtimes.get(name)
                if runtime is not None:
                    with contextlib.suppress(Exception):
                        await self._record_worker(name, runtime)
            finally:
                # Source inspection must neither suppress cleanup nor hide its errors.
                try:
                    released = await original_release(name, force=force)
                except BaseException as exc:
                    self.result['release_errors'][name] = f'{type(exc).__name__}: {exc}'
                    self.result['passed'] = False
                    raise
            return released

        wrapped_release = MethodType(observed_release, provider)

        def restore_release() -> None:
            if provider.release is wrapped_release:
                cast(Any, provider).release = original_release

        # This provider belongs to one runner invocation. SWE releases its workers
        # before on_run_output, so capture their source immediately before release.
        cast(Any, provider).release = wrapped_release
        self._restore_release = restore_release

    async def _record_worker(self, name: str, runtime: Any) -> None:
        try:
            code = 'from typing import Any\n' + inspect.getsource(source_fingerprint)
            code += f'\nreturn source_fingerprint({list(self.expected_hashes)!r})\n'
            fingerprint = await asyncio.wait_for(runtime.exec(code), timeout=120)
            self.result['workers'][name] = fingerprint
            self._check_hashes(fingerprint, f'Worker {name}')
        except BaseException as exc:
            self.result['worker_errors'][name] = f'{type(exc).__name__}: {exc}'
            self.result['passed'] = False
            raise

    async def on_run_output(self, ctx: ExecutionContext, output: RunOutput) -> None:
        self.result['passed'] = False
        # Also cover workers a task leaves for ExecutionContext to release.
        for name, runtime in list(ctx._runtimes.items()):
            with contextlib.suppress(Exception):
                await self._record_worker(name, runtime)
        if self.result['worker_errors']:
            raise ValueError('Worker source verification failed; see acceptance hook results')
        if self.result['release_errors']:
            raise ValueError('Worker release failed; see acceptance hook results')
        if not self.result['workers']:
            raise ValueError('No acquired worker was available for acceptance verification')
        self.result['passed'] = True

    async def on_context_closing(self, ctx: ExecutionContext) -> None:
        if self._restore_release is not None:
            self._restore_release()
            self._restore_release = None

    def get_result(self) -> dict[str, Any]:
        return self.result
