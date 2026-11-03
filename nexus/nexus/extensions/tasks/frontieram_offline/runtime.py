"""A Nexus Runtime for one strictly offline, hash-locked numerical workspace."""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import os
import re
import sys
import time
import uuid

from pathlib import Path
from typing import Any, cast

from nexus.runtime.client.base import Runtime
from nexus.runtime.executors.models.command import CommandRunStatus, RunCommandResponse
from nexus.runtime.models import InvokeRequest
from nexus.runtime.providers.base import RuntimeProvider
from nexus.runtime.providers.registry import runtime_provider_registry

from .assets import LockedAssets


LEASE_VERSION = 'frontieram-offline-runtime-v1'


async def _blocking(function: Any, *args: Any, **kwargs: Any) -> Any:
    """Finish lifecycle operations before cancellation can orphan their container."""
    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    cancelled = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as error:
            cancelled = error
        except Exception:
            break
    try:
        result = task.result()
    except BaseException as error:
        if cancelled is not None:
            cancelled.add_note(f'Lifecycle operation also failed: {type(error).__name__}')
            raise cancelled from None
        raise
    if cancelled is not None:
        raise cancelled
    return result


def _write_lease(path: Path, value: dict[str, Any], *, create: bool = False) -> None:
    flags = os.O_WRONLY | os.O_CREAT | (os.O_EXCL if create else os.O_TRUNC)
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, 'w') as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write('\n')


class OfflineDockerRuntime(Runtime):
    """Only command execution is exposed; arbitrary Nexus invoke is forbidden."""

    def __init__(
        self,
        *,
        assets: LockedAssets,
        docker: Any,
        container: str,
        output_dir: Path,
        configuration_checks: dict[str, bool],
    ) -> None:
        super().__init__()
        self.assets = assets
        self.docker = docker
        self.container = container
        self.output_dir = output_dir
        self.configuration_checks = configuration_checks
        self.resource_spec = {'cpu': 1, 'memory': 4}
        self.deadline_monotonic: float | None = None
        self._paused = False
        self._removed = False
        self._staged = False
        self._lease_path = output_dir / f'{container}.lease.json'
        self._lease: dict[str, Any] = {}
        self._lifecycle_lock = asyncio.Lock()

    async def initialize_lease(self) -> None:
        info = await self.inspect()
        self._lease = {
            'version': LEASE_VERSION,
            'source_root': str(self.assets.source_root),
            'docker_path': self.docker.prefix[0],
            'docker_host': self.docker.prefix[2],
            'container': self.container,
            'container_id': info['Id'],
            'image_id': self.assets.image_id,
            'cpuset': self.docker.cpuset,
            'cgroup_parent': self.docker.cgroup_parent,
            'deadline_monotonic': None,
            'deadline_unix_for_logging': None,
        }
        _write_lease(self._lease_path, self._lease, create=True)

    async def inspect(self) -> dict[str, Any]:
        if self._removed:
            raise RuntimeError('The offline workspace has been released')
        info = await _blocking(self.docker.inspect, self.container)
        self.docker.verify(info, self.assets.image_id)
        return info

    async def stage_public(self) -> dict[str, Any]:
        """Copy exactly the public tar through stdin; no host mount is created."""
        async with self._lifecycle_lock:
            self._check_active()
            if self._staged:
                raise RuntimeError('The public task package was already staged')
            self.assets.verify()
            data = self.assets.public_tar.read_bytes()
            await _blocking(
                self.docker.run,
                'exec',
                '-i',
                '--workdir',
                '/workspace',
                self.container,
                '/bin/tar',
                '--no-same-owner',
                '-xf',
                '-',
                input=data,
                timeout=90,
            )
            self._staged = True
            return {'public_tar': str(self.assets.public_tar), 'private_files_staged': False}

    def _check_active(self) -> None:
        if self._paused or self._removed:
            raise RuntimeError('The offline workspace is no longer open for development')
        if self.deadline_monotonic is not None and time.monotonic() >= self.deadline_monotonic:
            raise RuntimeError('The development deadline has expired')

    def set_deadline(self, *, deadline_monotonic: float | None = None, deadline_unix: float | None = None) -> float:
        if deadline_monotonic is None:
            if deadline_unix is None or not math.isfinite(deadline_unix):
                raise ValueError('A finite absolute development deadline is required')
            deadline_monotonic = time.monotonic() + deadline_unix - time.time()
        if not math.isfinite(deadline_monotonic) or deadline_monotonic <= time.monotonic():
            raise ValueError('The development deadline has expired or is invalid')
        if self.deadline_monotonic is not None and deadline_monotonic != self.deadline_monotonic:
            raise ValueError('An established development deadline cannot be changed')
        if self.deadline_monotonic is not None:
            return self.deadline_monotonic
        self.deadline_monotonic = deadline_monotonic
        self._lease['deadline_monotonic'] = deadline_monotonic
        self._lease['deadline_unix_for_logging'] = time.time() + deadline_monotonic - time.monotonic()
        _write_lease(self._lease_path, self._lease)
        return deadline_monotonic

    def mcp_config(
        self,
        *,
        deadline_monotonic: float | None = None,
        deadline_unix: float | None = None,
    ) -> dict[str, Any]:
        """One host-side MCP endpoint; candidate code never receives host access."""
        self._check_active()
        if deadline_monotonic is not None or deadline_unix is not None:
            self.set_deadline(deadline_monotonic=deadline_monotonic, deadline_unix=deadline_unix)
        if self.deadline_monotonic is None:
            raise ValueError('Set the shared development deadline before starting MCP')
        return {
            'name': 'workspace',
            'command': sys.executable,
            'args': ['-m', 'nexus.extensions.tasks.frontieram_offline.mcp', '--lease-file', str(self._lease_path)],
            'env': {'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONPATH': str(Path(__file__).resolve().parents[4])},
            'enabled_tools': ['exec'],
            'startup_timeout_sec': 30,
            'tool_timeout_sec': 130,
        }

    async def _invoke(self, request: InvokeRequest) -> Any:
        # InvokeRequest is constructed by trusted Nexus code; its pickle is never
        # accepted over an agent-facing network endpoint or from candidate bytes.
        if request.import_path != 'run_command' or request.venv_id is not None or request.execution_user is not None:
            raise ValueError('This offline Runtime exposes only the run_command executor')
        args, kwargs = request.unpack_payload()
        if len(args) != 1 or kwargs:
            raise ValueError('Unsupported run_command invocation shape')
        return await self._run_command(args[0])

    async def _run_command(self, value: Any) -> dict[str, Any]:
        from .mcp import DeadlineWorkspace

        self._check_active()
        if hasattr(value, 'model_dump'):
            value = value.model_dump()
        if not isinstance(value, dict) or set(value) - {
            'command',
            'cwd',
            'timeout',
            'stdin',
            'extra_env',
            'timeout_control_id',
        }:
            raise ValueError('Unsupported command parameters')
        if value.get('cwd') not in (None, '/workspace') or value.get('stdin') is not None:
            raise ValueError('Offline command cwd and stdin are fixed by the task contract')
        if value.get('extra_env') or value.get('timeout_control_id') is not None:
            raise ValueError('Offline command environment and deadline cannot be overridden')
        timeout = value.get('timeout', 60)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not float(timeout).is_integer():
            raise ValueError('Command timeout must be an integer in [1,120]')
        if self.deadline_monotonic is None:
            raise ValueError('Set the shared development deadline before running commands')
        workspace = DeadlineWorkspace.from_lease(self._lease_path)
        started = time.monotonic()
        workspace.start_watchdog()
        try:
            result = await _blocking(
                workspace.execute,
                {'command': value.get('command'), 'timeout_seconds': int(timeout), 'max_output_chars': 20000},
            )
        finally:
            await _blocking(workspace.stop_watchdog)
        status = (
            CommandRunStatus.TimeLimitExceeded
            if result['timed_out'] or result['exit_code'] == 124
            else CommandRunStatus.Finished
        )
        return RunCommandResponse(
            status=status,
            return_code=result['exit_code'],
            stdout=result['output'],
            stderr='',
            execution_time=time.monotonic() - started,
        ).model_dump()

    async def pause(self) -> dict[str, bool]:
        async with self._lifecycle_lock:
            if self._removed:
                raise RuntimeError('The offline workspace has already been released')
            await _blocking(self._pause_owned)
            self._paused = True
            return {'success': True, 'paused': True}

    def _pause_owned(self) -> None:
        try:
            self.docker.pause(self.container)
        except Exception:
            # The MCP watchdog can freeze between the controller's inspect and
            # pause calls. Accept only the verified postcondition on this lease.
            info = self.docker.inspect(self.container)
            self.docker.verify(info, self.assets.image_id)
            if (
                info.get('Id') != self._lease.get('container_id')
                or info.get('State', {}).get('Running') is not True
                or info.get('State', {}).get('Paused') is not True
            ):
                raise

    async def freeze(self, output_dir: str | Path | None = None, label: str = 'source') -> dict[str, Any]:
        """Pause before export; the reviewed exporter never resumes development."""
        if not re.fullmatch(r'[a-z][a-z0-9_-]{0,31}', label):
            raise ValueError('Invalid snapshot label')
        destination = Path(output_dir or self.output_dir).resolve()
        if destination == self.assets.source_root or self.assets.source_root in destination.parents:
            raise ValueError('Frozen source assets are read-only to this integration')
        destination.mkdir(parents=True, exist_ok=True)
        async with self._lifecycle_lock:
            if self._removed:
                raise RuntimeError('The offline workspace has already been released')
            await _blocking(self._pause_owned)
            self._paused = True
            return await _blocking(self._export, destination, label)

    def _export(self, destination: Path, label: str) -> dict[str, Any]:
        result = self.assets.controller.freeze(self.docker, self.container, destination, label)
        self._removed = True
        return result

    def _remove(self) -> None:
        self.docker.remove(self.container)
        self._removed = True

    async def close(self) -> None:
        async with self._lifecycle_lock:
            try:
                if not self._removed:
                    await _blocking(self._remove)
            finally:
                if self._removed:
                    self._lease_path.unlink(missing_ok=True)


@runtime_provider_registry.register('frontieram_offline')
class FrontierAMOfflineProvider(RuntimeProvider):
    """Nexus owns lifecycle; the locked primitives own strict Docker isolation."""

    def __init__(
        self,
        source_root: str | Path,
        docker_host: str,
        docker_path: str | Path | None = None,
        cpuset: str = '0',
        cgroup_parent: str | None = None,
        **config: Any,
    ) -> None:
        if config:
            raise ValueError('Unknown offline Runtime provider settings')
        super().__init__()
        if not docker_host.startswith('unix://'):
            raise ValueError('Only a local Docker daemon socket is permitted')
        self.assets = LockedAssets(source_root)
        binary = Path(docker_path or self.assets.source_root / 'runtime/bin/docker').resolve(strict=True)
        self.docker = self.assets.controller.Docker(binary, docker_host, cpuset, cgroup_parent)
        self._acquiring_names: set[str] = set()

    async def acquire(self, name: str, **params: Any) -> OfflineDockerRuntime:
        if name in self._active_runtimes or name in self._acquiring_names:
            raise ValueError('Offline workspaces cannot be reused across attempts')
        merged = self.get_params(name, **params)
        if set(merged) - {'output_dir', 'deadline_monotonic', 'deadline_unix', 'stage_public', 'run_id'}:
            raise ValueError('Unknown offline workspace settings')
        output_dir = Path(merged['output_dir']).expanduser().resolve()
        if output_dir == self.assets.source_root or self.assets.source_root in output_dir.parents:
            raise ValueError('Runtime evidence must stay outside the frozen source tree')
        output_dir.mkdir(parents=True, exist_ok=True)
        container = 'nexus-frontieram-' + uuid.uuid4().hex
        runtime: OfflineDockerRuntime | None = None
        self._acquiring_names.add(name)
        try:
            checks = await _blocking(self.docker.create, self.assets.image_id, container)
            runtime = OfflineDockerRuntime(
                assets=self.assets,
                docker=self.docker,
                container=container,
                output_dir=output_dir,
                configuration_checks=checks,
            )
            await runtime.initialize_lease()
            if merged.get('stage_public', True):
                await runtime.stage_public()
            if merged.get('deadline_monotonic') is not None or merged.get('deadline_unix') is not None:
                runtime.set_deadline(
                    deadline_monotonic=merged.get('deadline_monotonic'),
                    deadline_unix=merged.get('deadline_unix'),
                )
            self._active_runtimes[name] = runtime
            return runtime
        except BaseException:
            if runtime is not None:
                await runtime.close()
            else:
                # create() already cleans up after startup failure. This covers
                # cancellation after a successful create but before assignment.
                with contextlib.suppress(Exception):
                    await _blocking(self.docker.remove, container)
            raise
        finally:
            self._acquiring_names.discard(name)

    async def _release_impl(self, name: str) -> None:
        runtime = cast('OfflineDockerRuntime | None', self._active_runtimes.get(name))
        if runtime is not None:
            await runtime.close()
            self._active_runtimes.pop(name, None)

    async def _release_all_impl(self) -> None:
        errors = []
        for name in list(self._active_runtimes):
            try:
                await self._release_impl(name)
            except Exception as error:
                errors.append(error)
        if errors:
            raise ExceptionGroup('Failed to release offline workspaces', errors)

    def get_runtime_id(self, name: str) -> str | None:
        runtime = cast('OfflineDockerRuntime | None', self._active_runtimes.get(name))
        return runtime.container if runtime is not None else None
