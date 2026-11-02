"""Compose topology controlled through one trusted outer Nexus Runtime."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import random
import re
import shlex
import stat
import sys
import tarfile
import tempfile
import time

from collections.abc import Awaitable, Callable
from contextlib import suppress
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from nexus.runtime.executors.models.file import FileItem
from nexus.runtime.topology.base import ServiceHandle, TopologyHandle
from nexus.runtime.topology.models import (
    ComposeTopologySpec,
    ServiceStatus,
    TopologyServiceSpec,
    TopologyStatus,
)


logger = logging.getLogger(__name__)
_PROJECT_RE = re.compile(r'^[a-z0-9][a-z0-9_-]{0,62}$')
_ENV_NAME_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
_MIN_COMPOSE_VERSION = (2, 20, 0)
_MAX_TRANSFER_BYTES = 256 * 1024 * 1024
_AGENT_TRANSPORT_CONTAINER_DIR = '/opt/nexus-agent'
# Covers graceful down (shutdown_timeout_sec + 30), force-kill (60),
# force-down (120), directory cleanup (30), and scheduling slack.
_CLEANUP_ABANDON_SLACK_SEC = 300.0
_BEST_EFFORT_CLEANUP_TIMEOUT_SEC = 45.0
_TIMEOUT_ERRORS = (TimeoutError, asyncio.TimeoutError)
_SERVICE_RECEIVE_FILE_SCRIPT = (
    'target=$1; parent=${target%/*}; [ -n "$parent" ] || parent=/; '
    '[ -d "$parent" ] || { printf "%s\\n" "Target parent does not exist: $parent" >&2; exit 1; }; '
    'cat_bin=/nix/runtime/nix-env/bin/cat; '
    '[ -x "$cat_bin" ] || cat_bin="$(command -v cat)" || { '
    'printf "%s\\n" "Required tool not found: cat" >&2; exit 127; }; '
    '"$cat_bin" > "$target"'
)
_SERVICE_SEND_FILE_SCRIPT = (
    'source=$1; '
    '[ -f "$source" ] && [ ! -L "$source" ] || { '
    'printf "%s\\n" "Transfer source is not a regular file: $source" >&2; exit 1; }; '
    'cat_bin=/nix/runtime/nix-env/bin/cat; '
    '[ -x "$cat_bin" ] || cat_bin="$(command -v cat)" || { '
    'printf "%s\\n" "Required tool not found: cat" >&2; exit 127; }; '
    '"$cat_bin" "$source"'
)
_SERVICE_RECEIVE_DIR_SCRIPT = (
    'target=$1; '
    'mkdir_bin=/nix/runtime/nix-env/bin/mkdir; '
    '[ -x "$mkdir_bin" ] || mkdir_bin="$(command -v mkdir)" || { '
    'printf "%s\\n" "Required tool not found: mkdir" >&2; exit 127; }; '
    'tar_bin=/nix/runtime/nix-env/bin/tar; '
    '[ -x "$tar_bin" ] || tar_bin="$(command -v tar)" || { '
    'printf "%s\\n" "Required tool not found: tar" >&2; exit 127; }; '
    '"$mkdir_bin" -p "$target" && "$tar_bin" xzf - -C "$target"'
)
_SERVICE_SEND_DIR_SCRIPT = (
    'source=$1; '
    '[ -d "$source" ] && [ ! -L "$source" ] || { '
    'printf "%s\\n" "Transfer source is not a directory: $source" >&2; exit 1; }; '
    'find_bin=/nix/runtime/nix-env/bin/find; '
    '[ -x "$find_bin" ] || find_bin="$(command -v find)" || { '
    'printf "%s\\n" "Required tool not found: find" >&2; exit 127; }; '
    '[ -z "$("$find_bin" "$source" -type l -print -quit)" ] || { '
    'printf "%s\\n" "Transfer source contains symlinks: $source" >&2; exit 1; }; '
    'tar_bin=/nix/runtime/nix-env/bin/tar; '
    '[ -x "$tar_bin" ] || tar_bin="$(command -v tar)" || { '
    'printf "%s\\n" "Required tool not found: tar" >&2; exit 127; }; '
    '"$tar_bin" czf - -C "$source" .'
)


if TYPE_CHECKING:
    from nexus.runtime.executors.models.command import RunCommandResponse


def _positive_int_env(name: str, default: int = 0) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f'{name} must be an integer, got {raw!r}') from exc
    if value < 0:
        raise ValueError(f'{name} must be >= 0, got {value}')
    return value


def _float_env(name: str, default: float = 0.0) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f'{name} must be a number, got {raw!r}') from exc
    if value < 0:
        raise ValueError(f'{name} must be >= 0, got {value}')
    return value


def _consume_task_result(task: asyncio.Task[None]) -> None:
    with suppress(BaseException):
        task.result()


class _ComposeServiceStartSlots:
    """Bound simultaneous Compose startups across local batch worker processes."""

    def __init__(self, *, capacity: int, weight: int, gate_dir: Path) -> None:
        if capacity and weight > capacity:
            raise ValueError(
                'NEXUS_COMPOSE_START_MAX_INFLIGHT_SERVICES '
                f'({capacity}) must be at least the topology service count ({weight})'
            )
        self.capacity = capacity
        self.weight = max(weight, 1) if capacity else 0
        self.gate_dir = gate_dir
        self._handles: list[Any] = []

    async def __aenter__(self) -> _ComposeServiceStartSlots:
        if not self.capacity:
            return self
        self.gate_dir.mkdir(parents=True, exist_ok=True)
        while not self._try_acquire():
            await asyncio.sleep(0.25 + random.random() * 0.25)
        return self

    async def __aexit__(self, *_args: Any) -> None:
        self._release()

    def _try_acquire(self) -> bool:
        import fcntl

        coordinator = (self.gate_dir / 'coordinator.lock').open('a+b')
        acquired: list[Any] = []
        try:
            fcntl.flock(coordinator.fileno(), fcntl.LOCK_EX)
            for index in range(self.capacity):
                handle = (self.gate_dir / f'service-{index:04d}.lock').open('a+b')
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    handle.close()
                    continue
                acquired.append(handle)
                if len(acquired) == self.weight:
                    self._handles = acquired
                    return True
            return False
        finally:
            if not self._handles:
                for handle in acquired:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                    handle.close()
            fcntl.flock(coordinator.fileno(), fcntl.LOCK_UN)
            coordinator.close()

    def _release(self) -> None:
        import fcntl

        for handle in self._handles:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()
        self._handles.clear()


def _dependency_layers(service_specs: dict[str, TopologyServiceSpec]) -> list[list[str]]:
    remaining = {name: set(spec.depends_on) for name, spec in service_specs.items()}
    layers: list[list[str]] = []
    completed: set[str] = set()
    while remaining:
        layer = sorted(name for name, deps in remaining.items() if deps <= completed)
        if not layer:
            raise RuntimeError(f'Compose service dependency graph is cyclic: {remaining!r}')
        layers.append(layer)
        completed.update(layer)
        for name in layer:
            remaining.pop(name)
    return layers


def _is_start_container_hook_timeout(details: str) -> bool:
    normalized = details.lower()
    return 'startcontainer hook' in normalized and (
        'ran past specified timeout' in normalized or 'timed out' in normalized
    )


def _may_hide_start_container_hook_timeout(details: str) -> bool:
    normalized = details.lower()
    return 'dependency failed to start' in normalized and 'exited (255)' in normalized


class ComposeTopology(TopologyHandle):
    """Lifecycle owner for a Compose project inside a controller Runtime."""

    def __init__(
        self,
        *,
        runtime: Any,
        spec: ComposeTopologySpec,
        task_dir: Path,
        project_name: str,
        agent_helper_dir: Path | None = None,
        controller_root: str = '/nexus/compose',
        poll_interval_sec: float = 1.0,
    ) -> None:
        if not _PROJECT_RE.fullmatch(project_name):
            raise ValueError(f'Invalid Compose project name: {project_name!r}')
        self.runtime = runtime
        self.spec = spec
        self.task_dir = task_dir.resolve()
        self.project_name = project_name
        self.controller_root = controller_root.rstrip('/')
        self.controller_dir = f'{self.controller_root}/{project_name}'
        self.definition_path = f'{self.controller_dir}/{spec.definition}'
        transport_id = hashlib.sha256(project_name.encode('utf-8')).hexdigest()[:12]
        self.agent_transport_host_dir = f'/tmp/nexus-agent/{transport_id}'
        self.agent_transport_container_dir = _AGENT_TRANSPORT_CONTAINER_DIR
        self.agent_helper_dir = agent_helper_dir.resolve() if agent_helper_dir is not None else None
        if spec.agent_transport == 'socket-v1' and self.agent_helper_dir is None:
            raise ValueError('socket-v1 Agent transport requires agent_helper_dir')
        self.socket_bridge_host_path = f'{self.agent_transport_host_dir}/bin/nexus-socket-bridge'
        self.socket_bridge_container_path = f'{self.agent_transport_container_dir}/bin/nexus-socket-bridge'
        self.environment_override_path = (
            f'{self.controller_dir}/.nexus-runtime.override.json'
            if spec.primary_environment or spec.agent_transport != 'none'
            else None
        )
        self.poll_interval_sec = poll_interval_sec
        self._state: Literal['new', 'staged', 'started', 'ready', 'failed', 'deleted'] = 'new'
        self._startup_deadline: float | None = None
        self._delete_task: asyncio.Task[None] | None = None
        self._delete_deadline: float | None = None
        self._delete_timeout: float | None = None
        self._delete_abandoned = False
        self._service_specs = dict(spec.services)
        self._bind_paths = tuple(spec.bind_paths)
        self._services = {name: ComposeService(topology=self, service_name=name) for name in self._service_specs}

    @property
    def topology_id(self) -> str:
        return self.project_name

    @property
    def compose_argv(self) -> list[str]:
        argv = [
            'docker',
            'compose',
            '--project-name',
            self.project_name,
            '--file',
            self.definition_path,
        ]
        if self.environment_override_path is not None:
            argv.extend(['--file', self.environment_override_path])
        return argv

    @property
    def compose_files(self) -> tuple[str, ...]:
        files = [self.definition_path]
        if self.environment_override_path is not None:
            files.append(self.environment_override_path)
        return tuple(files)

    async def stage(self) -> None:
        if self._state != 'new':
            raise RuntimeError(f'Compose topology cannot stage from state {self._state!r}')
        version_result = await self._run_checked(
            ['docker', 'compose', 'version', '--short'],
            timeout=30,
            operation='Check Docker Compose version',
        )
        _require_compose_version(version_result.stdout or '')
        await self._run_checked(
            ['mkdir', '-p', self.controller_root],
            timeout=30,
            operation='Create Compose controller root',
        )
        await self._run_checked(
            ['mkdir', self.controller_dir],
            timeout=30,
            operation='Claim unique Compose controller directory',
        )

        archive_path: Path | None = None
        remote_archive = f'/tmp/.nexus_compose_stage_{uuid4().hex}.tar.gz'
        try:
            with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as archive_file:
                archive_path = Path(archive_file.name)
            with tarfile.open(archive_path, 'w:gz') as archive:
                for path in _minimal_stage_paths(
                    self.task_dir,
                    [self.spec.definition, *self._bind_paths],
                ):
                    archive.add(path, arcname=path.relative_to(self.task_dir).as_posix(), recursive=True)
            _require_transfer_size(archive_path)
            await _upload_outer_file(self.runtime, archive_path, remote_archive)
            await self._run_checked(
                [
                    'tar',
                    'xzf',
                    remote_archive,
                    '-C',
                    self.controller_dir,
                ],
                timeout=120,
                operation='Extract Compose task bundle',
            )
            # Use the definition staged from the live task directory. Some datasets
            # intentionally update that file after the task plan was compiled, so
            # the recorded digest is metadata rather than a runtime admission check.
            if self.environment_override_path is not None:
                primary_override: dict[str, Any] = {'environment': self.spec.primary_environment}
                if self.spec.agent_transport == 'socket-v1':
                    await self._run_checked(
                        ['mkdir', '-p', '/tmp/nexus-agent'],
                        timeout=30,
                        operation='Create Compose Agent transport root',
                    )
                    await self._run_checked(
                        ['mkdir', '-m', '0777', self.agent_transport_host_dir],
                        timeout=30,
                        operation='Create Compose Agent transport directory',
                    )
                    assert self.agent_helper_dir is not None
                    await self._run_checked(
                        ['mkdir', '-p', f'{self.agent_transport_host_dir}/bin'],
                        timeout=30,
                        operation='Create Compose Agent helper directory',
                    )
                    source = self.agent_helper_dir / 'nexus-socket-bridge'
                    if not source.is_file():
                        raise FileNotFoundError(f'Compose Agent helper not found: {source}')
                    await _upload_outer_file(self.runtime, source, self.socket_bridge_host_path)
                    await self._run_checked(
                        ['chmod', '0755', self.socket_bridge_host_path],
                        timeout=30,
                        operation='Protect Compose Agent helpers',
                    )
                    primary_override['volumes'] = [
                        {
                            'type': 'bind',
                            'source': '/nix',
                            'target': '/nix',
                            'read_only': True,
                            'bind': {'create_host_path': False},
                        },
                        {
                            'type': 'bind',
                            'source': self.agent_transport_host_dir,
                            'target': self.agent_transport_container_dir,
                            'bind': {'create_host_path': False},
                        },
                    ]
                with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False) as override_file:
                    json.dump(
                        {
                            'services': {
                                self.spec.primary_service: primary_override,
                            }
                        },
                        override_file,
                    )
                    override_path = Path(override_file.name)
                try:
                    await _upload_outer_file(self.runtime, override_path, self.environment_override_path)
                    await self._run_checked(
                        ['chmod', '600', self.environment_override_path],
                        timeout=30,
                        operation='Protect Compose primary environment override',
                    )
                finally:
                    override_path.unlink(missing_ok=True)
            self._state = 'staged'
        except BaseException:
            await _shielded_best_effort_cleanup(self.runtime, self.controller_dir)
            if self.spec.agent_transport == 'socket-v1':
                await _shielded_best_effort_cleanup(self.runtime, self.agent_transport_host_dir)
            self._state = 'new'
            raise
        finally:
            if archive_path is not None:
                archive_path.unlink(missing_ok=True)
            await _shielded_best_effort_cleanup(self.runtime, remote_archive, file_only=True)

    async def start(self) -> None:
        if self._state != 'staged':
            raise RuntimeError(f'Compose topology cannot start from state {self._state!r}')
        try:
            batch_size = _positive_int_env('NEXUS_COMPOSE_START_BATCH_SIZE')
            max_inflight_services = _positive_int_env('NEXUS_COMPOSE_START_MAX_INFLIGHT_SERVICES')
            retry_attempts = max(
                1,
                _positive_int_env('NEXUS_COMPOSE_START_RETRY_ATTEMPTS', self.spec.start_retry_attempts),
            )
            retry_backoff = _float_env('NEXUS_COMPOSE_START_RETRY_BACKOFF_SEC', self.spec.start_retry_backoff_sec)
            retry_jitter_default = (
                retry_backoff if self.spec.start_retry_jitter_max is None else self.spec.start_retry_jitter_max
            )
            retry_jitter_max = _float_env('NEXUS_COMPOSE_START_RETRY_JITTER_MAX', retry_jitter_default)
            jitter_max = _float_env('NEXUS_COMPOSE_STARTUP_JITTER_MAX', self.spec.startup_jitter_max)
            gate_dir = Path(os.getenv('NEXUS_COMPOSE_START_GATE_DIR', '/tmp/nexus-compose-start-gate'))
            # Match Gongfeng's startup_jitter_max semantics: de-synchronize the
            # start request before admission and before the per-topology startup
            # budget begins. The weighted gate remains an optional safety valve.
            if jitter_max:
                jitter = random.uniform(0, jitter_max)
                logger.debug(
                    'Compose startup jitter: sleeping %.1fs before topology start (max=%.1fs)',
                    jitter,
                    jitter_max,
                )
                await asyncio.sleep(jitter)
            async with _ComposeServiceStartSlots(
                capacity=max_inflight_services,
                weight=len(self._service_specs),
                gate_dir=gate_dir,
            ):
                # Queueing for an optional global startup slot is admission
                # control, not part of this topology's own startup budget.
                self._startup_deadline = time.monotonic() + self.spec.startup_timeout_sec
                if batch_size <= 0:
                    batches = [list(self._service_specs)]
                else:
                    batches = [
                        layer[offset : offset + batch_size]
                        for layer in _dependency_layers(self._service_specs)
                        for offset in range(0, len(layer), batch_size)
                    ]
                for batch in batches:
                    argv = [*self.compose_argv, 'up', '--detach', '--remove-orphans']
                    if batch_size > 0:
                        argv.extend(['--no-deps', *batch])
                    for attempt in range(1, retry_attempts + 1):
                        # Docker accepts Unix timestamps for --since. Subtract one
                        # second to avoid dropping logs emitted in the same second.
                        attempt_started_at = max(0, int(time.time()) - 1)
                        try:
                            await self._run_checked(
                                argv,
                                timeout=self._remaining_startup(),
                                operation=f'Start Compose service batch {batch!r}',
                            )
                            break
                        except RuntimeError as exc:
                            details = str(exc)
                            if (
                                attempt < retry_attempts
                                and not _is_start_container_hook_timeout(details)
                                and _may_hide_start_container_hook_timeout(details)
                            ):
                                # Compose often reports only exit 255 here, while the gVisor
                                # startContainer hook timeout is present in service logs. Restrict
                                # diagnostics to this attempt so an old hook failure cannot turn a
                                # later application error into a transient infrastructure retry.
                                diagnostics = await self._collect_compose_logs(
                                    tail=80,
                                    timeout=30.0,
                                    since=attempt_started_at,
                                )
                                details = f'{details}\n{diagnostics}'
                            transient = _is_start_container_hook_timeout(details)
                            if not transient or attempt >= retry_attempts:
                                raise
                            delay = retry_backoff * attempt + random.uniform(0, retry_jitter_max)
                            logger.warning(
                                'Transient inner-container start failure for %s (attempt %s/%s); retrying in %.1fs',
                                batch,
                                attempt,
                                retry_attempts,
                                delay,
                            )
                            await asyncio.sleep(min(delay, self._remaining_startup()))
                    if batch_size > 0:
                        await self._wait_services_ready(batch)
            self._state = 'started'
        except asyncio.CancelledError:
            self._state = 'failed'
            raise
        except Exception as exc:
            self._state = 'failed'
            diagnostics = await self._collect_compose_logs(tail=200, timeout=60.0, include_error=True)
            if diagnostics:
                raise RuntimeError(f'{exc}\nCompose service logs:\n{diagnostics}') from exc
            raise

    async def _collect_compose_logs(
        self,
        *,
        tail: int,
        timeout: float,
        include_error: bool = False,
        since: int | None = None,
    ) -> str:
        try:
            argv = [*self.compose_argv, 'logs', '--no-color', '--tail', str(tail)]
            if since is not None:
                argv.extend(['--since', str(since)])
            result = await self.runtime.run_command(
                command=shlex.join(argv),
                timeout=timeout,
            )
            return '\n'.join(part.strip() for part in (result.stdout, result.stderr) if part and part.strip())
        except Exception as exc:
            if include_error:
                return f'<failed to collect Compose logs: {exc}>'
            logger.debug('Failed to collect Compose startup diagnostics: %s', exc)
            return ''

    async def _wait_services_ready(self, service_names: list[str]) -> None:
        subset = {name: self._service_specs[name] for name in service_names}
        while True:
            remaining = self._remaining_startup()
            status = await self._inspect_remote(timeout=min(30.0, remaining))
            ready, failure = _readiness(subset, status.services)
            if failure is not None:
                raise RuntimeError(f'Compose service batch failed readiness: {failure}')
            if ready:
                return
            await asyncio.sleep(self.poll_interval_sec)

    async def wait_ready(self) -> TopologyStatus:
        if self._state not in {'started', 'ready'} or self._startup_deadline is None:
            raise RuntimeError(f'Compose topology cannot wait from state {self._state!r}')

        last_status: TopologyStatus | None = None
        while True:
            remaining = self._remaining_startup()
            last_status = await self._inspect_remote(timeout=min(30.0, remaining))
            ready, failure = _readiness(self._service_specs, last_status.services)
            if failure is not None:
                self._state = 'failed'
                raise RuntimeError(f'Compose topology failed readiness: {failure}')
            if ready:
                self._state = 'ready'
                return TopologyStatus(
                    topology_id=self.topology_id,
                    state='ready',
                    services=last_status.services,
                )
            if time.monotonic() >= self._startup_deadline:
                self._state = 'failed'
                raise TimeoutError(
                    f'Compose topology did not become ready within {self.spec.startup_timeout_sec}s: '
                    f'{last_status.model_dump_json()}'
                )
            await asyncio.sleep(self.poll_interval_sec)

    async def inspect(self) -> TopologyStatus:
        if self._state in {'new', 'staged', 'deleted'}:
            return TopologyStatus(
                topology_id=self.topology_id,
                state=self._state,
                services={},
            )
        return await self._inspect_remote(timeout=30.0)

    async def _inspect_remote(self, *, timeout: float) -> TopologyStatus:
        result = await self._run_checked(
            [*self.compose_argv, 'ps', '--all', '--format', 'json'],
            timeout=timeout,
            operation='Inspect Compose topology',
        )
        services = _parse_compose_ps(result.stdout or '')
        ready, failure = _readiness(self._service_specs, services)
        if failure is not None:
            observed_state = 'failed'
        elif ready:
            observed_state = 'ready'
        else:
            observed_state = 'started'
        self._state = observed_state
        return TopologyStatus(
            topology_id=self.topology_id,
            state=observed_state,
            services=services,
        )

    def _remaining_startup(self) -> float:
        if self._startup_deadline is None:
            raise RuntimeError('Compose topology startup deadline is not initialized')
        remaining = self._startup_deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f'Compose topology exceeded its {self.spec.startup_timeout_sec}s startup budget')
        return remaining

    def service(self, name: str) -> ServiceHandle:
        try:
            return self._services[name]
        except KeyError as exc:
            raise KeyError(f'Unknown Compose service {name!r}; expected one of {sorted(self._services)}') from exc

    def require_service_operations(self) -> None:
        if self._state not in {'started', 'ready'}:
            raise RuntimeError(f'Compose service operations require a started topology, got {self._state!r}')

    async def delete(self) -> None:
        if self._state == 'deleted' or self._delete_abandoned:
            return
        loop = asyncio.get_running_loop()
        delete_task = self._delete_task
        if delete_task is None or delete_task.done():
            timeout = self._cleanup_abandon_timeout()
            delete_task = asyncio.create_task(self._delete_impl())
            self._delete_task = delete_task
            self._delete_timeout = timeout
            self._delete_deadline = loop.time() + timeout
        else:
            assert self._delete_timeout is not None
            assert self._delete_deadline is not None
            timeout = self._delete_timeout
        deadline = self._delete_deadline
        assert deadline is not None
        # If delete() is awaited from a finally/except block while unwinding a
        # cancellation, its first await will not receive that same cancellation
        # again. Remember the active exception so a cleanup timeout cannot replace it.
        preserve_pending_cancellation = isinstance(sys.exc_info()[1], asyncio.CancelledError)

        async def wait_until_deadline() -> None:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError
            await asyncio.wait_for(asyncio.shield(delete_task), timeout=remaining)

        def abandon_delete() -> None:
            self._delete_abandoned = True
            self._state = 'failed'
            delete_task.cancel()
            delete_task.add_done_callback(_consume_task_result)

        try:
            await wait_until_deadline()
        except _TIMEOUT_ERRORS:
            # A TimeoutError raised by _delete_impl itself is a real cleanup
            # failure, not expiration of this outer connection-safe budget.
            if delete_task.done():
                raise
            abandon_delete()
            logger.warning(
                'Compose deletion for %s did not finish within %.1fs; abandoning it',
                self.project_name,
                timeout,
            )
            if preserve_pending_cancellation:
                # delete() may have been entered from a finally block after the
                # caller's cancellation was already delivered. Let that original
                # CancelledError resume instead of replacing it with cleanup timeout.
                return
            raise
        except asyncio.CancelledError:
            try:
                # Every step inside has its own budget, but those bound command execution,
                # not connecting to the runtime hosting it. A sandbox that stops accepting
                # connections leaves this waiting with no upper bound: measured once in 907,
                # holding a timed-out task open for a further 6h43m.
                await wait_until_deadline()
            except _TIMEOUT_ERRORS:
                if not delete_task.done():
                    abandon_delete()
                    logger.warning(
                        'Compose deletion for %s did not finish within %.1fs after cancellation; abandoning it',
                        self.project_name,
                        timeout,
                    )
            except asyncio.CancelledError:
                # A repeated cancellation is an explicit request to stop waiting.
                abandon_delete()
            finally:
                raise

    def _cleanup_abandon_timeout(self) -> float:
        """Total deletion budget, including runtime connection time and command execution."""
        return float(self.spec.shutdown_timeout_sec) + _CLEANUP_ABANDON_SLACK_SEC

    async def _delete_impl(self) -> None:
        if self._state != 'new':
            try:
                down_argv = [
                    *self.compose_argv,
                    'down',
                    '--timeout',
                    str(self.spec.shutdown_timeout_sec),
                    '--volumes',
                    '--remove-orphans',
                ]
                result = await self.runtime.run_command(
                    command=shlex.join(down_argv),
                    timeout=float(self.spec.shutdown_timeout_sec + 30),
                )
                if result.return_code is None:
                    # Some service processes do not honor TERM before the outer
                    # command budget expires. Force-kill the project, then ask
                    # Compose to remove containers/networks/volumes without an
                    # additional grace period. A timed-out graceful shutdown must
                    # not invalidate an otherwise completed evaluation.
                    logger.warning(
                        'Graceful Compose deletion timed out for %s; forcing project cleanup',
                        self.project_name,
                    )
                    kill_result = await self.runtime.run_command(
                        command=shlex.join([*self.compose_argv, 'kill']),
                        timeout=60,
                    )
                    if kill_result.return_code not in (0, None):
                        details = kill_result.stderr or kill_result.stdout or 'no output'
                        if 'is not running' in details and not _contains_non_idempotent_kill_error(details):
                            logger.debug(
                                'Compose force-kill for %s found containers already stopped; continuing cleanup',
                                self.project_name,
                            )
                        else:
                            raise RuntimeError(
                                f'Force-kill Compose topology failed with code {kill_result.return_code}: {details}'
                            )
                    await self._run_checked(
                        [
                            *self.compose_argv,
                            'down',
                            '--timeout',
                            '0',
                            '--volumes',
                            '--remove-orphans',
                        ],
                        timeout=120,
                        operation='Force-delete Compose topology',
                    )
                elif result.return_code != 0:
                    details = result.stderr or result.stdout or 'no output'
                    raise RuntimeError(f'Delete Compose topology failed with code {result.return_code}: {details}')
            except BaseException:
                self._state = 'failed'
                raise
        try:
            cleanup_paths = [self.controller_dir]
            if self.spec.agent_transport == 'socket-v1':
                cleanup_paths.append(self.agent_transport_host_dir)
            await self._run_checked(
                ['rm', '-rf', *cleanup_paths],
                timeout=30,
                operation='Clean Compose controller directories',
            )
        except BaseException:
            self._state = 'failed'
            raise
        self._state = 'deleted'

    async def _run_checked(
        self,
        argv: list[str],
        *,
        timeout: float,
        operation: str,
        stdin: str | None = None,
    ) -> RunCommandResponse:
        result = await self.runtime.run_command(
            command=shlex.join(argv),
            timeout=timeout,
            stdin=stdin,
        )
        if result.return_code != 0:
            details = result.stderr or result.stdout or 'no output'
            raise RuntimeError(f'{operation} failed with code {result.return_code}: {details}')
        return result


class ComposeService(ServiceHandle):
    """Service exec/copy operations routed through the outer controller."""

    def __init__(self, *, topology: ComposeTopology, service_name: str) -> None:
        self.topology = topology
        self.service_name = service_name

    @property
    def name(self) -> str:
        return self.service_name

    async def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | None = None,
        timeout: float = 180,
        stdin: str | None = None,
    ) -> RunCommandResponse:
        self.topology.require_service_operations()
        argv = [*self.topology.compose_argv, 'exec', '-T']
        if cwd is not None:
            _require_absolute_container_path(cwd, 'cwd')
            argv.extend(['--workdir', cwd])
        if user is not None:
            argv.extend(['--user', user])
        for key in env or {}:
            if not _ENV_NAME_RE.fullmatch(key):
                raise ValueError(f'Invalid service environment variable name: {key!r}')
            argv.extend(['--env', key])
        argv.extend([self.service_name, 'sh', '-c', command])
        return await self.topology.runtime.run_command(
            command=shlex.join(argv),
            timeout=timeout,
            stdin=stdin,
            extra_env=env or {},
        )

    async def stop(self, *, timeout: int = 30) -> None:
        self.topology.require_service_operations()
        await self.topology._run_checked(
            [
                *self.topology.compose_argv,
                'stop',
                '--timeout',
                str(timeout),
                self.service_name,
            ],
            timeout=float(timeout + 30),
            operation=f'Stop Compose service {self.service_name!r}',
        )

    async def _exec_transfer(
        self,
        script: str,
        path: str,
        *,
        operation: str,
        input_path: str | None = None,
        output_path: str | None = None,
    ) -> None:
        if input_path is not None and output_path is not None:
            raise ValueError('Compose exec transfer cannot use input_path and output_path together')
        if output_path is not None:
            # A failed docker cp may leave a partial file or symlink behind.
            # Remove it before opening the controller-side redirection target.
            await self.topology._run_checked(
                ['rm', '-rf', '--', output_path],
                timeout=30,
                operation='Prepare Compose exec transfer output',
            )
        argv = [
            *self.topology.compose_argv,
            'exec',
            '-T',
            '--user',
            'root',
            self.service_name,
            'sh',
            '-c',
            script,
            'sh',
            path,
        ]
        command = shlex.join(argv)
        if input_path is not None:
            command = f'{command} < {shlex.quote(input_path)}'
        if output_path is not None:
            command = f'{command} > {shlex.quote(output_path)}'
        result = await self.topology.runtime.run_command(command=command, timeout=120)
        if result.return_code != 0:
            details = result.stderr or result.stdout or 'no output'
            raise RuntimeError(f'{operation} failed with code {result.return_code}: {details}')

    async def _copy_with_exec_fallback(
        self,
        cp_argv: list[str],
        *,
        operation: str,
        fallback: Callable[[], Awaitable[None]],
    ) -> bool:
        result = await self.topology.runtime.run_command(
            command=shlex.join(cp_argv),
            timeout=120,
        )
        if result.return_code == 0:
            return False
        cp_details = result.stderr or result.stdout or 'no output'
        logger.info(
            '%s via docker compose cp failed for %s; trying in-service transfer: %s',
            operation,
            self.service_name,
            cp_details,
        )
        try:
            await fallback()
        except Exception as exc:
            raise RuntimeError(
                f'{operation} via docker compose cp failed with code {result.return_code}: {cp_details}; '
                f'in-service fallback also failed: {exc}'
            ) from exc
        return True

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        self.topology.require_service_operations()
        source = Path(source_path).resolve()
        if not source.is_file():
            raise FileNotFoundError(f'Service upload source is not a file: {source}')
        _validate_local_transfer_source(source)
        _require_transfer_size(source)
        _require_absolute_container_path(target_path, 'target_path')
        transfer_dir = f'{self.topology.controller_dir}/transfers/{uuid4().hex}'
        remote_source = f'{transfer_dir}/{source.name}'
        try:
            await self.topology._run_checked(
                ['mkdir', '-p', transfer_dir],
                timeout=30,
                operation='Create Compose transfer directory',
            )
            await _upload_outer_file(self.topology.runtime, source, remote_source)
            await self._copy_with_exec_fallback(
                [*self.topology.compose_argv, 'cp', remote_source, f'{self.service_name}:{target_path}'],
                operation=f'Upload file to Compose service {self.service_name!r}',
                fallback=lambda: self._exec_transfer(
                    _SERVICE_RECEIVE_FILE_SCRIPT,
                    target_path,
                    input_path=remote_source,
                    operation=f'Upload file in Compose service {self.service_name!r}',
                ),
            )
        finally:
            await _shielded_best_effort_cleanup(self.topology.runtime, transfer_dir)

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        self.topology.require_service_operations()
        source = Path(source_dir).resolve()
        if not source.is_dir():
            raise NotADirectoryError(f'Service upload source is not a directory: {source}')
        _validate_local_transfer_source(source)
        _require_absolute_container_path(target_dir, 'target_dir')
        transfer_dir = f'{self.topology.controller_dir}/transfers/{uuid4().hex}'
        remote_archive = f'{transfer_dir}/payload.tar.gz'
        with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as archive_file:
            archive_path = Path(archive_file.name)
        try:
            with tarfile.open(archive_path, 'w:gz') as archive:
                # Archive the contents, not the source root itself. The exec
                # fallback extracts directly into an existing mount such as
                # /tests; a "." member would overwrite that mount's mode,
                # owner, and timestamps with Host directory metadata.
                for child in sorted(source.iterdir(), key=lambda path: path.name):
                    archive.add(child, arcname=child.name, recursive=True)
            _require_transfer_size(archive_path)
            await self.topology._run_checked(
                ['mkdir', '-p', transfer_dir],
                timeout=30,
                operation='Create Compose transfer directory',
            )
            await _upload_outer_file(self.topology.runtime, archive_path, remote_archive)
            await self.topology._run_checked(
                ['mkdir', '-p', f'{transfer_dir}/payload'],
                timeout=30,
                operation='Create Compose transfer payload directory',
            )
            await self.topology._run_checked(
                ['tar', 'xzf', remote_archive, '-C', f'{transfer_dir}/payload'],
                timeout=120,
                operation='Extract Compose upload payload',
            )
            await self._copy_with_exec_fallback(
                [
                    *self.topology.compose_argv,
                    'cp',
                    f'{transfer_dir}/payload/.',
                    f'{self.service_name}:{target_dir}',
                ],
                operation=f'Upload directory to Compose service {self.service_name!r}',
                fallback=lambda: self._exec_transfer(
                    _SERVICE_RECEIVE_DIR_SCRIPT,
                    target_dir,
                    input_path=remote_archive,
                    operation=f'Upload directory in Compose service {self.service_name!r}',
                ),
            )
        finally:
            archive_path.unlink(missing_ok=True)
            await _shielded_best_effort_cleanup(self.topology.runtime, transfer_dir)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        self.topology.require_service_operations()
        _require_absolute_container_path(source_path, 'source_path')
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        transfer_dir = f'{self.topology.controller_dir}/transfers/{uuid4().hex}'
        remote_target = f'{transfer_dir}/payload'
        try:
            await self.topology._run_checked(
                ['mkdir', '-p', transfer_dir],
                timeout=30,
                operation='Create Compose transfer directory',
            )
            await self._copy_with_exec_fallback(
                [*self.topology.compose_argv, 'cp', f'{self.service_name}:{source_path}', remote_target],
                operation=f'Download file from Compose service {self.service_name!r}',
                fallback=lambda: self._exec_transfer(
                    _SERVICE_SEND_FILE_SCRIPT,
                    source_path,
                    output_path=remote_target,
                    operation=f'Download file in Compose service {self.service_name!r}',
                ),
            )
            await _reject_remote_symlinks(self.topology, remote_target, directory=False)
            await _download_outer_file(self.topology.runtime, remote_target, target)
        finally:
            await _shielded_best_effort_cleanup(self.topology.runtime, transfer_dir)

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        self.topology.require_service_operations()
        _require_absolute_container_path(source_dir, 'source_dir')
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        transfer_dir = f'{self.topology.controller_dir}/transfers/{uuid4().hex}'
        remote_payload = f'{transfer_dir}/payload'
        remote_archive = f'{transfer_dir}/payload.tar.gz'
        try:
            await self.topology._run_checked(
                ['mkdir', '-p', transfer_dir],
                timeout=30,
                operation='Create Compose transfer directory',
            )
            used_fallback = await self._copy_with_exec_fallback(
                [*self.topology.compose_argv, 'cp', f'{self.service_name}:{source_dir}/.', remote_payload],
                operation=f'Download directory from Compose service {self.service_name!r}',
                fallback=lambda: self._exec_transfer(
                    _SERVICE_SEND_DIR_SCRIPT,
                    source_dir,
                    output_path=remote_archive,
                    operation=f'Download directory in Compose service {self.service_name!r}',
                ),
            )
            if not used_fallback:
                await _reject_remote_symlinks(self.topology, remote_payload, directory=True)
                await self.topology._run_checked(
                    ['tar', 'czf', remote_archive, '-C', remote_payload, '.'],
                    timeout=120,
                    operation='Archive Compose download payload',
                )
            await _download_outer_archive(self.topology.runtime, remote_archive, target)
        finally:
            await _shielded_best_effort_cleanup(self.topology.runtime, transfer_dir)


def _contains_non_idempotent_kill_error(output: str) -> bool:
    # Compose prints progress lines such as ``Container foo Error while Killing``
    # before Docker's actual error response. Treat those progress lines as
    # descriptive only; classify the daemon/client error lines themselves.
    error_lines = [
        line.strip()
        for line in output.splitlines()
        if line.strip().lower().startswith(('error response from daemon:', 'error:'))
    ]
    if not error_lines:
        return True
    return any('is not running' not in line.lower() for line in error_lines)


def _parse_compose_ps(output: str) -> dict[str, ServiceStatus]:
    """Parse Docker Compose JSON array or newline-delimited status output."""
    stripped = output.strip()
    if not stripped:
        return {}
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        records = []
        for line in stripped.splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f'Invalid docker compose ps JSON: {line!r}') from exc
            records.append(item)
    else:
        records = payload if isinstance(payload, list) else [payload]

    statuses: dict[str, ServiceStatus] = {}
    for record in records:
        if not isinstance(record, dict):
            raise RuntimeError(f'Invalid docker compose ps record: {record!r}')
        normalized = {str(key).lower(): value for key, value in record.items()}
        service = normalized.get('service')
        if not isinstance(service, str) or not service:
            raise RuntimeError(f'docker compose ps record has no service: {record!r}')
        state = str(normalized.get('state') or normalized.get('status') or 'unknown').strip().lower()
        health = normalized.get('health')
        health_value = str(health).strip().lower() if health not in (None, '') else None
        exit_code = normalized.get('exitcode', normalized.get('exit_code'))
        try:
            parsed_exit_code = int(exit_code) if exit_code not in (None, '') else None
        except (TypeError, ValueError):
            parsed_exit_code = None
        statuses[service] = ServiceStatus(
            service=service,
            container_id=str(normalized.get('id') or normalized.get('containerid') or '') or None,
            state=state,
            health=health_value,
            exit_code=parsed_exit_code,
        )
    return statuses


def _readiness(
    service_specs: dict[str, TopologyServiceSpec],
    statuses: dict[str, ServiceStatus],
) -> tuple[bool, str | None]:
    for service_name, service_spec in service_specs.items():
        status = statuses.get(service_name)
        if status is None:
            return False, None
        if service_spec.one_shot:
            if status.state in {'exited', 'stopped', 'dead'}:
                if status.exit_code == 0:
                    continue
                return False, f'one-shot service {service_name!r} exited with code {status.exit_code}'
            if status.state in {'running', 'created', 'restarting'}:
                return False, None
            return False, f'one-shot service {service_name!r} entered unexpected state {status.state!r}'

        if status.state in {'exited', 'stopped', 'dead'}:
            return False, f'service {service_name!r} exited with code {status.exit_code}'
        if status.state != 'running':
            return False, None
        if service_spec.has_healthcheck:
            if status.health == 'unhealthy':
                return False, f'service {service_name!r} is unhealthy'
            if status.health != 'healthy':
                return False, None
    return True, None


def _require_compose_version(output: str) -> tuple[int, int, int]:
    match = re.search(r'v?(\d+)\.(\d+)(?:\.(\d+))?', output)
    if match is None:
        raise RuntimeError(f'Could not parse Docker Compose version: {output!r}')
    version = (int(match.group(1)), int(match.group(2)), int(match.group(3) or 0))
    if version < _MIN_COMPOSE_VERSION:
        required = '.'.join(str(part) for part in _MIN_COMPOSE_VERSION)
        actual = '.'.join(str(part) for part in version)
        raise RuntimeError(f'Docker Compose {required}+ is required, got {actual}')
    return version


def _minimal_stage_paths(task_dir: Path, relative_paths: list[str]) -> list[Path]:
    root = task_dir.resolve()
    validated: set[Path] = set()
    for relative in relative_paths:
        posix_path = PurePosixPath(relative)
        if posix_path.is_absolute() or '..' in posix_path.parts or '\\' in relative:
            raise ValueError(f'Compose stage path must be task-relative and safe: {relative!r}')
        candidate = root.joinpath(*posix_path.parts)
        _validate_task_local_path(root, candidate)
        validated.add(candidate)
    candidates = sorted(
        validated,
        key=lambda path: (len(path.relative_to(root).parts), path.relative_to(root).as_posix()),
    )
    selected: list[Path] = []
    for candidate in candidates:
        if not candidate.exists():
            raise FileNotFoundError(f'Compose stage input not found: {candidate}')
        if any(candidate == parent or candidate.is_relative_to(parent) for parent in selected):
            continue
        selected.append(candidate)
    return selected


def _validate_task_local_path(root: Path, path: Path) -> None:
    current = root
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f'Compose stage path escapes task directory: {path}') from exc
    for part in relative.parts:
        current /= part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError as exc:
            raise FileNotFoundError(f'Compose stage input not found: {current}') from exc
        if stat.S_ISLNK(mode):
            raise ValueError(f'Compose stage input must not contain symlinks: {current}')
        if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise ValueError(f'Compose stage input must contain only regular files and directories: {current}')
    if path.is_dir():
        for child in path.rglob('*'):
            mode = os.lstat(child).st_mode
            if stat.S_ISLNK(mode):
                raise ValueError(f'Compose stage input tree must not contain symlinks: {child}')
            if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                raise ValueError(f'Compose stage input tree must contain only regular files and directories: {child}')


def _validate_local_transfer_source(path: Path) -> None:
    root = path.parent.resolve()
    _validate_task_local_path(root, path)


def _require_transfer_size(path: Path) -> None:
    size = path.stat().st_size
    if size > _MAX_TRANSFER_BYTES:
        raise ValueError(f'Compose transfer exceeds {_MAX_TRANSFER_BYTES} bytes: {path} ({size} bytes)')


async def _upload_outer_file(runtime: Any, source: Path, target: str) -> None:
    _validate_local_transfer_source(source)
    _require_transfer_size(source)
    response = await runtime.upload_file(
        files=[
            FileItem(
                path=target,
                content=base64.b64encode(source.read_bytes()).decode('ascii'),
                encoding='base64',
            )
        ],
        overwrite=True,
    )
    failures = [item for item in response.results if not item.success]
    if failures:
        details = '; '.join(f'{item.path}: {item.message or "upload failed"}' for item in failures)
        raise RuntimeError(f'Failed to upload Compose controller file: {details}')


async def _download_outer_file(runtime: Any, source: str, target: Path) -> None:
    response = await runtime.download_file(files=[FileItem(path=source, encoding='base64')])
    if response.errors:
        details = '; '.join(f'{item.path}: {item.message or "download failed"}' for item in response.errors)
        raise RuntimeError(f'Failed to download Compose controller file: {details}')
    if not response.files or response.files[0].content is None:
        raise RuntimeError(f'Compose controller returned no file content for {source}')
    content = base64.b64decode(response.files[0].content)
    if len(content) > _MAX_TRANSFER_BYTES:
        raise ValueError(f'Compose download exceeds {_MAX_TRANSFER_BYTES} bytes: {source}')
    target.write_bytes(content)


async def _download_outer_archive(runtime: Any, source: str, target: Path) -> None:
    with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as archive_file:
        archive_path = Path(archive_file.name)
    try:
        await _download_outer_file(runtime, source, archive_path)
        with tarfile.open(archive_path, 'r:gz') as archive:
            members = archive.getmembers()
            total_size = 0
            for member in members:
                if member.issym() or member.islnk():
                    raise ValueError(f'Compose download archive contains a link: {member.name}')
                if not (member.isfile() or member.isdir()):
                    raise ValueError(f'Compose download archive contains a special file: {member.name}')
                total_size += member.size
                if total_size > _MAX_TRANSFER_BYTES:
                    raise ValueError(
                        f'Compose download archive exceeds {_MAX_TRANSFER_BYTES} extracted bytes: {source}'
                    )
            archive.extractall(path=target, members=members, filter='data')
    finally:
        archive_path.unlink(missing_ok=True)


async def _reject_remote_symlinks(
    topology: ComposeTopology,
    path: str,
    *,
    directory: bool,
) -> None:
    quoted = shlex.quote(path)
    if directory:
        command = (
            f'if [ ! -d {quoted} ] || [ -L {quoted} ] || '
            f'[ -n "$(find {quoted} -type l -print -quit)" ]; then exit 1; fi'
        )
    else:
        command = f'if [ ! -f {quoted} ] || [ -L {quoted} ]; then exit 1; fi'
    result = await topology.runtime.run_command(command=command, timeout=30)
    if result.return_code != 0:
        raise RuntimeError(f'Compose transfer source is missing, invalid, or contains symlinks: {path}')


async def _shielded_best_effort_cleanup(
    runtime: Any,
    path: str,
    *,
    file_only: bool = False,
    timeout: float = _BEST_EFFORT_CLEANUP_TIMEOUT_SEC,
) -> None:
    async def cleanup() -> None:
        flag = '-f' if file_only else '-rf'
        result = await runtime.run_command(
            command=shlex.join(['rm', flag, path]),
            timeout=30,
        )
        if result.return_code != 0:
            details = result.stderr or result.stdout or 'no output'
            logger.warning('Failed to clean Compose temporary path %s: %s', path, details)

    task = asyncio.create_task(cleanup())
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout

    async def wait_until_deadline() -> None:
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise TimeoutError
        await asyncio.wait_for(asyncio.shield(task), timeout=remaining)

    def abandon_cleanup() -> None:
        task.cancel()
        task.add_done_callback(_consume_task_result)

    try:
        await wait_until_deadline()
    except _TIMEOUT_ERRORS:
        if task.done():
            logger.exception('Failed to clean Compose temporary path %s', path)
            return
        abandon_cleanup()
        logger.warning('Timed out cleaning Compose temporary path %s after %.1fs; abandoning it', path, timeout)
    except asyncio.CancelledError:
        try:
            await wait_until_deadline()
        except _TIMEOUT_ERRORS:
            if not task.done():
                abandon_cleanup()
                logger.warning(
                    'Timed out cleaning Compose temporary path %s after cancellation; abandoning it',
                    path,
                )
        except asyncio.CancelledError:
            abandon_cleanup()
        finally:
            raise
    except Exception:
        logger.exception('Failed to clean Compose temporary path %s', path)


def _require_absolute_container_path(value: str, field: str) -> None:
    if not PurePosixPath(value).is_absolute() or '..' in PurePosixPath(value).parts:
        raise ValueError(f'{field} must be an absolute container path without "..": {value!r}')


__all__ = [
    'ComposeService',
    'ComposeTopology',
]
