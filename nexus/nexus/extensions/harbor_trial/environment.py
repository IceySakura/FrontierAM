"""Harbor environment backed by a direct or Docker-in-Docker Nexus Runtime."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import math
import re
import shlex
import tarfile
import tempfile

from pathlib import Path
from typing import Any, cast, override
from uuid import uuid4

from harbor.constants import MAIN_SERVICE_NAME
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities, EnvironmentResourceCapabilities
from harbor.environments.compose_service_ops import ComposeServiceOpsMixin, ComposeServiceTransport
from harbor.environments.definition import require_agent_environment_definition, should_use_prebuilt_docker_image
from harbor.environments.dind_compose import DinDComposeOps
from harbor.environments.docker import (
    COMPOSE_BUILD_PATH,
    COMPOSE_NO_NETWORK_PATH,
    COMPOSE_PREBUILT_PATH,
    ENV_COMPOSE_NAME,
    RESOURCES_COMPOSE_NAME,
    self_bind_mount,
    write_mounts_compose_file,
    write_resources_compose_file,
)
from harbor.environments.docker.compose_env import (
    ComposeInfraEnvVars,
    legacy_log_mount_env_vars,
    merge_compose_env,
)
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.config import ResourceMode, ServiceVolumeConfig
from harbor.models.trial.paths import EnvironmentPaths, TrialPaths
from harbor.utils.env import resolve_env_vars

from nexus.extensions.agents.installed_common.execution import (
    AgentExecutionTarget,
    ComposeExecutionTarget,
    DirectExecutionTarget,
)
from nexus.extensions.harbor_trial.context import require_harbor_trial_binding
from nexus.extensions.harbor_trial.network import network_capabilities, runtime_network_params
from nexus.runtime.executors.models.command import CommandRunStatus


_ENV_NAME_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
_RUNTIME_NAME_RE = re.compile(r'[^A-Za-z0-9_.-]+')
_PROJECT_NAME_RE = re.compile(r'[^a-z0-9_-]+')
_COMPOSE_DIR = '/harbor/compose'
_ENVIRONMENT_DIR = '/harbor/environment'
_MOUNTS_COMPOSE_NAME = 'docker-compose-mounts.json'
_NO_PROXY_COMPOSE_NAME = 'docker-compose-no-proxy.json'
_TRANSPORT_CONTAINER_DIR = '/opt/nexus-agent'


def _raise_for_upload_failures(response: Any, operation: str) -> None:
    results = getattr(response, 'results', None)
    if not results:
        raise RuntimeError(f'{operation} upload returned no per-file results')
    failed = [result for result in results if not result.success]
    if failed:
        details = '; '.join(f'{result.path}: {result.message or "upload failed"}' for result in failed)
        raise RuntimeError(f'Failed to upload {operation}: {details}')


def _downloaded_file_content(response: Any, operation: str) -> bytes:
    errors = getattr(response, 'errors', None) or []
    if errors:
        details = '; '.join(f'{error.path}: {error.message or "download failed"}' for error in errors)
        raise RuntimeError(f'Failed to download {operation}: {details}')
    files = getattr(response, 'files', None)
    if not files:
        raise RuntimeError(f'{operation} download returned no files')
    if len(files) != 1:
        raise RuntimeError(f'{operation} download returned {len(files)} files; expected exactly one')
    item = files[0]
    if item.content is None:
        raise RuntimeError(f'{operation} download returned empty content')
    if item.encoding == 'base64':
        return base64.b64decode(item.content, validate=True)
    return item.content.encode('utf-8')


def _merge_no_proxy_values(*values: str | None) -> str:
    entries: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value:
            continue
        for raw_entry in value.split(','):
            entry = raw_entry.strip()
            key = entry.casefold()
            if entry and key not in seen:
                seen.add(key)
                entries.append(entry)
    return ','.join(entries)


def _compose_internal_hosts(definition: dict[str, Any]) -> list[str]:
    hosts = ['localhost', '127.0.0.1', '::1']
    services = definition.get('services')
    if not isinstance(services, dict):
        return hosts

    for service_name in sorted(services):
        service = services[service_name]
        candidates: list[Any] = [service_name]
        if isinstance(service, dict):
            candidates.append(service.get('hostname'))
            networks = service.get('networks')
            if isinstance(networks, dict):
                for network_name in sorted(networks):
                    network = networks[network_name]
                    if isinstance(network, dict):
                        aliases = network.get('aliases')
                        if isinstance(aliases, list):
                            candidates.extend(aliases)
        for candidate in candidates:
            if not isinstance(candidate, str):
                continue
            host = candidate.strip()
            if host and ',' not in host and not any(character.isspace() for character in host):
                hosts.append(host)

    return _merge_no_proxy_values(*hosts).split(',')


def _service_environment_value(service: dict[str, Any], name: str) -> str | None:
    environment = service.get('environment')
    if isinstance(environment, dict):
        value = environment.get(name)
        return str(value) if value is not None else None
    if isinstance(environment, list):
        prefix = f'{name}='
        for item in environment:
            if isinstance(item, str) and item.startswith(prefix):
                return item.removeprefix(prefix)
    return None


def _runtime_name_for_session(session_id: str, main_runtime_name: str) -> str:
    if '__verifier__' not in session_id:
        return main_runtime_name
    safe = _RUNTIME_NAME_RE.sub('_', session_id).strip('._-') or 'verifier'
    digest = hashlib.sha1(session_id.encode()).hexdigest()[:8]
    return f'verifier_{safe[:48]}_{digest}'


def _compose_project_name(session_id: str) -> str:
    safe = _PROJECT_NAME_RE.sub('-', session_id.lower()).strip('-_') or 'harbor'
    digest = hashlib.sha1(session_id.encode()).hexdigest()[:8]
    return f'{safe[:53]}-{digest}'


def _helper_source(name: str) -> Path:
    repository_path = Path(__file__).resolve().parents[3] / 'deployment' / 'docker' / 'runtime-bin' / name
    if repository_path.is_file():
        return repository_path
    packaged_path = Path(__file__).resolve().parents[2] / '_bundle' / 'deployment' / 'docker' / 'runtime-bin' / name
    if packaged_path.is_file():
        return packaged_path
    raise FileNotFoundError(f'Compose Agent helper not found: {name}')


class _NexusDinD(DinDComposeOps):
    """Harbor Compose operations implemented on one DinD-enabled Nexus Runtime."""

    _SELF_BIND_LOG_DIRS = True

    def __init__(self, env: NexusEnvironment) -> None:
        self._env = env
        self._resolved_task_env = resolve_env_vars(env.task_env_config.env) if env.task_env_config.env else {}
        self._use_prebuilt = bool(env.task_env_config.docker_image)
        self._no_proxy_staged = False
        transport_id = hashlib.sha256(env.session_id.encode()).hexdigest()[:12]
        self.transport_host_dir = f'/tmp/nxa/{transport_id}'
        self.socket_bridge_host_path = f'{self.transport_host_dir}/bin/nexus-socket-bridge'
        self.socket_bridge_container_path = f'{_TRANSPORT_CONTAINER_DIR}/bin/nexus-socket-bridge'

    @property
    def project_name(self) -> str:
        return _compose_project_name(self._env.session_id)

    def compose_files(self) -> list[str]:
        build_or_prebuilt = COMPOSE_PREBUILT_PATH if self._use_prebuilt else COMPOSE_BUILD_PATH
        files = [
            f'{_COMPOSE_DIR}/{RESOURCES_COMPOSE_NAME}',
            f'{_COMPOSE_DIR}/{build_or_prebuilt.name}',
            f'{_COMPOSE_DIR}/{_MOUNTS_COMPOSE_NAME}',
        ]
        if self._env._environment_docker_compose_path.exists():
            files.append(f'{_ENVIRONMENT_DIR}/docker-compose.yaml')
        files.extend(self._staged_extra_compose_paths())
        files.append(f'{_COMPOSE_DIR}/{ENV_COMPOSE_NAME}')
        if self._no_proxy_staged:
            files.append(f'{_COMPOSE_DIR}/{_NO_PROXY_COMPOSE_NAME}')
        if self._env._network_disabled:
            files.append(f'{_COMPOSE_DIR}/{COMPOSE_NO_NETWORK_PATH.name}')
        return files

    def agent_execution_target(self, user: str | int | None = None) -> AgentExecutionTarget:
        return ComposeExecutionTarget(
            project_name=self.project_name,
            project_directory=_ENVIRONMENT_DIR,
            compose_files=self.compose_files(),
            service=MAIN_SERVICE_NAME,
            transport_host_dir=self.transport_host_dir,
            transport_container_dir=_TRANSPORT_CONTAINER_DIR,
            socket_bridge_host_path=self.socket_bridge_host_path,
            socket_bridge_container_path=self.socket_bridge_container_path,
            compose_env=self._compose_env_vars(),
            user=user,
        )

    def _compose_env_vars(self) -> dict[str, str]:
        infra = ComposeInfraEnvVars(
            main_image_name=f'hb__{self._env.environment_name}',
            context_dir=_ENVIRONMENT_DIR,
            prebuilt_image_name=(self._env.task_env_config.docker_image if self._use_prebuilt else None),
            cpus=self._env._effective_cpus,
            memory=(f'{memory_mb}M' if (memory_mb := self._env._effective_memory_mb) else None),
        ).to_env_dict()
        infra.update(legacy_log_mount_env_vars(self._resolve_volumes(), host_value='target'))
        user_env = {**self._resolved_task_env, **self._env._persistent_env}
        return merge_compose_env(user_env=user_env, infra_env=infra, logger=self._env.logger)

    def _compose_cmd(self, subcommand: list[str]) -> str:
        parts = [
            'docker',
            'compose',
            '--project-name',
            self.project_name,
            '--project-directory',
            _ENVIRONMENT_DIR,
        ]
        for compose_file in self.compose_files():
            parts.extend(['--file', compose_file])
        parts.extend(subcommand)
        return shlex.join(parts)

    @override
    async def _compose_exec(self, subcommand: list[str], timeout_sec: int | None = None) -> ExecResult:
        return await self._env._outer_exec(
            self._compose_cmd(subcommand),
            env=self._compose_env_vars(),
            timeout_sec=timeout_sec,
        )

    @override
    async def _host_exec(self, command: str, timeout_sec: int | None = None) -> ExecResult:
        return await self._env._outer_exec(command, timeout_sec=timeout_sec)

    @override
    async def _stage_file_to_host(self, source_path: Path | str, host_path: str) -> None:
        await self._env._upload_outer_file(source_path, host_path)

    @override
    async def _stage_dir_to_host(self, source_dir: Path | str, host_dir: str) -> None:
        await self._env._upload_outer_dir(source_dir, host_dir)

    @override
    async def _fetch_file_from_host(self, host_path: str, target_path: Path | str) -> None:
        await self._env._download_outer_file(host_path, target_path)

    @override
    async def _fetch_dir_from_host(self, host_dir: str, target_dir: Path | str) -> None:
        await self._env._download_outer_dir(host_dir, target_dir)

    def _staged_extra_compose_paths(self) -> list[str]:
        return [
            f'{_COMPOSE_DIR}/docker-compose-extra-{index}.yaml'
            for index, _ in enumerate(self._env.extra_docker_compose_paths)
        ]

    def _resolve_volumes(self) -> list[ServiceVolumeConfig]:
        volumes = [self_bind_mount(mount) if mount.get('type') == 'bind' else mount for mount in self._env._mounts]
        volumes.extend([
            cast(
                ServiceVolumeConfig,
                {
                    'type': 'bind',
                    'source': '/nix',
                    'target': '/nix',
                    'read_only': True,
                    'bind': {'create_host_path': False},
                },
            ),
            cast(
                ServiceVolumeConfig,
                {
                    'type': 'bind',
                    'source': self.transport_host_dir,
                    'target': _TRANSPORT_CONTAINER_DIR,
                    'bind': {'create_host_path': False},
                },
            ),
        ])
        return volumes

    async def start(self, force_build: bool) -> None:
        self._use_prebuilt = should_use_prebuilt_docker_image(
            self._env.environment_dir,
            docker_image=self._env.task_env_config.docker_image,
            force_build=force_build,
        )
        await self._wait_for_docker_daemon()
        await self._env._outer_exec_checked(
            f'mkdir -p {_COMPOSE_DIR} {_ENVIRONMENT_DIR} {shlex.quote(self.transport_host_dir + "/bin")}',
            timeout=30,
            operation='Create Harbor Compose directories',
        )
        build_or_prebuilt = COMPOSE_PREBUILT_PATH if self._use_prebuilt else COMPOSE_BUILD_PATH
        await self._stage_file_to_host(build_or_prebuilt, f'{_COMPOSE_DIR}/{build_or_prebuilt.name}')
        if self._env._network_disabled:
            await self._stage_file_to_host(COMPOSE_NO_NETWORK_PATH, f'{_COMPOSE_DIR}/{COMPOSE_NO_NETWORK_PATH.name}')
        await self._stage_resources_compose_file()
        await self._stage_env_compose_file(_COMPOSE_DIR)
        await self._stage_dir_to_host(self._env.environment_dir, _ENVIRONMENT_DIR)
        for source, target in zip(
            self._env.extra_docker_compose_paths,
            self._staged_extra_compose_paths(),
            strict=True,
        ):
            await self._stage_file_to_host(source, target)

        volumes = self._resolve_volumes()
        await self._stage_mounts_compose_file(volumes)
        bind_sources = [mount['source'] for mount in volumes if mount.get('type') == 'bind']
        if bind_sources:
            await self._env._outer_exec_checked(
                f'mkdir -p {" ".join(shlex.quote(path) for path in bind_sources)} && '
                f'chmod 777 {" ".join(shlex.quote(path) for path in bind_sources if path != "/nix")}',
                timeout=30,
                operation='Prepare Harbor Compose bind mounts',
            )

        await self._stage_file_to_host(_helper_source('nexus-socket-bridge'), self.socket_bridge_host_path)
        await self._env._outer_exec_checked(
            f'chmod 755 {shlex.quote(self.socket_bridge_host_path)}',
            timeout=30,
            operation='Install Compose Agent socket bridge',
        )

        definition = await self._resolved_compose_definition()
        self._validate_definition(definition)
        await self._stage_no_proxy_compose_file(definition)
        definition = await self._resolved_compose_definition()
        self._validate_definition(definition)
        # Harbor owns the cumulative environment.start() budget, including its timeout multiplier.
        result = await self._compose_exec(['build'], timeout_sec=None)
        if result.return_code != 0:
            raise RuntimeError(f'docker compose build failed: {result.stdout} {result.stderr}')
        result = await self._compose_exec(['up', '-d'], timeout_sec=None)
        if result.return_code != 0:
            raise RuntimeError(f'docker compose up failed: {result.stdout} {result.stderr}')
        await self._wait_for_main_container()
        await self._env._upload_environment_dir_after_start()

    async def stop(self) -> None:
        try:
            result = await self._compose_exec(['down', '--volumes', '--remove-orphans'], timeout_sec=60)
            if result.return_code != 0:
                self._env.logger.warning('docker compose down failed: %s', result.stderr or result.stdout)
        except Exception:
            self._env.logger.warning('docker compose down failed', exc_info=True)

    async def _wait_for_docker_daemon(self) -> None:
        await self._wait_for_readiness('docker info')

    async def _wait_for_main_container(self) -> None:
        await self._wait_for_readiness(
            self._compose_cmd(['exec', '-T', MAIN_SERVICE_NAME, 'true']),
            env=self._compose_env_vars(),
        )

    async def _wait_for_readiness(self, command: str, *, env: dict[str, str] | None = None) -> None:
        # Bound each probe, not the readiness phase; Harbor cancels the enclosing start().
        while True:
            try:
                result = await self._env._outer_exec(command, env=env, timeout_sec=10)
                if result.return_code == 0:
                    return
            except (asyncio.TimeoutError, TimeoutError):
                pass
            await asyncio.sleep(2)

    async def _resolved_compose_definition(self) -> dict[str, Any]:
        result = await self._compose_exec(['config', '--format', 'json'], timeout_sec=30)
        if result.return_code != 0:
            raise RuntimeError(f'docker compose config failed: {result.stderr or result.stdout}')
        definition = json.loads(result.stdout)
        if not isinstance(definition, dict):
            raise ValueError('docker compose config returned a non-object definition')
        return definition

    @staticmethod
    def _validate_definition(definition: dict[str, Any]) -> None:
        services = definition.get('services')
        if not isinstance(services, dict) or MAIN_SERVICE_NAME not in services:
            raise ValueError('Harbor Compose task must define a main service')
        main = services[MAIN_SERVICE_NAME]
        if not isinstance(main, dict) or not (main.get('image') or main.get('build')):
            raise ValueError('Harbor Compose main service must define image or build')

    async def _outer_no_proxy_values(self) -> tuple[str | None, str | None]:
        result = await self._host_exec(
            'printf \'NO_PROXY=%s\\nno_proxy=%s\\n\' "${NO_PROXY-}" "${no_proxy-}"',
            timeout_sec=10,
        )
        if result.return_code != 0:
            raise RuntimeError(f'Resolve outer NO_PROXY failed: {result.stderr or result.stdout}')
        values: dict[str, str] = {}
        for line in result.stdout.splitlines():
            name, separator, value = line.partition('=')
            if separator and name in {'NO_PROXY', 'no_proxy'}:
                values[name] = value
        return values.get('NO_PROXY'), values.get('no_proxy')

    async def _stage_no_proxy_compose_file(self, definition: dict[str, Any]) -> None:
        services = definition.get('services') or {}
        if not isinstance(services, dict):
            raise ValueError('docker compose config returned invalid services')

        outer_upper, outer_lower = await self._outer_no_proxy_values()
        internal_hosts = _compose_internal_hosts(definition)
        override_services: dict[str, Any] = {}
        for service_name, service in sorted(services.items()):
            if not isinstance(service, dict):
                raise ValueError(f'docker compose config returned invalid service {service_name!r}')
            no_proxy = _merge_no_proxy_values(
                outer_upper,
                outer_lower,
                _service_environment_value(service, 'NO_PROXY'),
                _service_environment_value(service, 'no_proxy'),
                *internal_hosts,
            )
            override_services[service_name] = {'environment': {'NO_PROXY': no_proxy, 'no_proxy': no_proxy}}

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / _NO_PROXY_COMPOSE_NAME
            path.write_text(json.dumps({'services': override_services}, indent=2))
            await self._stage_file_to_host(path, f'{_COMPOSE_DIR}/{_NO_PROXY_COMPOSE_NAME}')
        self._no_proxy_staged = True

    async def _stage_mounts_compose_file(self, volumes: list[ServiceVolumeConfig]) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = write_mounts_compose_file(Path(temp_dir) / _MOUNTS_COMPOSE_NAME, volumes)
            await self._stage_file_to_host(path, f'{_COMPOSE_DIR}/{_MOUNTS_COMPOSE_NAME}')

    async def _stage_resources_compose_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = write_resources_compose_file(
                Path(temp_dir) / RESOURCES_COMPOSE_NAME,
                cpu_request=self._env._resource_request_value('cpu', auto_mode=ResourceMode.REQUEST),
                cpu_limit=self._env._resource_limit_value('cpu', auto_mode=ResourceMode.REQUEST),
                memory_request_mb=self._env._resource_request_value('memory', auto_mode=ResourceMode.REQUEST),
                memory_limit_mb=self._env._resource_limit_value('memory', auto_mode=ResourceMode.REQUEST),
            )
            await self._stage_file_to_host(path, f'{_COMPOSE_DIR}/{RESOURCES_COMPOSE_NAME}')


class NexusEnvironment(ComposeServiceOpsMixin, BaseEnvironment):
    """Map Harbor environments to direct or DinD-enabled Nexus Runtimes."""

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths,
        task_env_config: EnvironmentConfig,
        logger: logging.Logger | None = None,
        runtime_name: str | None = None,
        extra_docker_compose: list[Path] | None = None,
        **kwargs: Any,
    ) -> None:
        self._binding = require_harbor_trial_binding()
        self._is_verifier_environment = '__verifier__' in session_id
        self._runtime_name = runtime_name or _runtime_name_for_session(session_id, self._binding.runtime_name)
        self._runtime: Any | None = None
        self._compose_mode = (
            (environment_dir / 'docker-compose.yaml').exists()
            or bool(extra_docker_compose)
            or not task_env_config.docker_image
        )
        self._dind: _NexusDinD | None = None
        super().__init__(
            environment_dir=environment_dir,
            environment_name=environment_name,
            session_id=session_id,
            trial_paths=trial_paths,
            task_env_config=task_env_config,
            logger=logger,
            extra_docker_compose=extra_docker_compose,
            **kwargs,
        )
        if self._compose_mode:
            self._dind = _NexusDinD(self)

    @staticmethod
    def type() -> str:
        return 'nexus'

    @property
    @override
    def _uses_compose(self) -> bool:
        return self._compose_mode

    @classmethod
    @override
    def resource_capabilities(cls) -> EnvironmentResourceCapabilities:
        return EnvironmentResourceCapabilities(cpu_request=True, memory_request=True)

    @property
    @override
    def capabilities(self) -> EnvironmentCapabilities:
        return network_capabilities(self._binding.execution_context.runtime_provider, compose_mode=self._compose_mode)

    @property
    def _environment_docker_compose_path(self) -> Path:
        return self.environment_dir / 'docker-compose.yaml'

    def agent_execution_target(self, user: str | int | None = None) -> AgentExecutionTarget | None:
        if self._dind is None:
            return DirectExecutionTarget(user=user) if user is not None else None
        return self._dind.agent_execution_target(user)

    @override
    def _validate_definition(self) -> None:
        require_agent_environment_definition(
            self.environment_dir,
            docker_image=self.task_env_config.docker_image,
            extra_docker_compose_paths=self.extra_docker_compose_paths,
        )
        if self.task_env_config.storage_mb is not None:
            self.logger.warning(
                'Ignoring [environment].storage_mb=%s because NexusEnvironment cannot enforce storage capacity',
                self.task_env_config.storage_mb,
            )

    @override
    async def start(self, force_build: bool) -> None:
        if self._runtime is not None:
            return
        if not self._compose_mode and not should_use_prebuilt_docker_image(
            self.environment_dir,
            docker_image=self.task_env_config.docker_image,
            force_build=force_build,
        ):
            self._compose_mode = True
            self._dind = _NexusDinD(self)
        self._validate_network_policy_support()

        context = self._binding.execution_context
        if context.runtime_provider is None:
            raise RuntimeError('harbor_trial_task requires a Nexus runtime provider')
        if self._runtime_name in context.runtime_provider.get_active_runtimes():
            raise RuntimeError(
                f'Nexus Runtime {self._runtime_name!r} was acquired before Harbor started the environment'
            )

        resource_spec: dict[str, int] = {}
        if self._effective_cpus is not None:
            resource_spec['cpu'] = int(self._effective_cpus)
        if self._effective_memory_mb is not None:
            resource_spec['memory'] = max(1, math.ceil(self._effective_memory_mb / 1024))

        params: dict[str, Any] = {
            'environment': {} if self._compose_mode else self._startup_env(),
            'enable_dind': self._compose_mode,
        }
        params.update(
            runtime_network_params(
                self.network_policy,
                context.runtime_provider,
                runtime_name=self._runtime_name,
                compose_mode=self._compose_mode,
            )
        )
        if resource_spec:
            params['resource_spec'] = resource_spec
        if not self._compose_mode:
            params['image'] = self.task_env_config.docker_image
            if self.task_env_config.workdir:
                params['working_dir'] = self.task_env_config.workdir

        self._runtime = await context.get_runtime(self._runtime_name, **params)
        if self._dind is not None:
            await self._dind.start(force_build=force_build)
        else:
            await self.ensure_dirs(self._mount_targets(writable_only=True))
            if self._is_verifier_environment:
                tests_dir = str(EnvironmentPaths.for_os(self.os).tests_dir)
                await self.ensure_dirs([tests_dir])
                await self.upload_dir(self.environment_dir, tests_dir)
            else:
                await self._upload_environment_dir_after_start()

    @override
    async def stop(self, delete: bool) -> None:
        if self._runtime is None:
            return
        if not delete:
            self.logger.debug('Keeping Nexus Runtime %s until ExecutionContext cleanup', self._runtime_name)
            return
        if self._dind is not None:
            await self._dind.stop()
        released = await self._binding.execution_context.release_runtime(self._runtime_name)
        if released:
            self._runtime = None

    def _require_runtime(self) -> Any:
        if self._runtime is None:
            raise RuntimeError('Nexus environment has not been started')
        return self._runtime

    @override
    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        user = self._resolve_user(user)
        merged_env = self._merge_env(env)
        effective_cwd = cwd or self.task_env_config.workdir
        if self._dind is not None:
            result = await self._dind.exec(
                command,
                cwd=effective_cwd,
                env=merged_env,
                timeout_sec=timeout_sec,
                user=user,
            )
        else:
            result = await self._outer_exec(
                command,
                cwd=effective_cwd,
                env=merged_env,
                timeout_sec=timeout_sec,
                user=user,
            )
        await self._publish_output(result)
        return result

    async def _outer_exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        runtime = self._require_runtime()
        remote_env_dir: str | None = None
        cleanup_error: str | None = None
        primary_error: BaseException | None = None
        try:
            if env:
                invalid_names = [name for name in env if not _ENV_NAME_RE.fullmatch(name)]
                if invalid_names:
                    raise ValueError(f'Invalid environment variable name(s): {invalid_names}')
                remote_env_dir = f'/tmp/.harbor_exec_env_{uuid4().hex}'
                remote_env_path = f'{remote_env_dir}/env'
                await self._outer_exec_checked(
                    f'install -d -m 700 {shlex.quote(remote_env_dir)}',
                    timeout=10,
                    operation='Create protected environment directory',
                )
                with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False) as env_file:
                    for key, value in env.items():
                        env_file.write(f'export {key}={shlex.quote(str(value))}\n')
                    local_env_path = Path(env_file.name)
                try:
                    await self._upload_outer_file(local_env_path, remote_env_path)
                finally:
                    local_env_path.unlink(missing_ok=True)
                permission_commands = [
                    f'chmod 700 {shlex.quote(remote_env_dir)}',
                    f'chmod 600 {shlex.quote(remote_env_path)}',
                ]
                if user not in (None, 'root', 0, '0'):
                    owner = f'"$(getent passwd {user} | cut -d: -f1)"' if isinstance(user, int) else shlex.quote(user)
                    permission_commands.insert(
                        0,
                        f'chown {owner} {shlex.quote(remote_env_dir)} {shlex.quote(remote_env_path)}',
                    )
                result = await runtime.run_command(command=' && '.join(permission_commands), timeout=10)
                if result.return_code != 0:
                    raise RuntimeError(
                        f'Secure environment file failed with code {result.return_code}: '
                        f'{result.stderr or result.stdout or "no output"}'
                    )

            inner = f'cd {shlex.quote(cwd)} && {command}' if cwd else command
            if remote_env_dir:
                inner = f'set -a && . {shlex.quote(remote_env_dir + "/env")} && set +a && {inner}'
            if user not in (None, 'root', 0, '0'):
                user_expr = f'"$(getent passwd {user} | cut -d: -f1)"' if isinstance(user, int) else shlex.quote(user)
                inner = f'su {user_expr} -s /bin/sh -c {shlex.quote(inner)}'
            result = await runtime.run_command(command=inner, timeout=timeout_sec)
            stdout, stderr = result.stdout or '', result.stderr or ''
            if result.status != CommandRunStatus.Finished or result.return_code is None:
                details = f'Command: {command}\nstdout:\n{stdout}\nstderr:\n{stderr}'
                if result.status == CommandRunStatus.TimeLimitExceeded:
                    duration = f' after {timeout_sec}s' if timeout_sec is not None else ''
                    raise asyncio.TimeoutError(f'Nexus command timed out{duration}\n{details}')
                if result.status == CommandRunStatus.Error:
                    raise RuntimeError(f'Nexus command execution failed\n{details}')
                raise RuntimeError(
                    f'Invalid Nexus command result: status={result.status}, return_code={result.return_code}\n{details}'
                )
            return ExecResult(stdout=stdout, stderr=stderr, return_code=result.return_code)
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            if remote_env_dir:
                try:
                    cleanup = await runtime.run_command(command=f'rm -rf {shlex.quote(remote_env_dir)}', timeout=10)
                    if cleanup.return_code != 0:
                        cleanup_error = cleanup.stderr or cleanup.stdout or 'unknown cleanup error'
                except Exception as exc:
                    cleanup_error = str(exc)
            if cleanup_error:
                if primary_error is None:
                    raise RuntimeError(f'Environment variable cleanup failed: {cleanup_error}')
                self.logger.warning('Environment variable cleanup failed after command error: %s', cleanup_error)

    async def _publish_output(self, result: ExecResult) -> None:
        callback = self._output_callback()
        if callback is None:
            return
        if result.stdout:
            await callback(result.stdout, 'stdout')
        if result.stderr:
            await callback(result.stderr, 'stderr')

    async def _outer_exec_checked(self, command: str, *, timeout: float, operation: str) -> Any:
        result = await self._outer_exec(command, timeout_sec=round(timeout))
        if result.return_code != 0:
            details = (result.stderr or result.stdout or '').strip() or 'no output'
            raise RuntimeError(f'{operation} failed with code {result.return_code}: {details}')
        return result

    async def _upload_outer_file(self, source_path: Path | str, target_path: str) -> None:
        from nexus.extensions.tasks.terminal.utils import read_local_file_to_file_item

        source = Path(source_path)
        if not source.is_file():
            raise FileNotFoundError(f'Upload source is not a file: {source}')
        response = await self._require_runtime().upload_file(
            files=[read_local_file_to_file_item(source, target_path)], overwrite=True
        )
        _raise_for_upload_failures(response, f'file {source}')

    async def _upload_outer_dir(self, source_dir: Path | str, target_dir: str) -> None:
        source = Path(source_dir)
        if not source.is_dir():
            raise FileNotFoundError(f'Upload source is not a directory: {source}')
        with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as temp_file:
            archive_path = Path(temp_file.name)
        remote_archive = f'/tmp/.harbor_upload_{uuid4().hex}.tar.gz'
        try:
            with tarfile.open(archive_path, 'w:gz') as archive:
                archive.add(source, arcname='.', recursive=True)
            await self._upload_outer_file(archive_path, remote_archive)
            await self._outer_exec_checked(
                f'mkdir -p {shlex.quote(target_dir)} && tar xzf {shlex.quote(remote_archive)} -C {shlex.quote(target_dir)}',
                timeout=600,
                operation=f'Upload directory to {target_dir}',
            )
        finally:
            archive_path.unlink(missing_ok=True)
            await self._outer_exec(f'rm -f {shlex.quote(remote_archive)}', timeout_sec=10)

    async def _download_outer_content(self, source_path: str, *, operation: str) -> bytes:
        from nexus.runtime.executors.models.file import FileItem

        response = await self._require_runtime().download_file(files=[FileItem(path=source_path, encoding='base64')])
        return _downloaded_file_content(response, operation)

    async def _download_outer_file(self, source_path: str, target_path: Path | str) -> None:
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(await self._download_outer_content(source_path, operation=f'file {source_path}'))

    async def _download_outer_dir(self, source_dir: str, target_dir: Path | str) -> None:
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        remote_archive = f'/tmp/.harbor_download_{uuid4().hex}.tar.gz'
        with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as temp_file:
            archive_path = Path(temp_file.name)
        try:
            await self._outer_exec_checked(
                f'tar czf {shlex.quote(remote_archive)} -C {shlex.quote(source_dir)} .',
                timeout=600,
                operation=f'Archive directory {source_dir}',
            )
            archive_path.write_bytes(
                await self._download_outer_content(remote_archive, operation=f'directory {source_dir}')
            )
            with tarfile.open(archive_path, 'r:gz') as archive:
                archive.extractall(path=target, filter='data')
        finally:
            archive_path.unlink(missing_ok=True)
            await self._outer_exec(f'rm -f {shlex.quote(remote_archive)}', timeout_sec=10)

    @override
    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        if self._dind is not None:
            await self._dind.upload_file(source_path, target_path)
        else:
            await self._upload_outer_file(source_path, target_path)

    @override
    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        if self._dind is not None:
            await self._dind.upload_dir(source_dir, target_dir)
        else:
            await self._upload_outer_dir(source_dir, target_dir)

    @override
    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        if self._dind is not None:
            await self._dind.download_file(source_path, target_path)
        else:
            await self._download_outer_file(source_path, target_path)

    @override
    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        if self._dind is not None:
            await self._dind.download_dir(source_dir, target_dir)
        else:
            await self._download_outer_dir(source_dir, target_dir)

    @override
    def _compose_service_transport(self, service: str | None) -> ComposeServiceTransport:
        if self._dind is None:
            raise self._compose_unsupported(service)
        return self._dind


__all__ = ['NexusEnvironment']
