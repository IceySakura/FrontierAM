"""Execution targets and transports shared by installed Agents."""

from __future__ import annotations

import json
import os
import re
import shlex
import sys

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator


PROXY_URL_PLACEHOLDER = '__NEXUS_AGENT_PROXY_URL__'
_PATH_COMPONENT_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]*$')


class DirectExecutionTarget(BaseModel):
    """Agent identity in the same sandbox as the privileged Worker."""

    model_config = ConfigDict(extra='forbid', strict=True, frozen=True)

    kind: Literal['direct'] = 'direct'
    user: str | int | None = None


class ComposeExecutionTarget(BaseModel):
    """Trusted controller coordinates for an Agent hosted in one Compose service."""

    model_config = ConfigDict(extra='forbid', strict=True, frozen=True)

    kind: Literal['compose'] = 'compose'
    project_name: str
    project_directory: str | None = None
    compose_files: list[str]
    service: str
    transport_host_dir: str
    transport_container_dir: str
    socket_bridge_host_path: str
    socket_bridge_container_path: str
    compose_env: dict[str, str] = Field(default_factory=dict)
    user: str | int | None = None

    @field_validator('project_name')
    @classmethod
    def _project_name_is_safe(cls, value: str) -> str:
        if not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,62}', value):
            raise ValueError('must be a bounded lowercase Compose project name')
        return value

    @field_validator('service')
    @classmethod
    def _service_name_is_safe(cls, value: str) -> str:
        if not _PATH_COMPONENT_RE.fullmatch(value):
            raise ValueError('must be a safe Compose service name')
        return value

    @field_validator('compose_files')
    @classmethod
    def _compose_files_are_absolute(cls, value: list[str]) -> list[str]:
        if not value or any(not path.startswith('/') or '\x00' in path or '\n' in path for path in value):
            raise ValueError('compose_files must contain safe absolute paths')
        return value

    @field_validator('project_directory')
    @classmethod
    def _optional_path_is_absolute(cls, value: str | None) -> str | None:
        if value is not None and (not value.startswith('/') or '\x00' in value or '\n' in value):
            raise ValueError('must be a safe absolute path')
        return value

    @field_validator(
        'transport_container_dir',
        'socket_bridge_host_path',
        'socket_bridge_container_path',
    )
    @classmethod
    def _paths_are_absolute(cls, value: str) -> str:
        if not value.startswith('/') or '\x00' in value or '\n' in value:
            raise ValueError('must be a safe absolute path')
        return value

    @field_validator('compose_env')
    @classmethod
    def _compose_env_is_safe(cls, value: dict[str, str]) -> dict[str, str]:
        if any(not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', name) for name in value):
            raise ValueError('compose_env contains an invalid variable name')
        if any('\x00' in item or '\n' in item for item in value.values()):
            raise ValueError('compose_env contains an invalid variable value')
        return value

    @field_validator('transport_host_dir')
    @classmethod
    def _transport_host_dir_is_socket_safe(cls, value: str) -> str:
        if not value.startswith('/') or '\x00' in value or '\n' in value:
            raise ValueError('must be a safe absolute path')
        if len(value.encode('utf-8')) > 48:
            raise ValueError('must be short enough for Unix socket paths')
        return value

    @model_validator(mode='after')
    def _helper_paths_are_scoped(self) -> ComposeExecutionTarget:
        host_prefix = f'{self.transport_host_dir.rstrip("/")}/'
        container_prefix = f'{self.transport_container_dir.rstrip("/")}/'
        if not self.socket_bridge_host_path.startswith(host_prefix):
            raise ValueError('socket_bridge_host_path must stay inside transport_host_dir')
        if not self.socket_bridge_container_path.startswith(container_prefix):
            raise ValueError('socket_bridge_container_path must stay inside transport_container_dir')
        return self


AgentExecutionTarget = Annotated[DirectExecutionTarget | ComposeExecutionTarget, Field(discriminator='kind')]
_TARGET_ADAPTER = TypeAdapter(AgentExecutionTarget)
_current_agent_execution_target: ContextVar[AgentExecutionTarget | None] = ContextVar(
    'agent_execution_target', default=None
)


@dataclass(frozen=True)
class PreparedExecution:
    """One process invocation prepared for the selected execution target."""

    command: list[str]
    cwd: str
    env: dict[str, str]


class ExecutionTransportError(RuntimeError):
    """Raised when an execution target cannot prepare or clean up a process."""


def resolve_agent_execution_target(value: AgentExecutionTarget | dict[str, Any] | None) -> AgentExecutionTarget | None:
    """Validate a serialized target received by a Runtime worker."""
    if value is None:
        return None
    if isinstance(value, (DirectExecutionTarget, ComposeExecutionTarget)):
        return value
    if 'kind' not in value:
        value = {'kind': 'compose', **value}
    return _TARGET_ADAPTER.validate_python(value)


def get_agent_execution_target() -> AgentExecutionTarget | None:
    """Return the target selected by the current task adapter, if any."""
    return _current_agent_execution_target.get()


@contextmanager
def agent_execution_target(target: AgentExecutionTarget | None) -> Iterator[None]:
    """Select an execution target for nested installed-Agent calls."""
    token = _current_agent_execution_target.set(target)
    try:
        yield
    finally:
        _current_agent_execution_target.reset(token)


def _require_path_component(value: str, name: str) -> str:
    if not _PATH_COMPONENT_RE.fullmatch(value):
        raise ValueError(f'{name} must be a safe path component: {value!r}')
    return value


class AgentExecutionTransport:
    """Dispatch installed-Agent work to local execution or a configured target."""

    def __init__(self, target: AgentExecutionTarget | dict[str, Any] | None = None) -> None:
        self.target = resolve_agent_execution_target(target)
        self._identity: dict[str, Any] | None = None

    @property
    def is_local(self) -> bool:
        return self.target is None

    @property
    def is_compose(self) -> bool:
        return isinstance(self.target, ComposeExecutionTarget)

    @property
    def has_identity(self) -> bool:
        return self._identity is not None

    def validate(self) -> None:
        if isinstance(self.target, DirectExecutionTarget) and (not hasattr(os, 'geteuid') or os.geteuid() != 0):
            raise ExecutionTransportError('Direct Agent execution requires a root Worker')
        if isinstance(self.target, ComposeExecutionTarget) and not os.access(
            self.target.socket_bridge_host_path, os.X_OK
        ):
            raise RuntimeError(
                f'Agent execution requires executable staged helper {self.target.socket_bridge_host_path}'
            )

    def _require_identity(self) -> dict[str, Any]:
        if self._identity is None:
            raise ExecutionTransportError('Initialize the Agent execution transport before resolving account paths')
        return self._identity

    def install_dir(self, *, agent_name: str, default: str) -> str:
        if self.is_local:
            return default
        identity = self._require_identity()
        name = _require_path_component(agent_name, 'agent_name')
        if isinstance(self.target, ComposeExecutionTarget):
            return f'{self.target.transport_container_dir}/agent-{identity["uid"]}/install/{name}'
        return f'{identity["home"]}/.local/share/nexus/agents/{name}'

    def home_dir(self, *, invocation_id: str, default: str | Path) -> Path:
        if self.is_local:
            return Path(default)
        return Path(self._require_identity()['home'])

    def shared_path(self, name: str, *, default: str | Path) -> Path:
        if not isinstance(self.target, ComposeExecutionTarget):
            return Path(default)
        component = _require_path_component(name, 'name')
        return Path(self.target.transport_host_dir) / component

    @asynccontextmanager
    async def prepare(
        self,
        command: list[str],
        *,
        working_dir: str,
        invocation_id: str,
        env: dict[str, str],
        proxy_files: list[str | Path] | None = None,
    ) -> AsyncIterator[PreparedExecution]:
        """Prepare one invocation while preserving the local path exactly by default."""
        if self.target is None:
            yield PreparedExecution(command=command, cwd=working_dir, env=env)
            return

        await self.initialize()
        if isinstance(self.target, DirectExecutionTarget):
            yield self._prepare_command(command, working_dir=working_dir, env=env)
            return

        from nexus.extensions.agents.installed_common.compose import ComposeExecutionTransport

        async with ComposeExecutionTransport(self.target).prepare(
            command,
            working_dir=working_dir,
            invocation_id=invocation_id,
            env=env,
            proxy_files=proxy_files,
            transport=self,
        ) as prepared:
            yield prepared

    def compose_command(self, *command: str) -> list[str]:
        if not isinstance(self.target, ComposeExecutionTarget):
            raise ExecutionTransportError('Not a Compose target')
        argv = ['docker', 'compose', '--project-name', self.target.project_name]
        if self.target.project_directory is not None:
            argv.extend(['--project-directory', self.target.project_directory])
        for compose_file in self.target.compose_files:
            argv.extend(['--file', compose_file])
        argv.extend(command)
        return argv

    def probe_command(self, script: str) -> list[str]:
        if not isinstance(self.target, ComposeExecutionTarget):
            return ['sh', '-c', script]
        return self.compose_command('exec', '-T', self.target.service, 'sh', '-c', script)

    async def initialize(self) -> None:
        """Resolve target identity and its effective HOME, never the Worker's HOME."""
        if self.target is None or self.has_identity:
            return
        self.validate()
        source = (
            'import json, os, pwd; '
            'account = pwd.getpwuid(os.geteuid()); '
            'print(json.dumps(dict(uid=account.pw_uid, gid=account.pw_gid, '
            'user=account.pw_name, home=account.pw_dir, path=os.environ.get("PATH", ""))))'
        )
        result = await self._execute([self.python_binary, '-c', source], env={}, check=True)
        identity = json.loads(result.stdout)
        if isinstance(self.target, ComposeExecutionTarget):
            from nexus.extensions.agents.installed_common import target_files

            # The transport mount stays writable even when the task rootfs and
            # account HOME do not. Create it as the Agent, not the root Worker.
            request = {
                'operation': 'prepare_home',
                'path': f'{self.target.transport_container_dir}/agent-{identity["uid"]}',
            }
            result = await self._execute(
                [self.python_binary, '-c', Path(target_files.__file__).read_text()],
                stdin=json.dumps(request),
                env={},
                check=True,
            )
            identity['home'] = json.loads(result.stdout)
        if not identity['home'].startswith('/'):
            raise ExecutionTransportError('Agent account must have an absolute HOME')
        identity['path'] = ':'.join(part for part in identity['path'].split(':') if part != '/nix/runtime/nix-env/bin')
        self._identity = identity

    @property
    def python_binary(self) -> str:
        runtime_python = '/nix/runtime/nix-env/bin/python3'
        if self.is_compose or Path(runtime_python).is_file():
            return runtime_python
        return '/usr/bin/python3' if Path('/usr/bin/python3').is_file() else sys.executable

    def agent_env(self, env: dict[str, str] | None = None) -> dict[str, str]:
        result = dict(env or {})
        if self._identity is not None:
            result.update(HOME=self._identity['home'], USER=self._identity['user'], LOGNAME=self._identity['user'])
            result.setdefault('PATH', self._identity['path'])
        return result

    def _prepare_command(
        self,
        command: list[str],
        *,
        working_dir: str = '/',
        env: dict[str, str] | None = None,
        root: bool = False,
    ) -> PreparedExecution:
        environment = dict(env or {}) if root else self.agent_env(env)
        if root and self.target is not None:
            environment['BASH_ENV'] = '/dev/null'
        if isinstance(self.target, ComposeExecutionTarget):
            argv = self.compose_command('exec', '-T')
            user = 'root' if root else self.target.user
            if user is not None:
                argv.extend(['--user', str(user)])
            argv.extend(['--workdir', working_dir])
            account_variables = {'HOME', 'PATH', 'USER', 'LOGNAME', 'BASH_ENV'}
            for name, value in environment.items():
                if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', name):
                    raise ValueError(f'Invalid environment variable: {name!r}')
                argv.extend(['--env', f'{name}={value}' if name in account_variables else name])
            argv.extend([self.target.service, *command])
            return PreparedExecution(argv, '/', self._management_env({**self.target.compose_env, **environment}))
        if isinstance(self.target, DirectExecutionTarget):
            import pwd

            user = 'root' if root else self.target.user
            if user is None:
                user = os.geteuid()
            account = pwd.getpwuid(int(user)) if isinstance(user, int) or str(user).isdigit() else pwd.getpwnam(user)
            environment.setdefault('BASH_ENV', '~/.bashrc')
            account_variables = {'HOME', 'PATH', 'USER', 'LOGNAME', 'BASH_ENV'}
            # Apply account variables only after switching identity. The outer root
            # shell must not source the Agent's BASH_ENV, even for a root Agent.
            inner_command = [
                'env',
                *[f'{name}={value}' for name, value in environment.items() if name in account_variables],
                *command,
            ]
            outer_env = self._management_env(environment)
            if account.pw_uid == os.geteuid():
                return PreparedExecution(inner_command, working_dir, outer_env)
            script = f'cd {shlex.quote(working_dir)} && exec ' + shlex.join(inner_command)
            argv = ['su', account.pw_name, '-s', '/bin/sh', '-c', script]
            return PreparedExecution(argv, '/', outer_env)
        return PreparedExecution(command, working_dir, environment)

    @staticmethod
    def _management_env(environment: dict[str, str]) -> dict[str, str]:
        result = dict(environment)
        for name in ('HOME', 'PATH', 'USER', 'LOGNAME'):
            result.pop(name, None)
            if name in os.environ:
                result[name] = os.environ[name]
        result['BASH_ENV'] = '/dev/null'
        return result

    async def _execute(
        self,
        command: list[str],
        *,
        working_dir: str = '/',
        env: dict[str, str] | None = None,
        timeout: float = 60,
        stdin: str | None = None,
        root: bool = False,
        check: bool = False,
    ) -> Any:
        from nexus.extensions.agents.installed_common import build_runtime_command, is_windows
        from nexus.runtime.context import execution_user
        from nexus.runtime.executors.command import CommandRunStatus, RunCommandRequest, run_command

        prepared = self._prepare_command(command, working_dir=working_dir, env=env, root=root)
        runtime_command = build_runtime_command(prepared.command)
        if self.is_local and is_windows():
            runtime_command += '; exit $LASTEXITCODE'
        with nullcontext() if self.is_local else execution_user(None):
            result = await run_command(
                RunCommandRequest(
                    command=runtime_command,
                    cwd=prepared.cwd,
                    extra_env=prepared.env,
                    timeout=timeout,
                    stdin=stdin,
                )
            )
        if check and (result.status != CommandRunStatus.Finished or result.return_code != 0):
            raise ExecutionTransportError(f'Target operation failed: {result.stderr or result.stdout or result.status}')
        return result

    async def exec_as_agent(self, command: list[str], **kwargs: Any) -> Any:
        await self.initialize()
        return await self._execute(command, **kwargs)

    async def exec_as_root(self, command: list[str], **kwargs: Any) -> Any:
        self.validate()
        return await self._execute(command, root=True, **kwargs)

    async def python(self, source: str, payload: Any, *, timeout: float = 60) -> Any:
        result = await self.exec_as_agent(
            [self.python_binary, '-c', source],
            stdin=json.dumps(payload),
            timeout=timeout,
            check=True,
        )
        return json.loads(result.stdout)

    async def file(self, operation: str, path: str | Path, **kwargs: Any) -> Any:
        from nexus.extensions.agents.installed_common import target_files

        request = {'operation': operation, 'path': str(path), **kwargs}
        if self.is_local:
            return target_files.operate(request)
        return await self.python(
            Path(target_files.__file__).read_text(),
            request,
            timeout=1800 if operation == 'download' else 60,
        )

    async def mkdir(self, path: str | Path) -> None:
        if self.target is None:
            Path(path).mkdir(parents=True, exist_ok=True)
        else:
            await self.file('mkdir', path)

    async def read_text(self, path: str | Path) -> str | None:
        if self.target is None:
            return Path(path).read_text(encoding='utf-8') if Path(path).exists() else None
        return await self.file('read', path)

    async def write_text(self, path: str | Path, content: str, *, mode: int = 0o600) -> None:
        if self.target is None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(content, encoding='utf-8')
        else:
            await self.file('write', path, content=content, mode=mode)

    async def remove(self, path: str | Path) -> None:
        if self.target is None:
            Path(path).unlink(missing_ok=True)
        else:
            await self.file('remove', path)

    async def find_binary(self, name: str) -> str:
        if self.is_local:
            from nexus.extensions.agents.installed_common import find_runtime_binary

            return find_runtime_binary(name) or name
        source = (
            'import json, shutil; from pathlib import Path; '
            'name=json.load(__import__("sys").stdin); '
            'fallback=Path("/nix/runtime/nix-env/bin")/name; '
            'print(json.dumps(shutil.which(name) or (str(fallback) if fallback.is_file() else name)))'
        )
        return await self.python(source, name)

    async def deliver_file(self, source: str | Path, destination: str | Path) -> None:
        """Hand a management-side upload to the account without making it world-readable."""
        await self.write_text(destination, Path(source).read_text(encoding='utf-8'))

    async def copy_directory_to_local(self, source: Path, destination: Path) -> None:
        """Copy a target directory's contents into a management-side directory."""
        import shutil

        if not self.is_compose:
            if source.is_dir():
                shutil.copytree(source, destination, dirs_exist_ok=True)
            return
        if await self.file('exists', source, directory=True):
            from nexus.runtime.context import execution_user
            from nexus.runtime.executors.command import RunCommandRequest, run_command

            destination.mkdir(parents=True, exist_ok=True)
            assert isinstance(self.target, ComposeExecutionTarget)
            argv = self.compose_command('cp', f'{self.target.service}:{source}/.', str(destination))
            with execution_user(None):
                result = await run_command(
                    RunCommandRequest(command=shlex.join(argv), extra_env=self.target.compose_env)
                )
            if result.return_code != 0:
                raise ExecutionTransportError(f'Copy target directory failed: {result.stderr}')

    def classify_failure(self, exit_code: int, stderr: str) -> tuple[str | None, str | None]:
        if exit_code == 0:
            return None, None
        infrastructure_markers = (
            'Cannot connect to the Docker daemon',
            'Error response from daemon',
            'is not running',
            'no such service',
        )
        if self.target is not None and any(marker in stderr for marker in infrastructure_markers):
            return 'infrastructure', 'execution_transport'
        return 'agent', 'process'


class LegacyAgentExecutionTransport(AgentExecutionTransport):
    """Explicit shared-install compatibility entry for Agents not yet migrated."""

    async def initialize(self) -> None:
        if not self.is_compose:
            await super().initialize()

    def install_dir(self, *, agent_name: str, default: str) -> str:
        if not isinstance(self.target, ComposeExecutionTarget):
            return default
        name = _require_path_component(agent_name, 'agent_name')
        return f'{self.target.transport_host_dir}/install/{name}'

    def home_dir(self, *, invocation_id: str, default: str | Path) -> Path:
        if not isinstance(self.target, ComposeExecutionTarget):
            return Path(default)
        invocation = _require_path_component(invocation_id, 'invocation_id')
        return Path(self.target.transport_host_dir) / 'home' / invocation

    @asynccontextmanager
    async def prepare(
        self,
        command: list[str],
        *,
        working_dir: str,
        invocation_id: str,
        env: dict[str, str],
        proxy_files: list[str | Path] | None = None,
    ) -> AsyncIterator[PreparedExecution]:
        if isinstance(self.target, ComposeExecutionTarget):
            from nexus.extensions.agents.installed_common.compose import ComposeExecutionTransport

            context = ComposeExecutionTransport(self.target).prepare_legacy(
                command, working_dir=working_dir, invocation_id=invocation_id, env=env, proxy_files=proxy_files
            )
        else:
            context = super().prepare(
                command, working_dir=working_dir, invocation_id=invocation_id, env=env, proxy_files=proxy_files
            )
        async with context as prepared:
            yield prepared


def execution_target_shared_path(
    target: AgentExecutionTarget | None,
    name: str,
    *,
    default: str | Path,
) -> Path:
    """Resolve a Host-uploaded file path visible to the selected target."""
    return AgentExecutionTransport(target).shared_path(name, default=default)


__all__ = [
    'PROXY_URL_PLACEHOLDER',
    'AgentExecutionTarget',
    'AgentExecutionTransport',
    'ComposeExecutionTarget',
    'DirectExecutionTarget',
    'ExecutionTransportError',
    'PreparedExecution',
    'agent_execution_target',
    'execution_target_shared_path',
    'get_agent_execution_target',
    'resolve_agent_execution_target',
]
