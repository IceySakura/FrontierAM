"""Gongfeng Environment: Harbor BaseEnvironment backed by nexus runtime.

This is the core bridge component. It implements harbor's BaseEnvironment
interface (exec, upload_file, upload_dir, download_file, download_dir, start, stop)
by delegating to nexus's Runtime object (which talks to the gongfeng sandbox API).

This allows harbor's Verifier, Agent, and Task code to work unchanged -
they just see a standard BaseEnvironment, while the actual execution happens
in a gongfeng sandbox managed by nexus.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import io
import logging
import re
import shlex
import tarfile
import tempfile

from pathlib import Path
from typing import Any
from uuid import uuid4

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths

from nexus.extensions.harbor_bridge.upload import raise_for_upload_failures
from nexus.runtime.executors.command import CommandRunStatus


logger = logging.getLogger(__name__)


_ENV_NAME_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')


def _build_directory_archive_item(source: Path, container_path: str) -> Any:
    """Pack a local directory into one base64 tar.gz FileItem; CPU-bound, run it off the event loop."""
    from nexus.runtime.executors.models.file import FileItem

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz') as tar:
        tar.add(str(source), arcname='.', recursive=True)
    content = base64.b64encode(buffer.getvalue()).decode('ascii')
    return FileItem(path=container_path, content=content, encoding='base64')


def _decode_base64_file_content(content: str | bytes, source_path: str) -> bytes:
    try:
        return base64.b64decode(content, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise RuntimeError(f'Download remote file {source_path} returned invalid base64 content') from exc


def _extract_directory_archive(archive: bytes, target: Path) -> None:
    with tarfile.open(fileobj=io.BytesIO(archive), mode='r:gz') as tar:
        tar.extractall(path=target, filter='data')


class GongfengEnvironment(BaseEnvironment):
    """Harbor BaseEnvironment implementation backed by nexus gongfeng Runtime.

    This environment delegates all operations to a nexus Runtime instance,
    translating harbor's environment interface to nexus's runtime API.
    """

    def __init__(
        self,
        runtime: Any,  # nexus Runtime instance
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths,
        task_env_config: EnvironmentConfig,
        default_timeout_sec: float = 600.0,
        logger_instance: logging.Logger | None = None,
        network_isolation_enabled: bool = False,
        harbor_network_policy: Any | None = None,
        gpus_enabled: bool = False,
        **kwargs: Any,
    ):
        """Initialize GongfengEnvironment.

        Args:
            runtime: nexus Runtime instance (HttpRuntime or SocketIoRuntime)
            environment_dir: Path to the environment directory (from harbor Task)
            environment_name: Name of the environment
            session_id: Session ID for this trial
            trial_paths: Trial paths for output
            task_env_config: Environment configuration from harbor task
            default_timeout_sec: Default command timeout when harbor does not pass one
            network_isolation_enabled: Whether a deny-by-default runtime network policy is active
        """
        self._runtime = runtime
        self._default_timeout_sec = default_timeout_sec
        self._network_isolation_enabled = network_isolation_enabled
        self._harbor_network_policy = harbor_network_policy
        self._gpus_enabled = gpus_enabled

        # Call parent init
        super().__init__(
            environment_dir=environment_dir,
            environment_name=environment_name,
            session_id=session_id,
            trial_paths=trial_paths,
            task_env_config=task_env_config,
            logger=logger_instance,
            network_policy=harbor_network_policy,
            **kwargs,
        )

    @staticmethod
    def type() -> str:
        return 'gongfeng'

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        """Gongfeng capabilities for the policy applied to this runtime."""
        network_mode = getattr(getattr(self, '_harbor_network_policy', None), 'network_mode', None)
        network_mode_value = getattr(network_mode, 'value', network_mode)
        allowlist_enabled = network_mode_value == 'allowlist'
        return EnvironmentCapabilities(
            gpus=getattr(self, '_gpus_enabled', False),
            disable_internet=self._network_isolation_enabled,
            network_allowlist=allowlist_enabled,
            network_allowlist_hostnames=allowlist_enabled,
            network_allowlist_wildcard_hostnames=allowlist_enabled,
            network_allowlist_ipv4_addresses=allowlist_enabled,
            network_allowlist_ipv4_cidrs=allowlist_enabled,
            mounted=False,  # Not mounted - files need upload/download
            windows=False,
            docker_compose=False,
        )

    def _validate_definition(self):
        """No filesystem validation needed for gongfeng environment."""
        pass

    async def start(self, force_build: bool = False) -> None:
        """Start the environment - runtime is already started by nexus."""
        logger.debug(f'GongfengEnvironment started for {self.environment_name}')

    async def stop(self, delete: bool = True) -> None:
        """Stop the environment - nexus manages runtime lifecycle."""
        logger.debug(f'GongfengEnvironment stopped for {self.environment_name}')

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        """Execute a command in the gongfeng sandbox.

        Translates harbor's exec interface to nexus Runtime.run_command().
        """
        user = self._resolve_user(user)
        if isinstance(user, int):
            raise ValueError(f'Numeric UID {user} is not supported by GongfengEnvironment; use a username.')
        merged_env = self._merge_env(env)

        # Do not inline verifier secrets into the command: nexus runtime logs command
        # text for diagnostics. Upload a short-lived env file inside a root-only
        # directory, source it in the sandbox, and log only its random path.
        remote_env_dir: str | None = None
        remote_env_path: str | None = None
        exec_result: ExecResult | None = None
        cleanup_error = ''

        try:
            if merged_env:
                invalid_names = [name for name in merged_env if not _ENV_NAME_RE.fullmatch(name)]
                if invalid_names:
                    raise ValueError(f'Invalid environment variable name(s): {invalid_names}')

                remote_env_dir = f'/tmp/.harbor_exec_env_{uuid4().hex}'
                remote_env_path = f'{remote_env_dir}/env'
                setup_result = await self._runtime.run_command(
                    command=f'install -d -m 700 {shlex.quote(remote_env_dir)}',
                    timeout=10,
                )
                if setup_result.return_code != 0:
                    raise RuntimeError(
                        f'Failed to create protected env directory: '
                        f'{setup_result.stderr or setup_result.stdout or "unknown error"}'
                    )

                with tempfile.NamedTemporaryFile(
                    mode='w',
                    encoding='utf-8',
                    prefix='harbor_exec_env_',
                    delete=False,
                ) as env_file:
                    for key, value in merged_env.items():
                        env_file.write(f'export {key}={shlex.quote(str(value))}\n')
                    local_env_path = Path(env_file.name)
                try:
                    from nexus.extensions.tasks.terminal.utils import read_local_file_to_file_item

                    upload_response = await self._runtime.upload_file(
                        files=[read_local_file_to_file_item(local_env_path, remote_env_path)],
                        overwrite=True,
                    )
                finally:
                    local_env_path.unlink(missing_ok=True)

                raise_for_upload_failures(upload_response, 'verifier env file')

                permission_commands = [
                    f'chmod 700 {shlex.quote(remote_env_dir)}',
                    f'chmod 600 {shlex.quote(remote_env_path)}',
                ]
                if user is not None and str(user) not in ('root', '0'):
                    owner = shlex.quote(str(user))
                    permission_commands.insert(
                        0,
                        f'chown {owner} {shlex.quote(remote_env_dir)} {shlex.quote(remote_env_path)}',
                    )
                permission_result = await self._runtime.run_command(
                    command=' && '.join(permission_commands),
                    timeout=10,
                )
                if permission_result.return_code != 0:
                    raise RuntimeError(
                        f'Failed to secure verifier env file: '
                        f'{permission_result.stderr or permission_result.stdout or "unknown error"}'
                    )

            # Build the full command without plaintext env values.
            inner_command = self._build_full_command(command, cwd=cwd)
            if remote_env_path:
                inner_command = f'set -a && . {shlex.quote(remote_env_path)} && set +a && {inner_command}'
            full_command = inner_command
            if user and user != 'root':
                full_command = f'su - {shlex.quote(str(user))} -c {shlex.quote(inner_command)}'

            timeout = timeout_sec if timeout_sec is not None else self._default_timeout_sec
            result = await self._runtime.run_command(
                command=full_command,
                timeout=timeout,
            )

            if result.return_code is not None:
                return_code = int(result.return_code)
            elif result.status == CommandRunStatus.TimeLimitExceeded:
                return_code = 124
            elif result.status == CommandRunStatus.Error:
                return_code = 1
            else:
                raise RuntimeError(
                    f'Finished command returned no exit code: {result.stderr or result.stdout or "no output"}'
                )

            stderr = result.stderr or ''
            if result.return_code is None:
                status = getattr(result.status, 'value', result.status)
                status_detail = f'runtime command status={status}; no process exit code was returned'
                stderr = f'{stderr}\n{status_detail}'.strip()
            exec_result = ExecResult(
                stdout=result.stdout or '',
                stderr=stderr,
                return_code=return_code,
            )
        except Exception:
            logger.exception('Runtime command transport failed')
            raise
        finally:
            if remote_env_dir:
                try:
                    cleanup_result = await self._runtime.run_command(
                        command=f'rm -rf {shlex.quote(remote_env_dir)}',
                        timeout=10,
                    )
                    if cleanup_result.return_code != 0:
                        cleanup_error = cleanup_result.stderr or cleanup_result.stdout or 'unknown cleanup error'
                except Exception as cleanup_exc:
                    cleanup_error = str(cleanup_exc)

        if cleanup_error:
            logger.error(f'Failed to remove verifier env directory: {cleanup_error}')
            if exec_result is not None and exec_result.return_code == 0:
                raise RuntimeError(f'Verifier env cleanup failed: {cleanup_error}')

        if exec_result is None:
            raise RuntimeError('Command execution produced no result')
        return exec_result

    def _build_full_command(
        self,
        command: str,
        cwd: str | None = None,
    ) -> str:
        """Prefix a sandbox command with its working directory when provided."""
        return f'cd {shlex.quote(cwd)} && {command}' if cwd else command

    async def _run_checked(self, *, command: str, timeout: float, operation: str) -> Any:
        """Run a sandbox command and raise on a nonzero exit status."""
        result = await self._runtime.run_command(command=command, timeout=timeout)
        if result.return_code != 0:
            details = result.stderr or result.stdout or 'no output'
            raise RuntimeError(f'{operation} failed with code {result.return_code}: {details}')
        return result

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        """Upload a single file through Nexus's native file API."""
        from nexus.extensions.tasks.terminal.utils import read_local_file_to_file_item

        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f'Source file not found: {source}')
        if not source.is_file():
            raise ValueError(f'Source path is not a file: {source}')

        file_item = await asyncio.to_thread(read_local_file_to_file_item, source, target_path)
        response = await self._runtime.upload_file(files=[file_item], overwrite=True)
        raise_for_upload_failures(response, f'file {source}')

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        """Upload a complete directory tree through a UUID-scoped tar archive."""
        source = Path(source_dir)
        if not source.exists():
            raise FileNotFoundError(f'Source directory not found: {source}')
        if not source.is_dir():
            raise NotADirectoryError(f'Source path is not a directory: {source}')

        remote_tar = f'/tmp/.harbor_upload_{uuid4().hex}.tar.gz'
        primary_error: BaseException | None = None

        try:
            archive_item = await asyncio.to_thread(_build_directory_archive_item, source, remote_tar)
            response = await self._runtime.upload_file(files=[archive_item], overwrite=True)
            raise_for_upload_failures(response, f'directory archive {source}')

            await self._run_checked(
                command=(
                    f'mkdir -p {shlex.quote(target_dir)} && '
                    f'tar xzf {shlex.quote(remote_tar)} -C {shlex.quote(target_dir)}'
                ),
                timeout=120,
                operation=f'Extract directory archive into {target_dir}',
            )
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            try:
                await self._run_checked(
                    command=f'rm -f {shlex.quote(remote_tar)}',
                    timeout=10,
                    operation='Clean up remote upload archive',
                )
            except Exception:
                if primary_error is None:
                    raise
                logger.exception('Failed to clean up remote upload archive after transfer failure')

    async def _download_runtime_file(self, source_path: str) -> bytes:
        """Download one binary file through Nexus's native file API."""
        from nexus.runtime.executors.models.file import FileItem

        try:
            response = await self._runtime.download_file(
                files=[FileItem(path=source_path, encoding='base64')],
            )
        except Exception as exc:
            raise RuntimeError(f'Download remote file {source_path} failed: {exc}') from exc

        errors = getattr(response, 'errors', None) or []
        if errors:
            details = '; '.join(
                f'{getattr(error, "path", source_path)}: {getattr(error, "message", None) or "unknown error"}'
                for error in errors
            )
            raise RuntimeError(f'Download remote file {source_path} failed: {details}')

        files = getattr(response, 'files', None) or []
        if len(files) != 1:
            raise RuntimeError(f'Download remote file {source_path} returned {len(files)} files, expected 1')

        content = getattr(files[0], 'content', None)
        if content is None:
            raise RuntimeError(f'Download remote file {source_path} returned no content')
        if not isinstance(content, (str, bytes)):
            raise TypeError(
                f'Download remote file {source_path} returned invalid content type {type(content).__name__}'
            )

        return await asyncio.to_thread(_decode_base64_file_content, content, source_path)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        """Download a file from the gongfeng sandbox through the native file API."""
        target = Path(target_path)
        await asyncio.to_thread(target.parent.mkdir, parents=True, exist_ok=True)
        content = await self._download_runtime_file(source_path)
        await asyncio.to_thread(target.write_bytes, content)

    async def download_file_tail(
        self,
        source_path: str,
        target_path: Path | str,
        *,
        max_bytes: int,
    ) -> None:
        """Download only the final bytes of a remote diagnostic file."""
        if max_bytes <= 0:
            raise ValueError(f'max_bytes must be positive, got {max_bytes}')

        remote_tail = f'/tmp/.harbor_tail_{uuid4().hex}'
        try:
            await self._run_checked(
                command=(f'tail -c {max_bytes} {shlex.quote(source_path)} > {shlex.quote(remote_tail)}'),
                timeout=30,
                operation=f'Read tail of remote file {source_path}',
            )
            await self.download_file(remote_tail, target_path)
        finally:
            try:
                await self._run_checked(
                    command=f'rm -f {shlex.quote(remote_tail)}',
                    timeout=10,
                    operation='Clean up remote diagnostic preview',
                )
            except Exception:
                logger.warning('Failed to clean up remote diagnostic preview %s', remote_tail, exc_info=True)

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        """Download a directory from the gongfeng sandbox using tar."""
        target = Path(target_dir)
        await asyncio.to_thread(target.mkdir, parents=True, exist_ok=True)

        remote_tar = f'/tmp/.harbor_download_{uuid4().hex}.tar.gz'
        primary_error: BaseException | None = None

        try:
            await self._run_checked(
                command=f'tar czf {shlex.quote(remote_tar)} -C {shlex.quote(source_dir)} .',
                timeout=120,
                operation=f'Archive remote directory {source_dir}',
            )
            tar_content = await self._download_runtime_file(remote_tar)
            if not tar_content:
                raise RuntimeError(f'Downloaded directory archive for {source_dir} is empty')
            await asyncio.to_thread(_extract_directory_archive, tar_content, target)
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            try:
                await self._run_checked(
                    command=f'rm -f {shlex.quote(remote_tar)}',
                    timeout=10,
                    operation='Clean up remote download archive',
                )
            except Exception:
                if primary_error is None:
                    raise
                logger.exception('Failed to clean up remote download archive after transfer failure')
