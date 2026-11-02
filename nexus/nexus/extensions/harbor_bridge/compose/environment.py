"""Harbor environment routing for a controller-owned Compose topology."""

from __future__ import annotations

import base64
import posixpath
import shlex
import tempfile

from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from harbor.environments.base import ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.utils.path_filter import filter_paths_by_patterns

from nexus.extensions.harbor_bridge.environment import GongfengEnvironment
from nexus.runtime.executors.models.file import (
    DownloadFileResponse,
    FileItem,
    FileOperationResult,
    UploadFileResponse,
)
from nexus.runtime.topology import ServiceHandle, TopologyHandle


class ComposeServiceRuntime:
    """Bounded Runtime-like command/file adapter for one Compose service."""

    def __init__(self, service: ServiceHandle) -> None:
        self.service = service

    async def run_command(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout: float = 180,
        stdin: str | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> Any:
        return await self.service.exec(
            command,
            cwd=cwd,
            env=extra_env,
            timeout=timeout,
            stdin=stdin,
        )

    async def upload_file(
        self,
        files: list[FileItem],
        *,
        overwrite: bool = True,
    ) -> UploadFileResponse:
        if not overwrite:
            raise ValueError('Compose service uploads require overwrite=true')
        results: list[FileOperationResult] = []
        for item in files:
            temp_path: Path | None = None
            try:
                if item.content is None:
                    raise ValueError('Upload FileItem must contain content')
                content = (
                    base64.b64decode(item.content, validate=True)
                    if item.encoding == 'base64'
                    else item.content.encode('utf-8')
                )
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_file.write(content)
                    temp_path = Path(temp_file.name)
                await self.service.upload_file(temp_path, item.path)
                results.append(FileOperationResult(path=item.path, success=True))
            except Exception as exc:
                results.append(FileOperationResult(path=item.path, success=False, message=str(exc)))
            finally:
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)
        return UploadFileResponse(results=results)

    async def download_file(self, files: list[FileItem]) -> DownloadFileResponse:
        downloaded: list[FileItem] = []
        errors: list[FileOperationResult] = []
        for item in files:
            temp_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_path = Path(temp_file.name)
                await self.service.download_file(item.path, temp_path)
                content = temp_path.read_bytes()
                encoded = (
                    base64.b64encode(content).decode('ascii') if item.encoding == 'base64' else content.decode('utf-8')
                )
                downloaded.append(FileItem(path=item.path, content=encoded, encoding=item.encoding))
            except Exception as exc:
                errors.append(FileOperationResult(path=item.path, success=False, message=str(exc)))
            finally:
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)
        return DownloadFileResponse(files=downloaded, errors=errors or None)


class ComposeHarborEnvironment(GongfengEnvironment):
    """Harbor BaseEnvironment backed by named services in a Compose topology."""

    def __init__(
        self,
        *,
        topology: TopologyHandle,
        primary_service: str,
        persistent_environment: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        self._topology = topology
        self._primary_service = primary_service
        self._primary_runtime = ComposeServiceRuntime(topology.service(primary_service))
        super().__init__(
            runtime=self._primary_runtime,
            persistent_env=persistent_environment,
            **kwargs,
        )

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return super().capabilities.model_copy(update={'docker_compose': True})

    @property
    def _uses_compose(self) -> bool:
        return True

    @property
    def primary_runtime(self) -> ComposeServiceRuntime:
        return self._primary_runtime

    def _service(self, service: str | None) -> ServiceHandle:
        return self._topology.service(self._primary_service if self.is_main_service(service) else str(service))

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        """Execute in the primary service without creating an env file in its ``/tmp``."""
        return await self.service_exec(
            command,
            service=None,
            cwd=cwd,
            env=env,
            timeout_sec=timeout_sec,
            user=user,
        )

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        """Upload through Controller staging rather than the primary service's ``/tmp``."""
        await self._service(None).upload_dir(source_dir, target_dir)

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        """Download through Controller staging rather than the primary service's ``/tmp``."""
        await self._service(None).download_dir(source_dir, target_dir)

    async def _download_dir_filtered_from_service(
        self,
        *,
        source_dir: str,
        target_dir: Path | str,
        service: str | None,
        include: Sequence[str] | None = None,
        exclude: Sequence[str] | None = None,
        protect: Sequence[str] | None = None,
    ) -> None:
        """List in-service files, filter on Host, and copy selected files via Controller staging."""
        quoted_source = shlex.quote(source_dir)
        result = await self.service_exec(
            (
                'find_bin=/nix/runtime/nix-env/bin/find; '
                'if [ ! -x "$find_bin" ]; then '
                'find_bin="$(command -v find)" || { '
                'printf "%s\\n" "Required tool not found: find" >&2; exit 127; }; '
                'fi; '
                f'cd {quoted_source} && "$find_bin" . -type f'
            ),
            service=service,
            timeout_sec=120,
            user='root',
        )
        if result.return_code != 0:
            details = result.stderr or result.stdout or 'no output'
            raise RuntimeError(f'Failed to list files in {source_dir!r}: {details}')

        paths: list[str] = []
        for line in (result.stdout or '').splitlines():
            relative = line.strip().removeprefix('./')
            relative_path = PurePosixPath(relative)
            if not relative or relative_path.is_absolute() or '..' in relative_path.parts:
                continue
            paths.append(relative_path.as_posix())

        selected = filter_paths_by_patterns(paths, include=include, exclude=exclude)
        if protect:
            selected_set = set(selected)
            selected.extend(path for path in paths if path in set(protect) - selected_set)
        if not selected:
            self.logger.warning(
                f'No files in {source_dir!r} matched include={include} exclude={exclude}; downloading nothing'
            )
            return

        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        handle = self._service(service)
        for relative in selected:
            relative_path = PurePosixPath(relative)
            destination = target.joinpath(*relative_path.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            await handle.download_file(
                posixpath.join(source_dir.rstrip('/') or '/', relative_path.as_posix()),
                destination,
            )

    async def download_dir_filtered(
        self,
        *,
        source_dir: str,
        target_dir: Path | str,
        include: Sequence[str] | None = None,
        exclude: Sequence[str] | None = None,
        protect: Sequence[str] | None = None,
    ) -> None:
        await self._download_dir_filtered_from_service(
            source_dir=source_dir,
            target_dir=target_dir,
            service=None,
            include=include,
            exclude=exclude,
            protect=protect,
        )

    async def download_dir_with_exclusions(
        self,
        *,
        source_dir: str,
        target_dir: Path | str,
        exclude: list[str],
    ) -> None:
        await self._download_dir_filtered_from_service(
            source_dir=source_dir,
            target_dir=target_dir,
            service=None,
            exclude=exclude,
        )

    async def service_download_dir_with_exclusions(
        self,
        *,
        source_dir: str,
        target_dir: Path | str,
        exclude: list[str],
        service: str | None = None,
    ) -> None:
        await self._download_dir_filtered_from_service(
            source_dir=source_dir,
            target_dir=target_dir,
            service=service,
            exclude=exclude,
        )

    async def download_file_tail(
        self,
        source_path: str,
        target_path: Path | str,
        *,
        max_bytes: int,
    ) -> None:
        """Read a diagnostic tail over Compose exec without a service-side temporary file."""
        if max_bytes <= 0:
            raise ValueError(f'max_bytes must be positive, got {max_bytes}')
        runtime_tail = '/nix/runtime/nix-env/bin/tail'
        tail_command = (
            f'tail_bin={shlex.quote(runtime_tail)}; '
            'if [ ! -x "$tail_bin" ]; then '
            'tail_bin="$(command -v tail)" || { '
            'printf "%s\\n" "Required tool not found: tail" >&2; exit 127; }; '
            'fi; '
            f'"$tail_bin" -c {max_bytes} {shlex.quote(source_path)}'
        )
        result = await self._service(None).exec(
            tail_command,
            timeout=30,
        )
        if result.return_code is None:
            raise RuntimeError(f'Read tail of Compose file {source_path} did not return an exit code')
        if result.return_code != 0:
            details = result.stderr or result.stdout or 'no output'
            raise RuntimeError(f'Read tail of Compose file {source_path} failed: {details}')
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(result.stdout or '', encoding='utf-8')

    async def service_exec(
        self,
        command: str,
        *,
        service: str | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        resolved_user = self._resolve_user(user)
        if isinstance(resolved_user, int):
            raise ValueError(f'Numeric UID {resolved_user} is not supported by Compose; use a username.')
        service_env = self._merge_env(env) if self.is_main_service(service) else (dict(env) if env else None)
        result = await self._service(service).exec(
            command,
            cwd=cwd,
            env=service_env,
            user=resolved_user,
            timeout=timeout_sec if timeout_sec is not None else self._default_timeout_sec,
        )
        if result.return_code is None:
            service_name = self._primary_service if self.is_main_service(service) else str(service)
            details = result.stderr or result.stdout or 'no output'
            raise RuntimeError(f'Compose service {service_name!r} command did not return an exit code: {details}')
        return ExecResult(
            stdout=result.stdout or '',
            stderr=result.stderr or '',
            return_code=int(result.return_code),
        )

    async def service_download_file(
        self,
        source_path: str,
        target_path: Path | str,
        *,
        service: str | None = None,
    ) -> None:
        await self._service(service).download_file(source_path, target_path)

    async def service_download_dir(
        self,
        source_dir: str,
        target_dir: Path | str,
        *,
        service: str | None = None,
    ) -> None:
        await self._service(service).download_dir(source_dir, target_dir)

    async def stop_service(self, service: str) -> None:
        await self._service(service).stop()


__all__ = ['ComposeHarborEnvironment', 'ComposeServiceRuntime']
