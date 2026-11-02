from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import sys
import tempfile

from dataclasses import dataclass
from pathlib import Path

from nexus.distributed.artifacts.files import file_identity
from nexus.distributed.artifacts.gateway import ApplicationGatewayArtifactTransport, ArtifactContentReader
from nexus.distributed.artifacts.models import ArtifactRef
from nexus.distributed.artifacts.ports import AllocationArtifactReader
from nexus.distributed.config import DistributedConfig
from nexus.distributed.dependency_build.wheelhouse import (
    WheelhouseMetadata,
    extract_wheelhouse_archive,
)
from nexus.distributed.environment.limits import source_package_limits_from_allocation_spec
from nexus.distributed.observability.event_loop import monitor_event_loop
from nexus.distributed.submission.source_package import PackageLimits, safe_extract_source_package
from nexus.distributed.submission.specs import resolve_source_package_extras
from nexus.distributed.utils.clock import monotonic_deadline_from_iso8601
from nexus.runtime.server.venv.manager import VenvManager
from nexus.runtime.server.venv.models import ArtifactSpec, VenvSpec


_MAX_WHEELHOUSE_FILES = 10_000


@dataclass(frozen=True, slots=True)
class PreparedEnvironment:
    workspace: Path
    venv: Path
    environment_id: str
    pool_id: str
    workload: str
    manager_labels: dict[str, str]
    python: Path | None = None


async def prepare_allocation_environment(
    allocation_id: str,
    *,
    environment_id: str,
    pool_id: str,
    workload: str,
    allocation_spec: dict,
    cache_root: Path,
    config: DistributedConfig,
    reader: AllocationArtifactReader | None = None,
    image_workspace: Path = Path('/workspace'),
) -> PreparedEnvironment:
    root = cache_root / allocation_id
    code = allocation_spec.get('code') or {'mode': 'config_only'}
    if code.get('mode') == 'image':
        if not image_workspace.is_dir():
            raise RuntimeError('image_workspace_unavailable')
        if not os.access(image_workspace, os.W_OK):
            image_workspace = await asyncio.to_thread(
                _copy_image_workspace,
                image_workspace,
                root,
                allocation_spec.get('workspace_size_bytes', PackageLimits().max_total_bytes),
            )
        return PreparedEnvironment(
            workspace=image_workspace,
            venv=Path(sys.prefix),
            python=Path(sys.executable),
            environment_id=environment_id,
            pool_id=pool_id,
            workload=workload,
            manager_labels={},
        )

    workspace = root / 'workspace'
    workspace.mkdir(parents=True, exist_ok=True)
    if code.get('mode') == 'config_only':
        return PreparedEnvironment(
            workspace=workspace,
            venv=Path(sys.prefix),
            python=Path(sys.executable),
            environment_id=environment_id,
            pool_id=pool_id,
            workload=workload,
            manager_labels={},
        )
    if code.get('mode') != 'source_package':
        raise ValueError('unsupported_allocation_code_mode')

    source_ref = ArtifactRef.model_validate(code.get('source_package'))
    nexus_wheel_ref = ArtifactRef.model_validate(code.get('nexus_wheel'))
    uv_lock_ref = ArtifactRef.model_validate(code.get('uv_lock'))
    if source_ref.kind != 'source_package' or nexus_wheel_ref.kind != 'nexus_wheel' or uv_lock_ref.kind != 'uv_lock':
        raise ValueError('allocation_code_artifact_kind_mismatch')
    raw_extras = code.get('extras') or []
    if not isinstance(raw_extras, list) or not all(isinstance(item, str) for item in raw_extras):
        raise ValueError('allocation_code_extras_invalid')
    extras = list(raw_extras)
    manager_extras = _manager_venv_extras(extras)
    wheelhouse_ref, wheelhouse_metadata = _ready_dependency_build(allocation_spec)
    token_file_value = os.getenv('NEXUS_DIST_ARTIFACT_READ_TOKEN_FILE')
    owned_transport: ApplicationGatewayArtifactTransport | None = None
    if reader is None:
        if not token_file_value:
            raise RuntimeError('artifact_read_credential_unavailable')
        owned_transport = ApplicationGatewayArtifactTransport(
            base_url=config.application_internal_base_url,
            content_ticket_file=Path(token_file_value),
        )
        reader = ArtifactContentReader(owned_transport)
    assert reader is not None
    deadline_value = os.getenv('NEXUS_DIST_ARTIFACT_DOWNLOAD_DEADLINE_AT', '').strip()
    download_deadline = monotonic_deadline_from_iso8601(deadline_value) if deadline_value else None
    artifacts = root / 'artifacts'
    artifacts.mkdir(parents=True, exist_ok=True)
    try:
        source_archive = artifacts / 'source.tar.zst'
        await _ensure_gateway_download(reader, source_archive, source_ref, deadline=download_deadline)
        wheelhouse_archive = artifacts / 'wheelhouse.tar.zst'
        await _ensure_gateway_download(reader, wheelhouse_archive, wheelhouse_ref, deadline=download_deadline)
        wheelhouse_root = artifacts / 'wheelhouse'
        nexus_wheel_path = await asyncio.to_thread(
            _extract_wheelhouse,
            wheelhouse_archive,
            wheelhouse_root,
            wheelhouse_metadata.nexus_wheel_path,
            max_total_bytes=_workspace_size_bytes(allocation_spec),
        )

        package_limits = source_package_limits_from_allocation_spec(allocation_spec)
        limits_identity = json.dumps(
            allocation_spec.get('source_package_limits') or {},
            sort_keys=True,
            separators=(',', ':'),
        )
        identity = (
            'dependency-check-v1\n'
            f'{source_ref.sha256}\n{wheelhouse_ref.sha256}\n{wheelhouse_metadata.build_key}\n'
            f'{canonicalize_extras(manager_extras)}\n{limits_identity}\n'
        )
        marker = root / '.prepared'
        manager_venv = root / 'manager-venvs' / 'manager'
        python = manager_venv / 'bin' / 'python'
        prepared = await asyncio.to_thread(_is_prepared, marker, identity, python)
        if not prepared:
            await asyncio.to_thread(marker.unlink, missing_ok=True)
            await asyncio.to_thread(_reset_workspace, workspace, source_archive, package_limits)
            manager = VenvManager(root=root / 'manager-venvs')
            await manager.destroy('manager')
            status = await manager.create(
                'manager',
                VenvSpec(
                    python_version=f'{sys.version_info.major}.{sys.version_info.minor}',
                    artifacts=[
                        ArtifactSpec(
                            name='nexus-agent',
                            whl_url=nexus_wheel_path.resolve().as_uri(),
                            whl_sha256=nexus_wheel_ref.sha256,
                            extras=manager_extras,
                        )
                    ],
                    extra_requirements=wheelhouse_metadata.module_host_requirements,
                    modules=[],
                ),
                wheelhouse=wheelhouse_root / 'wheels',
            )
            if status.state != 'ready':
                raise RuntimeError(f'manager_venv_create_failed:{status.error or "unknown"}')
            await _check_manager_dependencies(python)
            await asyncio.to_thread(marker.write_text, identity, encoding='utf-8')
        return PreparedEnvironment(
            workspace=workspace,
            venv=manager_venv,
            python=python,
            environment_id=environment_id,
            pool_id=pool_id,
            workload=workload,
            manager_labels={},
        )
    finally:
        if owned_transport is not None:
            await owned_transport.close()


def _ready_dependency_build(
    allocation_spec: dict,
) -> tuple[ArtifactRef, WheelhouseMetadata]:
    dependency_build = allocation_spec.get('dependency_build')
    if not isinstance(dependency_build, dict) or not dependency_build.get('dependency_build_id'):
        raise ValueError('allocation_dependency_build_unavailable')
    wheelhouse_ref = ArtifactRef.model_validate(dependency_build.get('wheelhouse'))
    if wheelhouse_ref.kind != 'wheelhouse':
        raise ValueError('allocation_dependency_build_artifact_kind_mismatch')
    metadata = WheelhouseMetadata.model_validate(dependency_build.get('metadata'))
    if dependency_build.get('build_key') != metadata.build_key:
        raise ValueError('allocation_dependency_build_key_mismatch')
    return wheelhouse_ref, metadata


def _copy_image_workspace(source: Path, root: Path, max_bytes: int) -> Path:
    """Prepare a writable process cwd when the workload image filesystem is read-only."""
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise ValueError('allocation_workspace_size_invalid')
    workspace = root / 'workspace'
    marker = workspace / '.nexus-image-workspace-ready'
    if marker.is_file():
        return workspace
    total = 0
    for path in source.rglob('*'):
        if path.is_symlink():
            continue
        if path.is_file():
            total += path.stat().st_size
            if total > max_bytes:
                raise ValueError('image_workspace_exceeds_byte_limit')
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.image-workspace-', dir=root) as temporary:
        staged = Path(temporary) / 'workspace'
        shutil.copytree(source, staged, symlinks=True)
        staged.chmod(staged.stat().st_mode | 0o700)
        (staged / marker.name).touch()
        staged.rename(workspace)
    return workspace


def _workspace_size_bytes(allocation_spec: dict) -> int:
    value = allocation_spec.get('workspace_size_bytes')
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError('allocation_workspace_size_invalid')
    return value


async def _ensure_gateway_download(
    reader: AllocationArtifactReader,
    path: Path,
    ref: ArtifactRef,
    *,
    deadline: float | None,
) -> None:
    if path.is_file() and await asyncio.to_thread(file_identity, path) == (ref.size_bytes, ref.sha256):
        return
    await reader.download(ref, path, deadline=deadline)


def _extract_wheelhouse(
    archive: Path,
    target: Path,
    nexus_wheel_relative_path: str,
    *,
    max_total_bytes: int,
) -> Path:
    extract_wheelhouse_archive(
        archive,
        target,
        max_files=_MAX_WHEELHOUSE_FILES,
        max_total_bytes=max_total_bytes,
    )
    nexus_wheel_path = target / nexus_wheel_relative_path
    if not nexus_wheel_path.is_file() or nexus_wheel_path.is_symlink():
        raise ValueError('allocation_nexus_wheel_missing_from_wheelhouse')
    return nexus_wheel_path


async def _check_manager_dependencies(python: Path) -> None:
    process = await asyncio.create_subprocess_exec(
        VenvManager._resolve_uv(),
        'pip',
        'check',
        '--no-config',
        '--python',
        str(python),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=VenvManager._uv_subprocess_env(),
    )
    output, _ = await process.communicate()
    if process.returncode != 0:
        detail = output.decode(errors='replace')[-4096:].strip()
        raise RuntimeError(f'manager_dependency_check_failed: {detail}')


def _is_prepared(marker: Path, identity: str, python: Path) -> bool:
    return marker.is_file() and marker.read_text(encoding='utf-8') == identity and python.is_file()


def _reset_workspace(workspace: Path, source_archive: Path, package_limits: PackageLimits) -> None:
    if workspace.exists():
        import shutil

        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)
    safe_extract_source_package(source_archive, workspace, limits=package_limits)


def canonicalize_extras(extras: list[str]) -> str:
    return ','.join(sorted(extras))


def _manager_venv_extras(extras: list[str]) -> list[str]:
    return resolve_source_package_extras(extras)


def exec_instance_manager(
    *,
    prepared: PreparedEnvironment,
    port: int,
    capacity: int,
    allocation_id: str,
    allocation_spec_hash: str | None = None,
    require_allocation_liveness_v1: bool = False,
    request_policy: dict | None = None,
) -> None:
    pod_name = os.getenv('POD_NAME') or os.getenv('HOSTNAME') or f'im-{allocation_id[:12]}'
    manager_id = _dynamic_manager_id(allocation_id, pod_name)
    python = prepared.python or (prepared.venv / 'bin' / 'python')
    argv = [
        str(python),
        '-m',
        'nexus.cli.main',
        'distributed',
        'instance-manager',
        '--host',
        '0.0.0.0',
        '--port',
        str(port),
        '--instance-manager-id',
        manager_id,
        '--capacity',
        str(capacity),
        '--environment-id',
        prepared.environment_id,
        '--pool-id',
        prepared.pool_id,
        '--bootstrap-version',
        '1',
        '--artifact-ref-version',
        '1',
        '--workload',
        prepared.workload,
        '--force-replace',
    ]
    if require_allocation_liveness_v1:
        argv.append('--require-allocation-liveness-v1')
    if isinstance(request_policy, dict) and request_policy.get('binding') and request_policy.get('scope'):
        argv.extend([
            '--request-policy-binding',
            str(request_policy['binding']),
            '--request-policy-scope',
            str(request_policy['scope']),
        ])
    labels = {**prepared.manager_labels, 'workload': prepared.workload}
    for key, value in sorted(labels.items()):
        argv.extend(['--label', f'{key}={value}'])
    environment = os.environ.copy()
    environment['PYTHONPATH'] = _workspace_pythonpath(prepared.workspace, environment.get('PYTHONPATH'))
    environment['NEXUS_DIST_MANAGER_ALLOCATION_ID'] = allocation_id
    environment['NEXUS_DIST_MANAGER_ENVIRONMENT_ID'] = prepared.environment_id
    environment['NEXUS_DIST_MANAGER_POOL_ID'] = prepared.pool_id
    environment['NEXUS_DIST_MANAGER_WORKLOAD'] = prepared.workload
    environment['NEXUS_DIST_MANAGER_ALLOCATION_SPEC_HASH'] = allocation_spec_hash or ''
    os.chdir(prepared.workspace)
    os.execve(argv[0], argv, environment)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--allocation-id', required=True)
    parser.add_argument('--require-allocation-liveness-v1', action='store_true')
    parser.add_argument('--environment-id', required=True)
    parser.add_argument('--pool-id', required=True)
    parser.add_argument('--workload', choices=['execution', 'orchestration'], required=True)
    parser.add_argument('--allocation-spec-hash', required=True)
    parser.add_argument('--port', type=int, default=8082)
    parser.add_argument('--capacity', type=int, default=1)
    parser.add_argument('--cache-root', type=Path, default=Path('/data/nexus-environment'))
    args = parser.parse_args()
    config = DistributedConfig.from_env()
    try:
        allocation_spec = json.loads(os.environ['NEXUS_DIST_ALLOCATION_SPEC_JSON'])
    except (KeyError, json.JSONDecodeError) as exc:
        raise RuntimeError('allocation_spec_unavailable') from exc
    from nexus.distributed.environment.liveness import startup_liveness

    startup_liveness(
        allocation_spec,
        required=args.require_allocation_liveness_v1,
        allocation_id=args.allocation_id,
        environment_id=args.environment_id,
        pool_id=args.pool_id,
        workload=args.workload,
    )
    prepared = asyncio.run(
        monitor_event_loop(
            'environment-bootstrap',
            prepare_allocation_environment(
                args.allocation_id,
                environment_id=args.environment_id,
                pool_id=args.pool_id,
                workload=args.workload,
                allocation_spec=allocation_spec,
                cache_root=args.cache_root,
                config=config,
            ),
        )
    )
    exec_instance_manager(
        prepared=prepared,
        port=args.port,
        capacity=args.capacity,
        allocation_id=args.allocation_id,
        allocation_spec_hash=args.allocation_spec_hash,
        require_allocation_liveness_v1=args.require_allocation_liveness_v1,
        request_policy=allocation_spec.get('request_policy')
        if isinstance(allocation_spec.get('request_policy'), dict)
        else None,
    )


def _workspace_pythonpath(workspace: Path, existing: str | None) -> str:
    return os.pathsep.join([str(workspace), existing]) if existing else str(workspace)


def _dynamic_manager_id(allocation_id: str, pod_name: str) -> str:
    match = re.search(r'-(\d+)$', pod_name)
    if match is None:
        raise ValueError('pod_ordinal_unavailable')
    return f'{allocation_id}-{int(match.group(1))}'


if __name__ == '__main__':
    main()
