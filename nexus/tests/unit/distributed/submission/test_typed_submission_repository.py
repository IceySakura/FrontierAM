from __future__ import annotations

import json

import pytest

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from nexus.distributed.access.run_trace_backends import RunTraceBackendAccessPolicy
from nexus.distributed.config import DistributedConfig
from nexus.distributed.errors import DomainConflictError, DomainValidationError
from nexus.distributed.persistence.orm import (
    ArtifactRefRow,
    ArtifactRow,
    BatchRunRow,
    DependencyBuildRow,
    EnvironmentAllocationRow,
    EnvironmentGrantRow,
    EnvironmentResourceUsageRow,
    EnvironmentRow,
    ExecutionCredentialRow,
    ExecutionRow,
    QuotaUsageRow,
    create_schema,
)
from nexus.distributed.repositories.access import AccessRepository
from nexus.distributed.repositories.environment import EnvironmentRepository
from nexus.distributed.repositories.provider_config import ProviderConfigRepository
from nexus.distributed.repositories.quota import QuotaExceededError
from nexus.distributed.repositories.run_trace_backend import RunTraceBackendGrantInput
from nexus.distributed.repositories.submission import SubmissionRepository, _resolve_submission_cpu_memory_ratio
from nexus.distributed.submission.specs import BatchSubmission, LiveSubmission, SingleSubmission
from nexus.distributed.utils.clock import utcnow
from nexus.distributed.utils.identity import canonical_sha256


GIB = 1024**3


def test_server_cpu_default_resolves_both_batch_managers_without_changing_request():
    submission = BatchSubmission.model_validate({
        'idempotency_key': 'auto-cpu-batch',
        'environment_id': 'env-dynamic',
        'code': {'mode': 'config_only'},
        'resources': {
            'batch_manager': {'cpu_cores': 1, 'memory_gib': 8},
            'managers': {'cpu_cores': 1, 'memory_gib': 17, 'count': 3, 'slots': 20},
        },
        'generation': {
            'mode': 'remote_materialize',
            'source': {
                'type': 'compose',
                'compose': {'parts': [{'kind': 'set', 'value': 'task_name=identity', 'source': 'test'}]},
            },
        },
    })
    resolved = _resolve_submission_cpu_memory_ratio(submission, auto_increase_cpu=True)
    normalized = resolved.resources.normalized()
    assert normalized.batch_manager.cpu_millicores == 2000
    assert normalized.managers.cpu_millicores == 15000
    assert submission.resources.batch_manager.cpu_cores == 1
    assert submission.resources.managers.cpu_cores == 1


def _config(**overrides) -> DistributedConfig:
    overrides.setdefault('environment_enabled_providers', ['kubernetes', 'tkex'])
    return DistributedConfig(**overrides)


@pytest.fixture
async def submission_session_factory(tmp_path):
    engine = create_async_engine(f'sqlite+aiosqlite:///{tmp_path / "submission.db"}')
    await create_schema(engine)
    try:
        yield async_sessionmaker(engine, expire_on_commit=False)
    finally:
        await engine.dispose()


async def _seed_environment(
    session,
    *,
    environment_id: str,
    allocation_type: str,
    provider_type: str,
    environment_limits: dict,
    grant_limits: dict,
    default_timeouts: dict | None = None,
) -> None:
    now = utcnow()
    access = AccessRepository(session)
    if await access.get_user('principal-submit') is None:
        await access.upsert_user(
            principal_id='principal-submit',
            issuer='tai',
            tenant='tenant',
            subject='submit-user',
            roles={'user'},
            display_name=None,
            actor='admin',
            now=now,
        )
    environments = EnvironmentRepository(session)
    if allocation_type == 'dynamic':
        provider_configs = ProviderConfigRepository(session)
        provider_config_ref = f'{provider_type}/default'
        try:
            await provider_configs.get(provider_config_ref)
        except KeyError:
            config = (
                {
                    'provider_type': 'kubernetes',
                    'namespace': 'default',
                    'auth_type': 'in_cluster',
                }
                if provider_type == 'kubernetes'
                else {
                    'provider_type': 'tkex',
                    'namespace': 'default',
                    'base_url': 'https://tkex.example',
                    'api_account': 'test-account',
                    'credential_ref': 'tkex-default',
                    'cluster_name': 'test-cluster',
                    'project_name': 'test-project',
                }
            )
            await provider_configs.create(
                provider_config_ref=provider_config_ref,
                config=config,
                status='enabled',
                actor='admin',
                now=now,
            )
    await environments.create_environment(
        environment_id=environment_id,
        name=environment_id,
        allocation_type=allocation_type,
        provider_type=provider_type,
        provider_config_ref=f'{provider_type}/default' if allocation_type == 'dynamic' else None,
        resource_limits=environment_limits,
        default_timeouts=default_timeouts or {},
        status='enabled',
        actor='admin',
        now=now,
    )
    await environments.upsert_grant(
        principal_id='principal-submit',
        environment_id=environment_id,
        limits=grant_limits,
        enabled=True,
        actor='admin',
        now=now,
    )


def _dynamic_single(
    *,
    run_value: str = 'first',
    idempotency_key: str = 'dynamic-single',
    host_env: list[dict[str, str]] | None = None,
) -> SingleSubmission:
    payload = {
        'submission_version': 1,
        'idempotency_key': idempotency_key,
        'environment_id': 'env-dynamic',
        'code': {'mode': 'config_only'},
        'resources': {
            'managers': {'count': 1, 'cpu_cores': 1.5, 'memory_gib': 2, 'slots': 1},
        },
        'run_input': {'task_name': 'identity', 'task_config': {'value': run_value}},
    }
    if host_env is not None:
        payload['host_env'] = host_env
    return SingleSubmission.model_validate(payload)


def _dynamic_single_with_trace_backends(
    sinks: list[str] | None,
    *,
    idempotency_key: str,
) -> SingleSubmission:
    payload = _dynamic_single(idempotency_key=idempotency_key).model_dump(mode='json', exclude_none=True)
    if sinks is not None:
        payload['run_trace_backends'] = {'sinks': sinks}
    return SingleSubmission.model_validate(payload)


@pytest.mark.parametrize(('server_default', 'client_option'), [(True, None), (False, True)])
async def test_submission_persists_adjusted_cpu_and_keeps_retry_stable(
    submission_session_factory,
    server_default,
    client_option,
):
    async with submission_session_factory() as session:
        await _seed_environment(
            session,
            environment_id='env-dynamic',
            allocation_type='dynamic',
            provider_type='tkex',
            environment_limits={},
            grant_limits={},
        )
        await session.commit()

    payload = _dynamic_single().model_dump(mode='json', exclude_none=True)
    payload['resources']['managers'].update(cpu_cores=1, memory_gib=17)
    if client_option is not None:
        payload['resources']['managers']['auto_increase_cpu'] = client_option
    submission = SingleSubmission.model_validate(payload)
    original_payload = submission.model_dump(mode='json')
    async with submission_session_factory() as session:
        result = await SubmissionRepository(
            session,
            config=_config(submission_auto_increase_cpu=server_default),
        ).create(
            submission,
            actor_principal_id='principal-submit',
            owner_principal_id='principal-submit',
            owner_tenant='tenant',
            now=utcnow(),
        )
        await session.commit()
        allocation = await session.get(EnvironmentAllocationRow, result.execution.allocation_id)
        assert allocation.cpu_millicores == 5000
        assert allocation.memory_bytes == 17 * GIB
        assert allocation.allocation_spec_json['resources']['cpu_millicores'] == 5000
        assert submission.model_dump(mode='json') == original_payload

    async with submission_session_factory() as session:
        retried = await SubmissionRepository(
            session,
            config=_config(submission_auto_increase_cpu=False),
        ).create(
            SingleSubmission.model_validate(payload),
            actor_principal_id='principal-submit',
            owner_principal_id='principal-submit',
            owner_tenant='tenant',
            now=utcnow(),
        )
        assert retried.reused
        assert retried.execution.execution_id == result.execution.execution_id


async def test_server_default_rejects_excess_memory_when_client_option_omitted(submission_session_factory):
    async with submission_session_factory() as session:
        await _seed_environment(
            session,
            environment_id='env-dynamic',
            allocation_type='dynamic',
            provider_type='tkex',
            environment_limits={},
            grant_limits={},
        )
        await session.commit()
        payload = _dynamic_single().model_dump(mode='json', exclude_none=True)
        payload['resources']['managers'].update(cpu_cores=1, memory_gib=16)
        with pytest.raises(DomainValidationError, match='invalid_manager_resources') as raised:
            await SubmissionRepository(session, config=_config(submission_auto_increase_cpu=False)).create(
                SingleSubmission.model_validate(payload),
                actor_principal_id='principal-submit',
                owner_principal_id='principal-submit',
                owner_tenant='tenant',
                now=utcnow(),
            )
        assert 'enable auto_increase_cpu' in raised.value.detail
        assert await session.scalar(select(ExecutionRow)) is None
        assert await session.scalar(select(EnvironmentAllocationRow)) is None


def _trace_backend_config(*, grants_required: bool = False) -> DistributedConfig:
    return _config(
        require_principal_run_trace_backend_grant=grants_required,
        run_trace_sinks={
            'archive': {
                'kind': 'cos',
                'displayName': 'Archive',
                'userSelectable': True,
                'sinkClass': 'nexus.runner.trace.local::RunTraceLocalSink',
                'sourceClass': 'nexus.runner.trace.local::RunTraceLocalSource',
            },
            'primary': {
                'kind': 'tdsql',
                'displayName': 'Primary',
                'userSelectable': True,
                'sinkClass': 'nexus.runner.trace.local::RunTraceLocalSink',
                'sourceClass': 'nexus.runner.trace.local::RunTraceLocalSource',
            },
        },
    )


def _source_package_code() -> dict:
    return {
        'mode': 'source_package',
        'source_package': {
            'artifact_id': 'artifact-source',
            'kind': 'source_package',
            'sha256': 'a' * 64,
            'size_bytes': 101,
        },
        'nexus_wheel': {
            'artifact_id': 'artifact-nexus-wheel',
            'kind': 'nexus_wheel',
            'sha256': 'b' * 64,
            'size_bytes': 102,
        },
        'uv_lock': {
            'artifact_id': 'artifact-uv-lock',
            'kind': 'uv_lock',
            'sha256': 'c' * 64,
            'size_bytes': 103,
        },
        'extras': ['runtime'],
    }


def _add_source_package_artifacts(session, *, environment_id: str, now) -> None:
    for artifact_id, kind, marker, size_bytes, filename in (
        ('artifact-source', 'source_package', 'a', 101, 'source.tar.zst'),
        ('artifact-nexus-wheel', 'nexus_wheel', 'b', 102, 'nexus-agent.whl'),
        ('artifact-uv-lock', 'uv_lock', 'c', 103, 'uv.lock'),
    ):
        session.add(
            ArtifactRow(
                artifact_id=artifact_id,
                owner_principal_id='principal-submit',
                environment_id=environment_id,
                owner_tenant='tenant',
                kind=kind,
                producer_key=f'test:{kind}',
                uri=f'file:///tmp/{filename}',
                sha256=marker * 64,
                size_bytes=size_bytes,
                provider_etag=None,
                status='ready',
                created_by='principal-submit',
                expires_at=None,
                created_at=now,
                updated_at=now,
            )
        )


def _add_local_batch_artifacts(
    session,
    *,
    environment_id: str,
    now,
    task_list_size: int = 120,
    data_list_size: int = 400,
    data_list_owner: str = 'principal-submit',
    data_list_environment: str | None = None,
    data_list_status: str = 'ready',
) -> None:
    for artifact_id, kind, marker, size_bytes, owner, artifact_environment, status, filename in (
        (
            'artifact-task-list',
            'task_list',
            'd',
            task_list_size,
            'principal-submit',
            environment_id,
            'ready',
            'tasks.jsonl',
        ),
        (
            'artifact-data-list',
            'data_list',
            'e',
            data_list_size,
            data_list_owner,
            data_list_environment or environment_id,
            data_list_status,
            'data_list.bin',
        ),
    ):
        session.add(
            ArtifactRow(
                artifact_id=artifact_id,
                owner_principal_id=owner,
                environment_id=artifact_environment,
                owner_tenant='tenant',
                kind=kind,
                producer_key=f'test:{kind}',
                uri=f'file:///tmp/{filename}',
                sha256=marker * 64,
                size_bytes=size_bytes,
                provider_etag=None,
                status=status,
                created_by='principal-submit',
                expires_at=None,
                created_at=now,
                updated_at=now,
            )
        )


def _local_v3_submission(
    *,
    environment_id: str,
    idempotency_key: str,
    task_list_size: int = 120,
    data_list_size: int = 400,
) -> BatchSubmission:
    return BatchSubmission.model_validate({
        'submission_version': 1,
        'idempotency_key': idempotency_key,
        'environment_id': environment_id,
        'code': {'mode': 'config_only'},
        'pool': 'children',
        'generation': {
            'mode': 'local_task_list',
            'task_list': {
                'ref': {
                    'artifact_id': 'artifact-task-list',
                    'kind': 'task_list',
                    'sha256': 'd' * 64,
                    'size_bytes': task_list_size,
                },
                'format': 'jsonl',
                'manifest': {
                    'version': 1,
                    'record_version': 3,
                    'item_count': 2,
                    'size_bytes': task_list_size,
                    'sha256': 'd' * 64,
                    'source': {'mode': 'local_compose'},
                    'datasets': [],
                },
            },
            'data_list': {
                'ref': {
                    'artifact_id': 'artifact-data-list',
                    'kind': 'data_list',
                    'sha256': 'e' * 64,
                    'size_bytes': data_list_size,
                },
                'format': 'nxdl',
                'manifest': {
                    'version': 1,
                    'frame_count': 1,
                    'size_bytes': data_list_size,
                    'sha256': 'e' * 64,
                },
            },
        },
        'batch': {'max_inflight': 4},
    })


async def test_dynamic_submission_rechecks_deployment_provider_allowlist(submission_session_factory) -> None:
    async with submission_session_factory() as session:
        await _seed_environment(
            session,
            environment_id='env-dynamic',
            allocation_type='dynamic',
            provider_type='kubernetes',
            environment_limits={
                'manager_count': 2,
                'cpu_millicores': 4_000,
                'memory_bytes': 8 * GIB,
                'slots': 2,
            },
            grant_limits={
                'manager_count': 2,
                'cpu_millicores': 4_000,
                'memory_bytes': 8 * GIB,
                'slots': 2,
            },
        )
        await session.commit()

    async with submission_session_factory() as session:
        with pytest.raises(DomainConflictError, match='provider_type_not_enabled'):
            await SubmissionRepository(
                session,
                config=_config(environment_enabled_providers=[]),
            ).create(
                _dynamic_single(idempotency_key='provider-disabled'),
                actor_principal_id='principal-submit',
                owner_principal_id='principal-submit',
                owner_tenant='tenant',
                now=utcnow(),
            )
        assert await session.scalar(select(func.count()).select_from(ExecutionRow)) == 0


async def test_submission_resolves_and_persists_authorized_run_trace_backends(
    submission_session_factory,
) -> None:
    config = _trace_backend_config()
    async with submission_session_factory() as session:
        await _seed_environment(
            session,
            environment_id='env-dynamic',
            allocation_type='dynamic',
            provider_type='kubernetes',
            environment_limits={
                'manager_count': 2,
                'cpu_millicores': 4_000,
                'memory_bytes': 8 * GIB,
                'slots': 2,
            },
            grant_limits={
                'manager_count': 2,
                'cpu_millicores': 4_000,
                'memory_bytes': 8 * GIB,
                'slots': 2,
            },
        )
        await RunTraceBackendAccessPolicy(session, config=config).replace_grants(
            'principal-submit',
            [
                RunTraceBackendGrantInput('archive', default_order=1),
                RunTraceBackendGrantInput('primary', default_order=0),
            ],
            actor='admin',
            now=utcnow(),
        )
        await session.commit()

    async with submission_session_factory() as session:
        result = await SubmissionRepository(session, config=config).create(
            _dynamic_single_with_trace_backends(
                ['archive', 'primary'],
                idempotency_key='trace-explicit',
            ),
            actor_principal_id='principal-submit',
            owner_principal_id='principal-submit',
            owner_tenant='tenant',
            now=utcnow(),
        )
        await session.commit()

    assert result.execution.driver_config == {
        'run_trace_backends': {
            'sinks': ['archive', 'primary'],
            'append_artifact_fallback': True,
        },
    }


async def test_submission_uses_principal_run_trace_backend_defaults(submission_session_factory) -> None:
    config = _trace_backend_config()
    async with submission_session_factory() as session:
        await _seed_environment(
            session,
            environment_id='env-dynamic',
            allocation_type='dynamic',
            provider_type='kubernetes',
            environment_limits={'manager_count': 2, 'cpu_millicores': 4_000, 'memory_bytes': 8 * GIB, 'slots': 2},
            grant_limits={'manager_count': 2, 'cpu_millicores': 4_000, 'memory_bytes': 8 * GIB, 'slots': 2},
        )
        await RunTraceBackendAccessPolicy(session, config=config).replace_grants(
            'principal-submit',
            [
                RunTraceBackendGrantInput('archive', default_order=1),
                RunTraceBackendGrantInput('primary', default_order=0),
            ],
            actor='admin',
            now=utcnow(),
        )
        await session.commit()

    async with submission_session_factory() as session:
        result = await SubmissionRepository(session, config=config).create(
            _dynamic_single_with_trace_backends(None, idempotency_key='trace-defaults'),
            actor_principal_id='principal-submit',
            owner_principal_id='principal-submit',
            owner_tenant='tenant',
            now=utcnow(),
        )

    assert result.execution.driver_config == {
        'run_trace_backends': {
            'sinks': ['primary', 'archive'],
            'append_artifact_fallback': True,
        },
    }


async def test_submission_requires_or_rejects_ungranted_run_trace_backend_selection(
    submission_session_factory,
) -> None:
    async with submission_session_factory() as session:
        await _seed_environment(
            session,
            environment_id='env-dynamic',
            allocation_type='dynamic',
            provider_type='kubernetes',
            environment_limits={'manager_count': 2, 'cpu_millicores': 4_000, 'memory_bytes': 8 * GIB, 'slots': 2},
            grant_limits={'manager_count': 2, 'cpu_millicores': 4_000, 'memory_bytes': 8 * GIB, 'slots': 2},
        )
        await session.commit()

    async with submission_session_factory() as session:
        with pytest.raises(DomainValidationError, match='run_trace_backend_selection_required'):
            await SubmissionRepository(session, config=_trace_backend_config(grants_required=True)).create(
                _dynamic_single_with_trace_backends(None, idempotency_key='trace-required'),
                actor_principal_id='principal-submit',
                owner_principal_id='principal-submit',
                owner_tenant='tenant',
                now=utcnow(),
            )

    async with submission_session_factory() as session:
        with pytest.raises(DomainValidationError, match='run_trace_backend_not_available'):
            await SubmissionRepository(session, config=_trace_backend_config()).create(
                _dynamic_single_with_trace_backends(['primary'], idempotency_key='trace-ungranted'),
                actor_principal_id='principal-submit',
                owner_principal_id='principal-submit',
                owner_tenant='tenant',
                now=utcnow(),
            )


async def test_submission_idempotency_includes_run_trace_backend_selection(submission_session_factory) -> None:
    config = _trace_backend_config()
    async with submission_session_factory() as session:
        await _seed_environment(
            session,
            environment_id='env-dynamic',
            allocation_type='dynamic',
            provider_type='kubernetes',
            environment_limits={'manager_count': 2, 'cpu_millicores': 4_000, 'memory_bytes': 8 * GIB, 'slots': 2},
            grant_limits={'manager_count': 2, 'cpu_millicores': 4_000, 'memory_bytes': 8 * GIB, 'slots': 2},
        )
        await RunTraceBackendAccessPolicy(session, config=config).replace_grants(
            'principal-submit',
            [RunTraceBackendGrantInput('archive'), RunTraceBackendGrantInput('primary')],
            actor='admin',
            now=utcnow(),
        )
        await session.commit()

    async with submission_session_factory() as session:
        await SubmissionRepository(session, config=config).create(
            _dynamic_single_with_trace_backends(['primary'], idempotency_key='trace-idempotency'),
            actor_principal_id='principal-submit',
            owner_principal_id='principal-submit',
            owner_tenant='tenant',
            now=utcnow(),
        )
        await session.commit()

    async with submission_session_factory() as session:
        with pytest.raises(DomainConflictError, match='submission_idempotency_conflict'):
            await SubmissionRepository(session, config=config).create(
                _dynamic_single_with_trace_backends(['archive'], idempotency_key='trace-idempotency'),
                actor_principal_id='principal-submit',
                owner_principal_id='principal-submit',
                owner_tenant='tenant',
                now=utcnow(),
            )


async def test_dynamic_single_admission_is_atomic_idempotent_and_reserves_compute(submission_session_factory):
    async with submission_session_factory() as session:
        await _seed_environment(
            session,
            environment_id='env-dynamic',
            allocation_type='dynamic',
            provider_type='kubernetes',
            environment_limits={
                'manager_count': 5,
                'cpu_millicores': 10_000,
                'memory_bytes': 20 * GIB,
                'slots': 10,
                'max_artifact_size_bytes': 64 * 1024 * 1024,
                'max_source_files': 123,
                'max_source_file_bytes': 4 * 1024 * 1024,
                'max_source_unpacked_bytes': 32 * 1024 * 1024,
                'max_workspace_bytes': 128 * 1024 * 1024,
            },
            grant_limits={
                'manager_count': 2,
                'cpu_millicores': 4_000,
                'memory_bytes': 8 * GIB,
                'slots': 4,
                'max_artifact_size_bytes': 32 * 1024 * 1024,
                'max_source_files': 100,
                'max_workspace_bytes': 64 * 1024 * 1024,
            },
            default_timeouts={
                'provisioning_timeout_seconds': 120,
                'queue_ttl_seconds': 300,
                'execution_ttl_seconds': 600,
                'stop_timeout_seconds': 30,
            },
        )
        await session.commit()

    config = _config(shard_count=8)
    async with submission_session_factory() as session:
        result = await SubmissionRepository(session, config=config).create(
            _dynamic_single(host_env=[{'name': 'HOST_ENV_MVP_PROBE', 'value': 'ok'}]),
            actor_principal_id='principal-submit',
            owner_principal_id='principal-submit',
            owner_tenant='tenant',
            now=utcnow(),
        )
        assert result.reused is False
        assert result.phase == 'waiting_resources'
        await session.commit()

    execution_id = result.execution.execution_id
    async with submission_session_factory() as session:
        execution = await session.get(ExecutionRow, execution_id)
        allocation = await session.get(EnvironmentAllocationRow, execution.allocation_id)
        environment_usage = await session.get(EnvironmentResourceUsageRow, 'environment:env-dynamic')
        grant_usage = await session.get(EnvironmentResourceUsageRow, 'grant:env-dynamic:principal-submit')

        assert execution.environment_id == 'env-dynamic'
        assert execution.workload == 'execution'
        assert execution.pool_id.startswith('pool-')
        assert execution.submission_sha256
        assert execution.allocation_spec_hash == allocation.allocation_spec_hash
        assert execution.first_schedulable_at is None
        assert execution.max_attempts == 2
        assert execution.queue_ttl_seconds == 300
        assert execution.execution_ttl_seconds == 600
        assert execution.stop_timeout_seconds == 30
        assert execution.spec_json['timeouts']['provisioning_timeout_seconds'] == 120

        assert allocation.status == 'requested'
        assert allocation.manager_count == 1
        assert allocation.cpu_millicores == 1_500
        assert allocation.memory_bytes == 2 * GIB
        assert allocation.slots == 1
        assert allocation.active_batches == 0
        assert allocation.dependency_build_id is None
        assert allocation.allocation_spec_json['provisioning_timeout_seconds'] == 120
        assert allocation.allocation_spec_json['allocation_type'] == 'dynamic'
        assert allocation.allocation_spec_json['provider_type'] == 'kubernetes'
        assert allocation.allocation_spec_json['provider_config_ref'] == 'kubernetes/default'
        assert allocation.allocation_spec_json['host_env'] == [
            {'name': 'HOST_ENV_MVP_PROBE', 'value': 'ok'},
        ]
        assert allocation.allocation_spec_json['provider_config'] == {
            'provider_type': 'kubernetes',
            'namespace': 'default',
            'auth_type': 'in_cluster',
            'api_url': '',
            'credential_ref': None,
            'tls_strict': True,
            'manager_resource_request_ratio': 1.0,
        }
        assert allocation.allocation_spec_json['workspace_size_bytes'] == 64 * 1024 * 1024
        assert allocation.allocation_spec_json['source_package_limits'] == {
            'max_files': 100,
            'max_single_file_bytes': 4 * 1024 * 1024,
            'max_total_bytes': 32 * 1024 * 1024,
            'max_archive_bytes': 32 * 1024 * 1024,
        }
        assert environment_usage.manager_count == grant_usage.manager_count == 1
        assert environment_usage.cpu_millicores == grant_usage.cpu_millicores == 1_500

    async with submission_session_factory() as session:
        repeated = await SubmissionRepository(session, config=config).create(
            _dynamic_single(host_env=[{'name': 'HOST_ENV_MVP_PROBE', 'value': 'ok'}]),
            actor_principal_id='principal-submit',
            owner_principal_id='principal-submit',
            owner_tenant='tenant',
            now=utcnow(),
        )
        assert repeated.reused is True
        with pytest.raises(ValueError, match='submission_idempotency_conflict'):
            await SubmissionRepository(session, config=config).create(
                _dynamic_single(
                    run_value='changed',
                    host_env=[{'name': 'HOST_ENV_MVP_PROBE', 'value': 'ok'}],
                ),
                actor_principal_id='principal-submit',
                owner_principal_id='principal-submit',
                owner_tenant='tenant',
                now=utcnow(),
            )
    assert repeated.execution.execution_id == execution_id


async def test_submission_without_trace_selection_reuses_legacy_idempotency_hash(
    submission_session_factory,
) -> None:
    async with submission_session_factory() as session:
        await _seed_environment(
            session,
            environment_id='env-dynamic',
            allocation_type='dynamic',
            provider_type='kubernetes',
            environment_limits={
                'manager_count': 2,
                'cpu_millicores': 4_000,
                'memory_bytes': 8 * GIB,
                'slots': 2,
            },
            grant_limits={
                'manager_count': 2,
                'cpu_millicores': 4_000,
                'memory_bytes': 8 * GIB,
                'slots': 2,
            },
        )
        await session.commit()

    submission = _dynamic_single(idempotency_key='legacy-idempotency-hash')
    legacy_payload = submission.model_dump(mode='json')
    legacy_payload.pop('run_trace_backends')
    legacy_hash = canonical_sha256(legacy_payload)
    config = _config()

    async with submission_session_factory() as session:
        created = await SubmissionRepository(session, config=config).create(
            submission,
            actor_principal_id='principal-submit',
            owner_principal_id='principal-submit',
            owner_tenant='tenant',
            now=utcnow(),
        )
        await session.commit()

    async with submission_session_factory() as session:
        execution = await session.get(ExecutionRow, created.execution.execution_id)
        assert execution is not None
        execution.submission_sha256 = legacy_hash
        await session.commit()

    async with submission_session_factory() as session:
        repeated = await SubmissionRepository(session, config=config).create(
            submission,
            actor_principal_id='principal-submit',
            owner_principal_id='principal-submit',
            owner_tenant='tenant',
            now=utcnow(),
        )

    assert repeated.reused is True
    assert repeated.execution.execution_id == created.execution.execution_id


async def test_submission_requires_matching_environment_grant_scope(submission_session_factory):
    async with submission_session_factory() as session:
        await _seed_environment(
            session,
            environment_id='env-dynamic',
            allocation_type='dynamic',
            provider_type='kubernetes',
            environment_limits={'manager_count': 2, 'cpu_millicores': 2_000, 'memory_bytes': 4 * GIB, 'slots': 2},
            grant_limits={'manager_count': 2, 'cpu_millicores': 2_000, 'memory_bytes': 4 * GIB, 'slots': 2},
        )
        await EnvironmentRepository(session).upsert_grant(
            principal_id='principal-submit',
            environment_id='env-dynamic',
            limits={'manager_count': 2, 'cpu_millicores': 2_000, 'memory_bytes': 4 * GIB, 'slots': 2},
            scopes={'submission:batch'},
            enabled=True,
            actor='admin',
            now=utcnow(),
        )
        await session.commit()

    async with submission_session_factory() as session:
        with pytest.raises(ValueError, match='environment_scope_required'):
            await SubmissionRepository(session, config=_config()).create(
                _dynamic_single(idempotency_key='single-with-batch-grant'),
                actor_principal_id='principal-submit',
                owner_principal_id='principal-submit',
                owner_tenant='tenant',
                now=utcnow(),
            )
        assert await session.scalar(select(func.count()).select_from(ExecutionRow)) == 0


async def test_typed_submission_admits_only_new_top_level_execution_against_queued_quota(
    submission_session_factory,
):
    async with submission_session_factory() as session:
        await _seed_environment(
            session,
            environment_id='env-dynamic',
            allocation_type='dynamic',
            provider_type='kubernetes',
            environment_limits={
                'manager_count': 10,
                'cpu_millicores': 20_000,
                'memory_bytes': 20 * GIB,
                'slots': 10,
            },
            grant_limits={
                'manager_count': 10,
                'cpu_millicores': 20_000,
                'memory_bytes': 20 * GIB,
                'slots': 10,
            },
        )
        await session.commit()

    config = _config(
        default_user_quota={
            'queued_executions': 1,
            'running_executions': 100,
            'reserved_slots': 100,
        }
    )
    async with submission_session_factory() as session:
        first = await SubmissionRepository(session, config=config).create(
            _dynamic_single(idempotency_key='quota-first'),
            actor_principal_id='principal-submit',
            owner_principal_id='principal-submit',
            owner_tenant='tenant',
            now=utcnow(),
        )
        await session.commit()

    async with submission_session_factory() as session:
        replay = await SubmissionRepository(session, config=config).create(
            _dynamic_single(idempotency_key='quota-first'),
            actor_principal_id='principal-submit',
            owner_principal_id='principal-submit',
            owner_tenant='tenant',
            now=utcnow(),
        )
        assert replay.reused is True
        assert replay.execution.execution_id == first.execution.execution_id

        with pytest.raises(QuotaExceededError, match='queued_execution_quota_exceeded'):
            await SubmissionRepository(session, config=config).create(
                _dynamic_single(idempotency_key='quota-second'),
                actor_principal_id='principal-submit',
                owner_principal_id='principal-submit',
                owner_tenant='tenant',
                now=utcnow(),
            )
        await session.rollback()

    async with submission_session_factory() as session:
        assert await session.scalar(select(func.count(ExecutionRow.execution_id))) == 1
        usage = await session.get(QuotaUsageRow, ('principal-submit', '*', '*', 'owner'))
        assert usage is not None
        assert usage.queued_executions == 1
        first_row = await session.get(ExecutionRow, first.execution.execution_id)
        assert first_row is not None
        first_row.status = 'expired'
        first_row.finished_at = utcnow()
        await session.commit()

    async with submission_session_factory() as session:
        second = await SubmissionRepository(session, config=config).create(
            _dynamic_single(idempotency_key='quota-second'),
            actor_principal_id='principal-submit',
            owner_principal_id='principal-submit',
            owner_tenant='tenant',
            now=utcnow(),
        )
        assert second.reused is False
        await session.commit()


async def test_submission_failure_after_admission_rolls_back_queued_quota(
    submission_session_factory,
):
    async with submission_session_factory() as session:
        await _seed_environment(
            session,
            environment_id='env-dynamic',
            allocation_type='dynamic',
            provider_type='kubernetes',
            environment_limits={
                'manager_count': 0,
                'cpu_millicores': 0,
                'memory_bytes': 0,
                'slots': 0,
            },
            grant_limits={
                'manager_count': 0,
                'cpu_millicores': 0,
                'memory_bytes': 0,
                'slots': 0,
            },
        )
        await session.commit()

    config = _config(
        default_user_quota={
            'queued_executions': 1,
            'running_executions': 100,
            'reserved_slots': 100,
        }
    )
    async with submission_session_factory() as session:
        with pytest.raises(QuotaExceededError, match='environment_manager_count_limit_exceeded'):
            await SubmissionRepository(session, config=config).create(
                _dynamic_single(idempotency_key='rollback-after-admission'),
                actor_principal_id='principal-submit',
                owner_principal_id='principal-submit',
                owner_tenant='tenant',
                now=utcnow(),
            )
        await session.rollback()

    async with submission_session_factory() as session:
        assert await session.scalar(select(func.count(ExecutionRow.execution_id))) == 0
        assert await session.get(QuotaUsageRow, ('principal-submit', '*', '*', 'owner')) is None
        limits = {
            'manager_count': 1,
            'cpu_millicores': 2_000,
            'memory_bytes': 2 * GIB,
            'slots': 1,
        }
        environment = await session.get(EnvironmentRow, 'env-dynamic')
        grant = await session.get(EnvironmentGrantRow, ('principal-submit', 'env-dynamic'))
        assert environment is not None
        assert grant is not None
        environment.resource_limits_json = limits
        grant.limits_json = limits
        await session.commit()

    async with submission_session_factory() as session:
        result = await SubmissionRepository(session, config=config).create(
            _dynamic_single(idempotency_key='rollback-after-admission'),
            actor_principal_id='principal-submit',
            owner_principal_id='principal-submit',
            owner_tenant='tenant',
            now=utcnow(),
        )
        assert result.reused is False
        await session.commit()


async def test_batch_submission_uses_top_level_queued_quota(submission_session_factory):
    async with submission_session_factory() as session:
        await _seed_environment(
            session,
            environment_id='env-quota-batch',
            allocation_type='dynamic',
            provider_type='kubernetes',
            environment_limits={
                'manager_count': 2,
                'cpu_millicores': 2_000,
                'memory_bytes': 2 * GIB,
                'slots': 2,
                'active_batches': 1,
            },
            grant_limits={
                'manager_count': 2,
                'cpu_millicores': 2_000,
                'memory_bytes': 2 * GIB,
                'slots': 2,
                'active_batches': 1,
            },
        )
        await session.commit()

    submission = BatchSubmission.model_validate({
        'idempotency_key': 'quota-batch',
        'environment_id': 'env-quota-batch',
        'code': {'mode': 'config_only'},
        'resources': {
            'batch_manager': {'cpu_cores': 1, 'memory_gib': 1},
            'managers': {'count': 1, 'cpu_cores': 1, 'memory_gib': 1, 'slots': 1},
        },
        'generation': {
            'mode': 'remote_materialize',
            'source': {
                'type': 'compose',
                'compose': {'parts': [{'kind': 'set', 'value': 'task_name=identity', 'source': 'test'}]},
            },
        },
        'batch': {'max_inflight': 1},
    })
    config = _config(
        default_user_quota={
            'queued_executions': 0,
            'running_executions': 100,
            'reserved_slots': 100,
        }
    )
    async with submission_session_factory() as session:
        with pytest.raises(QuotaExceededError, match='queued_execution_quota_exceeded'):
            await SubmissionRepository(session, config=config).create(
                submission,
                actor_principal_id='principal-submit',
                owner_principal_id='principal-submit',
                owner_tenant='tenant',
                now=utcnow(),
            )
        await session.rollback()

    async with submission_session_factory() as session:
        assert await session.scalar(select(func.count(ExecutionRow.execution_id))) == 0
        assert await session.scalar(select(func.count(BatchRunRow.parent_execution_id))) == 0
        assert await session.scalar(select(func.count(EnvironmentAllocationRow.allocation_id))) == 0


async def test_submission_credentials_are_stored_outside_public_execution_spec(submission_session_factory):
    limits = {
        'manager_count': 2,
        'cpu_millicores': 2_000,
        'memory_bytes': 2 * GIB,
        'slots': 2,
        'active_batches': 1,
    }
    async with submission_session_factory() as session:
        await _seed_environment(
            session,
            environment_id='env-credential-batch',
            allocation_type='dynamic',
            provider_type='kubernetes',
            environment_limits=limits,
            grant_limits=limits,
        )
        await session.commit()

    submission = BatchSubmission.model_validate({
        'idempotency_key': 'credential-batch',
        'environment_id': 'env-credential-batch',
        'code': {'mode': 'config_only'},
        'resources': {
            'batch_manager': {'cpu_cores': 1, 'memory_gib': 1},
            'managers': {'count': 1, 'cpu_cores': 1, 'memory_gib': 1, 'slots': 1},
        },
        'generation': {
            'mode': 'remote_materialize',
            'source': {
                'type': 'compose',
                'compose': {'parts': [{'kind': 'data', 'value': 'research/tasks@latest', 'source': 'test'}]},
            },
        },
        'batch': {'max_inflight': 1},
        'data_credentials': {
            'wedata': {'subject': 'alice', 'key': 'never-store-in-spec', 'id': 'epoch-1'},
        },
    })

    assert 'data_credentials' not in submission.model_dump(mode='json')
    assert 'never-store-in-spec' not in repr(submission)
    async with submission_session_factory() as session:
        result = await SubmissionRepository(session, config=_config()).create(
            submission,
            actor_principal_id='principal-submit',
            owner_principal_id='principal-submit',
            owner_tenant='tenant',
            now=utcnow(),
        )
        await session.commit()

    async with submission_session_factory() as session:
        execution = await session.get(ExecutionRow, result.execution.execution_id)
        credential = await session.get(ExecutionCredentialRow, (result.execution.execution_id, 'wedata'))

    assert execution is not None
    assert credential is not None
    assert credential.payload_json == {'subject': 'alice', 'key': 'never-store-in-spec', 'id': 'epoch-1'}
    assert 'never-store-in-spec' not in json.dumps(execution.spec_json)


async def test_batch_generation_known_limit_is_rejected_before_resources_are_created(
    submission_session_factory,
):
    async with submission_session_factory() as session:
        await _seed_environment(
            session,
            environment_id='env-generation-admission',
            allocation_type='dynamic',
            provider_type='kubernetes',
            environment_limits={
                'manager_count': 2,
                'cpu_millicores': 2_000,
                'memory_bytes': 2 * GIB,
                'slots': 2,
                'active_batches': 1,
                'max_items': 2,
                'max_task_list_bytes': 1024,
                'max_workspace_bytes': 1024,
            },
            grant_limits={
                'manager_count': 2,
                'cpu_millicores': 2_000,
                'memory_bytes': 2 * GIB,
                'slots': 2,
                'active_batches': 1,
                'max_items': 2,
                'max_task_list_bytes': 1024,
                'max_workspace_bytes': 1024,
            },
        )
        await session.commit()

    submission = BatchSubmission.model_validate({
        'idempotency_key': 'generation-admission',
        'environment_id': 'env-generation-admission',
        'code': {'mode': 'config_only'},
        'resources': {
            'batch_manager': {'cpu_cores': 1, 'memory_gib': 1},
            'managers': {'count': 1, 'cpu_cores': 1, 'memory_gib': 1, 'slots': 1},
        },
        'generation': {
            'mode': 'remote_materialize',
            'source': {
                'type': 'compose',
                'compose': {
                    'parts': [{'kind': 'set', 'value': 'task_name=identity', 'source': 'test'}],
                    'limit': 3,
                },
            },
        },
        'batch': {'max_inflight': 1},
    })
    async with submission_session_factory() as session:
        with pytest.raises(ValueError, match='batch_item_limit_exceeded'):
            await SubmissionRepository(session, config=_config()).create(
                submission,
                actor_principal_id='principal-submit',
                owner_principal_id='principal-submit',
                owner_tenant='tenant',
                now=utcnow(),
            )
        await session.rollback()

    async with submission_session_factory() as session:
        assert await session.scalar(select(func.count(ExecutionRow.execution_id))) == 0
        assert await session.scalar(select(func.count(EnvironmentAllocationRow.allocation_id))) == 0
        assert await session.scalar(select(func.count(BatchRunRow.parent_execution_id))) == 0


@pytest.mark.parametrize('submission_type', [SingleSubmission, LiveSubmission])
async def test_dynamic_source_package_run_creates_module_aware_dependency_build_and_binds_allocation(
    submission_session_factory,
    submission_type,
):
    now = utcnow()
    async with submission_session_factory() as session:
        await _seed_environment(
            session,
            environment_id='env-source-single',
            allocation_type='dynamic',
            provider_type='kubernetes',
            environment_limits={
                'manager_count': 1,
                'cpu_millicores': 2_000,
                'memory_bytes': 2 * GIB,
                'slots': 1,
                'max_source_files': 100,
                'max_source_file_bytes': 16 * 1024 * 1024,
                'max_source_unpacked_bytes': 64 * 1024 * 1024,
                'max_artifact_size_bytes': 32 * 1024 * 1024,
            },
            grant_limits={
                'manager_count': 1,
                'cpu_millicores': 2_000,
                'memory_bytes': 2 * GIB,
                'slots': 1,
                'max_source_files': 50,
                'max_source_file_bytes': 8 * 1024 * 1024,
                'max_source_unpacked_bytes': 48 * 1024 * 1024,
                'max_artifact_size_bytes': 24 * 1024 * 1024,
            },
        )
        _add_source_package_artifacts(session, environment_id='env-source-single', now=now)
        await session.commit()

    submission = submission_type.model_validate({
        'submission_version': 1,
        'idempotency_key': 'source-single',
        'environment_id': 'env-source-single',
        'code': _source_package_code(),
        'resources': {'managers': {'count': 1, 'cpu_cores': 1, 'memory_gib': 1, 'slots': 1}},
        'run_input': {'task_name': 'identity'},
    })
    config = _config(
        shard_count=8,
        environment_base_image=f'registry.example/workload@sha256:{"d" * 64}',
        environment_builder_image=f'registry.example/builder@sha256:{"e" * 64}',
    )
    async with submission_session_factory() as session:
        result = await SubmissionRepository(session, config=config).create(
            submission,
            actor_principal_id='principal-submit',
            owner_principal_id='principal-submit',
            owner_tenant='tenant',
            now=now,
        )
        await session.commit()

    async with submission_session_factory() as session:
        allocation = await session.get(EnvironmentAllocationRow, result.execution.allocation_id)
        dependency_build = await session.get(DependencyBuildRow, allocation.dependency_build_id)

        assert dependency_build.status == 'requested'
        assert dependency_build.owner_principal_id == 'principal-submit'
        assert dependency_build.environment_id == 'env-source-single'
        assert dependency_build.build_spec_json['workload_image'] == config.environment_base_image
        assert dependency_build.build_spec_json['builder_image'] == config.environment_builder_image
        assert dependency_build.build_spec_json['extras'] == ['distributed', 'runtime']
        assert dependency_build.build_spec_json['module_selectors'] == [
            {'kind': 'object', 'value': 'identity', 'object_type': 'tasks'}
        ]
        assert dependency_build.build_spec_json['source_package_limits'] == {
            'max_files': 50,
            'max_single_file_bytes': 8 * 1024 * 1024,
            'max_total_bytes': 48 * 1024 * 1024,
            'max_archive_bytes': 24 * 1024 * 1024,
        }
        assert (
            allocation.allocation_spec_json['source_package_limits']
            == dependency_build.build_spec_json['source_package_limits']
        )
        assert allocation.allocation_spec_json['dependency_build'] == {
            'dependency_build_id': dependency_build.dependency_build_id,
            'build_key': dependency_build.build_key,
        }


async def test_dynamic_source_package_run_rejects_invalid_module_import_as_domain_validation(
    submission_session_factory,
):
    now = utcnow()
    async with submission_session_factory() as session:
        await _seed_environment(
            session,
            environment_id='env-source-invalid-import',
            allocation_type='dynamic',
            provider_type='kubernetes',
            environment_limits={
                'manager_count': 1,
                'cpu_millicores': 2_000,
                'memory_bytes': 2 * GIB,
                'slots': 1,
            },
            grant_limits={
                'manager_count': 1,
                'cpu_millicores': 2_000,
                'memory_bytes': 2 * GIB,
                'slots': 1,
            },
        )
        _add_source_package_artifacts(session, environment_id='env-source-invalid-import', now=now)
        await session.commit()

    submission = SingleSubmission.model_validate({
        'submission_version': 1,
        'idempotency_key': 'source-invalid-import',
        'environment_id': 'env-source-invalid-import',
        'code': _source_package_code(),
        'resources': {'managers': {'count': 1, 'cpu_cores': 1, 'memory_gib': 1, 'slots': 1}},
        'run_input': {'task_name': 'identity', 'imports': ['nexus.extensions..broken']},
    })

    async with submission_session_factory() as session:
        with pytest.raises(DomainValidationError) as exc_info:
            await SubmissionRepository(session, config=_config()).create(
                submission,
                actor_principal_id='principal-submit',
                owner_principal_id='principal-submit',
                owner_tenant='tenant',
                now=now,
            )

    assert exc_info.value.code == 'module_import_invalid'
    assert exc_info.value.detail == 'nexus.extensions..broken'


async def test_dynamic_source_package_batch_parent_and_children_share_dependency_build(
    submission_session_factory,
):
    now = utcnow()
    async with submission_session_factory() as session:
        await _seed_environment(
            session,
            environment_id='env-source-batch',
            allocation_type='dynamic',
            provider_type='kubernetes',
            environment_limits={
                'manager_count': 3,
                'cpu_millicores': 6_000,
                'memory_bytes': 6 * GIB,
                'slots': 3,
                'active_batches': 1,
            },
            grant_limits={
                'manager_count': 3,
                'cpu_millicores': 6_000,
                'memory_bytes': 6 * GIB,
                'slots': 3,
                'active_batches': 1,
            },
        )
        _add_source_package_artifacts(session, environment_id='env-source-batch', now=now)
        await session.commit()

    submission = BatchSubmission.model_validate({
        'submission_version': 1,
        'idempotency_key': 'source-batch',
        'environment_id': 'env-source-batch',
        'code': _source_package_code(),
        'resources': {
            'batch_manager': {'cpu_cores': 1, 'memory_gib': 1},
            'managers': {'count': 2, 'cpu_cores': 1, 'memory_gib': 1, 'slots': 1},
        },
        'generation': {
            'mode': 'remote_materialize',
            'source': {
                'type': 'compose',
                'compose': {'parts': [{'kind': 'set', 'value': 'task_name=identity', 'source': 'test'}]},
            },
        },
        'batch': {'max_inflight': 2},
    })
    async with submission_session_factory() as session:
        result = await SubmissionRepository(session, config=_config(shard_count=8)).create(
            submission,
            actor_principal_id='principal-submit',
            owner_principal_id='principal-submit',
            owner_tenant='tenant',
            now=now,
        )
        await session.commit()

    async with submission_session_factory() as session:
        allocations = (
            await session.scalars(
                select(EnvironmentAllocationRow).where(
                    EnvironmentAllocationRow.top_level_execution_id == result.execution.execution_id
                )
            )
        ).all()
        dependency_build_ids = {allocation.dependency_build_id for allocation in allocations}
        batch = await session.get(BatchRunRow, result.execution.execution_id)

        assert len(allocations) == 2
        assert len(dependency_build_ids) == 1
        assert None not in dependency_build_ids
        assert batch is not None
        assert batch.dependency_status == 'pending'
        assert batch.module_dependencies_json is None
        assert batch.child_dependency_build_id in dependency_build_ids
        assert batch.child_template_json == result.execution.spec['batch']['child_template']
        for allocation in allocations:
            assert allocation.allocation_spec_json['dependency_build']['dependency_build_id'] == (
                allocation.dependency_build_id
            )


async def test_dynamic_image_batch_uses_one_resolved_image_without_dependency_build(
    submission_session_factory,
):
    now = utcnow()
    async with submission_session_factory() as session:
        await _seed_environment(
            session,
            environment_id='env-image-batch',
            allocation_type='dynamic',
            provider_type='kubernetes',
            environment_limits={
                'manager_count': 3,
                'cpu_millicores': 6_000,
                'memory_bytes': 6 * GIB,
                'slots': 3,
                'active_batches': 1,
            },
            grant_limits={
                'manager_count': 3,
                'cpu_millicores': 6_000,
                'memory_bytes': 6 * GIB,
                'slots': 3,
                'active_batches': 1,
            },
        )
        await session.commit()

    submission = BatchSubmission.model_validate({
        'submission_version': 1,
        'idempotency_key': 'image-batch',
        'environment_id': 'env-image-batch',
        'code': {'mode': 'image', 'image_version': 'user-build-20260802-001'},
        'resources': {
            'batch_manager': {'cpu_cores': 1, 'memory_gib': 1},
            'managers': {'count': 2, 'cpu_cores': 1, 'memory_gib': 1, 'slots': 1},
        },
        'generation': {
            'mode': 'remote_materialize',
            'source': {'type': 'generator', 'path': 'generators/tasks.py'},
        },
        'batch': {'max_inflight': 2},
    })
    base_image = f'registry.example:5000/team/workload@sha256:{"d" * 64}'
    expected_image = 'registry.example:5000/team/workload:user-build-20260802-001'
    async with submission_session_factory() as session:
        with pytest.raises(DomainValidationError, match='workload_image_repository_unavailable'):
            await SubmissionRepository(
                session,
                config=_config(
                    shard_count=8,
                    production_mode=True,
                    environment_base_image='registry.example:5000/team/workload:platform-v1',
                ),
            ).create(
                submission,
                actor_principal_id='principal-submit',
                owner_principal_id='principal-submit',
                owner_tenant='tenant',
                now=now,
            )

    async with submission_session_factory() as session:
        result = await SubmissionRepository(
            session,
            config=_config(shard_count=8, environment_base_image=base_image),
        ).create(
            submission,
            actor_principal_id='principal-submit',
            owner_principal_id='principal-submit',
            owner_tenant='tenant',
            now=now,
        )
        await session.commit()

    async with submission_session_factory() as session:
        allocations = (
            await session.scalars(
                select(EnvironmentAllocationRow).where(
                    EnvironmentAllocationRow.top_level_execution_id == result.execution.execution_id
                )
            )
        ).all()

        assert len(allocations) == 2
        assert await session.scalar(select(func.count()).select_from(DependencyBuildRow)) == 0
        for allocation in allocations:
            assert allocation.dependency_build_id is None
            assert allocation.allocation_spec_json['base_image'] == expected_image
            assert allocation.allocation_spec_json['code'] == {
                'mode': 'image',
                'image_version': 'user-build-20260802-001',
            }
            assert 'dependency_build' not in allocation.allocation_spec_json


async def test_predefined_batch_creates_ready_logical_allocations_and_initial_ledger(submission_session_factory):
    async with submission_session_factory() as session:
        await _seed_environment(
            session,
            environment_id='env-predefined',
            allocation_type='predefined',
            provider_type='predefined',
            environment_limits={
                'slots': 100,
                'active_batches': 2,
                'batch_max_inflight': 40,
                'max_items': 100_000,
                'max_task_list_bytes': 2 * GIB,
                'max_line_bytes': 8 * 1024**2,
                'max_workspace_bytes': 4 * GIB,
            },
            grant_limits={
                'slots': 30,
                'active_batches': 1,
                'batch_max_inflight': 25,
            },
            default_timeouts={
                'queue_ttl_seconds': 300,
                'execution_ttl_seconds': 3600,
                'stop_timeout_seconds': 30,
                'generation_timeout_seconds': 600,
                'child_queue_ttl_seconds': 120,
                'timeout_per_run': 20,
            },
        )
        await session.commit()

    submission = BatchSubmission.model_validate({
        'submission_version': 1,
        'idempotency_key': 'predefined-batch',
        'environment_id': 'env-predefined',
        'code': {'mode': 'config_only'},
        'pool': 'child-pool',
        'generation': {
            'mode': 'remote_materialize',
            'source': {
                'type': 'compose',
                'compose': {
                    'parts': [{'kind': 'set', 'value': 'task_name=identity', 'source': 'test'}],
                },
            },
        },
    })
    async with submission_session_factory() as session:
        result = await SubmissionRepository(session, config=_config(shard_count=8)).create(
            submission,
            actor_principal_id='principal-submit',
            owner_principal_id='principal-submit',
            owner_tenant='tenant',
            now=utcnow(),
        )
        assert result.reused is False
        assert result.batch_id == result.execution.execution_id
        assert result.phase == 'generating'
        await session.commit()

    async with submission_session_factory() as session:
        execution = await session.get(ExecutionRow, result.execution.execution_id)
        allocations = (
            await session.scalars(
                EnvironmentAllocationRow.__table__
                .select()
                .where(EnvironmentAllocationRow.top_level_execution_id == execution.execution_id)
                .order_by(EnvironmentAllocationRow.workload)
            )
        ).all()
        rows = (
            (
                await session.execute(
                    EnvironmentAllocationRow.__table__
                    .select()
                    .where(EnvironmentAllocationRow.top_level_execution_id == execution.execution_id)
                    .order_by(EnvironmentAllocationRow.workload)
                )
            )
            .mappings()
            .all()
        )
        by_workload = {row['workload']: row for row in rows}
        batch = await session.get(BatchRunRow, execution.execution_id)
        grant_usage = await session.get(EnvironmentResourceUsageRow, 'grant:env-predefined:principal-submit')

        assert len(allocations) == 2
        assert execution.workload == 'orchestration'
        assert execution.pool_id == 'orchestration'
        assert execution.first_schedulable_at is not None
        assert execution.spec_json['batch']['max_inflight'] == 25
        assert execution.spec_json['batch']['child_pool_id'] == 'child-pool'
        assert 'timeout_seconds' not in execution.spec_json['generation']
        assert 'limits' not in execution.spec_json['generation']
        assert execution.spec_json['generation_limits'] == {
            'max_items': 100_000,
            'max_task_list_bytes': 2 * GIB,
            'max_line_bytes': 8 * 1024**2,
            'max_workspace_bytes': 4 * GIB,
            'max_data_frame_bytes': 256 * 1024**2,
            'max_batch_task_record_bytes': 256 * 1024**2,
            'max_data_list_bytes': GIB,
            'timeout_seconds': 600,
        }
        assert execution.spec_json['batch']['child_queue_ttl_seconds'] == 120
        assert execution.spec_json['batch']['timeout_per_run'] == 20
        child_template = execution.spec_json['batch']['child_template']
        assert child_template['mode'] == 'job'
        assert child_template['spec_type'] == 'nexus_single'
        assert child_template['instance_mode'] == 'asyncio'
        assert child_template['workload'] == 'execution'
        assert child_template['environment_id'] == 'env-predefined'
        assert child_template['pool_id'] == 'child-pool'
        assert child_template['allocation_id'] == by_workload['execution']['allocation_id']
        assert child_template['allocation_spec_hash'] == by_workload['execution']['allocation_spec_hash']
        assert child_template['slot_cost'] == 1
        assert child_template['max_attempts'] == 2
        assert child_template['queue_ttl_seconds'] == 120
        assert child_template['execution_ttl_seconds'] == 20
        assert child_template['stop_timeout_seconds'] == 30
        assert child_template['spec_template']['code'] == {'mode': 'config_only'}

        assert by_workload['orchestration']['status'] == 'ready'
        assert by_workload['orchestration']['manager_count'] == 0
        assert by_workload['orchestration']['slots'] == 1
        assert by_workload['orchestration']['active_batches'] == 1
        assert by_workload['execution']['status'] == 'ready'
        assert by_workload['execution']['manager_count'] == 0
        assert by_workload['execution']['slots'] == 25
        assert grant_usage.manager_count == 0
        assert grant_usage.cpu_millicores == 0
        assert grant_usage.memory_bytes == 0
        assert grant_usage.slots == 26
        assert grant_usage.active_batches == 1

        assert batch.parent_execution_id == execution.execution_id
        assert batch.generation_epoch == 1
        assert batch.generation_status == 'pending'
        assert batch.dependency_status == 'not_required'
        assert batch.module_dependencies_json is None
        assert batch.child_dependency_build_id is None
        assert batch.child_template_json == child_template
        assert batch.materialized_count == 0
        assert batch.total_items == 0
        assert batch.submitted_items == 0
        if hasattr(batch, 'control_state'):
            assert batch.control_state == 'active'
        if hasattr(batch, 'materialize_cursor_bytes'):
            assert batch.materialize_cursor_bytes == 0

    async with submission_session_factory() as session:
        repeated = await SubmissionRepository(session, config=_config(shard_count=8)).create(
            submission,
            actor_principal_id='principal-submit',
            owner_principal_id='principal-submit',
            owner_tenant='tenant',
            now=utcnow(),
        )
        assert repeated.reused is True
        assert repeated.phase == 'generating'


async def test_dynamic_batch_reserves_parent_and_child_compute_without_double_counting(submission_session_factory):
    async with submission_session_factory() as session:
        await _seed_environment(
            session,
            environment_id='env-tkex',
            allocation_type='dynamic',
            provider_type='tkex',
            environment_limits={
                'manager_count': 10,
                'cpu_millicores': 20_000,
                'memory_bytes': 30 * GIB,
                'slots': 30,
                'active_batches': 2,
                'max_items': 50_000,
                'max_task_list_bytes': GIB,
            },
            grant_limits={
                'manager_count': 6,
                'cpu_millicores': 10_000,
                'memory_bytes': 15 * GIB,
                'slots': 20,
                'active_batches': 1,
            },
            default_timeouts={'generation_timeout_seconds': 900, 'child_queue_ttl_seconds': 120},
        )
        await session.commit()

    submission = BatchSubmission.model_validate({
        'submission_version': 1,
        'idempotency_key': 'dynamic-batch',
        'environment_id': 'env-tkex',
        'code': {'mode': 'config_only'},
        'resources': {
            'batch_manager': {'cpu_cores': 2, 'memory_gib': 3},
            'managers': {'count': 3, 'cpu_cores': 1.5, 'memory_gib': 2, 'slots': 4},
        },
        'generation': {
            'mode': 'remote_materialize',
            'source': {
                'type': 'compose',
                'compose': {'parts': [{'kind': 'set', 'value': 'task_name=identity', 'source': 'test'}]},
            },
        },
        'batch': {'max_inflight': 12},
        'host_env': [{'name': 'HOST_ENV_MVP_PROBE', 'value': 'ok'}],
    })
    async with submission_session_factory() as session:
        result = await SubmissionRepository(session, config=_config(shard_count=8)).create(
            submission,
            actor_principal_id='principal-submit',
            owner_principal_id='principal-submit',
            owner_tenant='tenant',
            now=utcnow(),
        )
        await session.commit()

    assert result.phase == 'waiting_resources'
    async with submission_session_factory() as session:
        execution = await session.get(ExecutionRow, result.execution.execution_id)
        rows = (
            (
                await session.execute(
                    EnvironmentAllocationRow.__table__.select().where(
                        EnvironmentAllocationRow.top_level_execution_id == execution.execution_id
                    )
                )
            )
            .mappings()
            .all()
        )
        by_workload = {row['workload']: row for row in rows}
        grant_usage = await session.get(EnvironmentResourceUsageRow, 'grant:env-tkex:principal-submit')

        assert execution.workload == 'orchestration'
        assert execution.allocation_id == by_workload['orchestration']['allocation_id']
        assert execution.pool_id.startswith('pool-')
        assert execution.first_schedulable_at is None
        assert execution.spec_json['batch']['max_inflight'] == 12
        assert execution.spec_json['batch']['child_pool_id'].startswith('pool-')

        assert by_workload['orchestration']['allocation_spec_json']['host_env'] == [
            {'name': 'HOST_ENV_MVP_PROBE', 'value': 'ok'},
        ]
        assert by_workload['execution']['allocation_spec_json']['host_env'] == [
            {'name': 'HOST_ENV_MVP_PROBE', 'value': 'ok'},
        ]

        assert by_workload['orchestration']['status'] == 'requested'
        assert by_workload['orchestration']['manager_count'] == 1
        assert by_workload['orchestration']['cpu_millicores'] == 2_000
        assert by_workload['orchestration']['memory_bytes'] == 3 * GIB
        assert by_workload['orchestration']['slots'] == 1
        assert by_workload['orchestration']['active_batches'] == 1
        assert by_workload['execution']['status'] == 'requested'
        assert by_workload['execution']['manager_count'] == 3
        assert by_workload['execution']['cpu_millicores'] == 4_500
        assert by_workload['execution']['memory_bytes'] == 6 * GIB
        assert by_workload['execution']['slots'] == 12
        assert by_workload['execution']['active_batches'] == 0

        assert grant_usage.manager_count == 4
        assert grant_usage.cpu_millicores == 6_500
        assert grant_usage.memory_bytes == 9 * GIB
        assert grant_usage.slots == 13
        assert grant_usage.active_batches == 1


async def test_predefined_single_and_live_reserve_one_ready_slot_per_top_level(submission_session_factory):
    async with submission_session_factory() as session:
        await _seed_environment(
            session,
            environment_id='env-existing',
            allocation_type='predefined',
            provider_type='predefined',
            environment_limits={'slots': 3},
            grant_limits={'slots': 2},
            default_timeouts={'queue_ttl_seconds': 60, 'stop_timeout_seconds': 10},
        )
        await session.commit()

    common = {
        'submission_version': 1,
        'idempotency_key': 'same-user-key',
        'environment_id': 'env-existing',
        'code': {'mode': 'config_only'},
        'pool': 'interactive',
        'run_input': {'task_name': 'identity'},
    }
    submissions = (SingleSubmission.model_validate(common), LiveSubmission.model_validate(common))
    results = []
    async with submission_session_factory() as session:
        repository = SubmissionRepository(session, config=_config(shard_count=8))
        for submission in submissions:
            results.append(
                await repository.create(
                    submission,
                    actor_principal_id='principal-submit',
                    owner_principal_id='principal-submit',
                    owner_tenant='tenant',
                    now=utcnow(),
                )
            )
        await session.commit()

    assert results[0].execution.execution_id != results[1].execution.execution_id
    async with submission_session_factory() as session:
        rows = (
            (
                await session.execute(
                    EnvironmentAllocationRow.__table__.select().where(
                        EnvironmentAllocationRow.top_level_execution_id.in_([
                            result.execution.execution_id for result in results
                        ])
                    )
                )
            )
            .mappings()
            .all()
        )
        executions = [await session.get(ExecutionRow, result.execution.execution_id) for result in results]
        grant_usage = await session.get(EnvironmentResourceUsageRow, 'grant:env-existing:principal-submit')

        assert {row['status'] for row in rows} == {'ready'}
        assert {row['manager_count'] for row in rows} == {0}
        assert {row['slots'] for row in rows} == {1}
        assert {row['pool_id'] for row in rows} == {'interactive'}
        assert all(execution.first_schedulable_at is not None for execution in executions)
        assert [execution.spec_type for execution in executions] == ['nexus_single', 'nexus_live']
        assert [execution.route_status for execution in executions] == ['none', 'active']
        assert all(execution.queue_ttl_seconds == 60 for execution in executions)
        assert all(execution.stop_timeout_seconds == 10 for execution in executions)
        assert grant_usage.manager_count == 0
        assert grant_usage.slots == 2


async def test_local_task_list_manifest_is_normalized_and_initializes_total_items(submission_session_factory):
    now = utcnow()
    async with submission_session_factory() as session:
        await _seed_environment(
            session,
            environment_id='env-local-list',
            allocation_type='predefined',
            provider_type='predefined',
            environment_limits={'slots': 10, 'active_batches': 1, 'batch_max_inflight': 4},
            grant_limits={'slots': 6, 'active_batches': 1, 'batch_max_inflight': 4},
            default_timeouts={'generation_timeout_seconds': 300},
        )
        session.add(
            ArtifactRow(
                artifact_id='artifact-task-list',
                owner_principal_id='principal-submit',
                environment_id='env-local-list',
                owner_tenant='tenant',
                kind='task_list',
                producer_key='test:local-task-list',
                uri='file:///tmp/tasks.jsonl',
                sha256='c' * 64,
                size_bytes=123,
                provider_etag=None,
                status='ready',
                created_by='principal-submit',
                expires_at=None,
                created_at=now,
                updated_at=now,
            )
        )
        await session.commit()

    manifest = {
        'version': 1,
        'item_count': 3,
        'size_bytes': 123,
        'sha256': 'c' * 64,
        'source': {'mode': 'local_generator', 'path': 'generator.py'},
        'datasets': [
            {
                'input_uri': 'namespace/dataset@latest',
                'resolved_uri': 'namespace/dataset@version-1',
            }
        ],
        'batch_metadata': {'group': 'local', 'tags': ['hyway']},
    }
    submission = BatchSubmission.model_validate({
        'submission_version': 1,
        'idempotency_key': 'local-task-list',
        'environment_id': 'env-local-list',
        'code': {'mode': 'config_only'},
        'pool': 'children',
        'generation': {
            'mode': 'local_task_list',
            'task_list': {
                'ref': {
                    'artifact_id': 'artifact-task-list',
                    'kind': 'task_list',
                    'sha256': 'c' * 64,
                    'size_bytes': 123,
                },
                'format': 'jsonl',
                'manifest': manifest,
            },
        },
        'batch': {'max_inflight': 4},
    })
    async with submission_session_factory() as session:
        result = await SubmissionRepository(session, config=_config(shard_count=8)).create(
            submission,
            actor_principal_id='principal-submit',
            owner_principal_id='principal-submit',
            owner_tenant='tenant',
            now=now,
        )
        await session.commit()

    async with submission_session_factory() as session:
        execution = await session.get(ExecutionRow, result.execution.execution_id)
        batch = await session.get(BatchRunRow, result.execution.execution_id)

        normalized_manifest = {**manifest, 'record_version': 1}
        assert execution.spec_json['generation']['task_list_manifest'] == normalized_manifest
        assert 'count' not in execution.spec_json['generation']['task_list_manifest']
        assert batch.total_items == 3
        assert batch.generation_status == 'materializing'
        if hasattr(batch, 'task_list_sha256'):
            assert batch.task_list_sha256 == manifest['sha256']
        if hasattr(batch, 'task_list_manifest_json'):
            assert batch.task_list_manifest_json == normalized_manifest


async def test_v3_data_list_admission_persists_effective_limits_and_links_both_artifacts(
    submission_session_factory,
):
    now = utcnow()
    async with submission_session_factory() as session:
        await _seed_environment(
            session,
            environment_id='env-local-v3',
            allocation_type='predefined',
            provider_type='predefined',
            environment_limits={
                'slots': 10,
                'active_batches': 1,
                'batch_max_inflight': 4,
                'max_artifact_size_bytes': 900,
                'max_task_list_bytes': 500,
                'max_workspace_bytes': 1_000,
                'max_data_frame_bytes': 700,
                'max_batch_task_record_bytes': 900,
                'max_data_list_bytes': 850,
            },
            grant_limits={
                'slots': 6,
                'active_batches': 1,
                'batch_max_inflight': 4,
                'max_artifact_size_bytes': 750,
                'max_workspace_bytes': 900,
                'max_data_frame_bytes': 600,
                'max_batch_task_record_bytes': 700,
                'max_data_list_bytes': 800,
            },
        )
        _add_local_batch_artifacts(session, environment_id='env-local-v3', now=now)
        await session.commit()

    submission = _local_v3_submission(
        environment_id='env-local-v3',
        idempotency_key='local-v3-effective-limits',
    )
    config = _config(
        artifact_upload_max_size_bytes=900,
        artifact_run_result_max_size_bytes=900,
        batch_data_frame_max_size_bytes=650,
        batch_task_record_max_size_bytes=750,
        batch_data_list_max_size_bytes=900,
    )
    async with submission_session_factory() as session:
        result = await SubmissionRepository(session, config=config).create(
            submission,
            actor_principal_id='principal-submit',
            owner_principal_id='principal-submit',
            owner_tenant='tenant',
            now=now,
        )
        await session.commit()

    async with submission_session_factory() as session:
        execution = await session.get(ExecutionRow, result.execution.execution_id)
        linked = set(
            await session.scalars(
                select(ArtifactRefRow.artifact_id).where(
                    ArtifactRefRow.resource_type == 'execution',
                    ArtifactRefRow.resource_id == result.execution.execution_id,
                )
            )
        )

    assert execution is not None
    assert {
        key: execution.spec_json['generation_limits'][key]
        for key in (
            'max_data_frame_bytes',
            'max_batch_task_record_bytes',
            'max_data_list_bytes',
            'max_workspace_bytes',
        )
    } == {
        'max_data_frame_bytes': 600,
        'max_batch_task_record_bytes': 700,
        'max_data_list_bytes': 750,
        'max_workspace_bytes': 900,
    }
    assert linked == {'artifact-task-list', 'artifact-data-list'}


@pytest.mark.parametrize(
    ('environment_limits', 'expected_error'),
    [
        ({'max_workspace_bytes': 500}, 'batch_workspace_limit_exceeded'),
        ({'max_data_list_bytes': 399}, 'batch_data_list_exceeds_byte_limit'),
    ],
)
async def test_v3_data_list_admission_rejects_effective_size_limits_before_execution(
    submission_session_factory,
    environment_limits,
    expected_error,
):
    now = utcnow()
    async with submission_session_factory() as session:
        await _seed_environment(
            session,
            environment_id='env-local-v3-limit',
            allocation_type='predefined',
            provider_type='predefined',
            environment_limits={
                'slots': 10,
                'active_batches': 1,
                'batch_max_inflight': 4,
                'max_artifact_size_bytes': 1_000,
                'max_task_list_bytes': 1_000,
                **environment_limits,
            },
            grant_limits={'slots': 6, 'active_batches': 1, 'batch_max_inflight': 4},
        )
        _add_local_batch_artifacts(
            session,
            environment_id='env-local-v3-limit',
            now=now,
            task_list_size=200,
            data_list_size=400,
        )
        await session.commit()

    submission = _local_v3_submission(
        environment_id='env-local-v3-limit',
        idempotency_key=f'local-v3-{expected_error}',
        task_list_size=200,
        data_list_size=400,
    )
    async with submission_session_factory() as session:
        with pytest.raises(DomainValidationError, match=expected_error):
            await SubmissionRepository(session, config=_config()).create(
                submission,
                actor_principal_id='principal-submit',
                owner_principal_id='principal-submit',
                owner_tenant='tenant',
                now=now,
            )
        await session.rollback()

    async with submission_session_factory() as session:
        assert await session.scalar(select(func.count(ExecutionRow.execution_id))) == 0


@pytest.mark.parametrize(
    'artifact_overrides',
    [
        {'data_list_owner': 'principal-other'},
        {'data_list_environment': 'env-other'},
        {'data_list_status': 'receiving'},
    ],
)
async def test_v3_data_list_ref_uses_task_list_owner_environment_and_ready_checks(
    submission_session_factory,
    artifact_overrides,
):
    now = utcnow()
    async with submission_session_factory() as session:
        await _seed_environment(
            session,
            environment_id='env-local-v3-ref',
            allocation_type='predefined',
            provider_type='predefined',
            environment_limits={
                'slots': 10,
                'active_batches': 1,
                'batch_max_inflight': 4,
                'max_artifact_size_bytes': 1_000,
            },
            grant_limits={'slots': 6, 'active_batches': 1, 'batch_max_inflight': 4},
        )
        _add_local_batch_artifacts(
            session,
            environment_id='env-local-v3-ref',
            now=now,
            **artifact_overrides,
        )
        await session.commit()

    submission = _local_v3_submission(
        environment_id='env-local-v3-ref',
        idempotency_key='local-v3-ref-checks',
    )
    async with submission_session_factory() as session:
        with pytest.raises((KeyError, DomainConflictError)):
            await SubmissionRepository(session, config=_config()).create(
                submission,
                actor_principal_id='principal-submit',
                owner_principal_id='principal-submit',
                owner_tenant='tenant',
                now=now,
            )


async def test_persisted_environment_type_rejects_the_opposite_submission_shape(submission_session_factory):
    async with submission_session_factory() as session:
        await _seed_environment(
            session,
            environment_id='env-authoritative-predefined',
            allocation_type='predefined',
            provider_type='predefined',
            environment_limits={'slots': 2},
            grant_limits={'slots': 2},
        )
        await _seed_environment(
            session,
            environment_id='env-authoritative-dynamic',
            allocation_type='dynamic',
            provider_type='kubernetes',
            environment_limits={'manager_count': 2, 'cpu_millicores': 4_000, 'memory_bytes': 4 * GIB, 'slots': 2},
            grant_limits={'manager_count': 2, 'cpu_millicores': 4_000, 'memory_bytes': 4 * GIB, 'slots': 2},
        )
        await session.commit()

    dynamic_shape = SingleSubmission.model_validate({
        'submission_version': 1,
        'idempotency_key': 'wrong-predefined-shape',
        'environment_id': 'env-authoritative-predefined',
        'code': {'mode': 'config_only'},
        'resources': {'managers': {'count': 1, 'cpu_cores': 1, 'memory_gib': 1, 'slots': 1}},
        'run_input': {'task_name': 'identity'},
    })
    predefined_shape = SingleSubmission.model_validate({
        'submission_version': 1,
        'idempotency_key': 'wrong-dynamic-shape',
        'environment_id': 'env-authoritative-dynamic',
        'code': {'mode': 'config_only'},
        'pool': 'existing',
        'run_input': {'task_name': 'identity'},
    })
    async with submission_session_factory() as session:
        repository = SubmissionRepository(session, config=_config(shard_count=8))
        with pytest.raises(ValueError, match='predefined_environment_requires_config_only_pool'):
            await repository.create(
                dynamic_shape,
                actor_principal_id='principal-submit',
                owner_principal_id='principal-submit',
                owner_tenant='tenant',
                now=utcnow(),
            )
        with pytest.raises(ValueError, match='dynamic_environment_requires_resources'):
            await repository.create(
                predefined_shape,
                actor_principal_id='principal-submit',
                owner_principal_id='principal-submit',
                owner_tenant='tenant',
                now=utcnow(),
            )


async def test_submission_timeout_cannot_exceed_environment_or_grant_maximum(submission_session_factory):
    async with submission_session_factory() as session:
        await _seed_environment(
            session,
            environment_id='env-timeout',
            allocation_type='dynamic',
            provider_type='kubernetes',
            environment_limits={
                'manager_count': 1,
                'cpu_millicores': 2_000,
                'memory_bytes': 2 * GIB,
                'slots': 1,
                'max_queue_ttl_seconds': 90,
            },
            grant_limits={
                'manager_count': 1,
                'cpu_millicores': 2_000,
                'memory_bytes': 2 * GIB,
                'slots': 1,
                'max_queue_ttl_seconds': 80,
            },
            default_timeouts={'queue_ttl_seconds': 60},
        )
        await session.commit()

    payload = {
        'submission_version': 1,
        'idempotency_key': 'timeout-limit',
        'environment_id': 'env-timeout',
        'code': {'mode': 'config_only'},
        'resources': {'managers': {'count': 1, 'cpu_cores': 1, 'memory_gib': 1, 'slots': 1}},
        'run_input': {'task_name': 'identity'},
        'queue_ttl_seconds': 81,
    }
    async with submission_session_factory() as session:
        with pytest.raises(ValueError, match='queue_ttl_seconds_exceeds_limit'):
            await SubmissionRepository(session, config=_config(shard_count=8)).create(
                SingleSubmission.model_validate(payload),
                actor_principal_id='principal-submit',
                owner_principal_id='principal-submit',
                owner_tenant='tenant',
                now=utcnow(),
            )
        assert (
            await session.scalar(
                select(func.count()).select_from(ExecutionRow).where(ExecutionRow.environment_id == 'env-timeout')
            )
            == 0
        )
