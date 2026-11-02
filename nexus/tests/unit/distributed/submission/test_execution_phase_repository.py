from __future__ import annotations

import logging

from datetime import timedelta

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from nexus.distributed.artifacts.run_result import run_result_producer_key
from nexus.distributed.execution.models import (
    ExecutionMode,
    ExecutionStatus,
)
from nexus.distributed.persistence.orm import (
    ArtifactRow,
    DependencyBuildRow,
    EnvironmentAllocationRow,
    ExecutionRow,
    create_schema,
)
from nexus.distributed.repositories.execution import CreateExecution, ExecutionRepository
from nexus.distributed.repositories.execution_phase import ExecutionPhaseRepository
from nexus.distributed.utils.clock import utcnow


async def test_execution_phase_repository_reads_artifacts_and_fails_closed_with_diagnostics(
    tmp_path,
    caplog,
):
    engine = create_async_engine(f'sqlite+aiosqlite:///{tmp_path / "phase.db"}')
    await create_schema(engine)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    now = utcnow()
    try:
        async with session_factory() as session:
            executions = ExecutionRepository(session, shard_count=8)
            for execution_id in ('uploading', 'invalid', 'missing'):
                await executions.create_execution(
                    CreateExecution(
                        execution_id=execution_id,
                        mode=ExecutionMode.JOB,
                        spec_type='nexus_single',
                        instance_mode='asyncio',
                        spec={},
                    ),
                    now=now,
                )
                row = await session.get(ExecutionRow, execution_id)
                assert row is not None
                row.status = ExecutionStatus.RUNNING
                row.current_attempt_id = f'attempt-{execution_id}'
                if execution_id == 'missing':
                    row.result_artifact_id = 'artifact-missing'
                session.add(
                    _allocation_row(
                        execution_id=execution_id,
                        allocation_id=f'allocation-{execution_id}',
                        status='mystery' if execution_id == 'invalid' else 'ready',
                        now=now,
                    )
                )

            session.add(
                ArtifactRow(
                    artifact_id='artifact-uploading',
                    owner_principal_id='legacy',
                    environment_id='environment',
                    owner_tenant=None,
                    kind='run_result',
                    producer_key=run_result_producer_key('uploading', 'attempt-uploading'),
                    uri='file:///tmp/run-result.json',
                    sha256='a' * 64,
                    size_bytes=1,
                    provider_etag=None,
                    status='uploading',
                    created_by='legacy',
                    expires_at=None,
                    created_at=now,
                    updated_at=now,
                )
            )
            session.add(
                ArtifactRow(
                    artifact_id='artifact-other-owner',
                    owner_principal_id='other-owner',
                    environment_id='environment',
                    kind='run_result',
                    producer_key=run_result_producer_key('uploading', 'attempt-uploading'),
                    uri='file:///tmp/other-result.json',
                    sha256='b' * 64,
                    size_bytes=1,
                    status='ready',
                    created_by='other-owner',
                    created_at=now,
                    updated_at=now,
                )
            )
            await session.flush()
            records = [
                await executions.get_execution('uploading'),
                await executions.get_execution('invalid'),
                await executions.get_execution('missing'),
            ]

            with caplog.at_level(logging.ERROR):
                phases = await ExecutionPhaseRepository(session).project_many(records)

        assert phases == {
            'uploading': 'uploading_result',
            'invalid': 'unknown',
            'missing': 'unknown',
        }
        assert 'execution_id=invalid' in caplog.text
        assert 'unknown_allocation_status:mystery' in caplog.text
        assert 'execution_id=missing' in caplog.text
        assert 'unknown_result_artifact_status:missing' in caplog.text
    finally:
        await engine.dispose()


async def test_execution_phase_repository_projects_retryable_dependency_failure_as_building(tmp_path):
    engine = create_async_engine(f'sqlite+aiosqlite:///{tmp_path / "retrying-phase.db"}')
    await create_schema(engine)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    now = utcnow()
    try:
        async with session_factory() as session:
            executions = ExecutionRepository(session, shard_count=8)
            await executions.create_execution(
                CreateExecution(
                    execution_id='retrying',
                    mode=ExecutionMode.JOB,
                    spec_type='nexus_single',
                    instance_mode='asyncio',
                    spec={},
                ),
                now=now,
            )
            session.add(
                DependencyBuildRow(
                    dependency_build_id='build-retrying',
                    build_key='a' * 64,
                    owner_principal_id='legacy',
                    environment_id='environment',
                    status='failed',
                    build_spec_json={},
                    attempt=1,
                    next_retry_at=now + timedelta(seconds=15),
                    created_at=now,
                    updated_at=now,
                )
            )
            session.add(
                _allocation_row(
                    execution_id='retrying',
                    allocation_id='allocation-retrying',
                    status='requested',
                    dependency_build_id='build-retrying',
                    now=now,
                )
            )
            await session.flush()
            record = await executions.get_execution('retrying')

            phase = await ExecutionPhaseRepository(session).project(record)

        assert phase == 'building_dependencies'
    finally:
        await engine.dispose()


async def test_execution_phase_repository_projects_waiting_policy(tmp_path):
    engine = create_async_engine(f'sqlite+aiosqlite:///{tmp_path / "waiting-policy-phase.db"}')
    await create_schema(engine)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    now = utcnow()
    try:
        async with session_factory() as session:
            executions = ExecutionRepository(session, shard_count=8)
            for execution_id in (
                'single-deferred',
                'single-capacity',
                'single-underscore-wildcard',
                'single-waiting-resources',
            ):
                await executions.create_execution(
                    CreateExecution(
                        execution_id=execution_id,
                        mode=ExecutionMode.JOB,
                        spec_type='nexus_single',
                        instance_mode='asyncio',
                        spec={},
                    ),
                    now=now,
                )

            single_deferred = await session.get(ExecutionRow, 'single-deferred')
            assert single_deferred is not None
            single_deferred.last_schedule_error = 'policy_deferred:quota_exhausted'

            single_capacity = await session.get(ExecutionRow, 'single-capacity')
            assert single_capacity is not None
            single_capacity.last_schedule_error = 'no_capacity'

            single_underscore = await session.get(ExecutionRow, 'single-underscore-wildcard')
            assert single_underscore is not None
            single_underscore.last_schedule_error = 'policyXdeferred:quota_exhausted'

            single_waiting_resources = await session.get(ExecutionRow, 'single-waiting-resources')
            assert single_waiting_resources is not None
            single_waiting_resources.last_schedule_error = 'policy_deferred:quota_exhausted'
            session.add(
                _allocation_row(
                    execution_id='single-waiting-resources',
                    allocation_id='allocation-waiting-resources',
                    status='requested',
                    now=now,
                )
            )
            await session.flush()

            records = [
                await executions.get_execution('single-deferred'),
                await executions.get_execution('single-capacity'),
                await executions.get_execution('single-underscore-wildcard'),
                await executions.get_execution('single-waiting-resources'),
            ]
            phases = await ExecutionPhaseRepository(session).project_many(records)

        assert phases == {
            'single-deferred': 'waiting_policy',
            'single-capacity': 'queued',
            'single-underscore-wildcard': 'queued',
            'single-waiting-resources': 'waiting_resources',
        }
    finally:
        await engine.dispose()


def _allocation_row(
    *,
    execution_id: str,
    allocation_id: str,
    status: str,
    now,
    dependency_build_id: str | None = None,
) -> EnvironmentAllocationRow:
    return EnvironmentAllocationRow(
        allocation_id=allocation_id,
        top_level_execution_id=execution_id,
        owner_principal_id='legacy',
        environment_id='environment',
        pool_id=allocation_id,
        workload='execution',
        status=status,
        desired_manager_count=1,
        ready_manager_count=1,
        manager_count=1,
        cpu_millicores=1000,
        memory_bytes=1024,
        slots=1,
        active_batches=0,
        dependency_build_id=dependency_build_id,
        provider_resource_ref_json=None,
        allocation_spec_json={},
        allocation_spec_hash='b' * 64,
        config_version=1,
        status_version=1,
        last_progress_at=now,
        created_at=now,
        updated_at=now,
    )
