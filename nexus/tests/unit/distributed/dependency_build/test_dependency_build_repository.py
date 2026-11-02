from __future__ import annotations

import asyncio

from datetime import timedelta

import pytest

from sqlalchemy import event, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from nexus.distributed.artifacts.models import ArtifactRef
from nexus.distributed.artifacts.tickets import ArtifactContentTicketSigner
from nexus.distributed.dependency_build.models import DependencyBuildStatus
from nexus.distributed.dependency_build.wheelhouse import (
    WheelhouseBuildSpec,
    WheelhouseMetadata,
    wheelhouse_build_key,
)
from nexus.distributed.persistence.orm import ArtifactRefRow, DependencyBuildRow, create_schema
from nexus.distributed.repositories.dependency_build import DependencyBuildRepository
from nexus.distributed.utils.clock import utcnow


def _ref(kind: str, marker: str) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=f'art-{kind}-{marker}',
        kind=kind,
        sha256=marker * 64,
        size_bytes=100,
    )


def _spec() -> WheelhouseBuildSpec:
    return WheelhouseBuildSpec(
        owner_principal_id='user-1',
        environment_id='env-1',
        source_package=_ref('source_package', 'a'),
        nexus_wheel=_ref('nexus_wheel', 'b'),
        uv_lock=_ref('uv_lock', 'c'),
        extras=['runtime', 'distributed'],
        workload_image=f'registry.example/workload@sha256:{"d" * 64}',
        builder_image=f'registry.example/builder@sha256:{"e" * 64}',
    )


def _provider_target() -> dict:
    return {
        'provider_type': 'kubernetes',
        'provider_config_ref': 'kubernetes/default',
        'provider_config_snapshot': {
            'provider_type': 'kubernetes',
            'namespace': 'default',
            'api_url': 'https://kubernetes.snapshot.example',
            'token_file': '/var/run/secrets/nexus/provider/kubernetes/token',
            'ca_file': '/var/run/secrets/nexus/provider/kubernetes/ca.crt',
        },
    }


async def _begin_build_attempt(repo: DependencyBuildRepository, *, now):
    record, _ = await repo.get_or_create(_spec(), now=now, **_provider_target())
    claim = await repo.claim_next(
        operation_owner='reconciler-a',
        lease_ttl=timedelta(minutes=2),
        now=now,
    )
    assert claim is not None
    await repo.begin_attempt(
        record.dependency_build_id,
        operation_id=claim.operation_id,
        attempt=1,
        provider_job_name='wheelhouse-build-1',
        now=now,
    )
    return record


@pytest.fixture
async def session_factory():
    engine = create_async_engine('sqlite+aiosqlite:///:memory:')
    await create_schema(engine)
    try:
        yield async_sessionmaker(engine, expire_on_commit=False)
    finally:
        await engine.dispose()


async def test_dependency_build_is_reused_by_owner_environment_and_build_key(session_factory):
    first_spec = _spec()
    repeated_upload_spec = first_spec.model_copy(
        update={
            'source_package': first_spec.source_package.model_copy(update={'artifact_id': 'art-source-reuploaded'}),
            'nexus_wheel': first_spec.nexus_wheel.model_copy(update={'artifact_id': 'art-wheel-reuploaded'}),
            'uv_lock': first_spec.uv_lock.model_copy(update={'artifact_id': 'art-lock-reuploaded'}),
        }
    )
    async with session_factory() as session:
        repo = DependencyBuildRepository(session)
        first, first_reused = await repo.get_or_create(first_spec, now=utcnow(), **_provider_target())
        second, second_reused = await repo.get_or_create(repeated_upload_spec, now=utcnow(), **_provider_target())
        await session.commit()

    assert first_reused is False
    assert second_reused is True
    assert first.dependency_build_id == second.dependency_build_id
    assert first.build_key == wheelhouse_build_key(first_spec)
    assert first.status == DependencyBuildStatus.REQUESTED
    assert first.provider_type == 'kubernetes'
    assert first.provider_config_ref == 'kubernetes/default'
    assert first.provider_config_snapshot == _provider_target()['provider_config_snapshot']


async def test_dependency_build_provider_snapshot_is_immutable_from_the_callers_mapping(session_factory):
    provider_target = _provider_target()
    async with session_factory() as session:
        record, _ = await DependencyBuildRepository(session).get_or_create(
            _spec(),
            now=utcnow(),
            **provider_target,
        )
        provider_target['provider_config_snapshot']['api_url'] = 'https://mutated.example'
        await session.commit()

    assert record.provider_config_snapshot is not None
    assert record.provider_config_snapshot['api_url'] == 'https://kubernetes.snapshot.example'


async def test_dependency_build_claim_sort_uses_narrow_candidate_projection(session_factory):
    async with session_factory() as session:
        record, _ = await DependencyBuildRepository(session).get_or_create(
            _spec(),
            now=utcnow(),
            **_provider_target(),
        )
        await session.commit()

    statements: list[str] = []
    engine = session_factory.kw['bind']

    def capture_sql(_conn, _cursor, statement, _parameters, _context, _executemany):
        statements.append(statement)

    event.listen(engine.sync_engine, 'before_cursor_execute', capture_sql)
    try:
        async with session_factory() as session:
            claim = await DependencyBuildRepository(session).claim_next(
                operation_owner='narrow-claim-controller',
                lease_ttl=timedelta(minutes=2),
                now=utcnow(),
            )
    finally:
        event.remove(engine.sync_engine, 'before_cursor_execute', capture_sql)

    assert claim is not None
    assert claim.record.dependency_build_id == record.dependency_build_id
    assert claim.record.build_spec == _spec()
    ordered = next(
        statement.lower()
        for statement in statements
        if 'order by distributed_dependency_builds.updated_at' in statement.lower()
    )
    assert 'distributed_dependency_builds.build_spec_json' not in ordered
    assert 'distributed_dependency_builds.provider_config_snapshot_json' not in ordered
    assert 'distributed_dependency_builds.manifest_json' not in ordered


async def test_dependency_build_concurrent_get_or_create_converges_without_unique_error(tmp_path):
    engine = create_async_engine(f'sqlite+aiosqlite:///{tmp_path / "dependency-build.db"}')
    await create_schema(engine)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    start = asyncio.Event()

    async def create_once():
        await start.wait()
        async with session_factory() as session:
            result = await DependencyBuildRepository(session).get_or_create(
                _spec(),
                now=utcnow(),
                **_provider_target(),
            )
            await session.commit()
            return result

    try:
        tasks = [asyncio.create_task(create_once()) for _ in range(2)]
        start.set()
        results = await asyncio.gather(*tasks)
        async with session_factory() as session:
            rows = (await session.scalars(select(DependencyBuildRow))).all()
    finally:
        await engine.dispose()

    assert len(rows) == 1
    assert {record.dependency_build_id for record, _reused in results} == {rows[0].dependency_build_id}
    assert sorted(reused for _record, reused in results) == [False, True]


async def test_dependency_build_claim_fences_state_transitions_and_links_output(session_factory):
    now = utcnow()
    async with session_factory() as session:
        repo = DependencyBuildRepository(session)
        record, _ = await repo.get_or_create(_spec(), now=now, **_provider_target())
        claim = await repo.claim_next(
            operation_owner='reconciler-a',
            lease_ttl=timedelta(minutes=2),
            now=now,
        )
        assert claim is not None
        assert claim.record.dependency_build_id == record.dependency_build_id
        assert claim.record.attempt == 0

        assert (
            await repo.claim_next(
                operation_owner='reconciler-b',
                lease_ttl=timedelta(minutes=2),
                now=now,
            )
            is None
        )

        building = await repo.begin_attempt(
            record.dependency_build_id,
            operation_id=claim.operation_id,
            attempt=1,
            provider_job_name='wheelhouse-build-1',
            now=now,
        )
        assert building.status == DependencyBuildStatus.BUILDING
        assert building.attempt == 1

        output = _ref('wheelhouse', 'f')
        metadata = WheelhouseMetadata(
            build_key=record.build_key,
            nexus_wheel_path='wheels/nexus_agent.whl',
        )
        staged = await repo.stage_output(
            record.dependency_build_id,
            attempt=1,
            wheelhouse=output,
            metadata=metadata,
            now=now,
        )
        assert staged.status == DependencyBuildStatus.BUILDING
        assert staged.wheelhouse == output
        assert staged.metadata == metadata
        ready = await repo.mark_ready(
            record.dependency_build_id,
            operation_id=claim.operation_id,
            wheelhouse=output,
            metadata=metadata,
            now=now,
        )
        await session.commit()

        linked = await session.get(
            ArtifactRefRow,
            {
                'artifact_id': output.artifact_id,
                'resource_type': 'dependency_build',
                'resource_id': record.dependency_build_id,
            },
        )

    assert ready.status == DependencyBuildStatus.READY
    assert ready.wheelhouse == output
    assert ready.operation_id is None
    assert linked is not None


async def test_dependency_build_stage_output_rejects_stale_attempt(session_factory):
    now = utcnow()
    async with session_factory() as session:
        repo = DependencyBuildRepository(session)
        record, _ = await repo.get_or_create(_spec(), now=now, **_provider_target())
        claim = await repo.claim_next(
            operation_owner='reconciler-a',
            lease_ttl=timedelta(minutes=2),
            now=now,
        )
        assert claim is not None
        await repo.begin_attempt(
            record.dependency_build_id,
            operation_id=claim.operation_id,
            attempt=1,
            provider_job_name='wheelhouse-build-1',
            now=now,
        )
        output = _ref('wheelhouse', 'f')
        metadata = WheelhouseMetadata(
            build_key=record.build_key,
            nexus_wheel_path='wheels/nexus_agent.whl',
        )

        with pytest.raises(ValueError, match='dependency_build_attempt_mismatch'):
            await repo.stage_output(
                record.dependency_build_id,
                attempt=0,
                wheelhouse=output,
                metadata=metadata,
                now=now,
            )


async def test_dependency_build_stages_metadata_idempotently(session_factory):
    now = utcnow()
    async with session_factory() as session:
        repo = DependencyBuildRepository(session)
        record = await _begin_build_attempt(repo, now=now)
        metadata = WheelhouseMetadata(
            build_key=record.build_key,
            nexus_wheel_path='wheels/nexus_agent.whl',
        )

        first = await repo.stage_metadata(
            record.dependency_build_id,
            attempt=1,
            metadata=metadata,
            now=now,
        )
        repeated = await repo.stage_metadata(
            record.dependency_build_id,
            attempt=1,
            metadata=metadata,
            now=now,
        )

    assert first.metadata == metadata
    assert first.wheelhouse is None
    assert repeated.metadata == metadata


async def test_dependency_build_stage_metadata_rejects_stale_attempt(session_factory):
    now = utcnow()
    async with session_factory() as session:
        repo = DependencyBuildRepository(session)
        record = await _begin_build_attempt(repo, now=now)
        metadata = WheelhouseMetadata(
            build_key=record.build_key,
            nexus_wheel_path='wheels/nexus_agent.whl',
        )

        with pytest.raises(ValueError, match='dependency_build_attempt_mismatch'):
            await repo.stage_metadata(
                record.dependency_build_id,
                attempt=0,
                metadata=metadata,
                now=now,
            )


async def test_dependency_build_stage_metadata_rejects_conflict(session_factory):
    now = utcnow()
    async with session_factory() as session:
        repo = DependencyBuildRepository(session)
        record = await _begin_build_attempt(repo, now=now)
        metadata = WheelhouseMetadata(
            build_key=record.build_key,
            nexus_wheel_path='wheels/nexus_agent.whl',
        )
        await repo.stage_metadata(
            record.dependency_build_id,
            attempt=1,
            metadata=metadata,
            now=now,
        )

        with pytest.raises(ValueError, match='dependency_build_metadata_conflict'):
            await repo.stage_metadata(
                record.dependency_build_id,
                attempt=1,
                metadata=metadata.model_copy(update={'nexus_wheel_path': 'wheels/other.whl'}),
                now=now,
            )


async def test_dependency_build_stage_output_must_match_staged_metadata(session_factory):
    now = utcnow()
    async with session_factory() as session:
        repo = DependencyBuildRepository(session)
        record = await _begin_build_attempt(repo, now=now)
        metadata = WheelhouseMetadata(
            build_key=record.build_key,
            nexus_wheel_path='wheels/nexus_agent.whl',
        )
        await repo.stage_metadata(
            record.dependency_build_id,
            attempt=1,
            metadata=metadata,
            now=now,
        )

        with pytest.raises(ValueError, match='dependency_build_output_conflict'):
            await repo.stage_output(
                record.dependency_build_id,
                attempt=1,
                wheelhouse=_ref('wheelhouse', 'f'),
                metadata=metadata.model_copy(update={'nexus_wheel_path': 'wheels/other.whl'}),
                now=now,
            )


async def test_dependency_build_stage_failure_is_attempt_fenced_and_bounded(session_factory):
    now = utcnow()
    async with session_factory() as session:
        repo = DependencyBuildRepository(session)
        record, _ = await repo.get_or_create(_spec(), now=now, **_provider_target())
        claim = await repo.claim_next(
            operation_owner='reconciler-a',
            lease_ttl=timedelta(minutes=2),
            now=now,
        )
        assert claim is not None
        await repo.begin_attempt(
            record.dependency_build_id,
            operation_id=claim.operation_id,
            attempt=1,
            provider_job_name='wheelhouse-build-1',
            now=now,
        )

        staged = await repo.stage_failure(
            record.dependency_build_id,
            attempt=1,
            error_code='OfflineInstallFailed' * 20,
            error_message='missing dependency wheel ' * 200,
            now=now,
        )

        assert staged.status == DependencyBuildStatus.BUILDING
        assert len(staged.error_code) == 128
        assert len(staged.error_message) == 2048
        with pytest.raises(ValueError, match='dependency_build_attempt_mismatch'):
            await repo.stage_failure(
                record.dependency_build_id,
                attempt=2,
                error_code='stale',
                error_message='stale',
                now=now,
            )


async def test_dependency_build_rejects_stale_operation(session_factory):
    async with session_factory() as session:
        repo = DependencyBuildRepository(session)
        record, _ = await repo.get_or_create(_spec(), now=utcnow(), **_provider_target())

        with pytest.raises(ValueError, match='dependency_build_lease_lost'):
            await repo.mark_failed(
                record.dependency_build_id,
                operation_id='stale-operation',
                error_code='DependencyBuildFailed',
                error_message='compile failed',
                now=utcnow(),
            )


def test_dependency_build_content_ticket_has_no_execution_or_control_scope():
    signer = ArtifactContentTicketSigner(
        keys={'current': 'secret'},
        current_key_id='current',
        ttl=timedelta(minutes=5),
    )

    claims = signer.verify(
        signer.issue_dependency_build(
            dependency_build_id='bld-1',
            attempt=2,
            environment_id='env-1',
            operations=['artifact:read', 'wheelhouse:write', 'dependency_build:report_failure'],
            artifact_ids=['source', 'wheel', 'lock'],
            now=utcnow(),
        ),
        now=utcnow(),
    )

    assert claims.dependency_build_id == 'bld-1'
    assert claims.dependency_build_attempt == 2
    assert claims.execution_id is None
    assert claims.allocation_id is None
    assert claims.operations == ['artifact:read', 'wheelhouse:write', 'dependency_build:report_failure']
