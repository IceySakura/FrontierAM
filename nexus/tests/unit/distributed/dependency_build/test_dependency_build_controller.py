from __future__ import annotations

import asyncio

from datetime import timedelta

import pytest

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from nexus.distributed.artifacts.models import ArtifactRef
from nexus.distributed.config import DistributedConfig
from nexus.distributed.dependency_build import controller as dependency_build_controller_module
from nexus.distributed.dependency_build.controller import DependencyBuildController, _job_outcome
from nexus.distributed.dependency_build.wheelhouse import (
    WheelhouseBuildSpec,
    WheelhouseMetadata,
    wheelhouse_build_key,
)
from nexus.distributed.environment.kubernetes import WorkloadNotFoundError, WorkloadPermanentError
from nexus.distributed.environment.providers import WorkloadClientResolver
from nexus.distributed.persistence.orm import DependencyBuildRow, EnvironmentRow, create_schema
from nexus.distributed.repositories.dependency_build import DependencyBuildRepository
from nexus.distributed.utils.clock import utcnow


def test_dependency_build_job_outcome_accepts_uncounted_success() -> None:
    assert (
        _job_outcome({
            'status': {
                'uncountedTerminatedPods': {
                    'succeeded': ['pod-uid'],
                }
            }
        })
        == 'succeeded'
    )


@pytest.mark.parametrize(
    ('backoff_limit', 'expected'),
    [
        (0, 'failed'),
        (1, 'active'),
    ],
)
def test_dependency_build_job_outcome_only_accepts_uncounted_failure_without_retries(
    backoff_limit: int,
    expected: str,
) -> None:
    assert (
        _job_outcome({
            'spec': {'backoffLimit': backoff_limit},
            'status': {
                'uncountedTerminatedPods': {
                    'failed': ['pod-uid'],
                }
            },
        })
        == expected
    )


class FakeBuildClient:
    def __init__(self) -> None:
        self.secrets: dict[str, dict] = {}
        self.jobs: dict[str, dict] = {}

    async def get_secret(self, name):
        if name not in self.secrets:
            raise WorkloadNotFoundError(name)
        return self.secrets[name]

    async def create_secret(self, body):
        self.secrets[body['metadata']['name']] = body
        return body

    async def patch_secret(self, name, body):
        self.secrets[name] = body
        return body

    async def delete_secret(self, name):
        self.secrets.pop(name, None)
        return {}

    async def get_job(self, name):
        if name not in self.jobs:
            raise WorkloadNotFoundError(name)
        return self.jobs[name]

    async def create_job(self, body):
        self.jobs[body['metadata']['name']] = body
        return body

    async def delete_job(self, name):
        self.jobs.pop(name, None)
        return {}

    async def list_job_pods(self, job_name):
        return []

    async def list_resource_events(self, resource_name):
        return []

    async def read_pod_log(self, pod_name, *, container, tail_lines, limit_bytes):
        del limit_bytes
        return ''

    async def close(self):
        return None


class ReferenceProtectingBuildClient(FakeBuildClient):
    def __init__(self) -> None:
        super().__init__()
        self.deleting_jobs: set[str] = set()

    async def get_job(self, name):
        if name in self.deleting_jobs:
            self.deleting_jobs.remove(name)
            self.jobs.pop(name, None)
            raise WorkloadNotFoundError(name)
        return await super().get_job(name)

    async def delete_job(self, name):
        self.deleting_jobs.add(name)
        return {}

    async def delete_secret(self, name):
        if self.jobs:
            raise WorkloadPermanentError('secret_referenced_by_job')
        return await super().delete_secret(name)


class InspectingBuildClient(FakeBuildClient):
    def __init__(self, session_factory: async_sessionmaker) -> None:
        super().__init__()
        self.session_factory = session_factory
        self.record_at_job_create = None

    async def create_job(self, body):
        build_id = body['metadata']['annotations']['nexus.io/dependency-build-id']
        async with self.session_factory() as session:
            self.record_at_job_create = await DependencyBuildRepository(session).get(build_id)
        return await super().create_job(body)


class TimeoutAfterCreateBuildClient(FakeBuildClient):
    def __init__(self) -> None:
        super().__init__()
        self.create_job_calls = 0

    async def create_job(self, body):
        self.create_job_calls += 1
        created = await super().create_job(body)
        if self.create_job_calls == 1:
            raise TimeoutError('provider response lost')
        return created


class BlockingCreateBuildClient(FakeBuildClient):
    def __init__(self) -> None:
        super().__init__()
        self.create_started = asyncio.Event()
        self.allow_create_return = asyncio.Event()

    async def create_job(self, body):
        created = await super().create_job(body)
        self.create_started.set()
        await self.allow_create_return.wait()
        return created


class DiagnosticBuildClient(FakeBuildClient):
    def __init__(self, session_factory: async_sessionmaker) -> None:
        super().__init__()
        self.session_factory = session_factory
        self.error_message_at_delete: str | None = None
        self.error_messages_at_delete: list[str | None] = []

    async def list_job_pods(self, job_name):
        assert job_name in self.jobs
        return [
            {
                'metadata': {'name': f'{job_name}-pod'},
                'status': {
                    'phase': 'Failed',
                    'reason': 'Evicted',
                    'message': 'node had disk pressure',
                    'containerStatuses': [
                        {
                            'name': 'builder',
                            'state': {
                                'terminated': {
                                    'exitCode': 137,
                                    'reason': 'OOMKilled',
                                    'message': 'container exceeded memory limit',
                                }
                            },
                        }
                    ],
                },
            }
        ]

    async def list_resource_events(self, resource_name):
        return [
            {
                'type': 'Warning',
                'reason': 'FailedMount',
                'message': f'could not mount projected token for {resource_name}',
                'count': 2,
            }
        ]

    async def read_pod_log(self, pod_name, *, container, tail_lines, limit_bytes):
        assert container == 'builder'
        assert tail_lines == 50
        assert limit_bytes == 256 * 1024
        return f'{pod_name}: token=super-secret-token\nfatal: build process exited'

    async def _record_error_message(self, name):
        build_id = self.jobs[name]['metadata']['annotations']['nexus.io/dependency-build-id']
        async with self.session_factory() as session:
            record = await DependencyBuildRepository(session).get(build_id)
        self.error_message_at_delete = record.error_message
        self.error_messages_at_delete.append(record.error_message)

    async def delete_job(self, name):
        await self._record_error_message(name)
        return await super().delete_job(name)


class RetryDeleteDiagnosticBuildClient(DiagnosticBuildClient):
    async def delete_job(self, name):
        await self._record_error_message(name)
        if len(self.error_messages_at_delete) == 1:
            raise TimeoutError('provider delete timed out')
        return await FakeBuildClient.delete_job(self, name)


class JobEventOnlyDiagnosticBuildClient(DiagnosticBuildClient):
    async def list_job_pods(self, job_name):
        assert job_name in self.jobs
        return []

    async def list_resource_events(self, resource_name):
        assert resource_name in self.jobs
        return [
            {
                'type': 'Warning',
                'reason': 'DeadlineExceeded',
                'message': 'Job was active longer than specified deadline',
                'count': 1,
            }
        ]


class HangingDiagnosticBuildClient(FakeBuildClient):
    async def list_job_pods(self, job_name):
        await asyncio.Event().wait()
        return []


class CapturingClientResolver:
    def __init__(self, client: FakeBuildClient) -> None:
        self.client = client
        self.calls: list[dict] = []

    def resolve(self, **kwargs):
        self.calls.append(kwargs)
        return type('ResolvedClient', (), {'client': self.client})()

    async def close(self):
        return None


@pytest.fixture
async def build_session_factory(tmp_path):
    engine = create_async_engine(f'sqlite+aiosqlite:///{tmp_path / "build-controller.db"}')
    await create_schema(engine)
    try:
        yield async_sessionmaker(engine, expire_on_commit=False)
    finally:
        await engine.dispose()


def _ref(kind: str, marker: str) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=f'art-{kind}',
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
        extras=['runtime'],
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
            'auth_type': 'secret',
            'api_url': 'https://kubernetes.snapshot.example',
            'credential_ref': 'kubernetes-default',
        },
    }


def _controller_config(**updates) -> DistributedConfig:
    return DistributedConfig(**updates)


def _controller(
    *,
    session_factory: async_sessionmaker,
    client: FakeBuildClient,
    config: DistributedConfig | None = None,
) -> DependencyBuildController:
    effective_config = config or _controller_config()
    resolver = WorkloadClientResolver(
        config=effective_config,
        clients={('kubernetes', 'kubernetes/default'): client},
    )
    return DependencyBuildController(
        session_factory=session_factory,
        config=effective_config,
        client_resolver=resolver,
    )


async def _create_build(build_session_factory):
    now = utcnow()
    async with build_session_factory() as session:
        session.add(
            EnvironmentRow(
                environment_id='env-1',
                name='env-1',
                allocation_type='dynamic',
                provider_type='kubernetes',
                provider_config_ref='kubernetes/default',
                resource_limits_json={},
                default_timeouts_json={},
                config_version=1,
                status='enabled',
                created_by='admin',
                updated_by='admin',
                created_at=now,
                updated_at=now,
            )
        )
        build, _ = await DependencyBuildRepository(session).get_or_create(
            _spec(),
            now=now,
            **_provider_target(),
        )
        await session.commit()
    return build


async def test_dependency_build_controller_waits_for_job_deletion_before_committing_staged_output(
    build_session_factory,
):
    now = utcnow()
    async with build_session_factory() as session:
        session.add(
            EnvironmentRow(
                environment_id='env-1',
                name='env-1',
                allocation_type='dynamic',
                provider_type='kubernetes',
                provider_config_ref='kubernetes/default',
                resource_limits_json={},
                default_timeouts_json={},
                config_version=1,
                status='enabled',
                created_by='admin',
                updated_by='admin',
                created_at=now,
                updated_at=now,
            )
        )
        build, _ = await DependencyBuildRepository(session).get_or_create(
            _spec(),
            now=now,
            **_provider_target(),
        )
        await session.commit()

    client = ReferenceProtectingBuildClient()
    config = _controller_config(
        dependency_build_poll_interval=timedelta(milliseconds=1),
        dependency_build_retry=timedelta(milliseconds=1),
        dependency_builder_index_url='https://pypi.internal.example/simple/',
        environment_image_pull_secrets=['csighub-yunzhiluo'],
        application_internal_base_url='https://nexus-application.internal.example',
    )
    controller = _controller(
        session_factory=build_session_factory,
        client=client,
        config=config,
    )

    assert await controller.reconcile_once() == 1
    async with build_session_factory() as session:
        building = await DependencyBuildRepository(session).get(build.dependency_build_id)
        assert building.status == 'building'
        assert building.attempt == 1
        assert building.operation_id is None
        assert building.provider_job_name in client.jobs
        job = client.jobs[building.provider_job_name]
        assert job['metadata']['annotations']['nexus.io/dependency-build-attempt'] == '1'
        secret = next(iter(client.secrets.values()))
        assert secret['metadata']['annotations']['nexus.io/dependency-build-attempt'] == '1'
        assert job['spec']['template']['spec']['automountServiceAccountToken'] is False
        assert job['spec']['template']['spec']['imagePullSecrets'] == [{'name': 'csighub-yunzhiluo'}]
        container = job['spec']['template']['spec']['containers'][0]
        assert container['image'] == _spec().builder_image
        assert container['command'] == ['/opt/nexus-venv/bin/python']
        assert container['args'] == ['-m', 'nexus.distributed.dependency_build.worker']
        env = {entry['name']: entry['value'] for entry in container['env'] if 'value' in entry}
        assert env['NEXUS_DIST_DEPENDENCY_BUILD_ATTEMPT'] == '1'
        assert env['NEXUS_DIST_DEPENDENCY_BUILD_KEY'] == building.build_key
        assert env['NEXUS_DIST_DEPENDENCY_BUILD_WORKSPACE'] == '/work/build'
        assert env['TMPDIR'] == '/work'
        assert env['UV_CACHE_DIR'] == '/work/uv-cache'
        assert env['PIP_INDEX_URL'] == 'https://pypi.internal.example/simple/'
        assert env['UV_INDEX_URL'] == 'https://pypi.internal.example/simple/'
        assert env['NEXUS_DIST_APPLICATION_INTERNAL_BASE_URL'] == 'https://nexus-application.internal.example'
        assert env['NEXUS_DIST_ARTIFACT_CONTENT_TOKEN_FILE'] == ('/var/run/secrets/nexus/build/artifact-content-token')
        assert (
            env['NEXUS_DIST_ARTIFACT_DOWNLOAD_DEADLINE_AT']
            == (building.started_at + timedelta(seconds=config.dependency_builder_active_deadline_seconds)).isoformat()
        )
        assert container['volumeMounts'][0] == {'name': 'workspace', 'mountPath': '/work'}
        assert container['volumeMounts'][1] == {
            'name': 'nexus-secrets',
            'mountPath': '/var/run/secrets/nexus',
            'readOnly': True,
        }
        nexus_secrets = next(
            volume for volume in job['spec']['template']['spec']['volumes'] if volume['name'] == 'nexus-secrets'
        )
        assert nexus_secrets == {
            'name': 'nexus-secrets',
            'projected': {
                'sources': [
                    {
                        'secret': {
                            'name': next(iter(client.secrets)),
                            'items': [
                                {'key': 'token', 'path': 'build/artifact-content-token'},
                                {'key': 'log-token', 'path': 'build/log-token'},
                            ],
                        }
                    }
                ]
            },
        }
        assert container['securityContext']['readOnlyRootFilesystem'] is True

        output = _ref('wheelhouse', 'f')
        metadata = WheelhouseMetadata(
            build_key=wheelhouse_build_key(_spec()),
            nexus_wheel_path='wheels/nexus_agent.whl',
        )
        await DependencyBuildRepository(session).stage_output(
            build.dependency_build_id,
            attempt=building.attempt,
            wheelhouse=output,
            metadata=metadata,
            now=utcnow(),
        )
        await session.commit()

    client.jobs[building.provider_job_name]['status'] = {'succeeded': 1}
    await asyncio.sleep(0.01)
    assert await controller.reconcile_once() == 1

    async with build_session_factory() as session:
        deleting = await DependencyBuildRepository(session).get(build.dependency_build_id)
        assert deleting.status == 'building'
    assert client.jobs
    assert client.secrets

    await asyncio.sleep(0.01)
    assert await controller.reconcile_once() == 1

    async with build_session_factory() as session:
        ready = await DependencyBuildRepository(session).get(build.dependency_build_id)
        assert ready.status == 'ready'
        assert ready.wheelhouse == output
    assert not client.jobs
    assert not client.secrets
    await controller.close()


async def test_dependency_build_controller_persists_attempt_before_creating_provider_job(build_session_factory):
    build = await _create_build(build_session_factory)
    client = InspectingBuildClient(build_session_factory)
    controller = _controller(
        session_factory=build_session_factory,
        client=client,
    )

    assert await controller.reconcile_once() == 1

    persisted = client.record_at_job_create
    assert persisted is not None
    assert persisted.dependency_build_id == build.dependency_build_id
    assert persisted.status == 'building'
    assert persisted.attempt == 1
    assert persisted.provider_job_name == next(iter(client.jobs))
    await controller.close()


async def test_dependency_build_controller_uses_submission_provider_snapshot_after_environment_changes(
    build_session_factory,
):
    build = await _create_build(build_session_factory)
    async with build_session_factory() as session:
        environment = await session.get(EnvironmentRow, 'env-1')
        assert environment is not None
        environment.provider_type = 'tkex'
        environment.provider_config_ref = 'tkex/replacement'
        await session.commit()

    client = FakeBuildClient()
    resolver = CapturingClientResolver(client)
    controller = DependencyBuildController(
        session_factory=build_session_factory,
        config=_controller_config(),
        client_resolver=resolver,
    )

    assert await controller.reconcile_once() == 1
    assert resolver.calls == [
        {
            'provider_type': 'kubernetes',
            'provider_config_ref': 'kubernetes/default',
            'snapshot': _provider_target()['provider_config_snapshot'],
        }
    ]
    assert client.jobs
    async with build_session_factory() as session:
        record = await DependencyBuildRepository(session).get(build.dependency_build_id)
    assert record.status == 'building'
    await controller.close()


async def test_dependency_build_controller_reuses_attempt_after_create_response_timeout(build_session_factory):
    build = await _create_build(build_session_factory)
    client = TimeoutAfterCreateBuildClient()
    controller = _controller(
        session_factory=build_session_factory,
        client=client,
        config=_controller_config(
            dependency_build_poll_interval=timedelta(milliseconds=1),
            dependency_build_retry=timedelta(milliseconds=1),
        ),
    )

    assert await controller.reconcile_once() == 1
    async with build_session_factory() as session:
        after_timeout = await DependencyBuildRepository(session).get(build.dependency_build_id)
    assert after_timeout.status == 'building'
    assert after_timeout.attempt == 1
    first_job_name = after_timeout.provider_job_name

    await asyncio.sleep(0.01)
    assert await controller.reconcile_once() == 1
    async with build_session_factory() as session:
        retried = await DependencyBuildRepository(session).get(build.dependency_build_id)
    assert retried.status == 'building'
    assert retried.attempt == 1
    assert retried.provider_job_name == first_job_name
    assert client.create_job_calls == 1
    await controller.close()


async def test_dependency_build_controller_stops_after_lease_is_stolen_during_provider_call(
    build_session_factory,
):
    build = await _create_build(build_session_factory)
    client = BlockingCreateBuildClient()
    controller = _controller(
        session_factory=build_session_factory,
        client=client,
        config=_controller_config(
            dependency_build_lease_ttl=timedelta(seconds=2),
            dependency_build_lease_renew_interval=timedelta(milliseconds=20),
        ),
    )

    reconcile = asyncio.create_task(controller.reconcile_once())
    await asyncio.wait_for(client.create_started.wait(), timeout=1)
    async with build_session_factory() as session:
        row = await session.get(DependencyBuildRow, build.dependency_build_id)
        assert row is not None
        row.operation_id = 'op-stolen'
        row.operation_owner = 'other-controller'
        row.operation_lease_until = utcnow() + timedelta(minutes=1)
        await session.commit()
    await asyncio.sleep(0.02)
    client.allow_create_return.set()

    assert await asyncio.wait_for(reconcile, timeout=1) == 1
    async with build_session_factory() as session:
        persisted = await DependencyBuildRepository(session).get(build.dependency_build_id)
    assert persisted.status == 'building'
    assert persisted.attempt == 1
    assert persisted.operation_id == 'op-stolen'
    assert persisted.operation_owner == 'other-controller'
    assert persisted.provider_job_name in client.jobs
    await controller.close()


async def test_dependency_build_controller_cleans_failed_attempt_before_retry(build_session_factory):
    now = utcnow()
    async with build_session_factory() as session:
        session.add(
            EnvironmentRow(
                environment_id='env-1',
                name='env-1',
                allocation_type='dynamic',
                provider_type='kubernetes',
                provider_config_ref='kubernetes/default',
                resource_limits_json={},
                default_timeouts_json={},
                config_version=1,
                status='enabled',
                created_by='admin',
                updated_by='admin',
                created_at=now,
                updated_at=now,
            )
        )
        build, _ = await DependencyBuildRepository(session).get_or_create(
            _spec(),
            now=now,
            **_provider_target(),
        )
        await session.commit()

    client = FakeBuildClient()
    config = _controller_config(
        dependency_build_poll_interval=timedelta(milliseconds=1),
        dependency_build_retry=timedelta(milliseconds=1),
    )
    controller = _controller(
        session_factory=build_session_factory,
        client=client,
        config=config,
    )

    assert await controller.reconcile_once() == 1
    async with build_session_factory() as session:
        building = await DependencyBuildRepository(session).get(build.dependency_build_id)
    client.jobs[building.provider_job_name]['status'] = {
        'failed': 1,
        'conditions': [{'type': 'Failed', 'status': 'True', 'message': 'build failed'}],
    }

    await asyncio.sleep(0.01)
    assert await controller.reconcile_once() == 1

    async with build_session_factory() as session:
        failed = await DependencyBuildRepository(session).get(build.dependency_build_id)
        assert failed.status == 'failed'
        assert failed.error_message == 'build failed'
        assert failed.next_retry_at is not None
    assert not client.jobs
    assert not client.secrets
    await controller.close()


async def test_dependency_build_controller_persists_pod_diagnostics_before_deleting_failed_job(
    build_session_factory,
):
    build = await _create_build(build_session_factory)
    client = DiagnosticBuildClient(build_session_factory)
    controller = _controller(
        session_factory=build_session_factory,
        client=client,
        config=_controller_config(
            dependency_build_poll_interval=timedelta(milliseconds=1),
            dependency_build_retry=timedelta(milliseconds=1),
        ),
    )

    assert await controller.reconcile_once() == 1
    async with build_session_factory() as session:
        building = await DependencyBuildRepository(session).get(build.dependency_build_id)
    client.jobs[building.provider_job_name]['status'] = {
        'failed': 1,
        'conditions': [
            {
                'type': 'Failed',
                'status': 'True',
                'reason': 'BackoffLimitExceeded',
                'message': 'Job has reached the specified backoff limit',
            }
        ],
    }

    await asyncio.sleep(0.01)
    assert await controller.reconcile_once() == 1

    assert client.error_message_at_delete is not None
    assert 'pod=' in client.error_message_at_delete
    assert 'phase=Failed' in client.error_message_at_delete
    assert 'pod_reason=Evicted' in client.error_message_at_delete
    assert 'exit_code=137' in client.error_message_at_delete
    assert 'termination_reason=OOMKilled' in client.error_message_at_delete
    assert 'termination_message=container exceeded memory limit' in client.error_message_at_delete
    assert 'event=Warning/FailedMount' in client.error_message_at_delete
    assert 'log_tail=' in client.error_message_at_delete
    assert 'fatal: build process exited' in client.error_message_at_delete
    assert 'super-secret-token' not in client.error_message_at_delete
    assert '<redacted>' in client.error_message_at_delete
    async with build_session_factory() as session:
        failed = await DependencyBuildRepository(session).get(build.dependency_build_id)
    assert failed.error_message == client.error_message_at_delete
    assert failed.next_retry_at is not None
    assert not client.jobs
    assert not client.secrets
    await controller.close()


async def test_dependency_build_controller_persists_job_events_when_failed_pods_are_gone(
    build_session_factory,
):
    build = await _create_build(build_session_factory)
    client = JobEventOnlyDiagnosticBuildClient(build_session_factory)
    controller = _controller(
        session_factory=build_session_factory,
        client=client,
        config=_controller_config(
            dependency_build_poll_interval=timedelta(milliseconds=1),
            dependency_build_retry=timedelta(milliseconds=1),
        ),
    )

    assert await controller.reconcile_once() == 1
    async with build_session_factory() as session:
        building = await DependencyBuildRepository(session).get(build.dependency_build_id)
    client.jobs[building.provider_job_name]['status'] = {
        'failed': 1,
        'conditions': [
            {
                'type': 'Failed',
                'status': 'True',
                'reason': 'DeadlineExceeded',
                'message': 'Job was active longer than specified deadline',
            }
        ],
    }

    await asyncio.sleep(0.01)
    assert await controller.reconcile_once() == 1

    assert client.error_message_at_delete is not None
    assert 'provider_diagnostics:' in client.error_message_at_delete
    assert 'event=Warning/DeadlineExceeded' in client.error_message_at_delete
    async with build_session_factory() as session:
        failed = await DependencyBuildRepository(session).get(build.dependency_build_id)
    assert failed.error_message == client.error_message_at_delete
    assert failed.next_retry_at is not None
    assert not client.jobs
    assert not client.secrets
    await controller.close()


async def test_dependency_build_controller_bounds_diagnostic_collection_time(build_session_factory, monkeypatch):
    build = await _create_build(build_session_factory)
    client = HangingDiagnosticBuildClient()
    controller = _controller(
        session_factory=build_session_factory,
        client=client,
        config=_controller_config(
            dependency_build_poll_interval=timedelta(milliseconds=1),
            dependency_build_retry=timedelta(milliseconds=1),
        ),
    )
    monkeypatch.setattr(dependency_build_controller_module, '_DIAGNOSTIC_REQUEST_TIMEOUT_SECONDS', 0.01, raising=False)

    try:
        assert await controller.reconcile_once() == 1
        async with build_session_factory() as session:
            building = await DependencyBuildRepository(session).get(build.dependency_build_id)
        client.jobs[building.provider_job_name]['status'] = {
            'failed': 1,
            'conditions': [{'type': 'Failed', 'status': 'True', 'message': 'build failed'}],
        }

        await asyncio.sleep(0.01)
        assert await asyncio.wait_for(controller.reconcile_once(), timeout=0.5) == 1
    finally:
        await controller.close()

    async with build_session_factory() as session:
        failed = await DependencyBuildRepository(session).get(build.dependency_build_id)
    assert failed.error_message == 'build failed'
    assert failed.next_retry_at is not None
    assert not client.jobs


async def test_dependency_build_controller_replaces_diagnostics_when_job_cleanup_retries(build_session_factory):
    build = await _create_build(build_session_factory)
    client = RetryDeleteDiagnosticBuildClient(build_session_factory)
    controller = _controller(
        session_factory=build_session_factory,
        client=client,
        config=_controller_config(
            dependency_build_poll_interval=timedelta(milliseconds=1),
            dependency_build_retry=timedelta(milliseconds=1),
        ),
    )

    try:
        assert await controller.reconcile_once() == 1
        async with build_session_factory() as session:
            building = await DependencyBuildRepository(session).get(build.dependency_build_id)
        client.jobs[building.provider_job_name]['status'] = {
            'failed': 1,
            'conditions': [{'type': 'Failed', 'status': 'True', 'message': 'build failed'}],
        }

        await asyncio.sleep(0.01)
        assert await controller.reconcile_once() == 1
        await asyncio.sleep(0.01)
        assert await controller.reconcile_once() == 1
    finally:
        await controller.close()

    assert len(client.error_messages_at_delete) == 2
    assert client.error_messages_at_delete[0] == client.error_messages_at_delete[1]
    assert client.error_messages_at_delete[1].count('provider_diagnostics:') == 1


async def test_dependency_build_controller_persists_builder_reported_error(build_session_factory):
    build = await _create_build(build_session_factory)
    client = FakeBuildClient()
    controller = _controller(
        session_factory=build_session_factory,
        client=client,
        config=_controller_config(
            dependency_build_poll_interval=timedelta(milliseconds=1),
            dependency_build_retry=timedelta(milliseconds=1),
        ),
    )

    assert await controller.reconcile_once() == 1
    async with build_session_factory() as session:
        building = await DependencyBuildRepository(session).get(build.dependency_build_id)
        await DependencyBuildRepository(session).stage_failure(
            build.dependency_build_id,
            attempt=building.attempt,
            error_code='dependency_offline_install_failed',
            error_message='missing dependency wheel; token=builder-secret-token',
            now=utcnow(),
        )
        await session.commit()
    client.jobs[building.provider_job_name]['status'] = {
        'failed': 1,
        'conditions': [{'type': 'Failed', 'status': 'True', 'message': 'Job failed'}],
    }

    await asyncio.sleep(0.01)
    assert await controller.reconcile_once() == 1

    async with build_session_factory() as session:
        failed = await DependencyBuildRepository(session).get(build.dependency_build_id)
    assert failed.status == 'failed'
    assert failed.error_code == 'dependency_offline_install_failed'
    assert failed.error_message == 'missing dependency wheel; token=<redacted>'
    await controller.close()


@pytest.mark.parametrize(
    'error_code', ['dependency_source_package_invalid', 'dependency_nexus_wheel_invalid', 'dependency_build_key_mismatch']
)
async def test_dependency_build_controller_does_not_retry_invalid_inputs(build_session_factory, error_code):
    build = await _create_build(build_session_factory)
    client = FakeBuildClient()
    controller = _controller(
        session_factory=build_session_factory,
        client=client,
        config=_controller_config(
            dependency_build_poll_interval=timedelta(milliseconds=1),
            dependency_build_retry=timedelta(milliseconds=1),
        ),
    )

    assert await controller.reconcile_once() == 1
    async with build_session_factory() as session:
        building = await DependencyBuildRepository(session).get(build.dependency_build_id)
        await DependencyBuildRepository(session).stage_failure(
            build.dependency_build_id,
            attempt=building.attempt,
            error_code=error_code,
            error_message='invalid immutable input',
            now=utcnow(),
        )
        await session.commit()
    client.jobs[building.provider_job_name]['status'] = {'failed': 1}

    await asyncio.sleep(0.01)
    assert await controller.reconcile_once() == 1

    async with build_session_factory() as session:
        failed = await DependencyBuildRepository(session).get(build.dependency_build_id)
    assert failed.status == 'failed'
    assert failed.error_code == error_code
    assert failed.next_retry_at is None
    await controller.close()


@pytest.mark.parametrize('exit_code', [0, 1])
@pytest.mark.parametrize('collector_state', [{'running': {}}, {'waiting': {}}, {}, {'terminated': {'exitCode': 1}}])
def test_builder_collection_wait_is_bounded_and_preserves_exit_code(exit_code, collector_state):
    from nexus.distributed.dependency_build.controller import _builder_outcome

    now = utcnow()
    pods = [
        {
            'status': {
                'containerStatuses': [
                    {
                        'name': 'builder',
                        'state': {'terminated': {'exitCode': exit_code, 'finishedAt': now.isoformat()}},
                    },
                    {'name': 'zhiyan-agent', 'state': collector_state},
                ]
            }
        }
    ]
    outcome = 'succeeded' if exit_code == 0 else 'failed'
    assert _builder_outcome(pods, now=now) == (
        outcome if exit_code == 0 or 'terminated' in collector_state else 'active'
    )
    assert _builder_outcome(pods, now=now + timedelta(seconds=31)) == outcome


async def test_builder_renderer_preserves_legacy_record_key(build_session_factory):
    from dataclasses import replace

    from nexus.distributed.dependency_build.rendering import DependencyBuildJobRenderer
    from nexus.distributed.dependency_build.wheelhouse import wheelhouse_build_key

    original = await _create_build(build_session_factory)
    legacy_key = wheelhouse_build_key(original.build_spec, version=3)
    build = replace(original, build_key=legacy_key, attempt=1, started_at=utcnow())
    renderer = DependencyBuildJobRenderer(config=_controller_config())
    container = renderer.render(build, token='token').job['spec']['template']['spec']['containers'][0]
    env = {item['name']: item['value'] for item in container['env'] if 'value' in item}

    assert legacy_key != wheelhouse_build_key(original.build_spec)
    assert env['NEXUS_DIST_DEPENDENCY_BUILD_KEY'] == legacy_key


async def test_builder_collection_uses_explicit_provider_configuration(build_session_factory):
    from dataclasses import replace

    from nexus.distributed.dependency_build.rendering import DependencyBuildJobRenderer

    build = replace(await _create_build(build_session_factory), attempt=1, started_at=utcnow())
    controller = _controller(session_factory=build_session_factory, client=FakeBuildClient())
    renderer = DependencyBuildJobRenderer(config=controller.config)
    assert len(renderer.render(build, token='token').job['spec']['template']['spec']['containers']) == 1
    config = {
        'log_name': 'build-flow',
        'agent_image': 'mirrors.tencent.com/zhiyan-agent/agent-integration:0.0.12',
        'volume_size_bytes': 16777216,
    }
    manager_only = renderer.render(
        replace(build, provider_type='tkex', provider_config_snapshot={'zhiyan_log': config}), token='token'
    ).job['spec']['template']['spec']
    assert [container['name'] for container in manager_only['containers']] == ['builder']
    pod = renderer.render(
        replace(build, provider_type='tkex', provider_config_snapshot={'builder_zhiyan_log': config}), token='token'
    ).job['spec']['template']['spec']
    builder, agent = pod['containers']
    assert agent['image'] == config['agent_image']
    assert {'name': 'ZHIYAN_LOG_NAME', 'value': 'build-flow'} in agent['env']
    assert {'name': 'log', 'mountPath': '/data/logs'} in agent['volumeMounts']
    assert {'name': 'log', 'mountPath': '/work/collected-logs'} in builder['volumeMounts']
    await controller.close()


@pytest.mark.parametrize('delete_target', ['delete_job', 'delete_secret'])
async def test_successful_builder_releases_scheduling_before_delayed_cleanup(
    build_session_factory, monkeypatch, delete_target
):
    build = await _create_build(build_session_factory)
    client = FakeBuildClient()
    now = utcnow()
    monkeypatch.setattr(dependency_build_controller_module, 'utcnow', lambda: now)
    config = _controller_config(dependency_build_poll_interval=timedelta(seconds=1))
    controller = _controller(session_factory=build_session_factory, client=client, config=config)
    await controller.reconcile_once()
    async with build_session_factory() as session:
        repo = DependencyBuildRepository(session)
        building = await repo.get(build.dependency_build_id)
        await repo.stage_output(
            build.dependency_build_id,
            attempt=1,
            wheelhouse=_ref('wheelhouse', 'f'),
            metadata=WheelhouseMetadata(build_key=wheelhouse_build_key(_spec()), nexus_wheel_path='wheels/nexus.whl'),
            now=now,
        )
        await session.commit()
    client.jobs[building.provider_job_name]['spec']['template']['spec']['containers'].append({'name': 'zhiyan-agent'})

    async def pods(_):
        return [
            {
                'status': {
                    'containerStatuses': [
                        {'name': 'builder', 'state': {'terminated': {'exitCode': 0, 'finishedAt': now.isoformat()}}},
                        {'name': 'zhiyan-agent', 'state': {'running': {}}},
                    ]
                }
            }
        ]

    client.list_job_pods = pods
    now += timedelta(seconds=2)
    await controller.reconcile_once()
    async with build_session_factory() as session:
        ready = await DependencyBuildRepository(session).get(build.dependency_build_id)
    assert ready.status == 'ready'
    assert ready.next_retry_at == now + timedelta(seconds=30)
    assert client.jobs and client.secrets
    assert await controller.reconcile_once() == 0
    # A new controller must recover scheduled cleanup without changing the result.
    controller = _controller(session_factory=build_session_factory, client=client, config=config)
    now += timedelta(seconds=31)

    async def failing_delete(_):
        raise WorkloadPermanentError('temporarily forbidden')

    original_delete = getattr(client, delete_target)
    setattr(client, delete_target, failing_delete)
    await controller.reconcile_once()
    async with build_session_factory() as session:
        assert (await DependencyBuildRepository(session).get(build.dependency_build_id)).status == 'ready'
    setattr(client, delete_target, original_delete)
    now += config.dependency_build_retry + timedelta(seconds=1)
    await controller.reconcile_once()
    async with build_session_factory() as session:
        cleaned = await DependencyBuildRepository(session).get(build.dependency_build_id)
    assert cleaned.status == 'ready'
    assert cleaned.next_retry_at is None
    assert not client.jobs and not client.secrets
