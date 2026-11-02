from __future__ import annotations

import inspect

from dataclasses import FrozenInstanceError
from datetime import timedelta

import pytest

from nexus.distributed.config import DistributedConfig, DistributedSecrets
from nexus.distributed.config.views import (
    artifact_config_view,
    dependency_build_config_view,
    environment_config_view,
    instance_manager_config_view,
    reconciler_config_view,
    scheduler_config_view,
)
from nexus.distributed.control_plane.reconciler import ReconcilerService, create_reconciler_service
from nexus.distributed.dependency_build.controller import DependencyBuildController
from nexus.distributed.environment.controller import EnvironmentController


def test_scheduler_config_view_is_read_only_and_excludes_unrelated_fields() -> None:
    config = DistributedConfig(
        scheduler_poll_interval=0.25,
        artifact_store_provider='generic_repository',
        artifact_generic_repository='nexus',
    )

    view = scheduler_config_view(config)

    assert view.scheduler_poll_interval == 0.25
    with pytest.raises(AttributeError):
        _ = view.artifact_store_provider
    with pytest.raises((AttributeError, FrozenInstanceError, TypeError)):
        view.scheduler_poll_interval = 1.0  # type: ignore[misc]


def test_config_view_copies_mutable_values() -> None:
    config = DistributedConfig(default_user_quota={'queued_executions': 3})

    view = scheduler_config_view(config)
    config.default_user_quota['queued_executions'] = 99

    assert view.default_user_quota == {'queued_executions': 3}


def test_manager_config_view_can_construct_compressed_result_backend() -> None:
    from nexus.distributed.artifacts.gateway import ApplicationGatewayArtifactTransport
    from nexus.distributed.execution.trace_backends import RunTraceSinkRequest, create_run_trace_backend_registry

    config = DistributedConfig(
        artifact_run_result_compression='zstd', artifact_run_result_max_decoded_size_bytes=123456
    )
    view = instance_manager_config_view(config)
    registry = create_run_trace_backend_registry(
        view, artifact_transport=ApplicationGatewayArtifactTransport(base_url='http://application', content_ticket='test')
    )
    resolved = registry.resolve(
        run_kind='single', driver_config={'run_trace_backends': {'sinks': ['artifact']}},
        request=RunTraceSinkRequest(execution_id='exec', assignment_id='asn', attempt_id='attempt'),
    )
    sink = resolved.backend_sinks['artifact']
    assert sink.compression == 'zstd'
    assert sink.max_decoded_size_bytes == 123456


def test_artifact_config_view_copies_gray_routes() -> None:
    config = DistributedConfig(
        artifact_store_routes=[
            {'target': 'generic_repository', 'weight_bps': 9500},
            {'target': 'artifact_service', 'weight_bps': 500},
        ]
    )

    view = artifact_config_view(config)
    config.artifact_store_routes[0].weight_bps = 1

    assert view.artifact_store_routes[0].weight_bps == 9500


def test_config_view_captures_internal_token_startup_snapshot() -> None:
    config = DistributedConfig(secrets={'internal_token': 'first'})
    view = scheduler_config_view(config)

    assert view.read_internal_token() == 'first'
    config.secrets = DistributedSecrets(internal_token='second')
    assert view.read_internal_token() == 'first'


def test_environment_config_view_captures_database_url_startup_snapshot(tmp_path) -> None:
    database_url_file = tmp_path / 'database-url'
    database_url_file.write_text('mysql+asyncmy://nexus:first@database/nexus')
    config = DistributedConfig(database_url_file=str(database_url_file))

    view = environment_config_view(config)
    database_url_file.write_text('mysql+asyncmy://nexus:second@database/nexus')

    assert view.read_database_url() == 'mysql+asyncmy://nexus:first@database/nexus'


def test_config_views_expose_only_declared_secret_readers() -> None:
    config = DistributedConfig(
        secrets={
            'internal_token': 'internal',
            'artifact_ticket_keys': {'v1': 'ticket'},
            'artifact_writer': {'username': 'writer', 'token': 'writer-token'},
            'provider_credentials': {
                'kubernetes/dev': {
                    'provider_type': 'kubernetes',
                    'token': 'provider-token',
                    'ca_certificate': 'provider-ca',
                }
            },
            'run_trace_credentials': {'trace-db': 'trace-password'},
        }
    )

    scheduler = scheduler_config_view(config)
    assert scheduler.read_internal_token() == 'internal'
    for method_name in (
        'artifact_ticket_keyring',
        'artifact_credential',
        'resolve_provider_credential',
        'read_run_trace_credential',
    ):
        with pytest.raises(AttributeError):
            getattr(scheduler, method_name)

    environment = environment_config_view(config)
    assert environment.artifact_ticket_keyring() == {'v1': 'ticket'}
    assert environment.resolve_provider_credential('kubernetes/dev') is not None
    for method_name in ('read_internal_token', 'artifact_credential', 'read_run_trace_credential'):
        with pytest.raises(AttributeError):
            getattr(environment, method_name)

    manager = instance_manager_config_view(config)
    assert manager.read_internal_token() == 'internal'
    assert manager.read_run_trace_credential('trace-db') == 'trace-password'
    assert manager.manager_executor_hang_timeout == timedelta(minutes=3)
    assert manager.manager_hard_exit_timeout == timedelta(seconds=5)
    assert manager.thread_executor_min_capacity_ratio == 0.5
    assert manager.batch_control_blocking_concurrency == 2
    assert manager.batch_data_frame_max_size_bytes == 256 * 1024**2
    assert manager.batch_task_record_max_size_bytes == 256 * 1024**2
    assert manager.batch_data_list_max_size_bytes == 4 * 1024**3
    assert manager.batch_task_store_write_chunk_bytes == 64 * 1024**2
    for method_name in ('artifact_ticket_keyring', 'artifact_credential', 'resolve_provider_credential'):
        with pytest.raises(AttributeError):
            getattr(manager, method_name)


def test_reconciler_components_require_composition_root_dependencies() -> None:
    assert inspect.signature(EnvironmentController).parameters['client_resolver'].default is inspect.Parameter.empty
    assert inspect.signature(DependencyBuildController).parameters['client_resolver'].default is inspect.Parameter.empty
    reconciler_parameters = inspect.signature(ReconcilerService).parameters
    for name in (
        'environment_controller',
        'dependency_build_controller',
        'workload_client_resolver',
        'batch_lifecycle',
    ):
        assert reconciler_parameters[name].default is inspect.Parameter.empty


@pytest.mark.parametrize(
    ('factory', 'expected_field', 'unrelated_field'),
    [
        (reconciler_config_view, 'reconciler_batch_size', 'environment_base_image'),
        (artifact_config_view, 'artifact_store_provider', 'scheduler_scan_limit'),
        (environment_config_view, 'environment_base_image', 'scheduler_scan_limit'),
        (dependency_build_config_view, 'dependency_builder_cpu_millicores', 'manager_start_batch_size'),
        (instance_manager_config_view, 'manager_start_batch_size', 'environment_builder_image'),
    ],
)
def test_component_config_views_expose_only_their_declared_surface(
    factory,
    expected_field: str,
    unrelated_field: str,
) -> None:
    view = factory(DistributedConfig())

    assert getattr(view, expected_field) is not None
    with pytest.raises(AttributeError):
        getattr(view, unrelated_field)


@pytest.mark.asyncio
async def test_reconciler_composition_projects_component_specific_configs(tmp_path) -> None:
    service = create_reconciler_service(
        session_factory=None,  # type: ignore[arg-type]
        config=DistributedConfig(
            artifact_local_root=str(tmp_path / 'artifacts'),
            artifact_maintenance_spool_dir=str(tmp_path / 'maintenance'),
        ),
        owner_id='reconciler-a',
    )
    try:
        assert service.config.reconciler_batch_size == 500
        assert service.environment_controller.config.environment_base_image
        assert service.dependency_build_controller.config.dependency_build_batch_size == 5
        assert service.batch_lifecycle.config.batch_lifecycle_page_size == 500
        assert service.artifact_maintenance is not None
        assert service.artifact_maintenance.config.artifact_store_provider == 'local'
        assert service.config.tdsql_partition_reconcile_enabled is False

        with pytest.raises(AttributeError):
            _ = service.config.environment_base_image
        with pytest.raises(AttributeError):
            _ = service.environment_controller.config.scheduler_scan_limit
        with pytest.raises(AttributeError):
            _ = service.dependency_build_controller.config.manager_start_batch_size
        with pytest.raises(AttributeError):
            _ = service.batch_lifecycle.config.environment_base_image
        with pytest.raises(AttributeError):
            _ = service.artifact_maintenance.config.scheduler_scan_limit
    finally:
        await service.close()
