from __future__ import annotations

import copy

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import timedelta
from types import MappingProxyType
from typing import Any, Literal, Protocol, TypeVar, cast

from nexus.distributed.config import (
    ArtifactStoreRoute,
    BasicCredential,
    DistributedConfig,
    ProviderCredential,
    RunTraceSinkConfig,
)
from nexus.distributed.policies.models import (
    PolicyExecutorConfig,
    RequestBudgetConfig,
    RequestPolicySelection,
)


class InternalTokenConfigView(Protocol):
    def read_internal_token(self) -> str: ...


class ArtifactGatewayConfigView(InternalTokenConfigView, Protocol):
    application_internal_base_url: str


class SchedulerConfigView(InternalTokenConfigView, Protocol):
    auth_mode: str
    default_user_quota: dict[str, int]
    manager_healthy_ttl: timedelta
    no_capacity_backoff: timedelta
    scheduler_lease_renew_interval: float
    scheduler_lease_ttl: timedelta
    scheduler_manager_lock_contention_backoff: timedelta
    scheduler_manager_lock_limit: int
    scheduler_max_assignments: int
    scheduler_poll_interval: float
    scheduler_scan_limit: int
    scheduler_shard_discover_interval: float
    scheduler_shard_discover_jitter: float
    scheduler_shard_ownership_limit: int
    policy_executor: PolicyExecutorConfig
    shard_count: int


class ReconcilerConfigView(Protocol):
    artifact_maintenance_reconcile_timeout: float
    artifact_maintenance_worker_count: int
    assign_ack_timeout: timedelta
    batch_lifecycle_poll_interval: float
    batch_lifecycle_reconcile_timeout: float
    batch_lifecycle_worker_count: int
    default_user_quota: dict[str, int]
    dependency_build_poll_interval: timedelta
    dependency_build_reconcile_timeout: float
    dependency_build_reconcile_worker_count: int
    environment_reconcile_timeout: float
    environment_reconcile_worker_count: int
    manager_dead_ttl: timedelta
    reconciler_batch_size: int
    reconciler_core_timeout: float
    reconciler_core_max_batches_per_tick: int
    reconciler_core_time_budget: float
    reconciler_failure_backoff: float
    reconciler_lease_renew_interval: float
    reconciler_lease_ttl: timedelta
    reconciler_poll_interval: float
    terminal_data_purge_enabled: bool
    terminal_purge_retention_days: int
    terminal_purge_reconcile_interval: float
    terminal_purge_partition_batch: int
    terminal_purge_ddl_timeout: float
    terminal_purge_reconcile_timeout: float
    reconciler_shard_discover_interval: float
    reconciler_shard_discover_jitter: float
    reconciler_shard_ownership_limit: int
    tdsql_partition_reconcile_enabled: bool
    tdsql_partition_reconcile_interval: float
    tdsql_partition_reconcile_timeout: float
    tdsql_partition_reconcile_horizon_days: int
    shard_count: int
    start_timeout: timedelta


class ArtifactConfigView(ArtifactGatewayConfigView, Protocol):
    artifact_store_routing_enabled: bool
    artifact_store_routing_seed: str
    artifact_store_routing_version: int
    artifact_store_routes: list[ArtifactStoreRoute]
    artifact_generic_base_url: str
    artifact_generic_repository: str
    artifact_local_root: str
    artifact_maintenance_batch_size: int
    artifact_maintenance_lease_renew_interval: timedelta
    artifact_maintenance_lease_ttl: timedelta
    artifact_maintenance_retry: timedelta
    artifact_maintenance_spool_dir: str
    artifact_service_base_url: str
    artifact_service_folder_cache_size: int
    artifact_service_namespace_prefix: str
    artifact_service_system_prefix: str
    artifact_store_provider: str
    artifact_store_read_fallback_providers: list[str]

    def artifact_credential(self, access: str) -> BasicCredential | None: ...

    def artifact_service_shared_secret(self) -> str | None: ...


class EnvironmentConfigView(Protocol):
    application_internal_base_url: str
    artifact_run_result_max_size_bytes: int
    artifact_run_result_publish_deadline: timedelta
    artifact_upload_ticket_current_key_id: str
    batch_reconcile_poll_interval: float
    batch_submit_batch_size: int
    config_store_required: bool
    default_user_quota: dict[str, int]
    environment_app_name: str
    environment_artifact_content_ticket_ttl: timedelta
    environment_base_image: str
    environment_cache_root: str
    environment_controller_batch_size: int
    environment_controller_owner_id: str
    environment_database_url_secret_key: str
    environment_database_url_secret_name: str
    environment_db_max_overflow: int
    environment_db_pool_size: int
    environment_extra_env: list[dict[str, Any]]
    environment_fake_runtime_base_url: str
    environment_image_pull_secrets: list[str]
    environment_manager_port: int
    environment_namespace: str
    environment_operation_lease_renew_interval: timedelta
    environment_operation_ttl: timedelta
    environment_orchestration_db_max_overflow: int
    environment_orchestration_db_pool_size: int
    environment_release_name: str
    environment_tmp_size_bytes: int
    manager_dead_ttl: timedelta
    manager_hard_exit_timeout: timedelta
    manager_heartbeat_interval: float
    manager_healthy_ttl: timedelta
    manager_self_fence_timeout: timedelta
    kubernetes_ca_file: str
    kubernetes_token_file: str
    shard_count: int

    def artifact_ticket_keyring(self) -> dict[str, str]: ...

    def read_database_url(self) -> str: ...

    def resolve_provider_credential(self, credential_ref: str) -> ProviderCredential | None: ...


class DependencyBuildConfigView(Protocol):
    application_internal_base_url: str
    artifact_upload_ticket_current_key_id: str
    dependency_build_batch_size: int
    dependency_build_lease_renew_interval: timedelta
    dependency_build_lease_ttl: timedelta
    dependency_build_max_attempts: int
    dependency_build_poll_interval: timedelta
    dependency_build_retry: timedelta
    dependency_builder_active_deadline_seconds: int
    dependency_builder_cpu_millicores: int
    dependency_builder_index_url: str
    dependency_builder_job_ttl_seconds: int
    dependency_builder_memory_bytes: int
    dependency_builder_workspace_bytes: int
    environment_app_name: str
    environment_artifact_content_ticket_ttl: timedelta
    environment_controller_owner_id: str
    environment_image_pull_secrets: list[str]
    environment_namespace: str
    environment_release_name: str

    def artifact_ticket_keyring(self) -> dict[str, str]: ...


class BatchRepositoryConfigView(Protocol):
    batch_item_metadata_max_size_bytes: int
    batch_submit_batch_size: int
    batch_task_store_write_chunk_bytes: int
    shard_count: int


class BatchControlConfigView(ArtifactGatewayConfigView, BatchRepositoryConfigView, Protocol):
    batch_control_blocking_concurrency: int
    batch_control_spool_dir: str
    batch_data_frame_max_size_bytes: int
    batch_data_list_max_size_bytes: int
    batch_task_record_max_size_bytes: int
    batch_task_store_write_chunk_bytes: int
    manager_executor_hang_timeout: timedelta


class InstanceManagerConfigView(BatchControlConfigView, Protocol):
    aiomonitor_base_port: int
    aiomonitor_console_base_port: int
    aiomonitor_enabled: bool
    aiomonitor_host: str
    aiomonitor_webui_base_port: int
    artifact_run_result_max_size_bytes: int
    artifact_run_result_compression: Literal['none', 'zstd']
    artifact_run_result_max_decoded_size_bytes: int
    artifact_run_result_publish_deadline: timedelta
    batch_reconcile_poll_interval: float
    batch_task_store_backend: str
    batch_task_store_database: str
    default_run_trace_sinks: list[str]
    manager_executor_hang_timeout: timedelta
    manager_hard_exit_timeout: timedelta
    manager_heartbeat_interval: float
    manager_healthy_ttl: timedelta
    manager_local_reconcile_interval: float
    manager_poll_interval: float
    manager_self_fence_timeout: timedelta
    manager_start_batch_size: int
    policy_executor: PolicyExecutorConfig
    request_budget: RequestBudgetConfig
    request_policy: RequestPolicySelection | None
    thread_executor_min_capacity_ratio: float
    run_trace_artifact_root: str
    run_trace_sinks: dict[str, RunTraceSinkConfig]

    def read_run_trace_credential(self, credential_ref: str) -> str: ...

    def read_database_url(self) -> str: ...


class BatchLifecycleConfigView(ArtifactGatewayConfigView, BatchRepositoryConfigView, Protocol):
    batch_lifecycle_claim_renew_interval: float
    batch_lifecycle_claim_ttl: timedelta
    batch_lifecycle_max_pages_per_parent: int
    batch_lifecycle_page_size: int
    batch_lifecycle_parent_limit: int
    batch_lifecycle_scan_limit: int
    batch_lifecycle_publish_backoff_base: float
    batch_lifecycle_publish_backoff_max: float
    reconciler_poll_interval: float


@dataclass(frozen=True)
class _ConfigProjection:
    _values: Mapping[str, Any]
    _secret_readers: Mapping[str, Callable[..., Any]] = field(repr=False)

    def __getattr__(self, name: str) -> Any:
        try:
            return self._values[name]
        except KeyError:
            try:
                return self._secret_readers[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

    def _secret_reader(self, name: str) -> Callable[..., Any]:
        try:
            return self._secret_readers[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


@dataclass(frozen=True)
class _SchedulerConfigProjection(_ConfigProjection, SchedulerConfigView):
    def read_internal_token(self) -> str:
        return self._secret_reader('read_internal_token')()


@dataclass(frozen=True)
class _ReconcilerConfigProjection(_ConfigProjection, ReconcilerConfigView):
    pass


@dataclass(frozen=True)
class _ArtifactConfigProjection(_ConfigProjection, ArtifactConfigView):
    def read_internal_token(self) -> str:
        return self._secret_reader('read_internal_token')()

    def artifact_credential(self, access: str) -> BasicCredential | None:
        return self._secret_reader('artifact_credential')(access)

    def artifact_service_shared_secret(self) -> str | None:
        return self._secret_reader('artifact_service_shared_secret')()


@dataclass(frozen=True)
class _EnvironmentConfigProjection(_ConfigProjection, EnvironmentConfigView):
    def artifact_ticket_keyring(self) -> dict[str, str]:
        return self._secret_reader('artifact_ticket_keyring')()

    def read_database_url(self) -> str:
        return self._secret_reader('read_database_url')()

    def resolve_provider_credential(self, credential_ref: str) -> ProviderCredential | None:
        return self._secret_reader('resolve_provider_credential')(credential_ref)


@dataclass(frozen=True)
class _DependencyBuildConfigProjection(_ConfigProjection, DependencyBuildConfigView):
    def artifact_ticket_keyring(self) -> dict[str, str]:
        return self._secret_reader('artifact_ticket_keyring')()


@dataclass(frozen=True)
class _InstanceManagerConfigProjection(_ConfigProjection, InstanceManagerConfigView):
    def read_database_url(self) -> str:
        return self._secret_reader('read_database_url')()

    def read_internal_token(self) -> str:
        return self._secret_reader('read_internal_token')()

    def read_run_trace_credential(self, credential_ref: str) -> str:
        return self._secret_reader('read_run_trace_credential')(credential_ref)


@dataclass(frozen=True)
class _BatchLifecycleConfigProjection(_ConfigProjection, BatchLifecycleConfigView):
    def read_internal_token(self) -> str:
        return self._secret_reader('read_internal_token')()


ViewT = TypeVar('ViewT')


def _project(
    config: DistributedConfig,
    view_type: type[ViewT],
    projection_type: type[_ConfigProjection],
) -> ViewT:
    field_names: set[str] = set()
    for base in reversed(view_type.__mro__):
        field_names.update(getattr(base, '__annotations__', {}))
    values = {
        field_name: copy.deepcopy(getattr(config, field_name))
        for field_name in field_names
        if not field_name.startswith('_')
    }
    secret_readers: dict[str, Callable[..., Any]] = {}
    secrets = config.secrets
    if hasattr(view_type, 'read_internal_token'):
        internal_token = secrets.read_internal_token()
        secret_readers['read_internal_token'] = lambda value=internal_token: value
    if hasattr(view_type, 'artifact_ticket_keyring'):
        ticket_keys = secrets.artifact_ticket_keyring()
        secret_readers['artifact_ticket_keyring'] = lambda value=ticket_keys: copy.deepcopy(value)
    if hasattr(view_type, 'read_database_url'):
        database_url = config.read_database_url()
        secret_readers['read_database_url'] = lambda value=database_url: value
    if hasattr(view_type, 'artifact_credential'):
        writer = copy.deepcopy(secrets.artifact_writer)
        maintenance = copy.deepcopy(secrets.artifact_maintenance)

        def artifact_credential(
            access: str,
            *,
            _writer: BasicCredential | None = writer,
            _maintenance: BasicCredential | None = maintenance,
        ) -> BasicCredential | None:
            if access == 'writer':
                return copy.deepcopy(_writer)
            if access == 'maintenance':
                return copy.deepcopy(_maintenance)
            if access == 'metadata':
                return None
            raise ValueError(f'unsupported artifact credential access: {access}')

        secret_readers['artifact_credential'] = artifact_credential
    if hasattr(view_type, 'artifact_service_shared_secret'):
        shared_secret = secrets.read_artifact_service_shared_secret()
        secret_readers['artifact_service_shared_secret'] = lambda value=shared_secret: value
    if hasattr(view_type, 'resolve_provider_credential'):
        provider_credentials = copy.deepcopy(secrets.provider_credentials)

        def resolve_provider_credential(
            credential_ref: str,
            *,
            _credentials: dict[str, ProviderCredential] = provider_credentials,
        ) -> ProviderCredential | None:
            return copy.deepcopy(_credentials.get(credential_ref))

        secret_readers['resolve_provider_credential'] = resolve_provider_credential
    if hasattr(view_type, 'read_run_trace_credential'):
        run_trace_credentials = {
            credential_ref: credential.get_secret_value()
            for credential_ref, credential in secrets.run_trace_credentials.items()
        }

        def read_run_trace_credential(
            credential_ref: str,
            *,
            _credentials: dict[str, str] = run_trace_credentials,
        ) -> str:
            credential = _credentials.get(credential_ref)
            if credential is None:
                raise ValueError(f'run trace credential {credential_ref!r} is not configured')
            return credential

        secret_readers['read_run_trace_credential'] = read_run_trace_credential
    return cast(
        ViewT,
        projection_type(MappingProxyType(values), MappingProxyType(secret_readers)),
    )


def scheduler_config_view(config: DistributedConfig) -> SchedulerConfigView:
    return _project(config, SchedulerConfigView, _SchedulerConfigProjection)


def reconciler_config_view(config: DistributedConfig) -> ReconcilerConfigView:
    return _project(config, ReconcilerConfigView, _ReconcilerConfigProjection)


def artifact_config_view(config: DistributedConfig) -> ArtifactConfigView:
    return _project(config, ArtifactConfigView, _ArtifactConfigProjection)


def environment_config_view(config: DistributedConfig) -> EnvironmentConfigView:
    return _project(config, EnvironmentConfigView, _EnvironmentConfigProjection)


def dependency_build_config_view(config: DistributedConfig) -> DependencyBuildConfigView:
    return _project(config, DependencyBuildConfigView, _DependencyBuildConfigProjection)


def instance_manager_config_view(config: DistributedConfig) -> InstanceManagerConfigView:
    return _project(config, InstanceManagerConfigView, _InstanceManagerConfigProjection)


def batch_lifecycle_config_view(config: DistributedConfig) -> BatchLifecycleConfigView:
    return _project(config, BatchLifecycleConfigView, _BatchLifecycleConfigProjection)


__all__ = [
    'ArtifactConfigView',
    'ArtifactGatewayConfigView',
    'BatchControlConfigView',
    'BatchRepositoryConfigView',
    'BatchLifecycleConfigView',
    'DependencyBuildConfigView',
    'EnvironmentConfigView',
    'InstanceManagerConfigView',
    'ReconcilerConfigView',
    'SchedulerConfigView',
    'artifact_config_view',
    'batch_lifecycle_config_view',
    'dependency_build_config_view',
    'environment_config_view',
    'instance_manager_config_view',
    'reconciler_config_view',
    'scheduler_config_view',
]
