from __future__ import annotations

import json
import logging
import os
import re

from datetime import timedelta
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator
from sqlalchemy.engine import make_url

from nexus.distributed.batch.task_list import DEFAULT_SERVER_BATCH_ITEM_METADATA_BYTES
from nexus.distributed.config.observability import (
    BusinessStatisticsConfig,
    MetricsHistoryConfig,
    PlatformMetricsConfig,
    StatusSamplingConfig,
)
from nexus.distributed.config.secrets import (
    BasicCredential,
    DistributedSecrets,
    ProviderCredential,
    TraceDatabaseCredential,
)
from nexus.distributed.policies.models import (
    PolicyExecutorConfig,
    RequestBudgetConfig,
    RequestPolicySelection,
)
from nexus.distributed.policies.request_rate import validate_manager_request_policy
from nexus.distributed.trace.models import TraceReadConfig, TraceSourceIdentity
from nexus.distributed.traffic.router import TrafficPolicy, TrafficTarget
from nexus.distributed.utils.identity import validate_run_trace_backend_name


logger = logging.getLogger(__name__)

_RUN_TRACE_SENSITIVE_PARTS = frozenset({'credential', 'passwd', 'password', 'secret', 'token'})

# Credentials interchangeable with one another: a Tencent Cloud secret id/key pair is not
# tied to a product, so the same pair signs calls for every product CAM allows it.
_TENCENT_CLOUD_PROVIDER_TYPES = frozenset({'tencent_cloud_dcdb', 'tencent_cloud_tdmysql'})


def _credential_matches_provider(credential_type: str, provider_type: str) -> bool:
    """Whether a credential can drive a provider, judged by shape rather than product.
    A MySQL password cannot sign a cloud API call, but the cloud itself makes no product split."""
    if credential_type == provider_type:
        return True
    return credential_type in _TENCENT_CLOUD_PROVIDER_TYPES and provider_type in _TENCENT_CLOUD_PROVIDER_TYPES


def _coerce_env_value(field_name: str, annotation, raw: str):
    if field_name in {
        'run_trace_sinks',
        'extend_run_trace_sinks',
        'trace_read',
        'trace_database_access',
        'extend_trace_database_access',
        'default_run_trace_sinks',
        'cors_origins',
        'admin_users',
        'default_user_quota',
        'environment_extra_env',
        'environment_enabled_providers',
        'environment_image_pull_secrets',
        'zhiyan_log_query',
        'artifact_store_routes',
        'metrics',
        'status_sampling',
        'metrics_history',
        'business_statistics',
        'policy_executor',
        'request_policy',
        'request_budget',
    }:
        return json.loads(raw)
    if annotation is bool:
        normalized = raw.strip().lower()
        if normalized in {'1', 'true', 'yes', 'on'}:
            return True
        if normalized in {'0', 'false', 'no', 'off'}:
            return False
    if annotation is timedelta:
        with_seconds = raw.strip()
        if with_seconds.replace('.', '', 1).isdigit():
            return timedelta(seconds=float(with_seconds))
    if field_name == 'log_level':
        return raw.upper()
    return raw


# Provisioning reads rows back through the source, so a backend with trace_database_access must
# use a source this codebase can drive. Boundless is a distinct class so older builds never see it.
_TRACE_DATABASE_ACCESS_SOURCE_CLASSES = frozenset({
    'nexus.runner.trace.tdsql::TDSQLTraceSource',
    'nexus.runner.trace.tdsql::TDSQLBoundlessTraceSource',
})


def _is_sensitive_run_trace_field(field_name: str) -> bool:
    normalized = re.sub(r'(?<!^)(?=[A-Z])', '_', field_name).replace('-', '_').lower()
    parts = {part for part in re.split(r'[^a-z0-9]+', normalized) if part}
    if normalized.endswith(('_ref', '_file', '_path')):
        return False
    return bool(parts.intersection(_RUN_TRACE_SENSITIVE_PARTS)) or any(
        pair <= parts for pair in ({'api', 'key'}, {'access', 'key'}, {'private', 'key'})
    )


class RunTraceSinkConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra='forbid')

    kind: str = Field(default='custom', min_length=1, max_length=64, pattern=r'^[A-Za-z0-9_-]+$')
    display_name: str | None = Field(default=None, min_length=1, max_length=128, alias='displayName')
    description: str | None = Field(default=None, min_length=1, max_length=512)
    user_selectable: bool = Field(default=False, alias='userSelectable')
    self_grantable: bool = Field(default=False, alias='selfGrantable')
    system_managed: bool = Field(default=False, alias='systemManaged')
    required: bool = False
    source_identity: TraceSourceIdentity | None = Field(default=None, alias='sourceIdentity')
    sink_class: str | None = Field(default=None, alias='sinkClass')
    source_class: str | None = Field(default=None, alias='sourceClass')
    config: dict[str, Any] = Field(default_factory=dict)
    secret_config: dict[str, str] = Field(default_factory=dict, alias='secretConfig')
    sink_config: dict[str, Any] = Field(default_factory=dict, alias='sinkConfig')
    source_config: dict[str, Any] = Field(default_factory=dict, alias='sourceConfig')

    @model_validator(mode='after')
    def validate_access_metadata(self) -> RunTraceSinkConfig:
        if self.system_managed and self.user_selectable:
            raise ValueError('system-managed run trace sinks cannot be user selectable')
        if self.self_grantable and not self.user_selectable:
            raise ValueError('self-grantable run trace sinks must be user selectable')
        if self.required and not self.system_managed:
            raise ValueError('required run trace sinks must be system managed')
        return self

    @model_validator(mode='after')
    def reject_cleartext_secrets(self) -> RunTraceSinkConfig:
        for values in (self.config, self.sink_config, self.source_config):
            for field_name in values:
                if _is_sensitive_run_trace_field(field_name):
                    raise ValueError(
                        f'sensitive run trace field {field_name!r} must be configured through secretConfig'
                    )
        return self


class ArtifactStoreRoute(BaseModel):
    model_config = ConfigDict(extra='forbid')

    target: str = Field(min_length=1, max_length=64, pattern=r'^[A-Za-z0-9_-]+$')
    weight_bps: int = Field(strict=True, ge=0, le=10000)


class MySQLTraceDatabaseAccountProviderConfig(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

    provider_type: Literal['mysql'] = 'mysql'


class TencentCloudDCDBTraceDatabaseAccountProviderConfig(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

    provider_type: Literal['tencent_cloud_dcdb']
    region: str = Field(min_length=1, pattern=r'^[a-z][a-z0-9]*(?:-[a-z0-9]+)+$')
    instance_id: str = Field(min_length=1, pattern=r'^(?:dcdbt|tdsql|tdsqlshard)-[A-Za-z0-9]+$')
    api_endpoint: Literal[
        'dcdb.tencentcloudapi.com',
        'dcdb.internal.tencentcloudapi.com',
    ] = 'dcdb.tencentcloudapi.com'
    account_host: str = Field(default='%', min_length=1, max_length=255)


class TencentCloudTDMySQLTraceDatabaseAccountProviderConfig(BaseModel):
    """TDSQL 3.0 (Boundless) accounts are managed by the ``tdmysql`` product API.
    A separate service from ``dcdb``: different endpoint, API version and action names."""

    model_config = ConfigDict(extra='forbid', frozen=True)

    provider_type: Literal['tencent_cloud_tdmysql']
    region: str = Field(min_length=1, pattern=r'^[a-z][a-z0-9]*(?:-[a-z0-9]+)+$')
    instance_id: str = Field(min_length=1, pattern=r'^tdsql[0-9]*-[A-Za-z0-9]+$')
    api_endpoint: Literal[
        'tdmysql.tencentcloudapi.com',
        'tdmysql.internal.tencentcloudapi.com',
    ] = 'tdmysql.tencentcloudapi.com'
    account_host: str = Field(default='%', min_length=1, max_length=255)
    # Production flows reach 'success' in 62-65s, against 2.4s on the test instance.
    flow_timeout_seconds: float = Field(default=180.0, gt=0, le=600)


TraceDatabaseAccountProviderConfig = Annotated[
    MySQLTraceDatabaseAccountProviderConfig
    | TencentCloudDCDBTraceDatabaseAccountProviderConfig
    | TencentCloudTDMySQLTraceDatabaseAccountProviderConfig,
    Field(discriminator='provider_type'),
]


class TraceDatabaseAccessConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

    read_host: str = Field(min_length=1)
    read_port: int = Field(default=3306, ge=1, le=65535)
    database: str = Field(min_length=1, pattern=r'^[A-Za-z0-9_]+$')
    maintenance_credential_ref: str = Field(min_length=1, max_length=128)
    account_provider: TraceDatabaseAccountProviderConfig = Field(
        default_factory=MySQLTraceDatabaseAccountProviderConfig
    )
    enabled: bool = True


class ZhiyanLogQueryConfig(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, hide_input_in_errors=True)

    enabled: bool = False
    endpoint: str = Field(default='http://openapi.zhiyan.woa.com', min_length=1)
    project_name: str = Field(default='', min_length=0, max_length=128)
    staff_name: str = Field(default='', min_length=0, max_length=128)
    credential_ref: str = Field(default='', min_length=0, max_length=128)
    storage_type: Literal['elasticsearch', 'ES', 'clickhouse', 'iceberg'] = 'elasticsearch'
    query_window_seconds: int = Field(default=6 * 60 * 60, ge=1, le=6 * 60 * 60)
    ingest_grace_seconds: int = Field(default=5 * 60, ge=0, le=60 * 60)
    max_lines: int = Field(default=1000, ge=1, le=1000)

    @model_validator(mode='after')
    def validate_endpoint_and_enabled_fields(self) -> ZhiyanLogQueryConfig:
        parsed = urlsplit(self.endpoint)
        if (
            parsed.scheme not in {'http', 'https'}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError('Zhiyan log endpoint must be an absolute HTTP URL without credentials or query')
        if self.enabled:
            for field_name in ('project_name', 'staff_name', 'credential_ref'):
                if not getattr(self, field_name).strip():
                    raise ValueError(f'{field_name} is required when Zhiyan log query is enabled')
        return self


# Fields persisted from this model are a cross-version contract: only add fields; never remove or repurpose them.
class DistributedConfig(BaseModel):
    model_config = ConfigDict(extra='forbid', hide_input_in_errors=True)

    # Records which entries the merges below folded in from each extend_* field: the merge is
    # destructive, and dump_for_persistence must put the folded ones back where they came from.
    _extension_folded_names: dict[str, tuple[str, ...]] = PrivateAttr(default_factory=dict)

    metrics: PlatformMetricsConfig = Field(default_factory=PlatformMetricsConfig)
    metrics_history: MetricsHistoryConfig = Field(default_factory=MetricsHistoryConfig)
    business_statistics: BusinessStatisticsConfig = Field(default_factory=BusinessStatisticsConfig)
    status_sampling: StatusSamplingConfig = Field(default_factory=StatusSamplingConfig)

    database_url: str = Field(
        default_factory=lambda: os.getenv('NEXUS_DIST_DATABASE_URL', ''),
        exclude=True,
        repr=False,
    )
    database_url_file: str = ''
    config_store_required: bool = Field(default=False, exclude=True)
    secrets: DistributedSecrets = Field(default_factory=DistributedSecrets, exclude=True, repr=False)
    run_trace_artifact_root: str = Field(
        default_factory=lambda: os.getenv(
            'NEXUS_DIST_RUN_TRACE_ARTIFACT_ROOT', '/data/nexus-distributed/run-trace-artifacts'
        )
    )
    trace_read: TraceReadConfig = Field(default_factory=TraceReadConfig)
    run_trace_sinks: dict[str, RunTraceSinkConfig] = Field(default_factory=dict)
    # Merged into run_trace_sinks below, yet invisible to older builds: an unknown top-level key
    # is stripped with a warning, while one inside a sink entry reaches and crashes the constructor.
    extend_run_trace_sinks: dict[str, RunTraceSinkConfig] = Field(default_factory=dict)
    default_run_trace_sinks: list[str] = Field(default_factory=lambda: ['local'])
    batch_task_store_backend: str = 'tdsql'
    batch_task_store_database: str = Field(default='nexus_runner', pattern=r'^[A-Za-z0-9_]+$')
    batch_item_metadata_max_size_bytes: int = Field(default=DEFAULT_SERVER_BATCH_ITEM_METADATA_BYTES, ge=1)
    submission_auto_increase_cpu: bool = True
    batch_data_frame_max_size_bytes: int = Field(default=256 * 1024**2, ge=1)
    batch_task_record_max_size_bytes: int = Field(default=256 * 1024**2, ge=1)
    batch_data_list_max_size_bytes: int = Field(default=4 * 1024**3, ge=1)
    batch_task_store_write_chunk_bytes: int = Field(default=64 * 1024**2, ge=1)
    require_principal_run_trace_backend_grant: bool = False
    trace_database_access: dict[str, TraceDatabaseAccessConfig] = Field(default_factory=dict)
    # Same purpose as extend_run_trace_sinks, sharper edge: an unknown provider fails as
    # union_tag_invalid, which the read-side strip misses -- a top-level key stays extra_forbidden.
    extend_trace_database_access: dict[str, TraceDatabaseAccessConfig] = Field(default_factory=dict)
    zhiyan_log_query: ZhiyanLogQueryConfig = Field(default_factory=ZhiyanLogQueryConfig)
    # Zhiyan dataflow name used to read terminal logs of predefined-pool instance managers.
    # Predefined environments have no provider config, so the binding cannot be derived per
    # allocation; when non-empty it is applied to every predefined allocation.
    predefined_manager_zhiyan_log_name: str = Field(default='', max_length=128)
    log_dir: str = Field(default_factory=lambda: os.getenv('NEXUS_DIST_LOG_DIR', ''))
    log_level: str = Field(default_factory=lambda: os.getenv('NEXUS_DIST_LOG_LEVEL', 'INFO').upper())
    aiomonitor_enabled: bool = True
    aiomonitor_host: str = '127.0.0.1'
    aiomonitor_base_port: int = 50100
    aiomonitor_webui_base_port: int = 50200
    aiomonitor_console_base_port: int = 50300
    shard_count: int = Field(default=64, ge=1, le=4096)
    db_pool_size: int | None = None
    db_max_overflow: int | None = None
    db_pool_timeout: float | None = None
    db_pool_recycle: int | None = None
    db_wait_timeout: float = 120.0
    slow_sql_log_threshold_ms: float = 0.0
    gateway_json_body_max_size_bytes: int = Field(default=1024**2, ge=1)
    gateway_proxy_body_max_size_bytes: int = Field(default=8 * 1024**2, ge=1)
    manager_healthy_ttl: timedelta = Field(default=timedelta(seconds=15), gt=timedelta(0))
    manager_dead_ttl: timedelta = Field(default=timedelta(seconds=45), gt=timedelta(0))
    assign_ack_timeout: timedelta = Field(default=timedelta(seconds=30), gt=timedelta(0))
    start_timeout: timedelta = Field(default=timedelta(seconds=120), gt=timedelta(0))
    reconciler_batch_size: int = Field(default=500, ge=1, le=100000)
    scheduler_lease_ttl: timedelta = Field(default=timedelta(seconds=20), gt=timedelta(0))
    scheduler_lease_renew_interval: float = Field(default=5.0, gt=0)
    scheduler_shard_ownership_limit: int = Field(default=0, ge=0, le=4096)
    scheduler_shard_discover_interval: float = Field(default=20.0, gt=0)
    scheduler_shard_discover_jitter: float = Field(default=0.2, ge=0, lt=1)
    policy_executor: PolicyExecutorConfig = Field(default_factory=PolicyExecutorConfig)
    request_policy: RequestPolicySelection | None = None
    request_budget: RequestBudgetConfig = Field(default_factory=RequestBudgetConfig)
    scheduler_poll_interval: float = Field(default=0.1, gt=0)
    scheduler_scan_limit: int = Field(default=250, ge=1)
    scheduler_max_assignments: int = Field(default=250, ge=1)
    scheduler_manager_lock_limit: int = Field(default=64, ge=1)
    scheduler_manager_lock_contention_backoff: timedelta = Field(
        default=timedelta(milliseconds=250),
        gt=timedelta(0),
    )
    reconciler_poll_interval: float = Field(default=5.0, gt=0)
    terminal_data_purge_enabled: bool = False
    terminal_purge_trace_backend: str | None = Field(default=None, min_length=1, max_length=64)
    terminal_purge_retention_days: int = Field(default=7, ge=0)
    terminal_purge_reconcile_interval: float = Field(default=3600.0, gt=0)
    terminal_purge_partition_batch: int = Field(default=20, ge=1, le=10000)
    terminal_purge_ddl_timeout: float = Field(default=60.0, gt=0)
    terminal_purge_reconcile_timeout: float = Field(default=300.0, gt=0)
    reconciler_lease_ttl: timedelta = Field(default=timedelta(seconds=20), gt=timedelta(0))
    reconciler_lease_renew_interval: float = Field(default=5.0, gt=0)
    reconciler_shard_ownership_limit: int = Field(default=0, ge=0, le=4096)
    reconciler_shard_discover_interval: float = Field(default=20.0, gt=0)
    reconciler_shard_discover_jitter: float = Field(default=0.2, ge=0, lt=1)
    reconciler_core_max_batches_per_tick: int = Field(default=2, ge=1, le=100)
    reconciler_core_time_budget: float = Field(default=2.0, gt=0)
    reconciler_core_timeout: float = Field(default=30.0, gt=0)
    environment_reconcile_worker_count: int = Field(default=2, ge=1, le=32)
    environment_reconcile_timeout: float = Field(default=300.0, gt=0)
    dependency_build_reconcile_worker_count: int = Field(default=2, ge=1, le=32)
    dependency_build_reconcile_timeout: float = Field(default=300.0, gt=0)
    artifact_maintenance_worker_count: int = Field(default=2, ge=1, le=32)
    artifact_maintenance_reconcile_timeout: float = Field(default=300.0, gt=0)
    batch_lifecycle_worker_count: int = Field(default=2, ge=1, le=32)
    batch_lifecycle_reconcile_timeout: float = Field(default=300.0, gt=0)
    reconciler_failure_backoff: float = Field(default=1.0, gt=0)
    # Boundless creates INTERVAL partitions lazily on out-of-range INSERTs; the reconciler
    # materializes a future window so concurrent writers do not race on DDL.
    tdsql_partition_reconcile_enabled: bool = False
    tdsql_partition_reconcile_interval: float = Field(default=3600.0, gt=0)
    tdsql_partition_reconcile_timeout: float = Field(default=60.0, gt=0)
    tdsql_partition_reconcile_horizon_days: int = Field(default=7, ge=1, le=31)
    manager_poll_interval: float = Field(default=1.0, gt=0)
    manager_heartbeat_interval: float = Field(default=5.0, gt=0)
    manager_self_fence_timeout: timedelta = Field(default=timedelta(seconds=30), gt=timedelta(0))
    manager_executor_hang_timeout: timedelta = Field(default=timedelta(minutes=3), gt=timedelta(0))
    manager_hard_exit_timeout: timedelta = Field(default=timedelta(seconds=5), gt=timedelta(0))
    thread_executor_min_capacity_ratio: float = Field(default=0.5, gt=0, le=1)
    manager_local_reconcile_interval: float = Field(default=15.0, gt=0)
    manager_start_batch_size: int = Field(default=10, ge=1, le=10000)
    batch_reconcile_poll_interval: float = Field(default=0.1, gt=0)
    batch_lifecycle_poll_interval: float = Field(default=0.5, gt=0)
    batch_lifecycle_page_size: int = Field(default=500, ge=1, le=10000)
    batch_lifecycle_parent_limit: int = Field(default=50, ge=1, le=10000)
    batch_lifecycle_max_pages_per_parent: int = Field(default=2, ge=1, le=100)
    batch_lifecycle_scan_limit: int = Field(default=5000, ge=1)
    batch_lifecycle_claim_ttl: timedelta = Field(default=timedelta(seconds=600), gt=timedelta(0))
    batch_lifecycle_claim_renew_interval: float = Field(default=30.0, gt=0)
    batch_lifecycle_publish_backoff_base: float = Field(default=1.0, gt=0)
    batch_lifecycle_publish_backoff_max: float = Field(default=30.0, gt=0)
    batch_submit_batch_size: int = Field(default=2000, ge=1, le=10000)
    no_capacity_backoff: timedelta = Field(default=timedelta(seconds=3), gt=timedelta(0))
    gateway_route_cache_ttl: float = 1.0
    auth_mode: str = 'disabled'
    tai_trust_x_rio_app_id: bool = False
    production_mode: bool = False
    tai_identity_issuer: str = 'tai'
    tai_identity_tenant: str = 'default'
    cors_origins: list[str] = Field(default_factory=list)
    admin_users: list[str] = Field(default_factory=list)
    submit_token_current_key_id: str = 'local'
    control_public_base_url: str = 'https://control-runner-nexus-agent.mcp.woa.com'
    console_public_base_url: str = 'https://cloud.nexus-agent.woa.com'
    application_public_base_url: str = 'http://127.0.0.1:8082'
    console_application_base_url: str | None = None
    cloud_console_welcome_animation_enabled: bool = False
    cloud_console_guide_url: str = 'https://nexus.pages.woa.com/cloud/overview/'
    application_internal_base_url: str = 'http://127.0.0.1:8082'
    artifact_store_routing_enabled: bool = False
    artifact_store_routing_seed: str = 'artifact-store-v1'
    artifact_store_routing_version: int = Field(default=1, ge=1)
    artifact_store_routes: list[ArtifactStoreRoute] = Field(default_factory=list)
    artifact_store_provider: str = 'local'
    artifact_store_read_fallback_providers: list[str] = Field(default_factory=list)
    artifact_local_root: str = '/tmp/nexus-distributed-artifacts'
    artifact_generic_base_url: str = 'https://mirrors.tencent.com/repository/generic'
    artifact_generic_repository: str = ''
    artifact_service_base_url: str = 'https://artifact-nexus-agent.mcp.woa.com'
    artifact_service_system_prefix: str = '/system'
    artifact_service_namespace_prefix: str = '/_nexus_dist'
    artifact_service_folder_cache_size: int = Field(default=4096, ge=0)
    artifact_upload_ticket_current_key_id: str = 'local'
    artifact_upload_ticket_ttl: timedelta = timedelta(minutes=10)
    artifact_upload_session_ttl: timedelta = timedelta(hours=1)
    artifact_upload_lease_ttl: timedelta = timedelta(minutes=15)
    artifact_upload_max_size_bytes: int = Field(default=1024 * 1024 * 1024, ge=1)
    artifact_upload_spool_dir: str = '/tmp/nexus-distributed-upload-spool'
    artifact_transfer_spool_capacity_bytes: int = Field(default=4 * 1024**3, ge=1)
    artifact_upload_total_concurrency: int = Field(default=12, ge=1)
    artifact_upload_input_concurrency: int = Field(default=4, ge=1)
    artifact_upload_result_concurrency: int = Field(default=8, ge=1)
    artifact_upload_max_waiters: int = Field(default=64, ge=0)
    artifact_upload_admission_timeout: float = Field(default=5.0, gt=0)
    artifact_upload_body_idle_timeout: float = Field(default=30.0, gt=0)
    artifact_upload_deadline: timedelta = Field(default=timedelta(minutes=30), gt=timedelta(0))
    artifact_upload_lease_renew_interval: timedelta = Field(default=timedelta(minutes=1), gt=timedelta(0))
    artifact_upload_retry_attempts: int = Field(default=8, ge=1)
    artifact_download_total_concurrency: int = Field(default=32, ge=1)
    artifact_download_input_concurrency: int = Field(default=24, ge=1)
    artifact_download_result_concurrency: int = Field(default=8, ge=1)
    artifact_download_max_waiters: int = Field(default=128, ge=0)
    artifact_download_admission_timeout: float = Field(default=5.0, gt=0)
    artifact_download_write_idle_timeout: float = Field(default=30.0, gt=0)
    artifact_download_deadline: timedelta = Field(default=timedelta(minutes=10), gt=timedelta(0))
    artifact_run_result_max_size_bytes: int = Field(default=16 * 1024**2, ge=1)
    artifact_run_result_compression: Literal['none', 'zstd'] = 'none'
    artifact_run_result_max_decoded_size_bytes: int = Field(default=2 * 1024**3, ge=1)
    artifact_run_result_decode_concurrency: int = Field(default=1, ge=1)
    artifact_run_result_max_inflight: int = Field(default=4, ge=1)
    artifact_run_result_max_waiters: int = Field(default=16, ge=0)
    artifact_run_result_admission_timeout: float = Field(default=5.0, gt=0)
    # Retained for stored-config compatibility; streaming results use bounded buffers instead.
    artifact_run_result_memory_budget_bytes: int = Field(default=16 * 1024**3, ge=1)
    artifact_run_result_memory_expansion_factor: int = Field(default=8, ge=1)
    artifact_run_result_response_chunk_bytes: int = Field(default=1024**2, ge=1, le=4 * 1024**2)
    artifact_run_result_publish_deadline: timedelta = Field(default=timedelta(minutes=30), gt=timedelta(0))
    artifact_default_ttl: timedelta = Field(default=timedelta(days=30), gt=timedelta(0))
    artifact_maintenance_batch_size: int = Field(default=50, ge=1)
    artifact_maintenance_enabled: bool = True
    artifact_maintenance_lease_ttl: timedelta = Field(default=timedelta(minutes=15), gt=timedelta(0))
    artifact_maintenance_lease_renew_interval: timedelta = Field(default=timedelta(minutes=1), gt=timedelta(0))
    artifact_maintenance_retry: timedelta = Field(default=timedelta(minutes=1), gt=timedelta(0))
    artifact_maintenance_spool_dir: str = '/tmp/nexus-distributed-artifact-maintenance'
    artifact_access_profile: str = 'default'
    artifact_access_version: int = Field(default=1, ge=1)
    environment_base_image: str = 'nexus-distributed-workload:latest'
    environment_builder_image: str = 'nexus-distributed-builder:latest'
    environment_image_pull_secrets: list[str] = Field(default_factory=list)
    environment_fixed_replicas: int = Field(default=1, ge=1)
    environment_slots_per_replica: int = Field(default=1, ge=1)
    environment_enabled_providers: list[str] = Field(default_factory=list)
    environment_controller_owner_id: str = 'reconciler'
    environment_controller_batch_size: int = Field(default=20, ge=1)
    environment_operation_ttl: timedelta = Field(default=timedelta(minutes=2), gt=timedelta(0))
    environment_operation_lease_renew_interval: timedelta = Field(
        default=timedelta(seconds=10),
        gt=timedelta(0),
    )
    dependency_build_batch_size: int = Field(default=5, ge=1)
    dependency_build_lease_ttl: timedelta = Field(default=timedelta(minutes=2), gt=timedelta(0))
    dependency_build_lease_renew_interval: timedelta = Field(default=timedelta(seconds=30), gt=timedelta(0))
    dependency_build_poll_interval: timedelta = Field(default=timedelta(seconds=2), gt=timedelta(0))
    dependency_build_retry: timedelta = Field(default=timedelta(seconds=15), gt=timedelta(0))
    dependency_build_max_attempts: int = Field(default=2, ge=1)
    dependency_builder_cpu_millicores: int = Field(default=2_000, ge=1)
    dependency_builder_memory_bytes: int = Field(default=4 * 1024**3, ge=1)
    dependency_builder_workspace_bytes: int = Field(default=8 * 1024**3, ge=1)
    dependency_builder_active_deadline_seconds: int = Field(default=3_600, ge=1)
    dependency_builder_job_ttl_seconds: int = Field(default=3_600, ge=1)
    dependency_builder_index_url: str = Field(
        default='https://mirrors.tencent.com/pypi/simple/',
        pattern=r'^https?://\S+$',
    )
    environment_artifact_content_ticket_ttl: timedelta = timedelta(hours=1)
    environment_retry_base: timedelta = timedelta(seconds=5)
    batch_control_spool_dir: str = '/tmp/nexus-batch-control'
    batch_control_blocking_concurrency: int = Field(default=2, ge=1, le=32)
    environment_namespace: str = 'default'
    environment_release_name: str = 'nexus-dist'
    environment_app_name: str = 'nexus-distributed'
    environment_manager_port: int = Field(default=8082, ge=1, le=65535)
    environment_manager_capacity: int = Field(default=1, ge=1)
    environment_db_pool_size: int = Field(default=1, ge=1)
    environment_db_max_overflow: int = Field(default=0, ge=0)
    environment_orchestration_db_pool_size: int = Field(default=4, ge=1)
    environment_orchestration_db_max_overflow: int = Field(default=2, ge=0)
    environment_database_url_secret_name: str = ''
    environment_database_url_secret_key: str = 'database-url'
    environment_cache_root: str = '/data/nexus-environment'
    environment_tmp_size_bytes: int = Field(default=1024**3, ge=1)
    environment_fake_runtime_base_url: str = ''
    environment_extra_env: list[dict[str, Any]] = Field(default_factory=list)
    kubernetes_token_file: str = '/var/run/secrets/kubernetes.io/serviceaccount/token'
    kubernetes_ca_file: str = '/var/run/secrets/kubernetes.io/serviceaccount/ca.crt'
    default_user_quota: dict[str, int] = Field(
        default_factory=lambda: {
            'queued_executions': 1000,
            'running_executions': 100,
            'reserved_slots': 100,
        }
    )

    @field_validator('cloud_console_guide_url')
    @classmethod
    def validate_cloud_console_guide_url(cls, value: str) -> str:
        value = value.strip()
        parsed = urlsplit(value)
        if parsed.scheme not in {'http', 'https'} or not parsed.hostname:
            raise ValueError('cloud_console_guide_url must be an absolute HTTP(S) URL')
        return value

    @field_validator('default_user_quota')
    @classmethod
    def validate_default_user_quota(cls, value: dict[str, int]) -> dict[str, int]:
        if any(limit < 0 for limit in value.values()):
            raise ValueError('default user quota limits must be non-negative')
        return value

    @field_validator('run_trace_sinks')
    @classmethod
    def validate_run_trace_sink_names(
        cls,
        value: dict[str, RunTraceSinkConfig],
    ) -> dict[str, RunTraceSinkConfig]:
        for name in value:
            validate_run_trace_backend_name(name)
        return value

    @field_validator('extend_run_trace_sinks')
    @classmethod
    def validate_extend_run_trace_sink_names(
        cls,
        value: dict[str, RunTraceSinkConfig],
    ) -> dict[str, RunTraceSinkConfig]:
        for name in value:
            validate_run_trace_backend_name(name)
        return value

    @model_validator(mode='after')
    def merge_extend_run_trace_sinks(self) -> DistributedConfig:
        """Fold the extension entries in so every reader of run_trace_sinks sees them.
        ``run_trace_sinks`` wins a collision, so an entry can be promoted out without a flag day."""
        if not self.extend_run_trace_sinks:
            return self
        promoted = sorted(set(self.extend_run_trace_sinks) & set(self.run_trace_sinks))
        if promoted:
            logger.warning(
                'run trace backends defined in both run_trace_sinks and extend_run_trace_sinks; '
                'using the run_trace_sinks definition for: %s',
                ', '.join(promoted),
            )
        merged = dict(self.extend_run_trace_sinks)
        merged.update(self.run_trace_sinks)
        self._extension_folded_names['run_trace_sinks'] = tuple(
            name for name in self.extend_run_trace_sinks if name not in self.run_trace_sinks
        )
        self.run_trace_sinks = merged
        return self

    @model_validator(mode='after')
    def merge_extend_trace_database_access(self) -> DistributedConfig:
        """Fold the extension entries in, mirroring merge_extend_run_trace_sinks.
        Runs before the cross-field checks, so an extension entry is validated like a canonical one."""
        if not self.extend_trace_database_access:
            return self
        promoted = sorted(set(self.extend_trace_database_access) & set(self.trace_database_access))
        if promoted:
            logger.warning(
                'trace database access defined in both trace_database_access and '
                'extend_trace_database_access; using the trace_database_access definition for: %s',
                ', '.join(promoted),
            )
        merged = dict(self.extend_trace_database_access)
        merged.update(self.trace_database_access)
        self._extension_folded_names['trace_database_access'] = tuple(
            name for name in self.extend_trace_database_access if name not in self.trace_database_access
        )
        self.trace_database_access = merged
        return self

    def dump_for_persistence(self, **kwargs: Any) -> dict[str, Any]:
        """model_dump with the merges above undone, for writing config back to storage.
        Saving the merged view would copy an extension entry where older builds can no longer skip it."""
        dumped = self.model_dump(**kwargs)
        for field_name, folded in self._extension_folded_names.items():
            value = dumped.get(field_name)
            if not folded or not isinstance(value, dict):
                continue
            dumped[field_name] = {name: entry for name, entry in value.items() if name not in folded}
        return dumped

    @field_validator('default_run_trace_sinks')
    @classmethod
    def validate_default_run_trace_sink_names(cls, value: list[str]) -> list[str]:
        for name in value:
            validate_run_trace_backend_name(name)
        return value

    @field_validator('batch_task_store_backend')
    @classmethod
    def validate_batch_task_store_backend(cls, value: str) -> str:
        return validate_run_trace_backend_name(value)

    @field_validator('environment_enabled_providers')
    @classmethod
    def validate_environment_enabled_providers(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)) or not set(value) <= {'kubernetes', 'tkex'}:
            raise ValueError('environment enabled providers must contain unique kubernetes or tkex values')
        return value

    @field_validator('environment_artifact_content_ticket_ttl')
    @classmethod
    def validate_environment_artifact_ticket_ttl(cls, value: timedelta) -> timedelta:
        if value.total_seconds() < 2:
            raise ValueError('environment artifact content ticket TTL must be at least two seconds')
        return value

    @model_validator(mode='after')
    def validate_artifact_transfer_limits(self) -> DistributedConfig:
        if (
            'manager_self_fence_timeout' not in self.model_fields_set
            and self.manager_healthy_ttl < self.manager_dead_ttl
            and not (self.manager_healthy_ttl < self.manager_self_fence_timeout < self.manager_dead_ttl)
        ):
            self.manager_self_fence_timeout = (
                self.manager_healthy_ttl + (self.manager_dead_ttl - self.manager_healthy_ttl) / 2
            )
        if self.scheduler_shard_ownership_limit > self.shard_count:
            raise ValueError('Scheduler shard ownership limit cannot exceed shard count')
        if self.reconciler_shard_ownership_limit > self.shard_count:
            raise ValueError('Reconciler shard ownership limit cannot exceed shard count')
        if self.scheduler_lease_renew_interval >= self.scheduler_lease_ttl.total_seconds():
            raise ValueError('Scheduler lease renew interval must be shorter than the lease TTL')
        if self.reconciler_lease_renew_interval >= self.reconciler_lease_ttl.total_seconds():
            raise ValueError('Reconciler lease renew interval must be shorter than the lease TTL')
        if self.manager_heartbeat_interval >= self.manager_healthy_ttl.total_seconds():
            raise ValueError('Manager heartbeat interval must be shorter than the healthy TTL')
        if self.manager_healthy_ttl >= self.manager_dead_ttl:
            raise ValueError('Manager healthy TTL must be shorter than the dead TTL')
        if self.manager_healthy_ttl >= self.manager_self_fence_timeout:
            raise ValueError('Manager healthy TTL must be shorter than the self-fence timeout')
        if self.manager_self_fence_timeout >= self.manager_dead_ttl:
            raise ValueError('Manager self-fence timeout must be shorter than the dead TTL')
        if self.batch_lifecycle_claim_renew_interval >= self.batch_lifecycle_claim_ttl.total_seconds():
            raise ValueError('Batch lifecycle claim renew interval must be shorter than the lease TTL')
        if self.batch_lifecycle_publish_backoff_max < self.batch_lifecycle_publish_backoff_base:
            raise ValueError('Batch lifecycle publish backoff max must be at least the backoff base')
        if self.artifact_upload_input_concurrency > self.artifact_upload_total_concurrency:
            raise ValueError('Artifact upload input concurrency cannot exceed total concurrency')
        if self.artifact_upload_result_concurrency > self.artifact_upload_total_concurrency:
            raise ValueError('Artifact upload result concurrency cannot exceed total concurrency')
        if self.artifact_download_input_concurrency > self.artifact_download_total_concurrency:
            raise ValueError('Artifact download input concurrency cannot exceed total concurrency')
        if self.artifact_download_result_concurrency > self.artifact_download_total_concurrency:
            raise ValueError('Artifact download result concurrency cannot exceed total concurrency')
        if self.artifact_upload_lease_renew_interval >= self.artifact_upload_lease_ttl:
            raise ValueError('Artifact upload lease renew interval must be shorter than the lease TTL')
        if self.artifact_maintenance_lease_renew_interval >= self.artifact_maintenance_lease_ttl:
            raise ValueError('Artifact maintenance lease renew interval must be shorter than the lease TTL')
        if self.dependency_build_lease_renew_interval >= self.dependency_build_lease_ttl:
            raise ValueError('Dependency build lease renew interval must be shorter than the lease TTL')
        if self.environment_operation_lease_renew_interval >= self.environment_operation_ttl:
            raise ValueError('Environment operation lease renew interval must be shorter than the lease TTL')
        if self.artifact_upload_deadline.total_seconds() <= self.artifact_upload_body_idle_timeout:
            raise ValueError('Artifact upload deadline must exceed the body idle timeout')
        if self.artifact_download_deadline.total_seconds() <= self.artifact_download_write_idle_timeout:
            raise ValueError('Artifact download deadline must exceed the write idle timeout')
        if self.artifact_run_result_max_size_bytes > self.artifact_upload_max_size_bytes:
            raise ValueError('Run result size limit cannot exceed the Artifact upload size limit')
        for backend_name, access in self.trace_database_access.items():
            backend = self.run_trace_sinks.get(backend_name)
            if backend is None:
                raise ValueError(f'trace database access backend is not configured: {backend_name}')
            if backend.source_class not in _TRACE_DATABASE_ACCESS_SOURCE_CLASSES:
                raise ValueError(f'trace database access backend must use a TDSQL trace source: {backend_name}')
            if backend.config.get('db') != access.database:
                raise ValueError(f'trace database access database must match trace backend: {backend_name}')
        if self.request_policy is not None and 'policy_executor' in self.model_fields_set:
            validate_manager_request_policy(self.policy_executor, self.request_policy)
        return self

    @model_validator(mode='after')
    def validate_artifact_store_routing(self) -> DistributedConfig:
        if self.artifact_store_routing_enabled:
            try:
                TrafficPolicy(
                    component='artifact_store',
                    seed=self.artifact_store_routing_seed,
                    version=self.artifact_store_routing_version,
                    targets=tuple(
                        TrafficTarget(target.target, target.weight_bps) for target in self.artifact_store_routes
                    ),
                )
            except ValueError as exc:
                raise ValueError(f'artifact store routing policy is invalid: {exc}') from exc
        return self

    @classmethod
    def values_from_env(cls) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for field_name, field in cls.model_fields.items():
            env_name = f'NEXUS_DIST_{field_name.upper()}'
            raw = os.getenv(env_name)
            if raw is None and field.annotation is timedelta:
                raw = os.getenv(f'{env_name}_SECONDS')
            if raw is None:
                continue
            values[field_name] = _coerce_env_value(field_name, field.annotation, raw)
        return values

    @classmethod
    def from_env(cls) -> DistributedConfig:
        values = cls.values_from_env()
        return cls(**values)

    def validate_for_service(self, service: str) -> None:
        artifact_ticket_services = {'control-gateway', 'application-gateway', 'reconciler'}
        artifact_store_services = {'control-gateway', 'application-gateway', 'reconciler'}
        database_services = {
            'db-init',
            'control-gateway',
            'application-gateway',
            'scheduler',
            'reconciler',
            'instance-manager',
        }
        internal_token_services = database_services - {'db-init'}
        if self.auth_mode not in {'disabled', 'identity', 'required'}:
            raise ValueError(f'unsupported auth mode: {self.auth_mode}')
        if service == 'instance-manager' and self.request_policy is not None:
            validate_manager_request_policy(self.policy_executor, self.request_policy)
        if service in artifact_ticket_services:
            self._validate_keyring(
                'artifact upload ticket',
                self.artifact_ticket_keyring(),
                self.artifact_upload_ticket_current_key_id,
            )
        if service in artifact_store_services:
            if self.artifact_store_provider not in {'local', 'generic_repository', 'artifact_service'}:
                raise ValueError(f'unsupported artifact store provider: {self.artifact_store_provider}')
            if self.artifact_store_provider == 'generic_repository' and not self.artifact_generic_repository:
                raise ValueError('generic artifact repository is required')
            for provider in self.artifact_store_read_fallback_providers:
                if provider not in {'local', 'generic_repository', 'artifact_service'}:
                    raise ValueError(f'unsupported artifact store read fallback provider: {provider}')
            if self.artifact_store_routing_enabled:
                allowed_targets = {'local', 'generic_repository', 'artifact_service'}
                for route in self.artifact_store_routes:
                    if route.target not in allowed_targets:
                        raise ValueError(f'unsupported artifact store route target: {route.target}')
                    if route.target == 'generic_repository' and not self.artifact_generic_repository:
                        raise ValueError('generic artifact repository is required')
        if service == 'control-gateway':
            if self.zhiyan_log_query.enabled:
                try:
                    self.read_zhiyan_log_credential(self.zhiyan_log_query.credential_ref)
                except ValueError as exc:
                    raise ValueError(
                        f'Zhiyan log credential is unavailable: {self.zhiyan_log_query.credential_ref}'
                    ) from exc
            database_url = self.read_database_url()
            control_database = make_url(database_url) if database_url else None
            for backend_name, access in self.trace_database_access.items():
                if not access.enabled:
                    continue
                credential = self.resolve_trace_database_credential(access.maintenance_credential_ref)
                if credential is None:
                    raise ValueError(f'trace database maintenance credential is unavailable: {backend_name}')
                # Compared by shape, not exact product: one Tencent Cloud key pair serves
                # every product it is authorised for, so a dcdb pair is equally valid here.
                if not _credential_matches_provider(credential.provider_type, access.account_provider.provider_type):
                    raise ValueError(f'trace database provider credential type mismatch: {backend_name}')
                backend = self.run_trace_sinks[backend_name]
                write_host = backend.config.get('host')
                write_port = backend.config.get('port', 3306)
                if (
                    control_database is not None
                    and isinstance(control_database.host, str)
                    and isinstance(write_host, str)
                    and control_database.host.rstrip('.').lower() == write_host.rstrip('.').lower()
                    and (control_database.port or 3306) == write_port
                    and control_database.database == access.database
                ):
                    raise ValueError('trace database must be different from the control database')
        if service == 'reconciler' and self.terminal_data_purge_enabled:
            backend_name = self.terminal_purge_trace_backend
            if not backend_name:
                raise ValueError('terminal_purge_trace_backend is required when terminal data purge is enabled')
            backend = self.run_trace_sinks.get(backend_name)
            access = self.trace_database_access.get(backend_name)
            if backend is None or access is None or not access.enabled:
                raise ValueError(f'terminal purge trace backend is not configured: {backend_name}')
            if 'TDSQLBoundlessTraceSink' not in (backend.sink_class or ''):
                raise ValueError(f'terminal purge backend must be a TDSQL Boundless sink: {backend_name}')
            if backend.config.get('db') != access.database:
                raise ValueError(f'terminal purge database must match trace backend: {backend_name}')
            # Only a MySQL maintenance credential is usable as a SQL login; every other kind
            # (and an unresolvable ref) makes the purge fall back to the backend's own writer
            # account, so require that fallback rather than the credential itself. Demanding a
            # resolvable credential here rejected the deployments that would never read it.
            credential = self.resolve_trace_database_credential(access.maintenance_credential_ref)
            if not (credential is not None and credential.provider_type == 'mysql') and (
                'password' not in backend.secret_config
            ):
                raise ValueError(f'terminal purge requires a SQL password fallback: {backend_name}')
        if not self.production_mode:
            return
        if self.auth_mode != 'required':
            raise ValueError('production mode requires auth_mode=required')
        if service in database_services and (not self.database_url_file or not Path(self.database_url_file).is_file()):
            raise ValueError('production mode requires a mounted database URL')
        if service in internal_token_services:
            internal_token = self.read_internal_token()
            if len(internal_token) < 32 or internal_token == 'local-token':
                raise ValueError('production mode requires a non-default internal token')
        if service in {'control-gateway', 'application-gateway'} and '*' in self.cors_origins:
            raise ValueError('production mode does not allow wildcard CORS')
        if (
            service in {'control-gateway', 'application-gateway'}
            and len(self.read_tai_application_token().encode('utf-8')) != 32
        ):
            raise ValueError('production mode requires a 32-byte Tai application token')
        if service in artifact_ticket_services and self._uses_default_key(
            self.artifact_ticket_keyring(),
            'local-dev-artifact-upload-key',
        ):
            raise ValueError('production mode requires a non-default artifact upload keyring')
        if (
            service in artifact_store_services
            and not self.artifact_store_routing_enabled
            and self.artifact_store_provider
            not in {
                'generic_repository',
                'artifact_service',
            }
        ):
            raise ValueError('production mode requires a shared artifact repository')
        if service in artifact_store_services:
            for provider in self.artifact_store_read_fallback_providers:
                if provider not in {'generic_repository', 'artifact_service'}:
                    raise ValueError(
                        'production mode artifact store read fallback providers must be shared repositories'
                    )
        if service == 'control-gateway' and not self.application_public_base_url.startswith('https://'):
            raise ValueError('production mode requires an HTTPS application upload base URL')
        if service == 'control-gateway' and not (
            self.console_application_base_url or self.application_public_base_url
        ).startswith('https://'):
            raise ValueError('production mode requires an HTTPS console application base URL')
        if service == 'control-gateway' and not self.control_public_base_url.startswith('https://'):
            raise ValueError('production mode requires an HTTPS control base URL')
        if service == 'control-gateway' and not self.console_public_base_url.startswith('https://'):
            raise ValueError('production mode requires an HTTPS console base URL')
        generic_route_enabled = self.artifact_store_routing_enabled and any(
            route.target == 'generic_repository' for route in self.artifact_store_routes
        )
        artifact_service_route_enabled = self.artifact_store_routing_enabled and any(
            route.target == 'artifact_service' for route in self.artifact_store_routes
        )
        if service in artifact_store_services and (
            (not self.artifact_store_routing_enabled and self.artifact_store_provider == 'generic_repository')
            or generic_route_enabled
        ):
            if not self.artifact_generic_base_url.startswith('https://'):
                raise ValueError('production mode requires an HTTPS generic artifact repository')
            required_access = {
                'application-gateway': 'writer',
                'reconciler': 'maintenance',
            }.get(service)
            if required_access and self.artifact_credential(required_access) is None:
                raise ValueError(f'production mode requires {required_access} artifact credentials')
        # The system store reaches the service through Taihu's no-auth /system
        # prefix and carries no employee identity; instead it must present a
        # verifiable ``X-System-Secret`` credential (checked below).
        # The system store no longer relies on the mere absence of an identity
        # header: it must present a verifiable ``X-System-Secret`` credential.
        # Require it whenever ``artifact_service`` is the active provider or a
        # route target in production, mirroring the HTTPS base-URL check above.
        if (
            service in artifact_store_services
            and (
                (not self.artifact_store_routing_enabled and self.artifact_store_provider == 'artifact_service')
                or artifact_service_route_enabled
            )
            and not self.artifact_service_base_url.startswith(('http://', 'https://'))
        ):
            raise ValueError('production mode requires an HTTP or HTTPS artifact service base URL')
        if (
            service in artifact_store_services
            and (
                (not self.artifact_store_routing_enabled and self.artifact_store_provider == 'artifact_service')
                or artifact_service_route_enabled
            )
            and not self.artifact_service_shared_secret()
        ):
            raise ValueError('production mode requires an artifact service system shared secret')
        dynamic_controller_enabled = bool(self.environment_enabled_providers)
        if service == 'reconciler' and dynamic_controller_enabled:
            if not re.fullmatch(r'.+@sha256:[0-9a-f]{64}', self.environment_base_image):
                raise ValueError('production dynamic environments require a digest-pinned workload image')
            if not re.fullmatch(r'.+@sha256:[0-9a-f]{64}', self.environment_builder_image):
                raise ValueError('production dynamic environments require a digest-pinned builder image')
            if not self.environment_database_url_secret_name:
                raise ValueError('production dynamic environments require a database URL Secret')
            if 'kubernetes' in self.environment_enabled_providers and (
                not Path(self.kubernetes_token_file).is_file() or not Path(self.kubernetes_ca_file).is_file()
            ):
                raise ValueError('production Kubernetes controller requires mounted service-account token and CA')

    @staticmethod
    def _validate_keyring(name: str, keys: dict[str, str], current_key_id: str) -> None:
        if not keys or current_key_id not in keys or not keys[current_key_id]:
            raise ValueError(f'{name} current key is unavailable')

    def read_internal_token(self) -> str:
        return self.secrets.read_internal_token()

    def read_database_url(self) -> str:
        if not self.database_url_file:
            return self.database_url
        try:
            database_url = Path(self.database_url_file).read_text(encoding='utf-8').strip()
        except OSError as exc:
            raise ValueError('database URL file is unavailable') from exc
        if not database_url:
            raise ValueError('database URL file is empty')
        return database_url

    def read_tai_application_token(self) -> str:
        return self.secrets.read_tai_application_token()

    def read_run_trace_credential(self, credential_ref: str) -> str:
        return self.secrets.read_run_trace_credential(credential_ref)

    def read_zhiyan_log_credential(self, credential_ref: str) -> str:
        return self.secrets.read_zhiyan_log_credential(credential_ref)

    def resolve_trace_database_credential(self, credential_ref: str) -> TraceDatabaseCredential | None:
        return self.secrets.resolve_trace_database_credential(credential_ref)

    def submit_token_keyring(self) -> dict[str, str]:
        return self.secrets.submit_token_keyring()

    def artifact_ticket_keyring(self) -> dict[str, str]:
        return self.secrets.artifact_ticket_keyring()

    def artifact_credential(self, access: str) -> BasicCredential | None:
        if access == 'writer':
            return self.secrets.artifact_writer
        if access == 'maintenance':
            return self.secrets.artifact_maintenance
        if access == 'metadata':
            return None
        raise ValueError(f'unsupported artifact credential access: {access}')

    def artifact_service_shared_secret(self) -> str | None:
        return self.secrets.read_artifact_service_shared_secret()

    def resolve_provider_credential(self, credential_ref: str) -> ProviderCredential | None:
        return self.secrets.provider_credentials.get(credential_ref)

    @staticmethod
    def _uses_default_key(keys: dict[str, str], default: str) -> bool:
        return not keys or any(not value or value == default for value in keys.values())
