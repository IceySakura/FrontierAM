from __future__ import annotations

from datetime import timedelta

import pytest

from pydantic import SecretStr, ValidationError

from nexus.distributed.config import BasicCredential, DistributedConfig, DistributedSecrets, ZhiyanLogQueryConfig


def test_distributed_secrets_are_typed_and_excluded_from_config_dump() -> None:
    secrets = DistributedSecrets(
        internal_token='internal-secret',
        artifact_writer={'username': 'writer', 'token': 'writer-token'},
        run_trace_credentials={'trajectory-db': 'trajectory-password'},
    )
    config = DistributedConfig(secrets=secrets)

    assert isinstance(secrets.internal_token, SecretStr)
    assert secrets.read_internal_token() == 'internal-secret'
    assert secrets.artifact_writer == BasicCredential(username='writer', token='writer-token')
    assert secrets.read_run_trace_credential('trajectory-db') == 'trajectory-password'
    assert 'internal-secret' not in repr(config)
    assert 'writer-token' not in repr(config)
    assert 'trajectory-password' not in repr(config)
    assert 'secrets' not in config.model_dump()


def test_submission_auto_cpu_default_can_be_configured(monkeypatch) -> None:
    assert DistributedConfig().submission_auto_increase_cpu is True
    monkeypatch.setenv('NEXUS_DIST_SUBMISSION_AUTO_INCREASE_CPU', 'false')
    assert DistributedConfig.from_env().submission_auto_increase_cpu is False


def test_terminal_data_purge_defaults() -> None:
    config = DistributedConfig()

    assert config.terminal_data_purge_enabled is False
    assert config.terminal_purge_trace_backend is None
    assert config.terminal_purge_retention_days == 7
    assert config.terminal_purge_reconcile_interval == 3600.0
    assert config.terminal_purge_partition_batch == 20
    assert config.terminal_purge_ddl_timeout == 60.0
    assert config.terminal_purge_reconcile_timeout == 300.0


def _terminal_purge_config(**overrides) -> DistributedConfig:
    values = {
        'terminal_data_purge_enabled': True,
        'terminal_purge_trace_backend': 'boundless',
        'run_trace_sinks': {
            'boundless': {
                'kind': 'tdsql',
                'sink_class': 'nexus.runner.trace.tdsql::TDSQLBoundlessTraceSink',
                'source_class': 'nexus.runner.trace.tdsql::TDSQLBoundlessTraceSource',
                'config': {'host': 'trace-host', 'db': 'trace', 'user': 'writer'},
                'secret_config': {'password': 'boundless-writer-password'},
            }
        },
        'trace_database_access': {
            'boundless': {
                'enabled': True,
                'database': 'trace',
                'read_host': 'trace-host',
                'maintenance_credential_ref': 'absent-ref',
            }
        },
    }
    values.update(overrides)
    return DistributedConfig(**values)


def test_terminal_purge_accepts_writer_password_without_maintenance_credential() -> None:
    """An unresolvable maintenance ref is fine: the purge falls back to the writer account."""
    config = _terminal_purge_config()

    assert config.resolve_trace_database_credential('absent-ref') is None
    config.validate_for_service('reconciler')


def test_terminal_purge_requires_a_sql_login() -> None:
    """With neither a MySQL maintenance credential nor a writer password there is no login."""
    config = _terminal_purge_config(
        run_trace_sinks={
            'boundless': {
                'kind': 'tdsql',
                'sink_class': 'nexus.runner.trace.tdsql::TDSQLBoundlessTraceSink',
                'source_class': 'nexus.runner.trace.tdsql::TDSQLBoundlessTraceSource',
                'config': {'host': 'trace-host', 'db': 'trace', 'user': 'writer'},
                'secret_config': {},
            }
        },
    )

    with pytest.raises(ValueError, match='terminal purge requires a SQL password fallback'):
        config.validate_for_service('reconciler')


def test_zhiyan_log_query_config_is_disabled_by_default_and_validates_enabled_identity() -> None:
    assert DistributedConfig().zhiyan_log_query.enabled is False
    config = DistributedConfig(
        zhiyan_log_query={
            'enabled': True,
            'endpoint': 'http://openapi.zhiyan.woa.com',
            'project_name': 'Nexus-Agent',
            'staff_name': 'service-account',
            'credential_ref': 'nexus-zhiyan-openapi',
        },
        secrets={'zhiyan_log_credentials': {'nexus-zhiyan-openapi': 'secret-token'}},
    )

    assert isinstance(config.zhiyan_log_query, ZhiyanLogQueryConfig)
    assert config.zhiyan_log_query.query_window_seconds == 21600
    assert config.read_zhiyan_log_credential('nexus-zhiyan-openapi') == 'secret-token'


@pytest.mark.parametrize(
    ('field', 'value', 'message'),
    [
        ('endpoint', 'ftp://openapi.zhiyan.woa.com', 'endpoint'),
        ('staff_name', '', 'staff_name'),
        ('credential_ref', '', 'credential_ref'),
    ],
)
def test_enabled_zhiyan_log_query_rejects_invalid_required_fields(field: str, value: str, message: str) -> None:
    values = {
        'enabled': True,
        'endpoint': 'http://openapi.zhiyan.woa.com',
        'project_name': 'Nexus-Agent',
        'staff_name': 'service-account',
        'credential_ref': 'nexus-zhiyan-openapi',
    }
    values[field] = value

    with pytest.raises(ValueError, match=message):
        DistributedConfig(zhiyan_log_query=values)


def test_enabled_zhiyan_log_query_fails_service_validation_without_secret() -> None:
    config = DistributedConfig(
        zhiyan_log_query={
            'enabled': True,
            'endpoint': 'http://openapi.zhiyan.woa.com',
            'project_name': 'Nexus-Agent',
            'staff_name': 'service-account',
            'credential_ref': 'nexus-zhiyan-openapi',
        }
    )

    with pytest.raises(ValueError, match='Zhiyan log credential'):
        config.validate_for_service('control-gateway')


def test_database_url_is_excluded_from_config_repr_and_dump() -> None:
    database_url = 'mysql+asyncmy://user:database-password@example/nexus'
    config = DistributedConfig(database_url=database_url)

    assert database_url not in repr(config)
    assert 'database_url' not in config.model_dump()


def test_distributed_secrets_reject_unknown_fields() -> None:
    with pytest.raises(ValueError, match='Extra inputs are not permitted'):
        DistributedSecrets.model_validate({'scheduler_poll_interval': 0.1})


def test_distributed_config_exposes_only_explicit_env_values(monkeypatch) -> None:
    monkeypatch.delenv('NEXUS_DIST_DATABASE_URL', raising=False)
    monkeypatch.delenv('NEXUS_DIST_MANAGER_POLL_INTERVAL', raising=False)
    monkeypatch.setenv('NEXUS_DIST_SHARD_COUNT', '17')

    values = DistributedConfig.values_from_env()

    assert values['shard_count'] == '17'
    assert 'database_url' not in values
    assert 'manager_poll_interval' not in values


def production_config(tmp_path, **updates) -> DistributedConfig:
    database_url = tmp_path / 'database-url'
    database_url.write_text('sqlite+aiosqlite:///production.db')
    secret_updates = updates.pop('secrets', {})
    secrets = {
        'internal_token': 'i' * 32,
        'tai_application_token': 't' * 32,
        'artifact_ticket_keys': {'v1': 'upload-key-production'},
        'submit_token_hash_keys': {'v1': 'submit-key-production'},
        **secret_updates,
    }
    values = {
        'production_mode': True,
        'auth_mode': 'required',
        'database_url_file': str(database_url),
        'secrets': secrets,
        'artifact_upload_ticket_current_key_id': 'v1',
        'submit_token_current_key_id': 'v1',
        'application_public_base_url': 'https://application.example',
        'artifact_store_provider': 'generic_repository',
        'artifact_generic_repository': 'nexus-agent',
        'environment_builder_image': f'registry.example/builder@sha256:{"b" * 64}',
    }
    values.update(updates)
    return DistributedConfig(**values)


def test_run_trace_artifact_root_does_not_read_legacy_live_env(monkeypatch) -> None:
    monkeypatch.delenv('NEXUS_DIST_RUN_TRACE_ARTIFACT_ROOT', raising=False)
    monkeypatch.setenv('NEXUS_DIST_LIVE_ARTIFACT_ROOT', '/legacy-live-root')

    assert DistributedConfig().run_trace_artifact_root == '/data/nexus-distributed/run-trace-artifacts'


def test_run_trace_artifact_root_reads_run_trace_env(monkeypatch) -> None:
    monkeypatch.setenv('NEXUS_DIST_RUN_TRACE_ARTIFACT_ROOT', '/run-trace-root')

    assert DistributedConfig().run_trace_artifact_root == '/run-trace-root'


def test_database_url_file_is_read_on_demand_for_secret_rotation(tmp_path) -> None:
    database_url_file = tmp_path / 'database-url'
    database_url_file.write_text('sqlite+aiosqlite:///first.db')
    config = DistributedConfig(
        database_url='sqlite+aiosqlite:///ignored.db',
        database_url_file=str(database_url_file),
    )

    assert config.read_database_url() == 'sqlite+aiosqlite:///first.db'

    database_url_file.write_text('sqlite+aiosqlite:///second.db')
    assert config.read_database_url() == 'sqlite+aiosqlite:///second.db'


def test_run_trace_sink_registry_config_reads_structured_env(monkeypatch) -> None:
    monkeypatch.setenv(
        'NEXUS_DIST_RUN_TRACE_SINKS',
        '{"local-fast":{'
        '"sinkClass":"nexus.runner.trace.local::RunTraceLocalSink",'
        '"sourceClass":"nexus.runner.trace.local::RunTraceLocalSource",'
        '"config":{"root":"/tmp/traces"},'
        '"sinkConfig":{"batch_size":20}}}',
    )
    monkeypatch.setenv('NEXUS_DIST_DEFAULT_RUN_TRACE_SINKS', '["local-fast"]')

    config = DistributedConfig.from_env()

    assert config.default_run_trace_sinks == ['local-fast']
    assert config.run_trace_sinks['local-fast'].sink_class == 'nexus.runner.trace.local::RunTraceLocalSink'
    assert config.run_trace_sinks['local-fast'].source_class == 'nexus.runner.trace.local::RunTraceLocalSource'
    assert config.run_trace_sinks['local-fast'].config == {'root': '/tmp/traces'}
    assert config.run_trace_sinks['local-fast'].sink_config == {'batch_size': 20}


def test_run_trace_sink_config_rejects_unknown_fields_outside_db_reads() -> None:
    with pytest.raises(ValueError, match='futureMetadata'):
        DistributedConfig(
            run_trace_sinks={
                'tdsql': {
                    'sinkClass': 'nexus.runner.trace.tdsql::TDSQLTraceSink',
                    'futureMetadata': {'owner': 'nexus', 'retentionDays': 30},
                }
            }
        )


def test_distributed_config_still_rejects_unknown_top_level_fields() -> None:
    with pytest.raises(ValueError, match='Extra inputs are not permitted'):
        DistributedConfig.model_validate({'future_top_level_setting': True})


def test_dependency_builder_defaults_to_tencent_pypi_mirror() -> None:
    assert DistributedConfig().dependency_builder_index_url == 'https://mirrors.tencent.com/pypi/simple/'


def test_console_public_base_url_defaults_to_cloud_console() -> None:
    assert DistributedConfig().console_public_base_url == 'https://cloud.nexus-agent.woa.com'


def test_control_public_base_url_defaults_to_cloud_control_api() -> None:
    assert DistributedConfig().control_public_base_url == 'https://control-runner-nexus-agent.mcp.woa.com'


@pytest.mark.parametrize(
    'field_name',
    ['control_public_base_url', 'console_public_base_url', 'console_application_base_url'],
)
def test_production_control_gateway_requires_https_public_urls(tmp_path, field_name: str) -> None:
    config = production_config(tmp_path, **{field_name: 'http://runner.example'})

    with pytest.raises(ValueError, match='requires an HTTPS'):
        config.validate_for_service('control-gateway')


def test_batch_load_defaults_match_validated_thousand_slot_profile() -> None:
    config = DistributedConfig()

    assert config.scheduler_poll_interval == 0.1
    assert config.scheduler_scan_limit == 250
    assert config.scheduler_max_assignments == 250
    assert config.scheduler_manager_lock_limit == 64
    assert config.scheduler_manager_lock_contention_backoff == timedelta(milliseconds=250)
    assert config.batch_reconcile_poll_interval == 0.1
    assert config.batch_submit_batch_size == 2000
    assert config.environment_db_pool_size == 1
    assert config.environment_db_max_overflow == 0
    assert config.environment_orchestration_db_pool_size == 4
    assert config.environment_orchestration_db_max_overflow == 2
    assert config.batch_lifecycle_poll_interval == 0.5
    assert config.batch_lifecycle_page_size == 500


def test_batch_data_list_pipeline_defaults_and_env_overrides(monkeypatch) -> None:
    config = DistributedConfig()

    assert config.batch_data_frame_max_size_bytes == 256 * 1024**2
    assert config.batch_task_record_max_size_bytes == 256 * 1024**2
    assert config.batch_data_list_max_size_bytes == 4 * 1024**3
    assert config.batch_task_store_write_chunk_bytes == 64 * 1024**2

    monkeypatch.setenv('NEXUS_DIST_BATCH_DATA_FRAME_MAX_SIZE_BYTES', '101')
    monkeypatch.setenv('NEXUS_DIST_BATCH_TASK_RECORD_MAX_SIZE_BYTES', '102')
    monkeypatch.setenv('NEXUS_DIST_BATCH_DATA_LIST_MAX_SIZE_BYTES', '103')
    monkeypatch.setenv('NEXUS_DIST_BATCH_TASK_STORE_WRITE_CHUNK_BYTES', '104')

    configured = DistributedConfig.from_env()
    assert configured.batch_data_frame_max_size_bytes == 101
    assert configured.batch_task_record_max_size_bytes == 102
    assert configured.batch_data_list_max_size_bytes == 103
    assert configured.batch_task_store_write_chunk_bytes == 104


def test_scheduler_manager_lock_contention_backoff_reads_fractional_seconds(monkeypatch) -> None:
    monkeypatch.setenv('NEXUS_DIST_SCHEDULER_MANAGER_LOCK_CONTENTION_BACKOFF_SECONDS', '0.4')

    config = DistributedConfig.from_env()

    assert config.scheduler_manager_lock_contention_backoff == timedelta(milliseconds=400)


def test_batch_submit_batch_size_allows_large_internal_stress_window(monkeypatch) -> None:
    monkeypatch.setenv('NEXUS_DIST_BATCH_SUBMIT_BATCH_SIZE', '8000')

    config = DistributedConfig.from_env()

    assert config.batch_submit_batch_size == 8000


def test_dependency_builder_index_url_reads_deployment_env(monkeypatch) -> None:
    monkeypatch.setenv('NEXUS_DIST_DEPENDENCY_BUILDER_INDEX_URL', 'https://pypi.internal.example/simple/')

    config = DistributedConfig.from_env()

    assert config.dependency_builder_index_url == 'https://pypi.internal.example/simple/'


def test_environment_image_pull_secrets_read_structured_env(monkeypatch) -> None:
    monkeypatch.setenv(
        'NEXUS_DIST_ENVIRONMENT_IMAGE_PULL_SECRETS',
        '["csighub-yunzhiluo","csighub-taolin"]',
    )

    config = DistributedConfig.from_env()

    assert config.environment_image_pull_secrets == ['csighub-yunzhiluo', 'csighub-taolin']


@pytest.mark.parametrize('index_url', ['', 'ftp://pypi.internal.example/simple/'])
def test_dependency_builder_index_url_requires_http(index_url: str) -> None:
    with pytest.raises(ValueError, match='String should match pattern'):
        DistributedConfig(dependency_builder_index_url=index_url)


def test_environment_enabled_providers_read_structured_env(monkeypatch) -> None:
    monkeypatch.setenv('NEXUS_DIST_ENVIRONMENT_ENABLED_PROVIDERS', '["kubernetes","tkex"]')

    config = DistributedConfig.from_env()

    assert config.environment_enabled_providers == ['kubernetes', 'tkex']


@pytest.mark.parametrize('providers', [['unknown'], ['kubernetes', 'kubernetes']])
def test_environment_enabled_providers_are_known_and_unique(providers: list[str]) -> None:
    with pytest.raises(ValueError, match='environment enabled providers'):
        DistributedConfig(environment_enabled_providers=providers)


def test_production_profile_fails_closed_with_development_defaults() -> None:
    with pytest.raises(ValueError, match='auth_mode=required'):
        DistributedConfig(production_mode=True).validate_for_service('control-gateway')


def test_database_backed_control_secrets_are_loaded_and_required_in_production(tmp_path) -> None:
    config = production_config(tmp_path)

    assert config.artifact_ticket_keyring() == {'v1': 'upload-key-production'}
    assert config.submit_token_keyring() == {'v1': 'submit-key-production'}
    config.validate_for_service('control-gateway')

    default_secrets = DistributedSecrets.model_validate({
        **config.secrets.model_dump(),
        'artifact_ticket_keys': {'local': 'local-dev-artifact-upload-key'},
    })
    default_key_config = config.model_copy(
        update={'secrets': default_secrets, 'artifact_upload_ticket_current_key_id': 'local'}
    )
    with pytest.raises(ValueError, match='non-default artifact upload keyring'):
        default_key_config.validate_for_service('control-gateway')


def test_production_roles_only_require_their_projected_secrets(tmp_path) -> None:
    database_url = tmp_path / 'database-url'
    database_url.write_text('sqlite+aiosqlite:///production.db')

    common = {
        'production_mode': True,
        'auth_mode': 'required',
        'database_url_file': str(database_url),
    }
    DistributedConfig(**common).validate_for_service('db-init')
    for service in ('scheduler', 'instance-manager'):
        DistributedConfig(**common, secrets={'internal_token': 'i' * 32}).validate_for_service(service)

    DistributedConfig(
        **common,
        secrets={
            'internal_token': 'i' * 32,
            'artifact_ticket_keys': {'v1': 'upload-key-production'},
            'artifact_maintenance': {'username': 'maintenance', 'token': 'token'},
        },
        artifact_upload_ticket_current_key_id='v1',
        artifact_store_provider='generic_repository',
        artifact_generic_repository='nexus-agent',
    ).validate_for_service('reconciler')


def test_production_database_services_require_mounted_database_url(tmp_path) -> None:
    config = production_config(tmp_path).model_copy(update={'database_url_file': ''})

    with pytest.raises(ValueError, match='mounted database URL'):
        config.validate_for_service('scheduler')


def test_production_identity_auth_requires_tai_application_token(tmp_path) -> None:
    config = production_config(tmp_path)
    config = config.model_copy(update={'secrets': config.secrets.model_copy(update={'tai_application_token': None})})

    with pytest.raises(ValueError, match='Tai application token'):
        config.validate_for_service('control-gateway')


@pytest.mark.parametrize('service', ['control-gateway', 'application-gateway'])
def test_gateway_service_validation_does_not_require_submit_token_keyring(service) -> None:
    config = DistributedConfig(
        auth_mode='required',
        secrets={
            'tai_application_token': 't' * 32,
            'submit_token_hash_keys': {},
        },
    )

    config.validate_for_service(service)


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('environment_fixed_replicas', 0),
        ('environment_manager_capacity', 0),
        ('scheduler_scan_limit', 0),
        ('scheduler_max_assignments', 0),
    ],
)
def test_cloud_capacity_fields_must_be_positive(field, value) -> None:
    with pytest.raises(ValueError, match='greater than or equal to 1'):
        DistributedConfig(**{field: value})


@pytest.mark.parametrize(
    'field',
    [
        'scheduler_poll_interval',
        'scheduler_shard_discover_interval',
        'scheduler_lease_renew_interval',
        'reconciler_poll_interval',
        'reconciler_failure_backoff',
        'tdsql_partition_reconcile_interval',
        'tdsql_partition_reconcile_timeout',
        'manager_poll_interval',
        'manager_heartbeat_interval',
        'manager_local_reconcile_interval',
    ],
)
def test_control_loop_intervals_must_be_positive(field: str) -> None:
    with pytest.raises(ValueError, match='greater than 0'):
        DistributedConfig(**{field: 0})


def test_tdsql_partition_reconcile_is_disabled_by_default() -> None:
    config = DistributedConfig()

    assert config.tdsql_partition_reconcile_enabled is False
    assert config.tdsql_partition_reconcile_interval == 3600.0


@pytest.mark.parametrize(
    'field',
    [
        'manager_healthy_ttl',
        'manager_dead_ttl',
        'assign_ack_timeout',
        'start_timeout',
        'scheduler_lease_ttl',
        'no_capacity_backoff',
    ],
)
def test_control_loop_durations_must_be_positive(field: str) -> None:
    with pytest.raises(ValueError, match='greater than 0'):
        DistributedConfig(**{field: timedelta(0)})


@pytest.mark.parametrize(
    'field',
    [
        'shard_count',
        'reconciler_batch_size',
        'manager_start_batch_size',
    ],
)
def test_control_loop_batch_and_shard_counts_must_be_positive(field: str) -> None:
    with pytest.raises(ValueError, match='greater than or equal to 1'):
        DistributedConfig(**{field: 0})


@pytest.mark.parametrize('horizon_days', [0, 32])
def test_tdsql_partition_reconcile_horizon_is_bounded(horizon_days: int) -> None:
    with pytest.raises(ValueError, match='greater than or equal to 1|less than or equal to 31'):
        DistributedConfig(tdsql_partition_reconcile_horizon_days=horizon_days)


@pytest.mark.parametrize('jitter', [-0.01, 1.0, 2.0])
def test_scheduler_discovery_jitter_must_be_a_fraction_below_one(jitter: float) -> None:
    with pytest.raises(ValueError, match='greater than or equal to 0|less than 1'):
        DistributedConfig(scheduler_shard_discover_jitter=jitter)


def test_scheduler_lease_renew_interval_must_be_shorter_than_ttl() -> None:
    with pytest.raises(ValueError, match='Scheduler lease renew interval'):
        DistributedConfig(
            scheduler_lease_ttl=timedelta(seconds=5),
            scheduler_lease_renew_interval=5,
        )


@pytest.mark.parametrize(
    'field',
    [
        'scheduler_shard_ownership_limit',
        'reconciler_shard_ownership_limit',
    ],
)
def test_shard_ownership_limit_cannot_exceed_shard_count(field: str) -> None:
    with pytest.raises(ValueError, match='ownership limit cannot exceed shard count'):
        DistributedConfig(shard_count=4, **{field: 5})


def test_invalid_scheduler_lease_environment_fails_config_loading(monkeypatch) -> None:
    monkeypatch.setenv('NEXUS_DIST_SCHEDULER_LEASE_TTL_SECONDS', '5')
    monkeypatch.setenv('NEXUS_DIST_SCHEDULER_LEASE_RENEW_INTERVAL', '5')

    with pytest.raises(ValueError, match='Scheduler lease renew interval'):
        DistributedConfig.from_env()


@pytest.mark.parametrize(
    ('healthy_ttl', 'dead_ttl', 'heartbeat_interval', 'message'),
    [
        (timedelta(seconds=5), timedelta(seconds=5), 1, 'healthy TTL must be shorter'),
        (timedelta(seconds=5), timedelta(seconds=10), 5, 'heartbeat interval must be shorter'),
        (timedelta(seconds=5), timedelta(seconds=10), 6, 'heartbeat interval must be shorter'),
    ],
)
def test_manager_liveness_intervals_must_be_strictly_ordered(
    healthy_ttl: timedelta,
    dead_ttl: timedelta,
    heartbeat_interval: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        DistributedConfig(
            manager_healthy_ttl=healthy_ttl,
            manager_dead_ttl=dead_ttl,
            manager_heartbeat_interval=heartbeat_interval,
        )


@pytest.mark.parametrize(
    ('healthy_ttl', 'self_fence_timeout', 'dead_ttl', 'message'),
    [
        (timedelta(seconds=10), timedelta(seconds=10), timedelta(seconds=20), 'healthy TTL must be shorter'),
        (timedelta(seconds=10), timedelta(seconds=20), timedelta(seconds=20), 'self-fence timeout'),
    ],
)
def test_manager_self_fence_timeout_is_between_healthy_and_dead_ttl(
    healthy_ttl: timedelta,
    self_fence_timeout: timedelta,
    dead_ttl: timedelta,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        DistributedConfig(
            manager_heartbeat_interval=1,
            manager_healthy_ttl=healthy_ttl,
            manager_self_fence_timeout=self_fence_timeout,
            manager_dead_ttl=dead_ttl,
        )


def test_manager_loop_resilience_defaults_are_conservative() -> None:
    config = DistributedConfig()

    assert config.manager_executor_hang_timeout == timedelta(minutes=3)
    assert config.manager_hard_exit_timeout == timedelta(seconds=5)
    assert config.thread_executor_min_capacity_ratio == 0.5
    assert config.batch_control_blocking_concurrency == 2


def test_batch_lifecycle_lease_renew_interval_must_be_shorter_than_ttl() -> None:
    with pytest.raises(ValueError, match='Batch lifecycle claim renew interval'):
        DistributedConfig(
            batch_lifecycle_claim_ttl=timedelta(seconds=10),
            batch_lifecycle_claim_renew_interval=10,
        )


def test_default_user_quota_rejects_negative_limits() -> None:
    with pytest.raises(ValueError, match='non-negative'):
        DistributedConfig(default_user_quota={'running_executions': -1})


def test_artifact_maintenance_lease_renew_interval_must_be_shorter_than_ttl() -> None:
    with pytest.raises(ValueError, match='maintenance lease renew interval'):
        DistributedConfig(
            artifact_maintenance_lease_ttl=timedelta(seconds=10),
            artifact_maintenance_lease_renew_interval=timedelta(seconds=10),
        )


def test_dependency_build_lease_renew_interval_reads_env(monkeypatch) -> None:
    monkeypatch.setenv('NEXUS_DIST_DEPENDENCY_BUILD_LEASE_RENEW_INTERVAL_SECONDS', '17')

    config = DistributedConfig.from_env()

    assert config.dependency_build_lease_renew_interval == timedelta(seconds=17)


def test_dependency_build_lease_renew_interval_must_be_shorter_than_ttl() -> None:
    with pytest.raises(ValueError, match='Dependency build lease renew interval'):
        DistributedConfig(
            dependency_build_lease_ttl=timedelta(seconds=10),
            dependency_build_lease_renew_interval=timedelta(seconds=10),
        )


@pytest.mark.parametrize(
    ('service', 'access'),
    [
        ('application-gateway', 'writer'),
        ('reconciler', 'maintenance'),
    ],
)
def test_production_artifact_services_require_access_credentials(tmp_path, service, access) -> None:
    config = production_config(tmp_path)

    with pytest.raises(ValueError, match=f'{access} artifact credentials'):
        config.validate_for_service(service)

    secret_field = f'artifact_{access}'
    config = config.model_copy(
        update={
            'secrets': config.secrets.model_copy(
                update={secret_field: BasicCredential(username='nexus-user', token='nexus-token')}
            )
        }
    )
    config.validate_for_service(service)


def test_production_instance_manager_uses_scoped_application_gateway_credential(tmp_path) -> None:
    production_config(tmp_path).validate_for_service('instance-manager')


def test_production_kubernetes_controller_requires_workload_secrets_and_tls_files(tmp_path) -> None:
    config = production_config(
        tmp_path,
        secrets={'artifact_maintenance': {'username': 'nexus-user', 'token': 'nexus-token'}},
        environment_enabled_providers=['kubernetes'],
        environment_base_image=f'registry.example/nexus@sha256:{"a" * 64}',
        kubernetes_token_file=str(tmp_path / 'kubernetes-token'),
        kubernetes_ca_file=str(tmp_path / 'kubernetes-ca.crt'),
    )

    with pytest.raises(ValueError, match='database URL Secret'):
        config.validate_for_service('reconciler')

    config = config.model_copy(
        update={
            'environment_database_url_secret_name': 'nexus-database',
            # Point at paths that cannot exist: the default locations are real files when the
            # test itself runs inside a Kubernetes pod, which would skip this check entirely.
            'kubernetes_token_file': str(tmp_path / 'missing-kubernetes-token'),
            'kubernetes_ca_file': str(tmp_path / 'missing-kubernetes-ca.crt'),
        }
    )
    with pytest.raises(ValueError, match='service-account token and CA'):
        config.validate_for_service('reconciler')

    kubernetes_token = tmp_path / 'kubernetes-token'
    kubernetes_ca = tmp_path / 'kubernetes-ca.crt'
    kubernetes_token.write_text('service-account-token')
    kubernetes_ca.write_text('test-ca')
    config = config.model_copy(
        update={
            'kubernetes_token_file': str(kubernetes_token),
            'kubernetes_ca_file': str(kubernetes_ca),
        }
    )
    config.validate_for_service('reconciler')


def test_production_dynamic_environment_requires_digest_pinned_image(tmp_path) -> None:
    config = production_config(
        tmp_path,
        secrets={'artifact_maintenance': {'username': 'nexus-user', 'token': 'nexus-token'}},
        environment_enabled_providers=['kubernetes'],
        environment_base_image='registry.example/nexus:latest',
    )

    with pytest.raises(ValueError, match='digest-pinned workload image'):
        config.validate_for_service('reconciler')

    config = config.model_copy(
        update={
            'environment_base_image': f'registry.example/workload@sha256:{"a" * 64}',
            'environment_builder_image': 'registry.example/builder:latest',
        }
    )
    with pytest.raises(ValueError, match='digest-pinned builder image'):
        config.validate_for_service('reconciler')


def test_trace_database_access_resolves_separate_maintenance_credential() -> None:
    config = DistributedConfig(
        database_url='mysql+asyncmy://control@db.example/nexus_control',
        run_trace_sinks={
            'tdsql': {
                'sinkClass': 'nexus.runner.trace.tdsql::TDSQLTraceSink',
                'sourceClass': 'nexus.runner.trace.tdsql::TDSQLTraceSource',
                'config': {'host': 'write.example', 'port': 3306, 'db': 'nexus_traces', 'user': 'writer'},
                'secretConfig': {'password': 'tdsql-writer-password'},
            }
        },
        trace_database_access={
            'tdsql': {
                'read_host': 'read.example',
                'read_port': 3307,
                'database': 'nexus_traces',
                'maintenance_credential_ref': 'tdsql-maintenance',
            }
        },
        secrets={
            'run_trace_credentials': {'tdsql-writer-password': 'writer-secret'},
            'trace_database_credentials': {
                'tdsql-maintenance': {'username': 'account-admin', 'password': 'maintenance-secret'}
            },
        },
    )

    config.validate_for_service('control-gateway')
    credential = config.resolve_trace_database_credential('tdsql-maintenance')

    assert credential is not None
    assert credential.read() == ('account-admin', 'maintenance-secret')
    assert 'maintenance-secret' not in repr(config)
    assert 'maintenance-secret' not in str(config.model_dump())


def test_trace_database_access_resolves_tencent_cloud_dcdb_provider_credential() -> None:
    config = DistributedConfig(
        run_trace_sinks={
            'tdsql': {
                'sinkClass': 'nexus.runner.trace.tdsql::TDSQLTraceSink',
                'sourceClass': 'nexus.runner.trace.tdsql::TDSQLTraceSource',
                'config': {'host': '21.248.112.134', 'port': 3306, 'db': 'runner', 'user': 'writer'},
            }
        },
        trace_database_access={
            'tdsql': {
                'read_host': '21.248.112.134',
                'database': 'runner',
                'maintenance_credential_ref': 'tdsql-cloud-maintenance',
                'account_provider': {
                    'provider_type': 'tencent_cloud_dcdb',
                    'region': 'eu-frankfurt',
                    'instance_id': 'tdsqlshard-9biily2z7z',
                    'api_endpoint': 'dcdb.internal.tencentcloudapi.com',
                },
            }
        },
        secrets={
            'trace_database_credentials': {
                'tdsql-cloud-maintenance': {
                    'provider_type': 'tencent_cloud_dcdb',
                    'secret_id': 'test-secret-id',
                    'secret_key': 'test-secret-key',
                }
            }
        },
    )

    config.validate_for_service('control-gateway')
    access = config.trace_database_access['tdsql']
    credential = config.resolve_trace_database_credential('tdsql-cloud-maintenance')

    assert access.account_provider.provider_type == 'tencent_cloud_dcdb'
    assert access.account_provider.region == 'eu-frankfurt'
    assert access.account_provider.instance_id == 'tdsqlshard-9biily2z7z'
    assert access.account_provider.api_endpoint == 'dcdb.internal.tencentcloudapi.com'
    assert credential is not None
    assert credential.provider_type == 'tencent_cloud_dcdb'
    assert credential.read() == ('test-secret-id', 'test-secret-key')
    assert 'test-secret-key' not in repr(config)


def test_trace_database_access_resolves_tencent_cloud_tdmysql_provider_for_boundless() -> None:
    config = DistributedConfig(
        extend_run_trace_sinks={
            'tdsql-boundless': {
                'sinkClass': 'nexus.runner.trace.tdsql::TDSQLBoundlessTraceSink',
                'sourceClass': 'nexus.runner.trace.tdsql::TDSQLBoundlessTraceSource',
                'config': {'host': '21.131.14.24', 'port': 3306, 'db': 'runner', 'user': 'nexus_dev'},
            }
        },
        trace_database_access={
            'tdsql-boundless': {
                'read_host': '21.131.14.24',
                'database': 'runner',
                'maintenance_credential_ref': 'tdsql-boundless-maintenance',
                'account_provider': {
                    'provider_type': 'tencent_cloud_tdmysql',
                    'region': 'ap-nanjing',
                    'instance_id': 'tdsql3-49qrr9xo6y',
                },
            }
        },
        secrets={
            'trace_database_credentials': {
                'tdsql-boundless-maintenance': {
                    'provider_type': 'tencent_cloud_tdmysql',
                    'secret_id': 'test-secret-id',
                    'secret_key': 'test-secret-key',
                }
            }
        },
    )

    config.validate_for_service('control-gateway')
    access = config.trace_database_access['tdsql-boundless']
    credential = config.resolve_trace_database_credential('tdsql-boundless-maintenance')

    assert access.account_provider.provider_type == 'tencent_cloud_tdmysql'
    assert access.account_provider.instance_id == 'tdsql3-49qrr9xo6y'
    assert access.account_provider.api_endpoint == 'tdmysql.tencentcloudapi.com'
    assert credential is not None
    assert credential.read() == ('test-secret-id', 'test-secret-key')
    assert 'test-secret-key' not in repr(config)


def test_trace_database_access_rejects_dcdb_instance_id_on_tdmysql_provider() -> None:
    with pytest.raises(ValueError, match='String should match pattern'):
        DistributedConfig(
            run_trace_sinks={
                'tdsql': {
                    'sinkClass': 'nexus.runner.trace.tdsql::TDSQLTraceSink',
                    'sourceClass': 'nexus.runner.trace.tdsql::TDSQLTraceSource',
                    'config': {'host': 'write.example', 'db': 'runner', 'user': 'writer'},
                }
            },
            trace_database_access={
                'tdsql': {
                    'read_host': 'read.example',
                    'database': 'runner',
                    'maintenance_credential_ref': 'tdsql-maintenance',
                    'account_provider': {
                        'provider_type': 'tencent_cloud_tdmysql',
                        'region': 'ap-nanjing',
                        'instance_id': 'dcdbt-9biily2z7z',
                    },
                }
            },
        )


def test_trace_database_access_rejects_non_tdsql_trace_source() -> None:
    with pytest.raises(ValueError, match='must use a TDSQL trace source'):
        DistributedConfig(
            run_trace_sinks={
                'custom': {
                    'sinkClass': 'nexus.runner.trace.local::LocalTraceSink',
                    'sourceClass': 'nexus.runner.trace.local::LocalTraceSource',
                    'config': {'host': 'write.example', 'db': 'runner', 'user': 'writer'},
                }
            },
            trace_database_access={
                'custom': {
                    'read_host': 'read.example',
                    'database': 'runner',
                    'maintenance_credential_ref': 'tdsql-maintenance',
                }
            },
        )


@pytest.mark.parametrize('empty_field', ['secret_id', 'secret_key'])
def test_trace_database_access_rejects_empty_tencent_cloud_credential(empty_field: str) -> None:
    credential = {
        'provider_type': 'tencent_cloud_dcdb',
        'secret_id': 'test-secret-id',
        'secret_key': 'test-secret-key',
    }
    credential[empty_field] = ''

    with pytest.raises(ValueError, match='at least 1 item'):
        DistributedSecrets(trace_database_credentials={'tdsql-cloud-maintenance': credential})


def test_trace_database_access_rejects_provider_credential_type_mismatch() -> None:
    config = DistributedConfig(
        run_trace_sinks={
            'tdsql': {
                'sinkClass': 'nexus.runner.trace.tdsql::TDSQLTraceSink',
                'sourceClass': 'nexus.runner.trace.tdsql::TDSQLTraceSource',
                'config': {'host': 'write.example', 'db': 'runner', 'user': 'writer'},
            }
        },
        trace_database_access={
            'tdsql': {
                'read_host': 'read.example',
                'database': 'runner',
                'maintenance_credential_ref': 'wrong-provider',
                'account_provider': {
                    'provider_type': 'tencent_cloud_dcdb',
                    'region': 'ap-nanjing',
                    'instance_id': 'tdsqlshard-9biily2z7z',
                },
            }
        },
        secrets={
            'trace_database_credentials': {
                'wrong-provider': {'username': 'account-admin', 'password': 'maintenance-secret'}
            }
        },
    )

    with pytest.raises(ValueError, match='provider credential type mismatch'):
        config.validate_for_service('control-gateway')


def test_trace_database_access_rejects_database_mismatch_with_trace_backend() -> None:
    with pytest.raises(ValueError, match='database must match'):
        DistributedConfig(
            run_trace_sinks={
                'tdsql': {
                    'sinkClass': 'nexus.runner.trace.tdsql::TDSQLTraceSink',
                    'sourceClass': 'nexus.runner.trace.tdsql::TDSQLTraceSource',
                    'config': {'host': 'write.example', 'db': 'nexus_traces', 'user': 'writer'},
                }
            },
            trace_database_access={
                'tdsql': {
                    'read_host': 'read.example',
                    'database': 'other_database',
                    'maintenance_credential_ref': 'tdsql-maintenance',
                }
            },
        )


def test_trace_database_access_allows_same_schema_name_on_different_endpoint() -> None:
    config = DistributedConfig(
        database_url='mysql+asyncmy://control@db.example/nexus_traces',
        run_trace_sinks={
            'tdsql': {
                'sinkClass': 'nexus.runner.trace.tdsql::TDSQLTraceSink',
                'sourceClass': 'nexus.runner.trace.tdsql::TDSQLTraceSource',
                'config': {'host': 'write.example', 'db': 'nexus_traces', 'user': 'writer'},
            }
        },
        trace_database_access={
            'tdsql': {
                'read_host': 'read.example',
                'database': 'nexus_traces',
                'maintenance_credential_ref': 'tdsql-maintenance',
            }
        },
        secrets={
            'trace_database_credentials': {
                'tdsql-maintenance': {'username': 'account-admin', 'password': 'maintenance-secret'}
            }
        },
    )

    config.validate_for_service('control-gateway')


def test_trace_database_access_rejects_control_database() -> None:
    config = DistributedConfig(
        database_url='mysql+asyncmy://control@write.example:3307/nexus_traces',
        run_trace_sinks={
            'tdsql': {
                'sinkClass': 'nexus.runner.trace.tdsql::TDSQLTraceSink',
                'sourceClass': 'nexus.runner.trace.tdsql::TDSQLTraceSource',
                'config': {'host': 'write.example', 'port': 3307, 'db': 'nexus_traces', 'user': 'writer'},
            }
        },
        trace_database_access={
            'tdsql': {
                'read_host': 'read.example',
                'database': 'nexus_traces',
                'maintenance_credential_ref': 'tdsql-maintenance',
            }
        },
        secrets={
            'trace_database_credentials': {
                'tdsql-maintenance': {'username': 'account-admin', 'password': 'maintenance-secret'}
            }
        },
    )

    with pytest.raises(ValueError, match='different from the control database'):
        config.validate_for_service('control-gateway')


def test_one_tencent_cloud_key_pair_serves_both_products() -> None:
    """A pair registered for dcdb must drive tdmysql too.

    Tencent Cloud authorises products on an existing secret id/key pair rather than
    issuing a new one per product, so requiring an exact provider match would force the
    same secret into two entries -- and a rotation that updates one leaves the other
    failing authentication with nothing to point at the cause.
    """
    config = DistributedConfig(
        extend_run_trace_sinks={
            'tdsql-boundless': {
                'sinkClass': 'nexus.runner.trace.tdsql::TDSQLBoundlessTraceSink',
                'sourceClass': 'nexus.runner.trace.tdsql::TDSQLBoundlessTraceSource',
                'config': {'host': '21.131.14.24', 'port': 3306, 'db': 'runner', 'user': 'nexus_dev'},
            }
        },
        trace_database_access={
            'tdsql-boundless': {
                'read_host': '21.131.14.24',
                'database': 'runner',
                # The pair already in use for the 2.0 instances.
                'maintenance_credential_ref': 'tdsql-maintenance',
                'account_provider': {
                    'provider_type': 'tencent_cloud_tdmysql',
                    'region': 'ap-nanjing',
                    'instance_id': 'tdsql3-49qrr9xo6y',
                },
            }
        },
        secrets={
            'trace_database_credentials': {
                'tdsql-maintenance': {
                    'provider_type': 'tencent_cloud_dcdb',
                    'secret_id': 'test-secret-id',
                    'secret_key': 'test-secret-key',
                }
            }
        },
    )

    config.validate_for_service('control-gateway')

    credential = config.resolve_trace_database_credential('tdsql-maintenance')
    assert credential is not None
    assert credential.read() == ('test-secret-id', 'test-secret-key')


def test_mysql_credential_cannot_drive_a_cloud_provider() -> None:
    """The distinction worth keeping: a database user cannot sign a cloud API call."""
    with pytest.raises(ValueError, match='credential type mismatch'):
        DistributedConfig(
            extend_run_trace_sinks={
                'tdsql-boundless': {
                    'sinkClass': 'nexus.runner.trace.tdsql::TDSQLBoundlessTraceSink',
                    'sourceClass': 'nexus.runner.trace.tdsql::TDSQLBoundlessTraceSource',
                    'config': {'host': '21.131.14.24', 'port': 3306, 'db': 'runner', 'user': 'nexus_dev'},
                }
            },
            trace_database_access={
                'tdsql-boundless': {
                    'read_host': '21.131.14.24',
                    'database': 'runner',
                    'maintenance_credential_ref': 'local-mysql',
                    'account_provider': {
                        'provider_type': 'tencent_cloud_tdmysql',
                        'region': 'ap-nanjing',
                        'instance_id': 'tdsql3-49qrr9xo6y',
                    },
                }
            },
            secrets={
                'trace_database_credentials': {
                    'local-mysql': {'provider_type': 'mysql', 'username': 'u', 'password': 'p'}
                }
            },
        ).validate_for_service('control-gateway')


def test_a_dcdb_registered_pair_builds_a_tdmysql_provider() -> None:
    """The config check is only half of it; the runtime factory must agree."""
    from nexus.distributed.access.trace_database import TraceDatabaseTarget, _create_provider
    from nexus.distributed.config import (
        TencentCloudTDMySQLTraceDatabaseAccountProviderConfig,
        TencentCloudTraceDatabaseCredential,
    )

    credential = TencentCloudTraceDatabaseCredential(
        provider_type='tencent_cloud_dcdb', secret_id='AKID', secret_key='key'
    )
    target = TraceDatabaseTarget(
        backend_name='tdsql-boundless',
        write_host='21.131.14.24',
        write_port=3306,
        read_host='21.131.14.24',
        read_port=3306,
        database='runner',
        account_provider=TencentCloudTDMySQLTraceDatabaseAccountProviderConfig(
            provider_type='tencent_cloud_tdmysql',
            region='ap-nanjing',
            instance_id='tdsql3-49qrr9xo6y',
        ),
    )

    provider = _create_provider(target, credential)

    assert type(provider).__name__ == 'TencentCloudTDMySQLTraceDatabaseAccountProvider'
    assert provider.flow_timeout_seconds == 180.0


def _tdmysql_target(**provider_kwargs):
    from nexus.distributed.access.trace_database import TraceDatabaseTarget
    from nexus.distributed.config import TencentCloudTDMySQLTraceDatabaseAccountProviderConfig

    return TraceDatabaseTarget(
        backend_name='tdsql-boundless',
        write_host='21.131.14.24',
        write_port=3306,
        read_host='21.131.14.24',
        read_port=3306,
        database='runner',
        account_provider=TencentCloudTDMySQLTraceDatabaseAccountProviderConfig(
            provider_type='tencent_cloud_tdmysql',
            region='ap-nanjing',
            instance_id='tdsql3-49qrr9xo6y',
            **provider_kwargs,
        ),
    )


def test_tdmysql_flow_timeout_is_configurable_and_reaches_the_provider() -> None:
    from nexus.distributed.access.trace_database import _create_provider
    from nexus.distributed.config import TencentCloudTraceDatabaseCredential

    credential = TencentCloudTraceDatabaseCredential(
        provider_type='tencent_cloud_tdmysql', secret_id='AKID', secret_key='key'
    )

    provider = _create_provider(_tdmysql_target(flow_timeout_seconds=420.0), credential)

    assert provider.flow_timeout_seconds == 420.0


@pytest.mark.parametrize('value', [0, -1, 601])
def test_tdmysql_flow_timeout_rejects_values_outside_the_supported_range(value) -> None:
    with pytest.raises(ValidationError):
        _tdmysql_target(flow_timeout_seconds=value)


def test_tdmysql_flow_timeout_round_trips_through_stored_config() -> None:
    """The value has to survive the dict form that config overrides are persisted as."""
    from nexus.distributed.config import TraceDatabaseAccessConfig

    access = TraceDatabaseAccessConfig.model_validate({
        'read_host': '21.131.14.24',
        'database': 'runner',
        'maintenance_credential_ref': 'tdsql-maintenance',
        'account_provider': {
            'provider_type': 'tencent_cloud_tdmysql',
            'region': 'ap-nanjing',
            'instance_id': 'tdsql3-49qrr9xo6y',
            'flow_timeout_seconds': 240,
        },
    })

    assert access.account_provider.flow_timeout_seconds == 240.0
    assert access.model_dump()['account_provider']['flow_timeout_seconds'] == 240.0


def test_stored_config_without_a_flow_timeout_still_loads_with_the_default() -> None:
    """Deploying the field must not require touching stored config: omitting it is valid.

    Shaped after the production entry, which predates the field. A missing key has to fall
    back rather than fail, since a rejected trace_database_access entry stops the service
    from starting at all.
    """
    from nexus.distributed.access.trace_database import TraceDatabaseTarget, _create_provider
    from nexus.distributed.config import TraceDatabaseAccessConfig, TencentCloudTraceDatabaseCredential

    access = TraceDatabaseAccessConfig.model_validate({
        'read_host': '21.131.12.30',
        'read_port': 3306,
        'database': 'runner',
        'maintenance_credential_ref': 'tdsql-maintenance',
        'account_provider': {
            'provider_type': 'tencent_cloud_tdmysql',
            'region': 'ap-guangzhou',
            'instance_id': 'tdsql3-m41nrs10q8',
            'api_endpoint': 'tdmysql.internal.tencentcloudapi.com',
            'account_host': '%',
        },
        'enabled': True,
    })

    assert access.account_provider.flow_timeout_seconds == 180.0

    provider = _create_provider(
        TraceDatabaseTarget(
            backend_name='tdsql-boundless',
            write_host='21.131.12.30',
            write_port=3306,
            read_host='21.131.12.30',
            read_port=3306,
            database='runner',
            account_provider=access.account_provider,
        ),
        TencentCloudTraceDatabaseCredential(provider_type='tencent_cloud_tdmysql', secret_id='AKID', secret_key='key'),
    )

    assert provider.flow_timeout_seconds == 180.0


def test_dcdb_provider_config_does_not_gain_the_tdmysql_flow_timeout() -> None:
    """The field is scoped to tdmysql: DCDB shows neither the pause nor the slow flows."""
    from nexus.distributed.config import TencentCloudDCDBTraceDatabaseAccountProviderConfig

    config = TencentCloudDCDBTraceDatabaseAccountProviderConfig(
        provider_type='tencent_cloud_dcdb', region='ap-nanjing', instance_id='tdsqlshard-lu0c3wfhxp'
    )

    assert not hasattr(config, 'flow_timeout_seconds')
    with pytest.raises(ValidationError):
        TencentCloudDCDBTraceDatabaseAccountProviderConfig(
            provider_type='tencent_cloud_dcdb',
            region='ap-nanjing',
            instance_id='tdsqlshard-lu0c3wfhxp',
            flow_timeout_seconds=240,
        )
