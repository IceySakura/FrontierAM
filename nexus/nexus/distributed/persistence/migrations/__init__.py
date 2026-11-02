from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


INITIAL_REVISION = '0001_initial'
CLEANUP_REVISION = '0002_add_execution_cleanup'
LEGACY_CLOUD_SUBMIT_REVISION = '0003_cloud_submit_foundation'
TRACE_CARRIER_REVISION = '0003_add_trace_carriers'
CONFIG_OVERRIDES_REVISION = '0004_add_config_overrides'
CLOUD_SUBMIT_REVISION = '0004_cloud_submit_foundation'
RUN_IDENTITY_REVISION = '0005_unify_run_identity'
EXECUTION_DEADLINES_REVISION = '0006_add_execution_deadlines'
BATCH_LIFECYCLE_REVISION = '0007_add_batch_lifecycle_claim'
BATCH_ITEM_HOT_PATH_REVISION = '0008_batch_item_hot_index'
BATCH_ITEM_COLLATION_REVISION = '0009_batch_item_bin_collation'
BATCH_LIFECYCLE_FAIRNESS_REVISION = '0010_batch_lifecycle_fair'
DEPENDENCY_PROVIDER_SNAPSHOT_REVISION = '0011_dependency_provider_snap'
LEGACY_ENVIRONMENT_CLEANUP_REVISION = '0012_remove_legacy_env_fields'
LEGACY_QUOTA_CLEANUP_REVISION = '0013_drop_legacy_quota_fields'
SHARD_LEASE_REVISION = '0014_rename_scheduler_leases'
MANAGER_LIVENESS_REVISION = '0015_add_manager_liveness'
BATCH_RUN_SHARD_REVISION = '0016_add_batch_run_shard'
RECONCILER_SHARD_INDEX_REVISION = '0017_reconciler_shard_indexes'
RESOURCE_USAGE_REVISION = '0018_precreate_resource_usage'
EXECUTION_PARENT_CURSOR_REVISION = '0019_add_execution_parent_cursor'
CONFIG_OVERRIDES_ENSURE_REVISION = '0020_ensure_config_overrides'
PROVIDER_CONFIG_REVISION = '0021_add_provider_configs'
GRANT_SCOPES_REVISION = '0022_add_grant_scopes'
CONFIG_SECRET_OVERRIDES_REVISION = '0023_add_config_secret_overrides'
EXECUTION_CREDENTIALS_REVISION = '0024_add_execution_credentials'
DEFAULT_SYSTEM_TOKEN_QUOTA_REVISION = '0025_default_token_quota'
SUBMIT_TOKEN_APP_BINDING_REVISION = '0026_submit_token_app_binding'
TRACE_DATABASE_ACCOUNTS_REVISION = '0027_trace_database_accounts'
BATCH_MODULE_DEPENDENCIES_REVISION = '0028_batch_module_dependencies'
RUN_TRACE_BACKEND_GRANTS_REVISION = '0029_run_trace_backend_grants'
PRINCIPAL_ACCESS_GRANTS_REVISION = '0030_principal_access_grants'
BUILD_LIST_INDEX_REVISION = '0031_build_list_index'
OBSERVABILITY_SNAPSHOTS_REVISION = '0032_observability_snapshots'
BUSINESS_STATISTICS_REVISION = '0033_business_statistics'
BATCH_FINISH_REVISION = '0034_batch_finish_backfill'
POLICY_SCOPES_REVISION = '0035_policy_scopes'
REQUEST_BUDGET_LEASES_REVISION = '0036_request_budget_leases'
HEAD_REVISION = REQUEST_BUDGET_LEASES_REVISION


@dataclass(frozen=True)
class Migration:
    revision: str
    down_revision: str | None
    module: str


MIGRATIONS: Sequence[Migration] = (
    Migration(INITIAL_REVISION, None, 'nexus.distributed.persistence.migrations.versions.0001_initial'),
    Migration(
        CLEANUP_REVISION,
        INITIAL_REVISION,
        'nexus.distributed.persistence.migrations.versions.0002_add_execution_cleanup',
    ),
    Migration(
        TRACE_CARRIER_REVISION,
        CLEANUP_REVISION,
        'nexus.distributed.persistence.migrations.versions.0003_add_trace_carriers',
    ),
    Migration(
        CONFIG_OVERRIDES_REVISION,
        TRACE_CARRIER_REVISION,
        'nexus.distributed.persistence.migrations.versions.0004_add_config_overrides',
    ),
    Migration(
        CLOUD_SUBMIT_REVISION,
        CONFIG_OVERRIDES_REVISION,
        'nexus.distributed.persistence.migrations.versions.0004_cloud_submit_foundation',
    ),
    Migration(
        RUN_IDENTITY_REVISION,
        CLOUD_SUBMIT_REVISION,
        'nexus.distributed.persistence.migrations.versions.0005_unify_distributed_run_identity',
    ),
    Migration(
        EXECUTION_DEADLINES_REVISION,
        RUN_IDENTITY_REVISION,
        'nexus.distributed.persistence.migrations.versions.0006_add_execution_deadlines',
    ),
    Migration(
        BATCH_LIFECYCLE_REVISION,
        EXECUTION_DEADLINES_REVISION,
        'nexus.distributed.persistence.migrations.versions.0007_add_batch_lifecycle_claim',
    ),
    Migration(
        BATCH_ITEM_HOT_PATH_REVISION,
        BATCH_LIFECYCLE_REVISION,
        'nexus.distributed.persistence.migrations.versions.0008_add_batch_item_hot_path_index',
    ),
    Migration(
        BATCH_ITEM_COLLATION_REVISION,
        BATCH_ITEM_HOT_PATH_REVISION,
        'nexus.distributed.persistence.migrations.versions.0009_add_batch_item_binary_collation',
    ),
    Migration(
        BATCH_LIFECYCLE_FAIRNESS_REVISION,
        BATCH_ITEM_COLLATION_REVISION,
        'nexus.distributed.persistence.migrations.versions.0010_add_batch_lifecycle_fairness',
    ),
    Migration(
        DEPENDENCY_PROVIDER_SNAPSHOT_REVISION,
        BATCH_LIFECYCLE_FAIRNESS_REVISION,
        'nexus.distributed.persistence.migrations.versions.0011_add_dependency_provider_snapshot',
    ),
    Migration(
        LEGACY_ENVIRONMENT_CLEANUP_REVISION,
        DEPENDENCY_PROVIDER_SNAPSHOT_REVISION,
        'nexus.distributed.persistence.migrations.versions.0012_remove_legacy_environment_fields',
    ),
    Migration(
        LEGACY_QUOTA_CLEANUP_REVISION,
        LEGACY_ENVIRONMENT_CLEANUP_REVISION,
        'nexus.distributed.persistence.migrations.versions.0013_drop_legacy_quota_fields',
    ),
    Migration(
        SHARD_LEASE_REVISION,
        LEGACY_QUOTA_CLEANUP_REVISION,
        'nexus.distributed.persistence.migrations.versions.0014_rename_scheduler_leases',
    ),
    Migration(
        MANAGER_LIVENESS_REVISION,
        SHARD_LEASE_REVISION,
        'nexus.distributed.persistence.migrations.versions.0015_add_manager_liveness',
    ),
    Migration(
        BATCH_RUN_SHARD_REVISION,
        MANAGER_LIVENESS_REVISION,
        'nexus.distributed.persistence.migrations.versions.0016_add_batch_run_shard',
    ),
    Migration(
        RECONCILER_SHARD_INDEX_REVISION,
        BATCH_RUN_SHARD_REVISION,
        'nexus.distributed.persistence.migrations.versions.0017_add_reconciler_shard_indexes',
    ),
    Migration(
        RESOURCE_USAGE_REVISION,
        RECONCILER_SHARD_INDEX_REVISION,
        'nexus.distributed.persistence.migrations.versions.0018_precreate_resource_usage',
    ),
    Migration(
        EXECUTION_PARENT_CURSOR_REVISION,
        RESOURCE_USAGE_REVISION,
        'nexus.distributed.persistence.migrations.versions.0019_add_execution_parent_cursor',
    ),
    Migration(
        CONFIG_OVERRIDES_ENSURE_REVISION,
        EXECUTION_PARENT_CURSOR_REVISION,
        'nexus.distributed.persistence.migrations.versions.0020_ensure_config_overrides',
    ),
    Migration(
        PROVIDER_CONFIG_REVISION,
        CONFIG_OVERRIDES_ENSURE_REVISION,
        'nexus.distributed.persistence.migrations.versions.0021_add_provider_configs',
    ),
    Migration(
        GRANT_SCOPES_REVISION,
        PROVIDER_CONFIG_REVISION,
        'nexus.distributed.persistence.migrations.versions.0022_add_grant_scopes',
    ),
    Migration(
        CONFIG_SECRET_OVERRIDES_REVISION,
        GRANT_SCOPES_REVISION,
        'nexus.distributed.persistence.migrations.versions.0023_add_config_secret_overrides',
    ),
    Migration(
        EXECUTION_CREDENTIALS_REVISION,
        CONFIG_SECRET_OVERRIDES_REVISION,
        'nexus.distributed.persistence.migrations.versions.0024_add_execution_credentials',
    ),
    Migration(
        DEFAULT_SYSTEM_TOKEN_QUOTA_REVISION,
        EXECUTION_CREDENTIALS_REVISION,
        'nexus.distributed.persistence.migrations.versions.0025_add_default_system_token_quota',
    ),
    Migration(
        SUBMIT_TOKEN_APP_BINDING_REVISION,
        DEFAULT_SYSTEM_TOKEN_QUOTA_REVISION,
        'nexus.distributed.persistence.migrations.versions.0026_add_submit_token_app_binding',
    ),
    Migration(
        TRACE_DATABASE_ACCOUNTS_REVISION,
        SUBMIT_TOKEN_APP_BINDING_REVISION,
        'nexus.distributed.persistence.migrations.versions.0027_add_trace_database_accounts',
    ),
    Migration(
        BATCH_MODULE_DEPENDENCIES_REVISION,
        TRACE_DATABASE_ACCOUNTS_REVISION,
        'nexus.distributed.persistence.migrations.versions.0028_add_batch_module_dependencies',
    ),
    Migration(
        RUN_TRACE_BACKEND_GRANTS_REVISION,
        BATCH_MODULE_DEPENDENCIES_REVISION,
        'nexus.distributed.persistence.migrations.versions.0029_add_run_trace_backend_grants',
    ),
    Migration(
        PRINCIPAL_ACCESS_GRANTS_REVISION,
        RUN_TRACE_BACKEND_GRANTS_REVISION,
        'nexus.distributed.persistence.migrations.versions.0030_add_principal_access_grants',
    ),
    Migration(
        BUILD_LIST_INDEX_REVISION,
        PRINCIPAL_ACCESS_GRANTS_REVISION,
        'nexus.distributed.persistence.migrations.versions.0031_add_build_list_index',
    ),
    Migration(
        OBSERVABILITY_SNAPSHOTS_REVISION,
        BUILD_LIST_INDEX_REVISION,
        'nexus.distributed.persistence.migrations.versions.0032_add_observability_snapshots',
    ),
    Migration(
        BUSINESS_STATISTICS_REVISION,
        OBSERVABILITY_SNAPSHOTS_REVISION,
        'nexus.distributed.persistence.migrations.versions.0033_add_business_statistics',
    ),
    Migration(
        BATCH_FINISH_REVISION,
        BUSINESS_STATISTICS_REVISION,
        'nexus.distributed.persistence.migrations.versions.0034_backfill_batch_finished_at',
    ),
    Migration(
        POLICY_SCOPES_REVISION,
        BATCH_FINISH_REVISION,
        'nexus.distributed.persistence.migrations.versions.0035_add_policy_scopes',
    ),
    Migration(
        REQUEST_BUDGET_LEASES_REVISION,
        POLICY_SCOPES_REVISION,
        'nexus.distributed.persistence.migrations.versions.0036_add_request_budget_leases',
    ),
)
