from __future__ import annotations

import json
import math
import re

from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from nexus.distributed.config.views import EnvironmentConfigView
from nexus.distributed.environment.naming import allocation_resource_names
from nexus.distributed.environment.ports import AllocationSnapshot
from nexus.distributed.environment.provider_config import TkexZhiyanLogConfig
from nexus.distributed.environment.zhiyan import add_zhiyan_log_sidecar


@dataclass(frozen=True, slots=True)
class RenderedAllocationWorkload:
    workload: dict[str, Any]


class AllocationWorkloadRenderer:
    def __init__(
        self,
        *,
        config: EnvironmentConfigView,
        cluster_name: str | None = None,
        workload_namespace: str | None = None,
        resource_request_ratio: float = 1.0,
    ) -> None:
        if (
            isinstance(resource_request_ratio, bool)
            or not isinstance(resource_request_ratio, (int, float))
            or not 0 < resource_request_ratio <= 1
        ):
            raise ValueError('resource_request_ratio must be greater than zero and at most one')
        self.config = config
        self.cluster_name = cluster_name
        self.workload_namespace = workload_namespace or config.environment_namespace
        self.resource_request_ratio = float(resource_request_ratio)

    def render(self, allocation: AllocationSnapshot, *, resource_kind: str) -> RenderedAllocationWorkload:
        workload_name, _ = allocation_resource_names(allocation.allocation_id)
        selector_value = allocation.allocation_spec_hash[:32]
        labels = {
            'app.kubernetes.io/name': self.config.environment_app_name,
            'app.kubernetes.io/instance': self.config.environment_release_name,
            'app.kubernetes.io/component': 'instance-manager',
            'app.kubernetes.io/managed-by': 'nexus',
            'nexus.io/allocation-key': selector_value,
            'nexus.io/workload': allocation.workload,
        }
        annotations = {
            'nexus.io/managed-by': 'nexus',
            'nexus.io/allocation-id': allocation.allocation_id,
            'nexus.io/environment-id': allocation.environment_id,
            'nexus.io/allocation-spec-hash': allocation.allocation_spec_hash,
        }
        if allocation.provider_config_ref:
            annotations['nexus.io/provider-config-ref'] = allocation.provider_config_ref
        workload = {
            'apiVersion': 'apps/v1',
            'kind': resource_kind,
            'metadata': {
                'name': workload_name,
                'namespace': self.workload_namespace,
                'labels': labels,
                'annotations': annotations,
            },
            'spec': {
                'serviceName': workload_name,
                'podManagementPolicy': 'Parallel',
                'replicas': allocation.desired_manager_count,
                'selector': {'matchLabels': {'nexus.io/allocation-key': selector_value}},
                'template': {
                    'metadata': {'labels': labels, 'annotations': annotations},
                    'spec': self._pod_spec(allocation),
                },
            },
        }
        if resource_kind == 'StatefulSetPlus':
            workload['apiVersion'] = 'platform.stke/v1alpha1'
        return RenderedAllocationWorkload(workload=workload)

    def _pod_spec(self, allocation: AllocationSnapshot) -> dict[str, Any]:
        config = self.config
        spec = allocation.allocation_spec
        code = spec.get('code') or {'mode': 'config_only'}
        if allocation.workload == 'orchestration':
            db_pool_size = config.environment_orchestration_db_pool_size
            db_max_overflow = config.environment_orchestration_db_max_overflow
        else:
            db_pool_size = config.environment_db_pool_size
            db_max_overflow = config.environment_db_max_overflow
        env: list[dict[str, Any]] = [
            {'name': 'POD_NAME', 'valueFrom': {'fieldRef': {'fieldPath': 'metadata.name'}}},
            {'name': 'POD_NAMESPACE', 'valueFrom': {'fieldRef': {'fieldPath': 'metadata.namespace'}}},
            {'name': 'POD_UID', 'valueFrom': {'fieldRef': {'fieldPath': 'metadata.uid'}}},
            {'name': 'POD_IP', 'valueFrom': {'fieldRef': {'fieldPath': 'status.podIP'}}},
            {'name': 'NEXUS_DIST_APPLICATION_INTERNAL_BASE_URL', 'value': config.application_internal_base_url},
            {'name': 'NEXUS_DIST_RUN_TRACE_ARTIFACT_ROOT', 'value': f'{config.environment_cache_root}/run-trace'},
            {'name': 'NEXUS_DIST_BATCH_CONTROL_SPOOL_DIR', 'value': f'{config.environment_cache_root}/batch-control'},
            {'name': 'TMPDIR', 'value': '/tmp'},
            {'name': 'TMP', 'value': '/tmp'},
            {'name': 'TEMP', 'value': '/tmp'},
            {
                'name': 'NEXUS_DIST_ALLOCATION_SPEC_JSON',
                'value': json.dumps(spec, sort_keys=True, separators=(',', ':')),
            },
            {'name': 'NEXUS_DIST_MANAGER_OWNER_PRINCIPAL_ID', 'value': allocation.owner_principal_id},
            {'name': 'NEXUS_DIST_MANAGER_ENVIRONMENT_ID', 'value': allocation.environment_id},
            {'name': 'NEXUS_DIST_MANAGER_POOL_ID', 'value': allocation.pool_id},
            {'name': 'NEXUS_DIST_MANAGER_ALLOCATION_ID', 'value': allocation.allocation_id},
            {'name': 'NEXUS_DIST_MANAGER_WORKLOAD', 'value': allocation.workload},
            {'name': 'NEXUS_DIST_MANAGER_ALLOCATION_SPEC_HASH', 'value': allocation.allocation_spec_hash},
            {'name': 'NEXUS_DIST_ENVIRONMENT_PROVIDER_TYPE', 'value': allocation.provider_type},
            {'name': 'NEXUS_DIST_SHARD_COUNT', 'value': str(config.shard_count)},
            {'name': 'NEXUS_DIST_MANAGER_HEARTBEAT_INTERVAL', 'value': str(config.manager_heartbeat_interval)},
            {
                'name': 'NEXUS_DIST_MANAGER_HEALTHY_TTL_SECONDS',
                'value': str(config.manager_healthy_ttl.total_seconds()),
            },
            {
                'name': 'NEXUS_DIST_MANAGER_SELF_FENCE_TIMEOUT_SECONDS',
                'value': str(config.manager_self_fence_timeout.total_seconds()),
            },
            {
                'name': 'NEXUS_DIST_MANAGER_DEAD_TTL_SECONDS',
                'value': str(config.manager_dead_ttl.total_seconds()),
            },
            {'name': 'NEXUS_DIST_DB_POOL_SIZE', 'value': str(db_pool_size)},
            {'name': 'NEXUS_DIST_DB_MAX_OVERFLOW', 'value': str(db_max_overflow)},
            {
                'name': 'NEXUS_DIST_ARTIFACT_RUN_RESULT_MAX_SIZE_BYTES',
                'value': str(config.artifact_run_result_max_size_bytes),
            },
            {
                'name': 'NEXUS_DIST_ARTIFACT_RUN_RESULT_PUBLISH_DEADLINE',
                'value': str(config.artifact_run_result_publish_deadline.total_seconds()),
            },
            {
                'name': 'NEXUS_DIST_BATCH_RECONCILE_POLL_INTERVAL',
                'value': str(config.batch_reconcile_poll_interval),
            },
            {
                'name': 'NEXUS_DIST_BATCH_SUBMIT_BATCH_SIZE',
                'value': str(config.batch_submit_batch_size),
            },
            {
                'name': 'NEXUS_DIST_CONFIG_STORE_REQUIRED',
                'value': '1' if config.config_store_required else '0',
            },
            {'name': 'PYTHONUNBUFFERED', 'value': '1'},
        ]
        if allocation.created_at is not None and allocation.provisioning_timeout_seconds is not None:
            env.append({
                'name': 'NEXUS_DIST_ARTIFACT_DOWNLOAD_DEADLINE_AT',
                'value': (
                    allocation.created_at + timedelta(seconds=allocation.provisioning_timeout_seconds)
                ).isoformat(),
            })
        if allocation.provider_config_ref:
            env.append({'name': 'NEXUS_DIST_PROVIDER_CONFIG_REF', 'value': allocation.provider_config_ref})
        if self.cluster_name:
            env.append({'name': 'NEXUS_DIST_CLUSTER_NAME', 'value': self.cluster_name})
        if not config.environment_database_url_secret_name:
            raise ValueError('dynamic_environment_database_secret_required')
        env.append({'name': 'NEXUS_DIST_DATABASE_URL_FILE', 'value': '/var/run/secrets/nexus/database/url'})
        if config.environment_fake_runtime_base_url:
            env.append({'name': 'NEXUS_DIST_FAKE_RUNTIME_BASE_URL', 'value': config.environment_fake_runtime_base_url})

        reserved_names = {item['name'] for item in env} | {
            'NEXUS_DIST_INTERNAL_TOKEN',
            'NEXUS_DIST_INTERNAL_TOKEN_FILE',
            'NEXUS_DIST_DATABASE_URL',
            'NEXUS_DIST_DATABASE_URL_FILE',
            'NEXUS_DIST_ARTIFACT_READ_TOKEN_FILE',
            'WEDATA_KEY',
            'WEDATA_KEY_FILE',
        }
        for item in config.environment_extra_env:
            name = str(item.get('name') or '')
            if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', name):
                raise ValueError('invalid_dynamic_environment_variable_name')
            if name in reserved_names or set(item) - {'name', 'value', 'valueFrom'}:
                raise ValueError(f'invalid_dynamic_environment_variable:{name}')
            if ('value' in item) == ('valueFrom' in item):
                raise ValueError(f'dynamic_environment_variable_requires_one_source:{name}')
            env.append(dict(item))
            reserved_names.add(name)

        host_env = spec.get('host_env') or []
        if not isinstance(host_env, list):
            raise ValueError('invalid_host_env')
        for item in host_env:
            if not isinstance(item, dict) or set(item) != {'name', 'value'}:
                raise ValueError('invalid_host_env')
            name = item.get('name')
            value = item.get('value')
            if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', name):
                raise ValueError('invalid_host_env_name')
            if name.startswith(('NEXUS_', 'POD_')) or name in {'TMPDIR', 'TMP', 'TEMP'}:
                raise ValueError('reserved_host_env_name')
            if not isinstance(value, str) or '\x00' in value:
                raise ValueError('invalid_host_env_value')
            if name in reserved_names:
                raise ValueError(f'host_env_conflicts_with_platform_variable:{name}')
            env.append({'name': name, 'value': value})
            reserved_names.add(name)

        empty_dir: dict[str, Any] = {}
        workspace_size_bytes = spec.get('workspace_size_bytes')
        if workspace_size_bytes is not None:
            if (
                isinstance(workspace_size_bytes, bool)
                or not isinstance(workspace_size_bytes, int)
                or workspace_size_bytes < 1
            ):
                raise ValueError('invalid_workspace_size_bytes')
            empty_dir['sizeLimit'] = str(workspace_size_bytes)
        secret_sources: list[dict[str, Any]] = [
            {
                'secret': {
                    'name': config.environment_database_url_secret_name,
                    'items': [{'key': config.environment_database_url_secret_key, 'path': 'database/url'}],
                }
            }
        ]
        volumes: list[dict[str, Any]] = [
            {'name': 'data', 'emptyDir': empty_dir},
            {'name': 'tmp', 'emptyDir': {'sizeLimit': str(config.environment_tmp_size_bytes)}},
        ]
        mounts = [
            {'name': 'data', 'mountPath': '/data'},
            {'name': 'tmp', 'mountPath': '/tmp'},
            {'name': 'nexus-secrets', 'mountPath': '/var/run/secrets/nexus', 'readOnly': True},
        ]
        credential = spec.get('artifact_read_credential') or (
            {
                'secret_name': allocation.artifact_credential_secret_name,
                'token_key': 'token',
            }
            if allocation.artifact_credential_secret_name
            else {}
        )
        if credential:
            secret_name = str(credential.get('secret_name') or '')
            token_key = str(credential.get('token_key') or 'token')
            if not secret_name:
                raise ValueError('artifact_read_credential_secret_required')
            secret_sources.append({
                'secret': {
                    'name': secret_name,
                    'items': [{'key': token_key, 'path': 'artifact/reader/token'}],
                }
            })
            env.append({
                'name': 'NEXUS_DIST_ARTIFACT_READ_TOKEN_FILE',
                'value': '/var/run/secrets/nexus/artifact/reader/token',
            })
        volumes.append({'name': 'nexus-secrets', 'projected': {'sources': secret_sources}})

        manager_count = allocation.desired_manager_count
        cpu_millicores = _per_manager_resource(
            allocation.cpu_millicores,
            manager_count=manager_count,
            field='cpu_millicores',
        )
        memory_bytes = _per_manager_resource(
            allocation.memory_bytes,
            manager_count=manager_count,
            field='memory_bytes',
        )
        slots = _per_manager_resource(
            allocation.slots,
            manager_count=manager_count,
            field='slots',
        )
        resources = {
            'limits': {'cpu': f'{cpu_millicores}m', 'memory': str(memory_bytes)},
            'requests': {
                'cpu': f'{max(1, math.ceil(cpu_millicores * self.resource_request_ratio))}m',
                'memory': str(max(1, math.ceil(memory_bytes * self.resource_request_ratio))),
            },
        }
        container: dict[str, Any] = {
            'name': 'instance-manager',
            'image': str(spec.get('base_image') or config.environment_base_image),
            'imagePullPolicy': 'IfNotPresent',
            'command': [sys_executable_for_workload()],
            'args': [
                '-m',
                'nexus.distributed.instance_manager.bootstrap',
                '--allocation-id',
                allocation.allocation_id,
                '--environment-id',
                allocation.environment_id,
                '--pool-id',
                allocation.pool_id,
                '--workload',
                allocation.workload,
                '--allocation-spec-hash',
                allocation.allocation_spec_hash,
                '--port',
                str(config.environment_manager_port),
                '--capacity',
                str(slots),
                '--cache-root',
                config.environment_cache_root,
            ],
            'env': env,
            'ports': [{'name': 'http', 'containerPort': config.environment_manager_port}],
            'volumeMounts': mounts,
            'resources': resources,
            'securityContext': {
                'allowPrivilegeEscalation': False,
                'capabilities': {'drop': ['ALL']},
                'readOnlyRootFilesystem': True,
            },
            'readinessProbe': {
                'tcpSocket': {'port': 'http'},
                'periodSeconds': 2,
                'failureThreshold': 30,
            },
        }
        from nexus.distributed.environment.liveness import LIVENESS_STARTUP_FLAG, allocation_liveness

        if allocation_liveness(spec) is not None:
            container['args'].append(LIVENESS_STARTUP_FLAG)
        if code.get('mode') == 'source_package' and not credential:
            raise ValueError('source_package_requires_scoped_artifact_read_credential')
        pod_spec = {
            'automountServiceAccountToken': False,
            'enableServiceLinks': False,
            'securityContext': {
                'runAsNonRoot': True,
                'runAsUser': 1000,
                'runAsGroup': 1000,
                'fsGroup': 1000,
                'seccompProfile': {'type': 'RuntimeDefault'},
            },
            'containers': [container],
            'volumes': volumes,
        }
        if config.environment_image_pull_secrets:
            pod_spec['imagePullSecrets'] = [
                {'name': secret_name} for secret_name in config.environment_image_pull_secrets
            ]
        return pod_spec


class TkexAllocationWorkloadRenderer(AllocationWorkloadRenderer):
    def __init__(
        self,
        *,
        config: EnvironmentConfigView,
        cluster_name: str,
        workload_namespace: str,
        zhiyan_log: TkexZhiyanLogConfig | None,
    ) -> None:
        super().__init__(
            config=config,
            cluster_name=cluster_name,
            workload_namespace=workload_namespace,
        )
        self.zhiyan_log = zhiyan_log

    def render(self, allocation: AllocationSnapshot, *, resource_kind: str) -> RenderedAllocationWorkload:
        if resource_kind != 'StatefulSetPlus':
            raise ValueError('tkex_renderer_requires_statefulsetplus')
        if allocation.allocation_type != 'dynamic' or allocation.provider_type != 'tkex':
            raise ValueError('tkex_renderer_requires_dynamic_tkex_allocation')
        rendered = super().render(allocation, resource_kind=resource_kind)
        if self.zhiyan_log is not None:
            self._add_zhiyan_log_sidecar(rendered.workload)
        return rendered

    def _add_zhiyan_log_sidecar(self, workload: dict[str, Any]) -> None:
        assert self.zhiyan_log is not None
        pod_spec = workload['spec']['template']['spec']
        manager = next(container for container in pod_spec['containers'] if container['name'] == 'instance-manager')
        manager_log_dir = f'{self.config.environment_cache_root}/logs'
        add_zhiyan_log_sidecar(
            pod_spec,
            manager,
            config=self.zhiyan_log,
            log_dir=manager_log_dir,
            log_dir_env='NEXUS_DIST_LOG_DIR',
        )
        pod_spec.setdefault('initContainers', []).append({
            'name': 'prepare-manager-cache-root',
            'image': manager['image'],
            'imagePullPolicy': 'IfNotPresent',
            'command': [sys_executable_for_workload()],
            'args': [
                '-c',
                'from pathlib import Path; import sys; Path(sys.argv[1]).mkdir(parents=True, exist_ok=True)',
                self.config.environment_cache_root,
            ],
            'volumeMounts': [{'name': 'data', 'mountPath': '/data'}],
            'securityContext': {
                'allowPrivilegeEscalation': False,
                'capabilities': {'drop': ['ALL']},
                'readOnlyRootFilesystem': True,
            },
        })
        manager['volumeMounts'].extend([
            {'name': 'cgroup', 'mountPath': '/sys/fs/cgroup', 'readOnly': True},
            {'name': 'shm', 'mountPath': '/dev/shm'},
        ])


def _per_manager_resource(total: int, *, manager_count: int, field: str) -> int:
    if manager_count < 1:
        raise ValueError('desired_manager_count_must_be_positive')
    per_manager, remainder = divmod(total, manager_count)
    if remainder:
        raise ValueError(f'{field}_not_divisible_by_manager_count')
    if per_manager < 1:
        raise ValueError(f'{field}_per_manager_must_be_positive')
    return per_manager


def sys_executable_for_workload() -> str:
    return '/opt/nexus-venv/bin/python'


__all__ = [
    'AllocationWorkloadRenderer',
    'RenderedAllocationWorkload',
    'TkexAllocationWorkloadRenderer',
    'sys_executable_for_workload',
]
