from __future__ import annotations

import base64

from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from nexus.distributed.config.views import DependencyBuildConfigView
from nexus.distributed.dependency_build.logs import BUILD_LOG_PORT, build_log_token
from nexus.distributed.environment.naming import dns_name
from nexus.distributed.environment.provider_config import TkexZhiyanLogConfig
from nexus.distributed.environment.zhiyan import add_zhiyan_log_sidecar
from nexus.distributed.repositories.dependency_build import DependencyBuildRecord
from nexus.distributed.utils.identity import canonical_sha256


@dataclass(frozen=True, slots=True)
class RenderedDependencyBuild:
    secret: dict[str, Any]
    job: dict[str, Any]
    secret_name: str
    job_name: str


class DependencyBuildJobRenderer:
    def __init__(self, *, config: DependencyBuildConfigView) -> None:
        self.config = config

    def render(self, record: DependencyBuildRecord, *, token: str) -> RenderedDependencyBuild:
        if record.attempt < 1:
            raise ValueError('dependency build attempt must be started before rendering')
        if record.started_at is None:
            raise ValueError('dependency build started_at is required before rendering')
        download_deadline_at = record.started_at + timedelta(
            seconds=self.config.dependency_builder_active_deadline_seconds
        )
        secret_name = self.secret_name(record.dependency_build_id, record.attempt)
        job_name = self.job_name(record.dependency_build_id, record.attempt)
        labels = {
            'app.kubernetes.io/name': self.config.environment_app_name,
            'app.kubernetes.io/instance': self.config.environment_release_name,
            'app.kubernetes.io/component': 'dependency-builder',
            'app.kubernetes.io/managed-by': 'nexus',
        }
        annotations = {
            'nexus.io/managed-by': 'nexus',
            'nexus.io/dependency-build-id': record.dependency_build_id,
            'nexus.io/dependency-build-key': record.build_key,
            'nexus.io/dependency-build-attempt': str(record.attempt),
        }
        key_id = self.config.artifact_upload_ticket_current_key_id
        read_token = build_log_token(
            self.config.artifact_ticket_keyring()[key_id], record.dependency_build_id, record.attempt
        )
        annotations['nexus.io/build-log-key-id'] = key_id
        token_hash = canonical_sha256(token + read_token)
        secret = {
            'apiVersion': 'v1',
            'kind': 'Secret',
            'metadata': {
                'name': secret_name,
                'namespace': self.config.environment_namespace,
                'labels': labels,
                'annotations': {**annotations, 'nexus.io/credential-sha256': token_hash},
            },
            'type': 'Opaque',
            'data': {
                'token': base64.b64encode(token.encode()).decode(),
                'log-token': base64.b64encode(read_token.encode()).decode(),
            },
        }
        pod_spec: dict[str, Any] = {
            'automountServiceAccountToken': False,
            'enableServiceLinks': False,
            'restartPolicy': 'Never',
            'securityContext': {
                'runAsNonRoot': True,
                'runAsUser': 1000,
                'runAsGroup': 1000,
                'fsGroup': 1000,
                'seccompProfile': {'type': 'RuntimeDefault'},
            },
            'containers': [
                {
                    'name': 'builder',
                    'ports': [{'name': 'build-logs', 'containerPort': BUILD_LOG_PORT}],
                    'image': record.build_spec.builder_image,
                    'imagePullPolicy': 'IfNotPresent',
                    'command': ['/opt/nexus-venv/bin/python'],
                    'args': ['-m', 'nexus.distributed.dependency_build.worker'],
                    'env': [
                        {
                            'name': 'NEXUS_DIST_DEPENDENCY_BUILD_SPEC_JSON',
                            'value': record.build_spec.model_dump_json(),
                        },
                        {'name': 'NEXUS_DIST_DEPENDENCY_BUILD_ID', 'value': record.dependency_build_id},
                        {'name': 'NEXUS_DIST_DEPENDENCY_BUILD_KEY', 'value': record.build_key},
                        {'name': 'NEXUS_DIST_DEPENDENCY_BUILD_ATTEMPT', 'value': str(record.attempt)},
                        {
                            'name': 'NEXUS_DIST_APPLICATION_INTERNAL_BASE_URL',
                            'value': self.config.application_internal_base_url,
                        },
                        {
                            'name': 'NEXUS_DIST_ARTIFACT_CONTENT_TOKEN_FILE',
                            'value': '/var/run/secrets/nexus/build/artifact-content-token',
                        },
                        {
                            'name': 'NEXUS_DIST_ARTIFACT_DOWNLOAD_DEADLINE_AT',
                            'value': download_deadline_at.isoformat(),
                        },
                        {
                            'name': 'NEXUS_DIST_DEPENDENCY_BUILD_LOG_TOKEN_FILE',
                            'value': '/var/run/secrets/nexus/build/log-token',
                        },
                        {'name': 'NEXUS_DIST_DEPENDENCY_BUILD_WORKSPACE', 'value': '/work/build'},
                        {'name': 'TMPDIR', 'value': '/work'},
                        {'name': 'UV_CACHE_DIR', 'value': '/work/uv-cache'},
                        {'name': 'PIP_INDEX_URL', 'value': self.config.dependency_builder_index_url},
                        {'name': 'UV_INDEX_URL', 'value': self.config.dependency_builder_index_url},
                        {'name': 'PYTHONUNBUFFERED', 'value': '1'},
                    ],
                    'volumeMounts': [
                        {'name': 'workspace', 'mountPath': '/work'},
                        {
                            'name': 'nexus-secrets',
                            'mountPath': '/var/run/secrets/nexus',
                            'readOnly': True,
                        },
                    ],
                    'resources': {
                        'requests': {
                            'cpu': f'{self.config.dependency_builder_cpu_millicores}m',
                            'memory': str(self.config.dependency_builder_memory_bytes),
                        },
                        'limits': {
                            'cpu': f'{self.config.dependency_builder_cpu_millicores}m',
                            'memory': str(self.config.dependency_builder_memory_bytes),
                        },
                    },
                    'securityContext': {
                        'allowPrivilegeEscalation': False,
                        'capabilities': {'drop': ['ALL']},
                        'readOnlyRootFilesystem': True,
                    },
                }
            ],
            'volumes': [
                {
                    'name': 'workspace',
                    'emptyDir': {'sizeLimit': str(self.config.dependency_builder_workspace_bytes)},
                },
                {
                    'name': 'nexus-secrets',
                    'projected': {
                        'sources': [
                            {
                                'secret': {
                                    'name': secret_name,
                                    'items': [
                                        {'key': 'token', 'path': 'build/artifact-content-token'},
                                        {'key': 'log-token', 'path': 'build/log-token'},
                                    ],
                                }
                            }
                        ]
                    },
                },
            ],
        }
        snapshot = record.provider_config_snapshot or {}
        if record.provider_type == 'tkex' and snapshot.get('builder_zhiyan_log'):
            add_zhiyan_log_sidecar(
                pod_spec,
                pod_spec['containers'][0],
                config=TkexZhiyanLogConfig.model_validate(snapshot['builder_zhiyan_log']),
                log_dir='/work/collected-logs',
                log_dir_env='NEXUS_DIST_DEPENDENCY_BUILD_LOG_DIR',
            )
        if self.config.environment_image_pull_secrets:
            pod_spec['imagePullSecrets'] = [
                {'name': secret_name} for secret_name in self.config.environment_image_pull_secrets
            ]
        job = {
            'apiVersion': 'batch/v1',
            'kind': 'Job',
            'metadata': {
                'name': job_name,
                'namespace': self.config.environment_namespace,
                'labels': labels,
                'annotations': annotations,
            },
            'spec': {
                'backoffLimit': 0,
                'activeDeadlineSeconds': self.config.dependency_builder_active_deadline_seconds,
                'ttlSecondsAfterFinished': self.config.dependency_builder_job_ttl_seconds,
                'template': {
                    'metadata': {'labels': labels, 'annotations': annotations},
                    'spec': pod_spec,
                },
            },
        }
        return RenderedDependencyBuild(secret=secret, job=job, secret_name=secret_name, job_name=job_name)

    @staticmethod
    def secret_name(dependency_build_id: str, attempt: int) -> str:
        return dns_name(
            f'nexus-build-{dependency_build_id}-{attempt}-artifacts',
            fallback='nexus-build',
        )

    @staticmethod
    def job_name(dependency_build_id: str, attempt: int) -> str:
        return dns_name(f'nexus-build-{dependency_build_id}-{attempt}', fallback='nexus-build')


__all__ = ['DependencyBuildJobRenderer', 'RenderedDependencyBuild']
