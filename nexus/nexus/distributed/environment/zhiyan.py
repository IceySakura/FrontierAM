from __future__ import annotations

from typing import Any

from nexus.distributed.environment.provider_config import TkexZhiyanLogConfig


def add_zhiyan_log_sidecar(
    pod_spec: dict[str, Any],
    container: dict[str, Any],
    *,
    config: TkexZhiyanLogConfig,
    log_dir: str,
    log_dir_env: str,
) -> None:
    expected = {'name': log_dir_env, 'value': log_dir}
    existing = next((item for item in container['env'] if item.get('name') == log_dir_env), None)
    if existing is None:
        container['env'].append(expected)
    elif existing != expected:
        raise ValueError('tkex_zhiyan_log_dir_conflicts_with_environment_extra_env')
    container['volumeMounts'].append({'name': 'log', 'mountPath': log_dir})
    pod_spec['volumes'].extend([
        {'name': 'cgroup', 'hostPath': {'path': '/sys/fs/cgroup', 'type': ''}},
        {'name': 'shm', 'emptyDir': {'medium': 'Memory'}},
        {'name': 'log', 'emptyDir': {'sizeLimit': str(config.volume_size_bytes)}},
    ])
    pod_spec['containers'].append(_zhiyan_agent_container(config))


def _zhiyan_agent_container(config: TkexZhiyanLogConfig) -> dict[str, Any]:
    return {
        'name': 'zhiyan-agent',
        'image': config.agent_image,
        'imagePullPolicy': 'Always',
        'stdin': True,
        'env': [
            {'name': 'ZHIYAN_LOG_NAME', 'value': config.log_name},
            {
                'name': 'POD_NAME',
                'valueFrom': {'fieldRef': {'apiVersion': 'v1', 'fieldPath': 'metadata.name'}},
            },
        ],
        'resources': {
            'limits': {'cpu': '0', 'memory': '0'},
            'requests': {'cpu': '0', 'memory': '0'},
        },
        'securityContext': {'runAsNonRoot': False, 'runAsUser': 0},
        'volumeMounts': [
            {'name': 'cgroup', 'mountPath': '/sys/fs/cgroup', 'readOnly': True},
            {'name': 'shm', 'mountPath': '/dev/shm'},
            {'name': 'log', 'mountPath': '/data/logs'},
        ],
    }
