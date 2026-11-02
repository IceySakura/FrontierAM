#!/usr/bin/env python3
from __future__ import annotations

import base64
import copy
import hashlib
import os
import shutil
import signal
import subprocess
import sys

from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import yaml


try:
    from .toolathlon_k8s_sql import apply_mysql_script, apply_schema_sql
    from .toolathlon_mock_utils import mutate_store, read_store, utc_now_iso
except ImportError:
    from toolathlon_k8s_sql import apply_mysql_script, apply_schema_sql
    from toolathlon_mock_utils import mutate_store, read_store, utc_now_iso


_T = TypeVar('_T')

STORE_PATH_ENV = 'NEXUS_TOOLATHLON_K8S_STATE_PATH'
ROOT_PATH_ENV = 'NEXUS_TOOLATHLON_K8S_ROOT'
_FORWARDER_NAME = 'toolathlon_k8s_forwarder.py'

_KIND_ALIASES = {
    'namespace': 'namespace',
    'namespaces': 'namespace',
    'ns': 'namespace',
    'configmap': 'configmap',
    'configmaps': 'configmap',
    'cm': 'configmap',
    'secret': 'secret',
    'secrets': 'secret',
    'deployment': 'deployment',
    'deployments': 'deployment',
    'deploy': 'deployment',
    'statefulset': 'statefulset',
    'statefulsets': 'statefulset',
    'sts': 'statefulset',
    'pod': 'pod',
    'pods': 'pod',
    'service': 'service',
    'services': 'service',
    'svc': 'service',
    'serviceaccount': 'serviceaccount',
    'serviceaccounts': 'serviceaccount',
    'sa': 'serviceaccount',
    'endpoints': 'endpoints',
    'ep': 'endpoints',
    'node': 'node',
    'nodes': 'node',
}

_NAMESPACED_KINDS = {
    'configmap',
    'secret',
    'deployment',
    'statefulset',
    'pod',
    'service',
    'serviceaccount',
    'endpoints',
}
_RESOURCE_STORE_KEYS = {
    'namespace': 'namespaces',
    'configmap': 'configmaps',
    'secret': 'secrets',
    'deployment': 'deployments',
    'statefulset': 'statefulsets',
    'pod': 'pods',
    'service': 'services',
    'serviceaccount': 'serviceaccounts',
}


class K8sStateError(RuntimeError):
    pass


def case_root_from_cwd() -> Path:
    cwd = Path.cwd().resolve()
    if cwd.name == 'agent_workspace':
        return cwd.parent
    return cwd


def k8s_root_path() -> Path:
    configured = os.getenv(ROOT_PATH_ENV)
    if configured:
        return Path(configured)
    return case_root_from_cwd() / '.nexus_k8s'


def k8s_state_path() -> str:
    configured = os.getenv(STORE_PATH_ENV)
    if configured:
        return configured
    return str(case_root_from_cwd() / '.nexus_k8s_state.json')


def _resource_key(namespace: str | None, name: str) -> str:
    return name if namespace in {None, ''} else f'{namespace}/{name}'


def _canonical_kind(kind: str) -> str:
    normalized = (kind or '').strip().lower()
    if normalized in _KIND_ALIASES:
        return _KIND_ALIASES[normalized]
    raise K8sStateError(f'Unsupported resource kind: {kind}')


def _store_key(kind: str) -> str:
    canonical = _canonical_kind(kind)
    if canonical not in _RESOURCE_STORE_KEYS:
        raise K8sStateError(f'Unsupported stored resource kind: {kind}')
    return _RESOURCE_STORE_KEYS[canonical]


def _generated_annotation(owner: str) -> dict[str, str]:
    return {'nexus.generatedBy': owner}


def _pod_hash(namespace: str, name: str) -> str:
    return hashlib.sha1(f'{namespace}/{name}'.encode()).hexdigest()[:6]


def _pod_ip(name: str, namespace: str, index: int = 0) -> str:
    digest = hashlib.sha1(f'{namespace}/{name}:{index}'.encode()).digest()
    return f'10.{digest[0] % 250}.{digest[1] % 250}.{max(2, digest[2] % 250)}'


def _cluster_ip(name: str, namespace: str) -> str:
    digest = hashlib.md5(f'{namespace}/{name}'.encode()).digest()
    return f'172.19.{digest[0] % 250}.{max(2, digest[1] % 250)}'


def _process_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _kill_pid(pid: int | None) -> None:
    if not _process_alive(pid):
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return


def _node_object(name: str = 'mock-control-plane') -> dict[str, Any]:
    now = utc_now_iso()
    return {
        'apiVersion': 'v1',
        'kind': 'Node',
        'metadata': {
            'name': name,
            'creationTimestamp': now,
            'labels': {
                'kubernetes.io/hostname': name,
                'kubernetes.io/os': 'linux',
                'node-role.kubernetes.io/control-plane': 'true',
            },
            'annotations': {},
        },
        'spec': {},
        'status': {
            'conditions': [{'type': 'Ready', 'status': 'True'}],
            'addresses': [{'type': 'InternalIP', 'address': '172.18.0.2'}],
            'nodeInfo': {'kubeletVersion': 'v1.30.0-mock'},
        },
    }


def default_k8s_state() -> dict[str, Any]:
    return {
        'version': 1,
        'current_cluster': None,
        'helm_repos': {},
        'clusters': {},
    }


def read_k8s_state() -> dict[str, Any]:
    return read_store(k8s_state_path(), default_k8s_state)


def mutate_k8s_state(mutator: Callable[[dict[str, Any]], _T]) -> _T:
    return mutate_store(k8s_state_path(), default_k8s_state, mutator)


def _cluster_runtime_root(cluster_name: str) -> Path:
    return k8s_root_path() / 'clusters' / cluster_name


def _pod_fs_root(cluster_name: str, namespace: str, pod_name: str) -> Path:
    return _cluster_runtime_root(cluster_name) / 'pods' / namespace / pod_name


def _mysql_db_path(cluster_name: str) -> Path:
    return _cluster_runtime_root(cluster_name) / 'mysql' / 'f1.sqlite3'


def _ensure_cluster_defaults(cluster: dict[str, Any], cluster_name: str) -> dict[str, Any]:
    cluster.setdefault('name', cluster_name)
    cluster.setdefault('created_at', utc_now_iso())
    cluster.setdefault('kubeconfigs', [])
    cluster.setdefault('extra_port_mappings', [])
    cluster.setdefault('resources', {})
    for store_name in _RESOURCE_STORE_KEYS.values():
        cluster['resources'].setdefault(store_name, {})
    cluster.setdefault('nodes', {'mock-control-plane': _node_object()})
    cluster.setdefault('port_forwards', {})
    cluster.setdefault('http_bindings', {})
    cluster.setdefault('helm_releases', {})
    cluster.setdefault(
        'mysql',
        {
            'db_path': str(_mysql_db_path(cluster_name)),
            'users': {},
        },
    )
    namespaces = cluster['resources']['namespaces']
    for namespace_name in ('default', 'kube-system'):
        namespaces.setdefault(namespace_name, _namespace_object(namespace_name))
    return cluster


def _namespace_object(name: str) -> dict[str, Any]:
    return {
        'apiVersion': 'v1',
        'kind': 'Namespace',
        'metadata': {
            'name': name,
            'creationTimestamp': utc_now_iso(),
            'labels': {},
            'annotations': {},
        },
        'status': {'phase': 'Active'},
    }


def _metadata_for_resource(
    existing: dict[str, Any] | None, payload: dict[str, Any], *, namespace: str | None = None
) -> dict[str, Any]:
    existing_meta = copy.deepcopy((existing or {}).get('metadata', {}))
    payload_meta = copy.deepcopy(payload.get('metadata', {}))
    payload_meta.setdefault('labels', {})
    payload_meta.setdefault('annotations', {})
    if namespace is not None:
        payload_meta['namespace'] = namespace
    meta = existing_meta
    meta.update(payload_meta)
    meta.setdefault('creationTimestamp', (existing_meta or {}).get('creationTimestamp') or utc_now_iso())
    meta.setdefault('labels', {})
    meta.setdefault('annotations', {})
    return meta


def _resource_store(cluster: dict[str, Any], kind: str) -> dict[str, Any]:
    return cluster['resources'][_store_key(kind)]


def _ensure_namespace(cluster: dict[str, Any], namespace: str) -> None:
    namespaces = cluster['resources']['namespaces']
    if namespace and namespace not in namespaces:
        namespaces[namespace] = _namespace_object(namespace)


def _service_key(namespace: str, name: str) -> str:
    return _resource_key(namespace, name)


def _ensure_service_defaults(
    existing: dict[str, Any] | None, payload: dict[str, Any], *, namespace: str
) -> dict[str, Any]:
    spec = copy.deepcopy(payload.get('spec', {}))
    ports = copy.deepcopy(spec.get('ports', []))
    for index, port in enumerate(ports):
        port.setdefault('protocol', 'TCP')
        port.setdefault('targetPort', port.get('port'))
        port.setdefault('name', f'port-{index}')
        if spec.get('type') == 'NodePort' and 'nodePort' not in port:
            port['nodePort'] = int(port.get('port') or 30000)
    spec['ports'] = ports
    if str(spec.get('clusterIP')) == 'None':
        spec['clusterIP'] = 'None'
    elif existing is not None and existing.get('spec', {}).get('clusterIP') and 'clusterIP' not in spec:
        spec['clusterIP'] = existing['spec']['clusterIP']
    elif 'clusterIP' not in spec:
        spec['clusterIP'] = _cluster_ip(payload['metadata']['name'], namespace)
    return {
        'apiVersion': payload.get('apiVersion', 'v1'),
        'kind': 'Service',
        'metadata': _metadata_for_resource(existing, payload, namespace=namespace),
        'spec': spec,
        'status': copy.deepcopy((existing or {}).get('status', {})),
    }


def _secret_payload(existing: dict[str, Any] | None, payload: dict[str, Any], *, namespace: str) -> dict[str, Any]:
    data = copy.deepcopy((existing or {}).get('data', {}))
    data.update(copy.deepcopy(payload.get('data', {})))
    for key, value in (payload.get('stringData') or {}).items():
        data[key] = base64.b64encode(str(value).encode('utf-8')).decode('ascii')
    return {
        'apiVersion': payload.get('apiVersion', 'v1'),
        'kind': 'Secret',
        'metadata': _metadata_for_resource(existing, payload, namespace=namespace),
        'type': payload.get('type', (existing or {}).get('type') or 'Opaque'),
        'data': data,
    }


def _configmap_payload(existing: dict[str, Any] | None, payload: dict[str, Any], *, namespace: str) -> dict[str, Any]:
    data = copy.deepcopy((existing or {}).get('data', {}))
    data.update(copy.deepcopy(payload.get('data', {})))
    return {
        'apiVersion': payload.get('apiVersion', 'v1'),
        'kind': 'ConfigMap',
        'metadata': _metadata_for_resource(existing, payload, namespace=namespace),
        'data': data,
    }


def _serviceaccount_payload(
    existing: dict[str, Any] | None, payload: dict[str, Any], *, namespace: str
) -> dict[str, Any]:
    return {
        'apiVersion': payload.get('apiVersion', 'v1'),
        'kind': 'ServiceAccount',
        'metadata': _metadata_for_resource(existing, payload, namespace=namespace),
        'secrets': copy.deepcopy(payload.get('secrets', [])),
    }


def _pod_status_for_spec(spec: dict[str, Any], pod_name: str, namespace: str) -> dict[str, Any]:
    containers = spec.get('containers', []) or []
    container_statuses = [
        {
            'name': str(container.get('name') or f'container-{index}'),
            'image': str(container.get('image') or ''),
            'ready': True,
            'restartCount': 0,
            'state': {'running': {'startedAt': utc_now_iso()}},
        }
        for index, container in enumerate(containers)
    ]
    return {
        'phase': 'Running',
        'podIP': _pod_ip(pod_name, namespace),
        'containerStatuses': container_statuses,
        'conditions': [
            {'type': 'Initialized', 'status': 'True'},
            {'type': 'Ready', 'status': 'True'},
            {'type': 'ContainersReady', 'status': 'True'},
            {'type': 'PodScheduled', 'status': 'True'},
        ],
    }


def _manual_pod_payload(existing: dict[str, Any] | None, payload: dict[str, Any], *, namespace: str) -> dict[str, Any]:
    spec = copy.deepcopy(payload.get('spec', {}))
    spec.setdefault('nodeName', 'mock-control-plane')
    pod_name = str(payload['metadata']['name'])
    return {
        'apiVersion': payload.get('apiVersion', 'v1'),
        'kind': 'Pod',
        'metadata': _metadata_for_resource(existing, payload, namespace=namespace),
        'spec': spec,
        'status': _pod_status_for_spec(spec, pod_name, namespace),
    }


def _deployment_payload(existing: dict[str, Any] | None, payload: dict[str, Any], *, namespace: str) -> dict[str, Any]:
    spec = copy.deepcopy(payload.get('spec', {}))
    spec.setdefault('replicas', 1)
    return {
        'apiVersion': payload.get('apiVersion', 'apps/v1'),
        'kind': 'Deployment',
        'metadata': _metadata_for_resource(existing, payload, namespace=namespace),
        'spec': spec,
        'status': copy.deepcopy((existing or {}).get('status', {})),
    }


def _statefulset_payload(existing: dict[str, Any] | None, payload: dict[str, Any], *, namespace: str) -> dict[str, Any]:
    spec = copy.deepcopy(payload.get('spec', {}))
    spec.setdefault('replicas', 1)
    return {
        'apiVersion': payload.get('apiVersion', 'apps/v1'),
        'kind': 'StatefulSet',
        'metadata': _metadata_for_resource(existing, payload, namespace=namespace),
        'spec': spec,
        'status': copy.deepcopy((existing or {}).get('status', {})),
    }


def _normalize_resource_document(
    cluster: dict[str, Any], document: dict[str, Any]
) -> tuple[str, str | None, str, dict[str, Any]]:
    if not isinstance(document, dict) or not document:
        raise K8sStateError('Manifest document must be a mapping')
    kind = _canonical_kind(str(document.get('kind') or ''))
    payload = copy.deepcopy(document)
    metadata = payload.setdefault('metadata', {})
    name = str(metadata.get('name') or '').strip()
    if not name:
        raise K8sStateError(f'{kind} manifest missing metadata.name')
    namespace = str(metadata.get('namespace') or 'default').strip() if kind in _NAMESPACED_KINDS else None
    if kind == 'namespace':
        namespace = None
    if namespace:
        _ensure_namespace(cluster, namespace)
    return kind, namespace, name, payload


def _selector_matches(labels: dict[str, Any], selector: dict[str, Any]) -> bool:
    if not selector:
        return False
    return all(str(labels.get(key)) == str(value) for key, value in selector.items())


def _iter_namespaced_resources(
    cluster: dict[str, Any], kind: str, namespace: str | None = None
) -> list[dict[str, Any]]:
    store = _resource_store(cluster, kind)
    values = list(store.values())
    if namespace in {None, ''} or kind == 'namespace':
        return values
    return [item for item in values if str(item.get('metadata', {}).get('namespace') or '') == namespace]


def _sync_generated_workload_pods(cluster: dict[str, Any], cluster_name: str) -> None:
    pods_store = _resource_store(cluster, 'pod')
    desired: dict[str, dict[str, Any]] = {}

    def _controller_pods(owner_kind: str, owner: dict[str, Any]) -> list[dict[str, Any]]:
        namespace = str(owner['metadata'].get('namespace') or 'default')
        name = str(owner['metadata']['name'])
        spec = owner.get('spec', {})
        template = copy.deepcopy(spec.get('template', {}))
        pod_spec = template.setdefault('spec', {})
        pod_spec.setdefault('nodeName', 'mock-control-plane')
        template_meta = template.setdefault('metadata', {})
        labels = copy.deepcopy(template_meta.get('labels', {}))
        template_meta['labels'] = labels
        replicas = int(spec.get('replicas') or 0)
        items: list[dict[str, Any]] = []
        for index in range(replicas):
            if owner_kind == 'statefulset':
                pod_name = f'{name}-{index}'
            else:
                suffix = _pod_hash(namespace, name)
                pod_name = f'{name}-{suffix}' if replicas == 1 else f'{name}-{suffix}-{index}'
            key = _resource_key(namespace, pod_name)
            existing = pods_store.get(key)
            metadata = _metadata_for_resource(
                existing,
                {
                    'metadata': {
                        'name': pod_name,
                        'namespace': namespace,
                        'labels': labels,
                        'annotations': _generated_annotation(f'{owner_kind}/{namespace}/{name}'),
                    }
                },
                namespace=namespace,
            )
            metadata['annotations'].update(_generated_annotation(f'{owner_kind}/{namespace}/{name}'))
            pod_object = {
                'apiVersion': 'v1',
                'kind': 'Pod',
                'metadata': metadata,
                'spec': copy.deepcopy(pod_spec),
                'status': _pod_status_for_spec(pod_spec, pod_name, namespace),
            }
            items.append((key, pod_object))
        return [item for _, item in items]

    for deployment in _iter_namespaced_resources(cluster, 'deployment'):
        namespace = str(deployment['metadata'].get('namespace') or 'default')
        name = str(deployment['metadata']['name'])
        spec = deployment.get('spec', {})
        template = copy.deepcopy(spec.get('template', {}))
        pod_spec = template.setdefault('spec', {})
        pod_spec.setdefault('nodeName', 'mock-control-plane')
        template_meta = template.setdefault('metadata', {})
        labels = copy.deepcopy(template_meta.get('labels', {}))
        replicas = int(spec.get('replicas') or 0)
        for index in range(replicas):
            suffix = _pod_hash(namespace, name)
            pod_name = f'{name}-{suffix}' if replicas == 1 else f'{name}-{suffix}-{index}'
            key = _resource_key(namespace, pod_name)
            existing = pods_store.get(key)
            metadata = _metadata_for_resource(
                existing,
                {
                    'metadata': {
                        'name': pod_name,
                        'namespace': namespace,
                        'labels': labels,
                        'annotations': _generated_annotation(f'deployment/{namespace}/{name}'),
                    }
                },
                namespace=namespace,
            )
            metadata['annotations'].update(_generated_annotation(f'deployment/{namespace}/{name}'))
            desired[key] = {
                'apiVersion': 'v1',
                'kind': 'Pod',
                'metadata': metadata,
                'spec': copy.deepcopy(pod_spec),
                'status': _pod_status_for_spec(pod_spec, pod_name, namespace),
            }
        deployment['status'] = {
            'observedGeneration': 1,
            'replicas': replicas,
            'updatedReplicas': replicas,
            'readyReplicas': replicas,
            'availableReplicas': replicas,
            'conditions': [
                {'type': 'Available', 'status': 'True'},
                {'type': 'Progressing', 'status': 'True'},
            ],
        }

    for statefulset in _iter_namespaced_resources(cluster, 'statefulset'):
        namespace = str(statefulset['metadata'].get('namespace') or 'default')
        name = str(statefulset['metadata']['name'])
        spec = statefulset.get('spec', {})
        template = copy.deepcopy(spec.get('template', {}))
        pod_spec = template.setdefault('spec', {})
        pod_spec.setdefault('nodeName', 'mock-control-plane')
        template_meta = template.setdefault('metadata', {})
        labels = copy.deepcopy(template_meta.get('labels', {}))
        replicas = int(spec.get('replicas') or 0)
        for index in range(replicas):
            pod_name = f'{name}-{index}'
            key = _resource_key(namespace, pod_name)
            existing = pods_store.get(key)
            metadata = _metadata_for_resource(
                existing,
                {
                    'metadata': {
                        'name': pod_name,
                        'namespace': namespace,
                        'labels': labels,
                        'annotations': _generated_annotation(f'statefulset/{namespace}/{name}'),
                    }
                },
                namespace=namespace,
            )
            metadata['annotations'].update(_generated_annotation(f'statefulset/{namespace}/{name}'))
            desired[key] = {
                'apiVersion': 'v1',
                'kind': 'Pod',
                'metadata': metadata,
                'spec': copy.deepcopy(pod_spec),
                'status': _pod_status_for_spec(pod_spec, pod_name, namespace),
            }
        statefulset['status'] = {
            'observedGeneration': 1,
            'replicas': replicas,
            'readyReplicas': replicas,
            'availableReplicas': replicas,
            'currentReplicas': replicas,
            'updatedReplicas': replicas,
        }

    for key, pod in list(pods_store.items()):
        owner = str(pod.get('metadata', {}).get('annotations', {}).get('nexus.generatedBy') or '')
        if owner and key not in desired:
            pods_store.pop(key, None)
    pods_store.update(desired)


def _service_endpoints(cluster: dict[str, Any], service: dict[str, Any]) -> list[dict[str, Any]]:
    selector = copy.deepcopy(service.get('spec', {}).get('selector', {}))
    if not selector:
        return []
    namespace = str(service.get('metadata', {}).get('namespace') or 'default')
    pods = _iter_namespaced_resources(cluster, 'pod', namespace)
    addresses: list[dict[str, Any]] = []
    for pod in pods:
        labels = copy.deepcopy(pod.get('metadata', {}).get('labels', {}))
        if not _selector_matches(labels, selector):
            continue
        addresses.append({
            'ip': str(pod.get('status', {}).get('podIP') or _pod_ip(str(pod['metadata']['name']), namespace)),
            'targetRef': {
                'kind': 'Pod',
                'name': str(pod['metadata']['name']),
                'namespace': namespace,
            },
        })
    return addresses


def _sync_service_status(cluster: dict[str, Any]) -> None:
    services = _iter_namespaced_resources(cluster, 'service')
    for service in services:
        namespace = str(service.get('metadata', {}).get('namespace') or 'default')
        ports = copy.deepcopy(service.get('spec', {}).get('ports', []))
        addresses = _service_endpoints(cluster, service)
        service['status'] = {
            'loadBalancer': {},
            'addresses': addresses,
        }
        service['nexus_endpoints'] = {
            'subsets': [
                {
                    'addresses': addresses,
                    'ports': [
                        {
                            'name': str(port.get('name') or ''),
                            'port': int(port.get('port') or 0),
                            'protocol': str(port.get('protocol') or 'TCP'),
                        }
                        for port in ports
                    ],
                }
            ]
            if addresses
            else []
        }
        if str(service.get('spec', {}).get('clusterIP') or '') == 'None':
            service['spec']['clusterIP'] = 'None'
        elif not service.get('spec', {}).get('clusterIP'):
            service['spec']['clusterIP'] = _cluster_ip(str(service['metadata']['name']), namespace)


def _sync_cluster(cluster: dict[str, Any], cluster_name: str) -> None:
    _ensure_cluster_defaults(cluster, cluster_name)
    _sync_generated_workload_pods(cluster, cluster_name)
    _sync_service_status(cluster)


def _cluster_for_state(
    state: dict[str, Any], *, cluster_name: str | None = None, kubeconfig_path: str | None = None
) -> tuple[str, dict[str, Any]]:
    if cluster_name:
        if cluster_name not in state['clusters']:
            raise K8sStateError(f'Unknown cluster: {cluster_name}')
        cluster = _ensure_cluster_defaults(state['clusters'][cluster_name], cluster_name)
        return cluster_name, cluster
    if kubeconfig_path:
        for name, cluster in state['clusters'].items():
            cluster = _ensure_cluster_defaults(cluster, name)
            if kubeconfig_path in cluster.get('kubeconfigs', []):
                return name, cluster
    current = state.get('current_cluster')
    if current and current in state['clusters']:
        cluster = _ensure_cluster_defaults(state['clusters'][current], current)
        return current, cluster
    if len(state['clusters']) == 1:
        name, cluster = next(iter(state['clusters'].items()))
        cluster = _ensure_cluster_defaults(cluster, name)
        state['current_cluster'] = name
        return name, cluster
    raise K8sStateError('No active cluster is available')


def write_kubeconfig(cluster_name: str, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'apiVersion': 'v1',
        'kind': 'Config',
        'clusters': [{'name': cluster_name, 'cluster': {'server': f'https://mock-{cluster_name}.local'}}],
        'contexts': [{'name': cluster_name, 'context': {'cluster': cluster_name, 'user': cluster_name}}],
        'current-context': cluster_name,
        'users': [{'name': cluster_name, 'user': {'token': 'mock-token'}}],
    }
    target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding='utf-8')


def read_kubeconfig_cluster_name(path: str | Path) -> str:
    payload = yaml.safe_load(Path(path).read_text(encoding='utf-8')) or {}
    current = str(payload.get('current-context') or '').strip()
    if current:
        return current
    contexts = payload.get('contexts') or []
    if contexts:
        return str(contexts[0].get('name') or '').strip()
    raise K8sStateError(f'Unable to resolve cluster name from kubeconfig: {path}')


def create_cluster(
    name: str, kubeconfig_path: str | Path, *, extra_port_mappings: list[dict[str, int]] | None = None
) -> None:
    mapping_payload = copy.deepcopy(extra_port_mappings or [])

    def _mutate(state: dict[str, Any]) -> None:
        cluster = _ensure_cluster_defaults(state['clusters'].setdefault(name, {}), name)
        cluster['kubeconfigs'] = [str(kubeconfig_path)]
        cluster['extra_port_mappings'] = mapping_payload
        state['current_cluster'] = name
        _sync_cluster(cluster, name)

    mutate_k8s_state(_mutate)
    write_kubeconfig(name, kubeconfig_path)


def delete_cluster(name: str) -> None:
    state_path = k8s_state_path()
    cluster_root = _cluster_runtime_root(name)

    def _mutate(state: dict[str, Any]) -> tuple[list[int], list[int]]:
        cluster = state['clusters'].pop(name, None)
        if not cluster:
            return [], []
        if state.get('current_cluster') == name:
            state['current_cluster'] = None
        forward_pids = [int(item.get('pid') or 0) for item in cluster.get('port_forwards', {}).values()]
        binding_pids = [int(item.get('pid') or 0) for item in cluster.get('http_bindings', {}).values()]
        return forward_pids, binding_pids

    forward_pids, binding_pids = mutate_k8s_state(_mutate)
    for pid in forward_pids + binding_pids:
        _kill_pid(pid)
    shutil.rmtree(cluster_root, ignore_errors=True)
    if not Path(state_path).exists():
        return


def list_cluster_names() -> list[str]:
    state = read_k8s_state()
    return sorted(state.get('clusters', {}).keys())


def _spawn_forwarder(*, cluster_name: str, namespace: str, service_name: str, bind_port: int, protocol: str) -> int:
    helper_path = Path(__file__).with_name(_FORWARDER_NAME)
    env = dict(os.environ)
    env.setdefault(STORE_PATH_ENV, k8s_state_path())
    env.setdefault(ROOT_PATH_ENV, str(k8s_root_path()))
    process = subprocess.Popen(
        [
            sys.executable,
            str(helper_path),
            '--state-path',
            k8s_state_path(),
            '--cluster',
            cluster_name,
            '--namespace',
            namespace,
            '--service',
            service_name,
            '--port',
            str(bind_port),
            '--protocol',
            protocol,
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return int(process.pid)


def _ensure_nodeport_bindings(cluster_name: str) -> None:
    state = read_k8s_state()
    if cluster_name not in state.get('clusters', {}):
        return
    cluster = _ensure_cluster_defaults(state['clusters'][cluster_name], cluster_name)
    desired: list[tuple[str, str, str, int]] = []
    mappings = cluster.get('extra_port_mappings', [])
    mapping_lookup = {int(item.get('containerPort') or 0): int(item.get('hostPort') or 0) for item in mappings}
    for service in _iter_namespaced_resources(cluster, 'service'):
        if str(service.get('spec', {}).get('type') or 'ClusterIP') != 'NodePort':
            continue
        namespace = str(service.get('metadata', {}).get('namespace') or 'default')
        service_name = str(service['metadata']['name'])
        for port in service.get('spec', {}).get('ports', []) or []:
            node_port = int(port.get('nodePort') or 0)
            host_port = mapping_lookup.get(node_port)
            if host_port:
                desired.append((namespace, service_name, str(node_port), host_port))

    started: list[tuple[str, int]] = []

    def _needs_start(binding: dict[str, Any] | None, host_port: int, namespace: str, service_name: str) -> bool:
        if not binding:
            return True
        if int(binding.get('port') or 0) != host_port:
            return True
        if str(binding.get('namespace') or '') != namespace:
            return True
        if str(binding.get('service') or '') != service_name:
            return True
        return not _process_alive(int(binding.get('pid') or 0))

    for namespace, service_name, _node_port, host_port in desired:
        binding_key = f'{namespace}/{service_name}:{host_port}'
        binding = cluster.get('http_bindings', {}).get(binding_key)
        if _needs_start(binding, host_port, namespace, service_name):
            started.append((
                binding_key,
                _spawn_forwarder(
                    cluster_name=cluster_name,
                    namespace=namespace,
                    service_name=service_name,
                    bind_port=host_port,
                    protocol='http',
                ),
            ))

    if not started:
        return

    def _record(state: dict[str, Any]) -> None:
        cluster_state = _ensure_cluster_defaults(state['clusters'][cluster_name], cluster_name)
        bindings = cluster_state.setdefault('http_bindings', {})
        for binding_key, pid in started:
            namespace, remainder = binding_key.split('/', 1)
            service_name, port_text = remainder.split(':', 1)
            bindings[binding_key] = {
                'pid': pid,
                'namespace': namespace,
                'service': service_name,
                'port': int(port_text),
            }

    mutate_k8s_state(_record)


def apply_documents(
    documents: list[dict[str, Any]], *, cluster_name: str | None = None, kubeconfig_path: str | None = None
) -> list[str]:
    applied_refs: list[str] = []
    resolved_cluster_name: str | None = None

    def _mutate(state: dict[str, Any]) -> list[str]:
        nonlocal resolved_cluster_name
        resolved_cluster_name, cluster = _cluster_for_state(
            state, cluster_name=cluster_name, kubeconfig_path=kubeconfig_path
        )
        refs: list[str] = []
        for document in documents:
            if not document:
                continue
            kind, namespace, name, payload = _normalize_resource_document(cluster, document)
            refs.append(_resource_key(namespace, name))
            store = _resource_store(cluster, kind)
            key = _resource_key(namespace, name)
            existing = copy.deepcopy(store.get(key))
            if kind == 'namespace':
                store[name] = _namespace_object(name)
                store[name]['metadata'] = _metadata_for_resource(existing, payload)
            elif kind == 'configmap':
                store[key] = _configmap_payload(existing, payload, namespace=namespace or 'default')
            elif kind == 'secret':
                store[key] = _secret_payload(existing, payload, namespace=namespace or 'default')
                secret = store[key]
                if name == 'mysql-f1-secret':
                    password_b64 = secret.get('data', {}).get('MYSQL_ROOT_PASSWORD')
                    if password_b64:
                        cluster['mysql']['users']['root'] = {'password': base64.b64decode(password_b64).decode('utf-8')}
            elif kind == 'serviceaccount':
                store[key] = _serviceaccount_payload(existing, payload, namespace=namespace or 'default')
            elif kind == 'pod':
                store[key] = _manual_pod_payload(existing, payload, namespace=namespace or 'default')
            elif kind == 'deployment':
                store[key] = _deployment_payload(existing, payload, namespace=namespace or 'default')
            elif kind == 'statefulset':
                store[key] = _statefulset_payload(existing, payload, namespace=namespace or 'default')
            elif kind == 'service':
                store[key] = _ensure_service_defaults(existing, payload, namespace=namespace or 'default')
            else:
                raise K8sStateError(f'Unsupported manifest kind: {kind}')
        _sync_cluster(cluster, resolved_cluster_name)
        return refs

    applied_refs = mutate_k8s_state(_mutate)
    if resolved_cluster_name:
        _ensure_nodeport_bindings(resolved_cluster_name)
    return applied_refs


def parse_yaml_documents(text: str) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for document in yaml.safe_load_all(text):
        if document:
            docs.append(document)
    return docs


def get_resource(
    kind: str,
    name: str,
    *,
    namespace: str | None = None,
    cluster_name: str | None = None,
    kubeconfig_path: str | None = None,
) -> dict[str, Any]:
    state = read_k8s_state()
    resolved_cluster_name, cluster = _cluster_for_state(
        state, cluster_name=cluster_name, kubeconfig_path=kubeconfig_path
    )
    _sync_cluster(cluster, resolved_cluster_name)
    canonical = _canonical_kind(kind)
    if canonical == 'node':
        node = copy.deepcopy(cluster.get('nodes', {}).get(name))
        if node is None:
            raise K8sStateError(f'Node not found: {name}')
        return node
    if canonical == 'endpoints':
        service = get_resource('service', name, namespace=namespace, cluster_name=resolved_cluster_name)
        return {
            'apiVersion': 'v1',
            'kind': 'Endpoints',
            'metadata': {
                'name': name,
                'namespace': namespace or 'default',
                'creationTimestamp': service.get('metadata', {}).get('creationTimestamp') or utc_now_iso(),
                'labels': {},
                'annotations': {},
            },
            'subsets': copy.deepcopy(service.get('nexus_endpoints', {}).get('subsets', [])),
        }
    store = _resource_store(cluster, canonical)
    key = _resource_key(namespace, name)
    if key not in store:
        raise K8sStateError(f'{canonical} not found: {key}')
    return copy.deepcopy(store[key])


def list_resources(
    kind: str,
    *,
    namespace: str | None = None,
    label_selector: str | None = None,
    cluster_name: str | None = None,
    kubeconfig_path: str | None = None,
) -> list[dict[str, Any]]:
    state = read_k8s_state()
    resolved_cluster_name, cluster = _cluster_for_state(
        state, cluster_name=cluster_name, kubeconfig_path=kubeconfig_path
    )
    _sync_cluster(cluster, resolved_cluster_name)
    canonical = _canonical_kind(kind)
    if canonical == 'node':
        items = [copy.deepcopy(item) for item in cluster.get('nodes', {}).values()]
    elif canonical == 'endpoints':
        services = _iter_namespaced_resources(cluster, 'service', namespace)
        items = [
            {
                'apiVersion': 'v1',
                'kind': 'Endpoints',
                'metadata': {
                    'name': str(service['metadata']['name']),
                    'namespace': str(service['metadata'].get('namespace') or 'default'),
                    'creationTimestamp': service.get('metadata', {}).get('creationTimestamp') or utc_now_iso(),
                    'labels': {},
                    'annotations': {},
                },
                'subsets': copy.deepcopy(service.get('nexus_endpoints', {}).get('subsets', [])),
            }
            for service in services
        ]
    else:
        items = [copy.deepcopy(item) for item in _iter_namespaced_resources(cluster, canonical, namespace)]
    selector = _parse_label_selector(label_selector)
    if selector:
        items = [
            item
            for item in items
            if _selector_matches(copy.deepcopy(item.get('metadata', {}).get('labels', {})), selector)
        ]
    return items


def _parse_label_selector(selector: str | None) -> dict[str, str]:
    parsed: dict[str, str] = {}
    if not selector:
        return parsed
    for raw_part in selector.split(','):
        part = raw_part.strip()
        if not part or '=' not in part:
            continue
        key, value = part.split('=', 1)
        parsed[key.strip()] = value.strip()
    return parsed


def delete_resource(
    kind: str,
    name: str,
    *,
    namespace: str | None = None,
    cluster_name: str | None = None,
    kubeconfig_path: str | None = None,
) -> None:
    resolved_cluster_name: str | None = None

    def _mutate(state: dict[str, Any]) -> None:
        nonlocal resolved_cluster_name
        resolved_cluster_name, cluster = _cluster_for_state(
            state, cluster_name=cluster_name, kubeconfig_path=kubeconfig_path
        )
        canonical = _canonical_kind(kind)
        if canonical == 'namespace':
            cluster['resources']['namespaces'].pop(name, None)
            for store_name in ('configmaps', 'secrets', 'deployments', 'statefulsets', 'pods', 'services'):
                store = cluster['resources'][store_name]
                for key in list(store.keys()):
                    if key.startswith(f'{name}/'):
                        store.pop(key, None)
        else:
            store = _resource_store(cluster, canonical)
            store.pop(_resource_key(namespace, name), None)
        _sync_cluster(cluster, resolved_cluster_name)

    mutate_k8s_state(_mutate)
    if resolved_cluster_name:
        _ensure_nodeport_bindings(resolved_cluster_name)


def scale_workload(
    kind: str,
    name: str,
    replicas: int,
    *,
    namespace: str | None = None,
    cluster_name: str | None = None,
    kubeconfig_path: str | None = None,
) -> None:
    canonical = _canonical_kind(kind)
    if canonical not in {'deployment', 'statefulset'}:
        raise K8sStateError(f'Cannot scale kind: {kind}')
    resolved_cluster_name: str | None = None

    def _mutate(state: dict[str, Any]) -> None:
        nonlocal resolved_cluster_name
        resolved_cluster_name, cluster = _cluster_for_state(
            state, cluster_name=cluster_name, kubeconfig_path=kubeconfig_path
        )
        store = _resource_store(cluster, canonical)
        key = _resource_key(namespace, name)
        if key not in store:
            raise K8sStateError(f'{canonical} not found: {key}')
        store[key].setdefault('spec', {})['replicas'] = int(replicas)
        _sync_cluster(cluster, resolved_cluster_name)

    mutate_k8s_state(_mutate)
    if resolved_cluster_name:
        _ensure_nodeport_bindings(resolved_cluster_name)


def patch_resource_json(
    kind: str,
    name: str,
    patch_ops: list[dict[str, Any]],
    *,
    namespace: str | None = None,
    cluster_name: str | None = None,
    kubeconfig_path: str | None = None,
) -> dict[str, Any]:
    resolved_cluster_name: str | None = None

    def _apply(target: Any, path_tokens: list[str], value: Any | None, op: str) -> None:
        parent = target
        for token in path_tokens[:-1]:
            if isinstance(parent, list):
                parent = parent[int(token)]
                continue
            parent = parent.setdefault(token, {})
        leaf = path_tokens[-1]
        if isinstance(parent, list):
            index = int(leaf)
            if op == 'remove':
                parent.pop(index)
            elif op == 'add':
                if index >= len(parent):
                    parent.append(value)
                else:
                    parent.insert(index, value)
            else:
                parent[index] = value
            return
        if op == 'remove':
            parent.pop(leaf, None)
        else:
            parent[leaf] = value

    def _mutate(state: dict[str, Any]) -> dict[str, Any]:
        nonlocal resolved_cluster_name
        resolved_cluster_name, cluster = _cluster_for_state(
            state, cluster_name=cluster_name, kubeconfig_path=kubeconfig_path
        )
        canonical = _canonical_kind(kind)
        store = _resource_store(cluster, canonical)
        key = _resource_key(namespace, name)
        if key not in store:
            raise K8sStateError(f'{canonical} not found: {key}')
        payload = store[key]
        for operation in patch_ops:
            op = str(operation.get('op') or '').strip().lower()
            path_text = str(operation.get('path') or '').strip()
            value = copy.deepcopy(operation.get('value'))
            if not op or not path_text.startswith('/'):
                raise K8sStateError(f'Unsupported patch operation: {operation}')
            tokens = [token for token in path_text.lstrip('/').split('/') if token]
            _apply(payload, tokens, value, op)
        _sync_cluster(cluster, resolved_cluster_name)
        return copy.deepcopy(payload)

    result = mutate_k8s_state(_mutate)
    if resolved_cluster_name:
        _ensure_nodeport_bindings(resolved_cluster_name)
    return result


def rollout_ready(
    kind: str,
    name: str,
    *,
    namespace: str | None = None,
    cluster_name: str | None = None,
    kubeconfig_path: str | None = None,
) -> bool:
    resource = get_resource(kind, name, namespace=namespace, cluster_name=cluster_name, kubeconfig_path=kubeconfig_path)
    status = copy.deepcopy(resource.get('status', {}))
    spec = copy.deepcopy(resource.get('spec', {}))
    replicas = int(spec.get('replicas') or 0)
    if _canonical_kind(kind) == 'statefulset':
        return int(status.get('readyReplicas') or 0) == replicas
    ready = int(status.get('readyReplicas') or 0)
    available = int(status.get('availableReplicas') or 0)
    return ready == replicas and available == replicas


def wait_pods_ready(
    *,
    namespace: str | None = None,
    name: str | None = None,
    label_selector: str | None = None,
    cluster_name: str | None = None,
    kubeconfig_path: str | None = None,
) -> bool:
    if name:
        pod = get_resource('pod', name, namespace=namespace, cluster_name=cluster_name, kubeconfig_path=kubeconfig_path)
        statuses = pod.get('status', {}).get('containerStatuses') or []
        return all(bool(item.get('ready')) for item in statuses)
    pods = list_resources(
        'pod',
        namespace=namespace,
        label_selector=label_selector,
        cluster_name=cluster_name,
        kubeconfig_path=kubeconfig_path,
    )
    if not pods:
        return True
    for pod in pods:
        statuses = pod.get('status', {}).get('containerStatuses') or []
        if not all(bool(item.get('ready')) for item in statuses):
            return False
    return True


def cluster_info_text(cluster_name: str | None = None, kubeconfig_path: str | None = None) -> str:
    state = read_k8s_state()
    resolved_cluster_name, cluster = _cluster_for_state(
        state, cluster_name=cluster_name, kubeconfig_path=kubeconfig_path
    )
    node = next(iter(cluster.get('nodes', {}).values()))
    return (
        f'Kubernetes control plane is running for {resolved_cluster_name}\n'
        f'CoreDNS is running at 10.96.0.10\n'
        f'Node: {node["metadata"]["name"]}'
    )


def pod_copy_in(
    src_path: str | Path,
    dest_spec: str,
    *,
    namespace: str | None = None,
    cluster_name: str | None = None,
    kubeconfig_path: str | None = None,
) -> str:
    source = Path(src_path).resolve()
    if ':' not in dest_spec:
        raise K8sStateError(f'Invalid pod copy target: {dest_spec}')
    pod_name, remote_path = dest_spec.split(':', 1)
    state = read_k8s_state()
    resolved_cluster_name, cluster = _cluster_for_state(
        state, cluster_name=cluster_name, kubeconfig_path=kubeconfig_path
    )
    pod = get_resource('pod', pod_name, namespace=namespace, cluster_name=resolved_cluster_name)
    pod_namespace = str(pod.get('metadata', {}).get('namespace') or namespace or 'default')
    target_root = _pod_fs_root(resolved_cluster_name, pod_namespace, pod_name)
    destination = target_root / remote_path.lstrip('/')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        destination.mkdir(parents=True, exist_ok=True)
        for child in source.iterdir():
            target = destination / child.name
            if child.is_dir():
                shutil.copytree(child, target, dirs_exist_ok=True)
            else:
                shutil.copy2(child, target)
    else:
        if source.name == '.':
            destination.mkdir(parents=True, exist_ok=True)
        else:
            shutil.copy2(source, destination)
    return str(destination)


def mysql_exec_import(
    *,
    cluster_name: str | None = None,
    kubeconfig_path: str | None = None,
    namespace: str | None = None,
    pod_name: str | None = None,
    sql_text: str,
    csv_root: str | Path | None = None,
) -> None:
    state = read_k8s_state()
    resolved_cluster_name, cluster = _cluster_for_state(
        state, cluster_name=cluster_name, kubeconfig_path=kubeconfig_path
    )
    db_path = cluster['mysql']['db_path']
    apply_mysql_script(db_path, sql_text, csv_root=csv_root)


def mysql_load_schema(*, cluster_name: str | None = None, kubeconfig_path: str | None = None, sql_text: str) -> None:
    state = read_k8s_state()
    resolved_cluster_name, cluster = _cluster_for_state(
        state, cluster_name=cluster_name, kubeconfig_path=kubeconfig_path
    )
    apply_schema_sql(cluster['mysql']['db_path'], sql_text)


def set_mysql_user(
    username: str, password: str, *, cluster_name: str | None = None, kubeconfig_path: str | None = None
) -> None:
    def _mutate(state: dict[str, Any]) -> None:
        resolved_cluster_name, cluster = _cluster_for_state(
            state, cluster_name=cluster_name, kubeconfig_path=kubeconfig_path
        )
        cluster['mysql']['users'][username] = {'password': password}
        _sync_cluster(cluster, resolved_cluster_name)

    mutate_k8s_state(_mutate)


def mysql_users(*, cluster_name: str | None = None, kubeconfig_path: str | None = None) -> dict[str, dict[str, Any]]:
    state = read_k8s_state()
    _, cluster = _cluster_for_state(state, cluster_name=cluster_name, kubeconfig_path=kubeconfig_path)
    return copy.deepcopy(cluster.get('mysql', {}).get('users', {}))


def mysql_db_path(*, cluster_name: str | None = None, kubeconfig_path: str | None = None) -> str:
    state = read_k8s_state()
    _, cluster = _cluster_for_state(state, cluster_name=cluster_name, kubeconfig_path=kubeconfig_path)
    return str(cluster['mysql']['db_path'])


def register_port_forward(
    *,
    local_port: int,
    namespace: str,
    resource: str,
    pid: int,
    kind: str = 'service',
    cluster_name: str | None = None,
    kubeconfig_path: str | None = None,
) -> None:
    def _mutate(state: dict[str, Any]) -> None:
        resolved_cluster_name, cluster = _cluster_for_state(
            state, cluster_name=cluster_name, kubeconfig_path=kubeconfig_path
        )
        key = f'{namespace}/{resource}:{local_port}'
        cluster.setdefault('port_forwards', {})[key] = {
            'pid': int(pid),
            'namespace': namespace,
            'resource': resource,
            'port': int(local_port),
            'kind': kind,
        }
        _sync_cluster(cluster, resolved_cluster_name)

    mutate_k8s_state(_mutate)


def resolve_service_content(
    namespace: str, service_name: str, *, cluster_name: str | None = None, kubeconfig_path: str | None = None
) -> tuple[str, str]:
    state = read_k8s_state()
    resolved_cluster_name, cluster = _cluster_for_state(
        state, cluster_name=cluster_name, kubeconfig_path=kubeconfig_path
    )
    service = get_resource('service', service_name, namespace=namespace, cluster_name=resolved_cluster_name)
    selector = copy.deepcopy(service.get('spec', {}).get('selector', {}))
    if not selector:
        return '<html><body>No selector</body></html>', 'text/html; charset=utf-8'
    pods = list_resources('pod', namespace=namespace, cluster_name=resolved_cluster_name)
    pod = next(
        (
            item
            for item in pods
            if _selector_matches(copy.deepcopy(item.get('metadata', {}).get('labels', {})), selector)
        ),
        None,
    )
    if pod is None:
        return '<html><body>No backend pods</body></html>', 'text/html; charset=utf-8'
    owner = str(pod.get('metadata', {}).get('annotations', {}).get('nexus.generatedBy') or '')
    workload: dict[str, Any] | None = None
    if owner.startswith('deployment/'):
        _, workload_namespace, workload_name = owner.split('/', 2)
        workload = get_resource(
            'deployment', workload_name, namespace=workload_namespace, cluster_name=resolved_cluster_name
        )
    elif owner.startswith('statefulset/'):
        _, workload_namespace, workload_name = owner.split('/', 2)
        workload = get_resource(
            'statefulset', workload_name, namespace=workload_namespace, cluster_name=resolved_cluster_name
        )
    if workload is None:
        return '<html><body>Mock service is running</body></html>', 'text/html; charset=utf-8'
    template_spec = copy.deepcopy(workload.get('spec', {}).get('template', {}).get('spec', {}))
    volumes = {str(item.get('name') or ''): item for item in template_spec.get('volumes', []) or []}
    for container in template_spec.get('containers', []) or []:
        for mount in container.get('volumeMounts', []) or []:
            mount_path = str(mount.get('mountPath') or '')
            if not mount_path.endswith('index.html'):
                continue
            volume = volumes.get(str(mount.get('name') or ''))
            if not volume or 'configMap' not in volume:
                continue
            configmap_name = str(volume['configMap'].get('name') or '').strip()
            configmap = get_resource(
                'configmap', configmap_name, namespace=namespace, cluster_name=resolved_cluster_name
            )
            sub_path = str(mount.get('subPath') or 'index.html')
            data = str(configmap.get('data', {}).get(sub_path) or '')
            if data:
                return data, 'text/html; charset=utf-8'
            if configmap.get('data'):
                first_value = next(iter(configmap['data'].values()))
                return str(first_value), 'text/html; charset=utf-8'
    return '<html><body>Mock service is running</body></html>', 'text/html; charset=utf-8'


def add_helm_repo(name: str, url: str) -> None:
    def _mutate(state: dict[str, Any]) -> None:
        state.setdefault('helm_repos', {})[name] = url

    mutate_k8s_state(_mutate)


def list_helm_releases(
    *, namespace: str | None = None, cluster_name: str | None = None, kubeconfig_path: str | None = None
) -> list[dict[str, Any]]:
    state = read_k8s_state()
    _, cluster = _cluster_for_state(state, cluster_name=cluster_name, kubeconfig_path=kubeconfig_path)
    releases = [copy.deepcopy(item) for item in cluster.get('helm_releases', {}).values()]
    if namespace:
        releases = [item for item in releases if str(item.get('namespace') or '') == namespace]
    releases.sort(key=lambda item: (str(item.get('namespace') or ''), str(item.get('name') or '')))
    return releases


def get_helm_values(
    release_name: str,
    *,
    namespace: str | None = None,
    cluster_name: str | None = None,
    kubeconfig_path: str | None = None,
) -> dict[str, Any]:
    releases = list_helm_releases(namespace=namespace, cluster_name=cluster_name, kubeconfig_path=kubeconfig_path)
    for release in releases:
        if str(release.get('name') or '') == release_name:
            return copy.deepcopy(release.get('values') or {})
    raise K8sStateError(f'Helm release not found: {release_name}')


def _helm_generated_annotation(release_name: str) -> dict[str, str]:
    return {'nexus.generatedByHelm': release_name}


def _upsert_resource_object(store: dict[str, Any], key: str, obj: dict[str, Any]) -> None:
    existing = copy.deepcopy(store.get(key))
    obj['metadata'] = _metadata_for_resource(existing, obj)
    store[key] = obj


def _sync_redis_release(cluster: dict[str, Any], release: dict[str, Any]) -> None:
    namespace = str(release['namespace'])
    release_name = str(release['name'])
    values = copy.deepcopy(release.get('values') or {})
    _ensure_namespace(cluster, namespace)

    for store_name in ('pods', 'services', 'secrets'):
        store = cluster['resources'][store_name]
        for key in list(store.keys()):
            if (
                str(store[key].get('metadata', {}).get('annotations', {}).get('nexus.generatedByHelm') or '')
                == release_name
            ):
                store.pop(key, None)

    secret_store = cluster['resources']['secrets']
    password = str(values.get('auth', {}).get('password') or 'simple123')
    secret_key = _resource_key(namespace, release_name)
    secret_store[secret_key] = {
        'apiVersion': 'v1',
        'kind': 'Secret',
        'metadata': {
            'name': release_name,
            'namespace': namespace,
            'creationTimestamp': utc_now_iso(),
            'labels': {'app.kubernetes.io/instance': release_name},
            'annotations': _helm_generated_annotation(release_name),
        },
        'type': 'Opaque',
        'data': {'redis-password': base64.b64encode(password.encode('utf-8')).decode('ascii')},
    }

    service_store = cluster['resources']['services']
    common_labels = {'app.kubernetes.io/instance': release_name, 'app.kubernetes.io/name': 'redis'}
    common_labels.update({str(key): str(value) for key, value in (values.get('commonLabels') or {}).items()})
    service_payloads = [
        {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {
                'name': f'{release_name}-master',
                'namespace': namespace,
                'labels': common_labels,
                'annotations': _helm_generated_annotation(release_name),
            },
            'spec': {
                'type': str(values.get('service', {}).get('type') or 'ClusterIP'),
                'selector': {'app.kubernetes.io/instance': release_name, 'app.kubernetes.io/component': 'master'},
                'ports': [
                    {'name': 'redis', 'port': int(values.get('service', {}).get('port') or 6379), 'targetPort': 6379}
                ],
                'annotations': copy.deepcopy(values.get('service', {}).get('annotations') or {}),
            },
        },
        {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {
                'name': f'{release_name}-headless',
                'namespace': namespace,
                'labels': common_labels,
                'annotations': _helm_generated_annotation(release_name),
            },
            'spec': {
                'clusterIP': 'None',
                'selector': {'app.kubernetes.io/instance': release_name},
                'ports': [{'name': 'redis', 'port': 6379, 'targetPort': 6379}],
            },
        },
        {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {
                'name': f'{release_name}-replicas',
                'namespace': namespace,
                'labels': common_labels,
                'annotations': _helm_generated_annotation(release_name),
            },
            'spec': {
                'selector': {'app.kubernetes.io/instance': release_name, 'app.kubernetes.io/component': 'replica'},
                'ports': [{'name': 'redis', 'port': 6379, 'targetPort': 6379}],
            },
        },
    ]
    for payload in service_payloads:
        service_key = _service_key(namespace, str(payload['metadata']['name']))
        service_store[service_key] = _ensure_service_defaults(
            service_store.get(service_key), payload, namespace=namespace
        )

    pod_store = cluster['resources']['pods']
    master_labels = dict(common_labels)
    master_labels.update({'app.kubernetes.io/component': 'master'})
    replica_labels = dict(common_labels)
    replica_labels.update({'app.kubernetes.io/component': 'replica'})
    pod_payloads: list[dict[str, Any]] = [
        {
            'apiVersion': 'v1',
            'kind': 'Pod',
            'metadata': {
                'name': f'{release_name}-master-0',
                'namespace': namespace,
                'labels': master_labels,
                'annotations': _helm_generated_annotation(release_name),
            },
            'spec': {
                'containers': [
                    {
                        'name': 'redis',
                        'image': f'bitnami/redis:{release["version"]}',
                        'ports': [{'containerPort': 6379}],
                    }
                ],
                'nodeName': 'mock-control-plane',
            },
        }
    ]
    replica_count = int(values.get('replica', {}).get('replicaCount') or 0)
    for index in range(replica_count):
        pod_payloads.append({
            'apiVersion': 'v1',
            'kind': 'Pod',
            'metadata': {
                'name': f'{release_name}-replicas-{index}',
                'namespace': namespace,
                'labels': replica_labels,
                'annotations': _helm_generated_annotation(release_name),
            },
            'spec': {
                'containers': [
                    {
                        'name': 'redis',
                        'image': f'bitnami/redis:{release["version"]}',
                        'ports': [{'containerPort': 6379}],
                    }
                ],
                'nodeName': 'mock-control-plane',
            },
        })
    for payload in pod_payloads:
        pod_key = _resource_key(namespace, str(payload['metadata']['name']))
        pod_store[pod_key] = _manual_pod_payload(pod_store.get(pod_key), payload, namespace=namespace)
        pod_store[pod_key]['metadata']['annotations'].update(_helm_generated_annotation(release_name))


def helm_install_or_upgrade(
    release_name: str,
    chart_ref: str,
    *,
    namespace: str,
    version: str,
    values: dict[str, Any],
    reuse_values: bool = False,
    cluster_name: str | None = None,
    kubeconfig_path: str | None = None,
) -> dict[str, Any]:
    resolved_cluster_name: str | None = None

    def _mutate(state: dict[str, Any]) -> dict[str, Any]:
        nonlocal resolved_cluster_name
        resolved_cluster_name, cluster = _cluster_for_state(
            state, cluster_name=cluster_name, kubeconfig_path=kubeconfig_path
        )
        releases = cluster.setdefault('helm_releases', {})
        key = f'{namespace}/{release_name}'
        existing = copy.deepcopy(releases.get(key) or {})
        current_values = copy.deepcopy(existing.get('values') or {})
        merged_values = copy.deepcopy(current_values if reuse_values else {})
        merged_values.update(copy.deepcopy(values))
        release = {
            'name': release_name,
            'namespace': namespace,
            'chart': f'{chart_ref.split("/", 1)[-1]}-{version}',
            'chart_ref': chart_ref,
            'status': 'deployed',
            'revision': int(existing.get('revision') or 0) + 1,
            'version': version,
            'values': merged_values,
        }
        releases[key] = release
        _sync_redis_release(cluster, release)
        _sync_cluster(cluster, resolved_cluster_name)
        return copy.deepcopy(release)

    release = mutate_k8s_state(_mutate)
    if resolved_cluster_name:
        _ensure_nodeport_bindings(resolved_cluster_name)
    return release


__all__ = [
    'K8sStateError',
    'STORE_PATH_ENV',
    'ROOT_PATH_ENV',
    'add_helm_repo',
    'apply_documents',
    'case_root_from_cwd',
    'cluster_info_text',
    'create_cluster',
    'delete_cluster',
    'get_helm_values',
    'get_resource',
    'helm_install_or_upgrade',
    'k8s_root_path',
    'k8s_state_path',
    'list_cluster_names',
    'list_helm_releases',
    'list_resources',
    'mysql_db_path',
    'mysql_exec_import',
    'mysql_load_schema',
    'mysql_users',
    'parse_yaml_documents',
    'patch_resource_json',
    'pod_copy_in',
    'read_k8s_state',
    'read_kubeconfig_cluster_name',
    'register_port_forward',
    'resolve_service_content',
    'rollout_ready',
    'scale_workload',
    'set_mysql_user',
    'wait_pods_ready',
    'write_kubeconfig',
]
