from __future__ import annotations

from typing import Any

from .exceptions import ApiException


try:
    from toolathlon_k8s_state import K8sStateError, get_resource, list_resources
except ImportError:
    from ...toolathlon_k8s_state import K8sStateError, get_resource, list_resources

from ..config import current_cluster_name, current_kubeconfig_path


def _camelize(value: str) -> str:
    head, *tail = value.split('_')
    return head + ''.join(part.capitalize() for part in tail)


class _ObjectView:
    def __init__(self, payload: Any):
        self._payload = payload

    def __iter__(self):
        if isinstance(self._payload, dict):
            yield from self._payload
            return
        if isinstance(self._payload, list):
            for item in self._payload:
                yield _wrap(item)
            return
        raise TypeError('Object is not iterable')

    def __getitem__(self, key: str) -> Any:
        if isinstance(self._payload, dict):
            return _wrap(self._payload[key])
        if isinstance(self._payload, list):
            return _wrap(self._payload[key])
        raise TypeError('Unsupported index operation')

    def __len__(self) -> int:
        if isinstance(self._payload, (list, dict)):
            return len(self._payload)
        raise TypeError('Object has no length')

    def __bool__(self) -> bool:
        return bool(self._payload)

    def __contains__(self, item: object) -> bool:
        if isinstance(self._payload, dict):
            return item in self._payload
        if isinstance(self._payload, list):
            return item in self._payload
        return False

    def get(self, key: str, default: Any = None) -> Any:
        if not isinstance(self._payload, dict):
            raise TypeError('Object is not a mapping')
        if key in self._payload:
            return _wrap(self._payload[key])
        camel = _camelize(key)
        if camel in self._payload:
            return _wrap(self._payload[camel])
        return default

    def keys(self):
        if not isinstance(self._payload, dict):
            raise TypeError('Object is not a mapping')
        return self._payload.keys()

    def values(self):
        if not isinstance(self._payload, dict):
            raise TypeError('Object is not a mapping')
        return [_wrap(value) for value in self._payload.values()]

    def __getattr__(self, item: str) -> Any:
        if isinstance(self._payload, dict):
            if item in self._payload:
                return _wrap(self._payload[item])
            camel = _camelize(item)
            if camel in self._payload:
                return _wrap(self._payload[camel])
            return None
        raise AttributeError(item)

    def to_dict(self) -> Any:
        return self._payload

    def __repr__(self) -> str:
        return repr(self._payload)


def _wrap(payload: Any) -> Any:
    if isinstance(payload, dict):
        return _ObjectView(payload)
    if isinstance(payload, list):
        return [_wrap(item) for item in payload]
    return payload


def _current_kwargs() -> dict[str, Any]:
    return {
        'cluster_name': current_cluster_name(),
        'kubeconfig_path': current_kubeconfig_path(),
    }


def _get(kind: str, name: str, *, namespace: str | None = None) -> _ObjectView:
    try:
        return _wrap(get_resource(kind, name, namespace=namespace, **_current_kwargs()))
    except K8sStateError as exc:
        raise ApiException(status=404, reason=str(exc)) from exc


def _list(kind: str, *, namespace: str | None = None, label_selector: str | None = None) -> _ObjectView:
    try:
        return _wrap({
            'items': list_resources(kind, namespace=namespace, label_selector=label_selector, **_current_kwargs())
        })
    except K8sStateError as exc:
        raise ApiException(status=500, reason=str(exc)) from exc


class CoreV1Api:
    def list_namespace(self):
        return _list('namespace')

    def list_namespaced_pod(self, namespace: str, label_selector: str | None = None):
        return _list('pod', namespace=namespace, label_selector=label_selector)

    def read_namespaced_secret(self, name: str, namespace: str):
        return _get('secret', name, namespace=namespace)

    def read_namespaced_config_map(self, name: str, namespace: str):
        return _get('configmap', name, namespace=namespace)

    def read_namespaced_service(self, name: str, namespace: str):
        return _get('service', name, namespace=namespace)

    def read_namespaced_pod(self, name: str, namespace: str):
        return _get('pod', name, namespace=namespace)


class AppsV1Api:
    def read_namespaced_deployment(self, name: str, namespace: str):
        return _get('deployment', name, namespace=namespace)

    def read_namespaced_stateful_set(self, name: str, namespace: str):
        return _get('statefulset', name, namespace=namespace)


__all__ = ['ApiException', 'AppsV1Api', 'CoreV1Api']
