from __future__ import annotations


try:
    from toolathlon_k8s_state import read_kubeconfig_cluster_name
except ImportError:
    from ..toolathlon_k8s_state import read_kubeconfig_cluster_name


_ACTIVE_CONTEXT: dict[str, str | None] = {'cluster_name': None, 'kubeconfig_path': None}


def load_kube_config(config_file: str | None = None, context: str | None = None, **_: object) -> None:
    cluster_name = context or (_ACTIVE_CONTEXT.get('cluster_name') if config_file is None else None)
    if not cluster_name and config_file:
        cluster_name = read_kubeconfig_cluster_name(config_file)
    _ACTIVE_CONTEXT['cluster_name'] = cluster_name
    _ACTIVE_CONTEXT['kubeconfig_path'] = config_file


def current_cluster_name() -> str | None:
    return _ACTIVE_CONTEXT.get('cluster_name')


def current_kubeconfig_path() -> str | None:
    return _ACTIVE_CONTEXT.get('kubeconfig_path')


__all__ = ['current_cluster_name', 'current_kubeconfig_path', 'load_kube_config']
