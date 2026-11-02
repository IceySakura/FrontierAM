import sys

from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def ray(monkeypatch):
    ray = SimpleNamespace(
        is_initialized=Mock(return_value=True),
        init=Mock(),
        nodes=Mock(
            return_value=[
                {'Alive': False, 'NodeManagerAddress': '192.0.2.9', 'Resources': {'node:__internal_head__': 1}},
                {'Alive': True, 'NodeManagerAddress': '192.0.2.2', 'Resources': {'CPU': 8}},
                {'Alive': True, 'NodeManagerAddress': '192.0.2.1', 'Resources': {'node:__internal_head__': 1}},
            ]
        ),
    )
    monkeypatch.setitem(sys.modules, 'ray', ray)
    return ray


@pytest.mark.parametrize('chief_ip', [None, '192.0.2.2', '203.0.113.1'])
def test_head_discovery_ignores_node_order_dead_heads_and_chief_ip(monkeypatch, ray, chief_ip):
    from nexus.gateway.trajectory_store_backends.redis_store.utils import get_ray_head_ip

    if chief_ip is None:
        monkeypatch.delenv('CHIEF_IP', raising=False)
    else:
        monkeypatch.setenv('CHIEF_IP', chief_ip)
    assert get_ray_head_ip() == '192.0.2.1'
    ray.init.assert_not_called()


def test_discovery_does_not_connect_or_start_ray(ray):
    from nexus.gateway.trajectory_store_backends.redis_store.utils import get_ray_head_ip

    ray.is_initialized.return_value = False
    with pytest.raises(RuntimeError, match='initialized Ray'):
        get_ray_head_ip()
    ray.init.assert_not_called()
    ray.nodes.assert_not_called()


def test_missing_alive_head_fails_explicitly(ray):
    from nexus.gateway.trajectory_store_backends.redis_store.utils import get_ray_head_ip

    ray.nodes.return_value = ray.nodes.return_value[:2]
    with pytest.raises(RuntimeError, match='alive Ray head'):
        get_ray_head_ip()
    ray.init.assert_not_called()


def test_discovery_does_not_cache_a_previous_cluster(ray):
    from nexus.gateway.trajectory_store_backends.redis_store.utils import get_ray_head_ip

    assert get_ray_head_ip() == '192.0.2.1'
    ray.nodes.return_value[-1]['NodeManagerAddress'] = '198.51.100.1'
    assert get_ray_head_ip() == '198.51.100.1'
