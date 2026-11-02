from __future__ import annotations

import asyncio

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nexus.distributed.config import DistributedConfig
from nexus.distributed.observability.history import HISTORY_CATALOG, MetricHistoryReader
from nexus.distributed.observability.sampler import _group_gauges
from nexus.distributed.repositories.observability import ObservabilityRepository


@pytest.mark.asyncio
async def test_database_server_reads_global_counters_and_limit():
    session = MagicMock()
    session.get_bind.return_value.dialect.name = 'mysql'
    session.execute = AsyncMock(
        side_effect=[
            SimpleNamespace(all=lambda: [('Threads_connected', '12'), ('Threads_running', '2')]),
            SimpleNamespace(scalar_one=lambda: 100),
        ]
    )
    value = await ObservabilityRepository(session, DistributedConfig()).database_server()
    assert value == {'data_status': 'ready', 'connected': 12, 'running': 2, 'limit': 100}
    assert len(session.execute.await_args_list) == 2
    gauges = _group_gauges('database_server', 'inventory', {'database_server': value}, {'environment': 'all'})
    assert gauges == [
        (12, {'environment': 'all', 'kind': 'connected'}),
        (2, {'environment': 'all', 'kind': 'running'}),
        (100, {'environment': 'all', 'kind': 'limit'}),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize('timeout', [False, True])
async def test_database_server_failure_is_isolated_and_has_no_fabricated_gauge(timeout):
    session = MagicMock()
    session.get_bind.return_value.dialect.name = 'mysql'

    async def fail(*args):
        if timeout:
            await asyncio.sleep(10)
        raise PermissionError('private database details')

    session.execute = AsyncMock(side_effect=fail)
    session.rollback = AsyncMock()
    config = DistributedConfig(status_sampling={'query_timeout_seconds': 0.1})
    value = await ObservabilityRepository(session, config).database_server()
    assert value == {'data_status': 'error', 'error_code': 'TimeoutError' if timeout else 'PermissionError'}
    session.rollback.assert_awaited_once()
    assert _group_gauges('database_server', 'inventory', {'database_server': value}, {}) == []


@pytest.mark.asyncio
async def test_database_server_skips_unsupported_dialects():
    session = MagicMock()
    session.get_bind.return_value.dialect.name = 'sqlite'
    assert await ObservabilityRepository(session, DistributedConfig()).database_server() == {
        'data_status': 'not_supported'
    }
    session.execute.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize('sample, expected', [(None, 'not_configured'), (0, 'ready'), (3, 'ready')])
async def test_disabled_trace_database_preserves_existing_history(monkeypatch, sample, expected):
    reader = MetricHistoryReader(
        DistributedConfig(
            metrics_history={
                'enabled': True,
                'project_name': 'test',
                'project_id': 1,
                'app_mark': 'test',
                'environment': 'test',
            },
            secrets={'metrics_query': {'username': 'test', 'token': 'test'}},
        )
    )
    query = AsyncMock(return_value={'points': [{'time': '2026-09-14T00:00:00', 'value': sample}]})
    monkeypatch.setattr(reader, '_query', query)
    try:
        value = await reader.get(panel='database_pool', hours=1, filters={'source': 'trace_db'})
        assert value['data_status'] == expected
        assert value['series'][0]['points'][0]['value'] == sample
        await reader.get(panel='database_server', hours=1, filters={'environment': 'selected'}, view='running')
        assert query.await_args.kwargs['filters'] == {'environment': 'all', 'kind': 'running'}
    finally:
        await reader.close()


def test_catalog_excludes_unused_quota_and_unsupported_budget_views():
    assert 'artifact_quota' not in HISTORY_CATALOG
    assert HISTORY_CATALOG['database_server']['scope'] == 'global'
    assert len(HISTORY_CATALOG['stage_unknown_budget']['views']) == 7
