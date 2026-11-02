from __future__ import annotations

import asyncio

from datetime import datetime

import pytest

from aiohttp import web
from aiohttp.test_utils import TestServer

from nexus.distributed.config import DistributedConfig
from nexus.distributed.observability.history import MetricHistoryReader


NOW = datetime(2026, 9, 14, 0, 0)


def config(endpoint):
    return DistributedConfig(
        metrics_history={
            'enabled': True,
            'endpoint': endpoint,
            'project_name': 'test-project',
            'project_id': 1,
            'app_mark': 'test-app',
            'environment': 'test',
        },
        secrets={'metrics_query': {'username': 'reader', 'token': 'private-query-token'}},
    )


@pytest.mark.asyncio
async def test_history_query_is_scoped_coalesced_and_preserves_missing_samples(monkeypatch):
    from nexus.distributed.observability import history

    monkeypatch.setattr(history, 'utcnow', lambda: NOW)
    monkeypatch.setenv('OTEL_EXPORTER_OTLP_HEADERS', 'Authorization=unrelated-trace-token')
    requests = []

    async def handler(request):
        body = await request.json()
        requests.append((body, dict(request.headers)))
        await asyncio.sleep(0.02)
        metric = body['metric_name']
        values = [2, 8] if metric == 'gateway_requests' else [0, 1] if metric == 'gateway_server_errors' else [0.1, 0.3]
        return web.json_response(
            {
                'code': 0,
                'data': {
                    'chart_info': [
                        {
                            'cond': {'metric': [metric], 'app_mark': 'test-app_test_nexus_cloud_abcd'},
                            'unit': 'seconds' if metric.endswith('p95') else 'count',
                            'detail_data_list': [
                                {'time': '2026-09-14 07:58', 'current': values[0]},
                                {'time': '2026-09-14 07:59', 'current': values[1]},
                                {'time': '2026-09-14 08:00', 'current': None},
                            ],
                        }
                    ]
                },
            }
        )

    app = web.Application()
    app.router.add_post('/query', handler)
    server = TestServer(app)
    await server.start_server()
    reader = MetricHistoryReader(config(str(server.make_url('/query'))))
    try:
        values = await asyncio.gather(
            *(
                reader.get(panel='gateway', hours=1, filters={'environment': 'one', 'service': 'control-gateway'})
                for _ in range(20)
            )
        )
        assert len(requests) == 3
        value = values[0]
        assert value['completed'] == 10
        assert value['failed'] == 1
        assert value['failure_ratio'] == 0.1
        assert value['series'][2]['points'][-1]['value'] is None
        assert value['series'][2]['points'][-2]['value'] == 0.3
        assert 'private-query-token' not in str(value)
        for body, headers in requests:
            assert body['app_mark'] == 'test-app'
            assert body['env'] == 'test'
            assert body['sec_lvl_name'] == 'nexus_cloud'
            assert body['tag_set'] == [
                {'key': 'environment', 'value': ['one'], 'op': '='},
                {'key': 'service', 'value': ['control-gateway'], 'op': '='},
            ]
            assert headers['token'] == 'private-query-token'
            assert 'Authorization' not in headers
        gauge = await reader.get(panel='queue_depth', hours=1, filters={})
        assert gauge['metric']['name'] == 'queue_depth'
        assert gauge['metric']['aggregation'] == 'avg'
        assert gauge['completed'] is None
        assert gauge['failure_ratio'] is None
        assert requests[-1][0]['tag_set'] == [
            {'key': 'environment', 'value': ['all'], 'op': '='},
            {'key': 'kind', 'value': ['schedulable'], 'op': '='},
        ]
        await reader.get(panel='slots', hours=1, filters={'environment': 'one'})
        assert requests[-1][0]['tag_set'][-1] == {'key': 'kind', 'value': ['available'], 'op': '='}
        await reader.get(panel='slots', hours=1, filters={'environment': 'one'}, view='total')
        assert requests[-1][0]['tag_set'][-1] == {'key': 'kind', 'value': ['total'], 'op': '='}
        await reader.get(panel='active_executions', hours=1, filters={'environment': 'one'}, view='queued')
        assert requests[-1][0]['tag_set'][-1] == {'key': 'state', 'value': ['queued'], 'op': '='}
        with pytest.raises(ValueError, match='invalid_history_view'):
            await reader.get(panel='slots', hours=1, filters={}, view='not-a-kind')
        with pytest.raises(ValueError, match='invalid_history_query'):
            await reader.get(panel='unregistered_metric', hours=1, filters={})
        values[0]['series'].clear()
        assert (
            len(
                (
                    await reader.get(
                        panel='gateway', hours=1, filters={'environment': 'one', 'service': 'control-gateway'}
                    )
                )['series']
            )
            == 3
        )
    finally:
        await reader.close()
        await server.close()


@pytest.mark.asyncio
async def test_history_does_not_call_upstream_when_disabled():
    reader = MetricHistoryReader(DistributedConfig())
    assert await reader.get(panel='gateway', hours=1, filters={}) == {
        'source': 'zhiyan',
        'data_status': 'unavailable',
        'error_code': 'history_not_configured',
    }
    assert reader._client is None


def test_error_ratio_requires_matching_sample_coverage():
    from nexus.distributed.observability.history import _matching_coverage, _total

    denominator = [{'time': 'first', 'value': 10}, {'time': 'second', 'value': 10}]
    numerator = [{'time': 'first', 'value': 0}, {'time': 'second', 'value': None}]
    assert not _matching_coverage(denominator, numerator)
    assert _total([{'time': 'first', 'value': None}]) is None
