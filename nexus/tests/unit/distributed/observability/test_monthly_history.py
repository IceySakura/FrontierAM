from __future__ import annotations

import asyncio
import time

from datetime import datetime, timedelta

import pytest

from aiohttp import web
from aiohttp.test_utils import TestServer

from nexus.distributed.config import DistributedConfig
from nexus.distributed.observability.history import MetricHistoryReader
from nexus.distributed.observability.history_transport import HISTORY_RANGES, query_parts


NOW = datetime(2026, 9, 14, 9, 41)


@pytest.fixture
async def destination(monkeypatch):
    from nexus.distributed.observability import history

    clock = [NOW]
    monkeypatch.setattr(history, 'utcnow', lambda: clock[0])
    state = {'requests': [], 'active': 0, 'maximum': 0, 'fail_begin': None, 'times': [], 'delay': 0.005, 'transient': []}

    async def handle(request):
        body = await request.json()
        state['requests'].append(body)
        state['times'].append(time.monotonic())
        state['active'] += 1
        state['maximum'] = max(state['maximum'], state['active'])
        try:
            await asyncio.sleep(state['delay'])
            if state['transient']:
                failure = state['transient'].pop(0)
                return web.json_response({'code': 50001 if failure == 200 else 1}, status=failure)
            if body['begin_time'] == state['fail_begin']:
                return web.json_response({'code': 1}, status=503)
            begin = datetime.strptime(body['begin_time'], '%Y-%m-%d %H:%M:%S')
            end = datetime.strptime(body['end_time'], '%Y-%m-%d %H:%M:%S')
            assert end - begin <= timedelta(days=7)
            step = timedelta(minutes=body['gap'])
            metric = body['metric_name']
            value = 0.125 if metric.endswith('_p95') else body['gap'] * (2 if metric == 'gateway_requests' else 1)
            points = []
            while begin <= end:
                points.append({'time': begin.strftime('%Y-%m-%d %H:%M'), 'current': 999999 if begin == end else value})
                begin += step
            return web.json_response(
                {
                    'code': 0,
                    'data': {
                        'chart_info': [
                            {
                                'cond': {'metric': [metric], 'app_mark': 'test_dev_nexus_cloud_scope'},
                                'unit': 'seconds' if metric.endswith('_p95') else 'count',
                                'detail_data_list': points,
                            }
                        ]
                    },
                }
            )
        finally:
            state['active'] -= 1

    app = web.Application()
    app.router.add_post('/query', handle)
    server = TestServer(app)
    await server.start_server()
    config = DistributedConfig(
        metrics_history={
            'enabled': True,
            'endpoint': str(server.make_url('/query')),
            'project_name': 'test',
            'project_id': 1,
            'app_mark': 'test',
            'environment': 'dev',
        },
        secrets={'metrics_query': {'username': 'reader', 'token': 'test'}},
    )
    yield config, state, clock
    await server.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('hours', [168, 720])
async def test_long_range_native_aggregation_has_complete_axis_no_boundary_double_counts(destination, hours):
    config, state, _ = destination
    reader = MetricHistoryReader(config)
    try:
        result = await reader.get(panel='gateway', hours=hours, filters={'environment': 'chosen'})
        assert result['data_status'] == 'ready'
        assert result['granularity_seconds'] == HISTORY_RANGES[hours]
        assert result['completed'] == hours * 60 * 2
        assert result['failed'] == hours * 60
        assert result['failure_ratio'] == 0.5
        assert result['requested_window_end'] == NOW.isoformat()
        assert result['window_end'].endswith('09:30:00' if hours == 168 else '09:00:00')
        for series in result['series']:
            assert len(series['points']) == hours * 3600 // HISTORY_RANGES[hours] + 1
            assert series['points'][0]['time'] == result['window_start']
            assert series['points'][-1]['time'] == result['window_end']
            assert series['points'][-1]['value'] is None
        assert all(point['value'] == 0.125 for point in result['series'][2]['points'][:-1])
        assert state['maximum'] <= 12
        assert all(body['gap'] == HISTORY_RANGES[hours] // 60 for body in state['requests'])
        assert all(
            body['tag_set'] == [{'key': 'environment', 'value': ['chosen'], 'op': '='}] for body in state['requests']
        )
    finally:
        await reader.close()


@pytest.mark.asyncio
async def test_failed_segment_is_visible_and_retry_reuses_only_successful_segments(destination):
    config, state, _ = destination
    reader = MetricHistoryReader(config)
    end = NOW.replace(minute=0)
    parts = query_parts(end - timedelta(days=30), end, 3600, 168)
    state['fail_begin'] = (parts[1][0] + timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S')
    try:
        partial = await reader.get(panel='gateway', hours=720, filters={})
        assert partial['data_status'] == 'partial'
        assert partial['failure_ratio'] is None
        assert len(partial['failed_intervals']) == 3
        assert partial['completed'] < 720 * 60 * 2
        previous = len(state['requests'])
        state['fail_begin'] = None
        complete = await reader.get(panel='gateway', hours=720, filters={})
        assert complete['data_status'] == 'ready'
        assert complete['completed'] == 720 * 60 * 2
        assert len(state['requests']) - previous == 3
    finally:
        await reader.close()


@pytest.mark.asyncio
async def test_aligned_windows_share_closed_segments_but_refresh_new_boundaries(destination):
    config, state, clock = destination
    reader = MetricHistoryReader(config)
    try:
        values = await asyncio.gather(*(reader.get(panel='gateway', hours=720, filters={}) for _ in range(10)))
        assert all(value == values[0] for value in values)
        first = len(state['requests'])
        assert (
            first == len(query_parts(NOW.replace(minute=0) - timedelta(days=30), NOW.replace(minute=0), 3600, 168)) * 3
        )
        clock[0] += timedelta(hours=1)
        second = await reader.get(panel='gateway', hours=720, filters={})
        assert second['window_end'] != values[0]['window_end']
        assert len(state['requests']) - first == 6
    finally:
        await reader.close()


@pytest.mark.asyncio
async def test_unsupported_long_histogram_does_not_invent_p95_or_hide_counter_ratio(destination):
    config, state, _ = destination
    config = config.model_copy(
        update={'metrics_history': config.metrics_history.model_copy(update={'histogram_aggregation_supported': False})}
    )
    reader = MetricHistoryReader(config)
    try:
        result = await reader.get(panel='gateway', hours=720, filters={})
        assert result['data_status'] == 'partial'
        assert result['series'][2]['data_status'] == 'not_supported'
        assert result['series'][2]['points'] == []
        assert result['failure_ratio'] == 0.5
        assert not any(body['metric_name'].endswith('_p95') for body in state['requests'])
        with pytest.raises(ValueError, match='invalid_history_query'):
            await reader.get(panel='gateway', hours=744, filters={})
    finally:
        await reader.close()


def test_segment_boundaries_are_stable_and_unaligned_native_windows_are_rejected():
    end = NOW.replace(minute=0)
    parts = query_parts(end - timedelta(days=30), end, 3600, 168)
    assert all(left[1] == right[0] for left, right in zip(parts[:-1], parts[1:], strict=True))
    assert sum((stop - start for start, stop, _ in parts), timedelta()) == timedelta(days=30)
    with pytest.raises(ValueError, match='align'):
        query_parts(NOW - timedelta(days=30), NOW, 3600, 168)


@pytest.mark.asyncio
@pytest.mark.parametrize('status', [429, 503, 200])
async def test_upstream_transient_rejection_retries_once_within_the_read_budget(destination, status):
    config, state, _ = destination
    state['transient'] = [status]
    reader = MetricHistoryReader(config)
    try:
        result = await reader.get(panel='gateway_requests', hours=1, filters={})
        assert result['data_status'] == 'ready'
        assert len(state['requests']) == 2
        assert not result['failed_intervals']
    finally:
        await reader.close()


@pytest.mark.asyncio
async def test_one_second_source_response_does_not_fail_the_short_range(destination):
    config, state, _ = destination
    state['delay'] = 1.0
    reader = MetricHistoryReader(config)
    try:
        result = await reader.get(panel='gateway_requests', hours=1, filters={})
        assert result['data_status'] == 'ready'
        assert len(state['requests']) == 1
    finally:
        await reader.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('now, request_count', [(NOW, 45), (datetime(2026, 9, 3, 9, 41), 54)])
async def test_concurrent_month_queries_share_one_upstream_request_rate(destination, now, request_count):
    config, state, clock = destination
    clock[0] = now
    state['delay'] = 0.4
    reader = MetricHistoryReader(config)
    try:
        results = await asyncio.gather(*(reader.get(panel=panel, hours=720, filters={}) for panel in ('gateway', 'terminal', 'delivery')))
        assert all(result['data_status'] == 'ready' for result in results)
        assert len(state['times']) == request_count
        # Pacing is applied before the client sends, but `times` is stamped when the
        # in-process handler starts, so every gap also carries event-loop scheduling
        # jitter. That jitter is larger than the slack in a single interval, so assert
        # what the limiter actually guarantees -- the rate across all requests -- plus
        # a typical gap, instead of requiring every adjacent pair to clear the bar.
        interval = 60 / config.metrics_history.upstream_requests_per_minute
        gaps = sorted(b - a for a, b in zip(state['times'], state['times'][1:]))
        assert state['times'][-1] - state['times'][0] >= (request_count - 1) * interval * 0.9
        assert gaps[len(gaps) // 2] >= interval * 0.9
    finally:
        await reader.close()
