"""Bounded, shared reads of native time aggregates from the metrics destination."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import math
import time

from collections import OrderedDict
from datetime import datetime, timedelta

import aiohttp


HISTORY_RANGES = {1: 60, 24: 60, 168: 1800, 720: 3600}
EPOCH = datetime(1970, 1, 1)
UPSTREAM_TIMEOUT_SECONDS = 1.5
logger = logging.getLogger(__name__)


class HistorySourceError(RuntimeError):
    def __init__(self, code, *, http_status=None, upstream_code=None):
        super().__init__(code)
        self.code = code
        self.http_status = http_status
        self.upstream_code = upstream_code

    @property
    def retryable(self):
        return (
            self.http_status == 429
            or self.upstream_code == 50001
            or (self.http_status is not None and 500 <= self.http_status < 600)
        )


def align_time(value: datetime, grain: int) -> datetime:
    step = timedelta(seconds=grain)
    return EPOCH + ((value - EPOCH) // step) * step


def query_parts(start: datetime, end: datetime, grain: int, segment_hours: int):
    """Use non-overlapping complete native buckets, split at stable UTC segment boundaries."""
    if start != align_time(start, grain) or end != align_time(end, grain):
        raise ValueError('history_window_must_align_with_grain')
    result = []
    span = timedelta(hours=segment_hours)
    while start < end:
        boundary = EPOCH + (((start - EPOCH) // span) + 1) * span
        stop = min(boundary, end)
        result.append((start, stop, grain))
        start = stop
    return result


class HistoryTransport:
    def __init__(self, config, credential):
        self.config = config
        self.credential = credential
        self.client = None
        self._slots = asyncio.Semaphore(12)
        self._cache = OrderedDict()
        self._pending = {}
        self._waiters = {}
        self._pacing_lock = asyncio.Lock()
        self._next_request_at = 0.0
        self._request_interval = 60 / getattr(config, 'upstream_requests_per_minute', 750)

    async def query(self, metric, *, now, hours, filters):
        grain = HISTORY_RANGES[hours]
        start = now - timedelta(hours=hours)
        parts = query_parts(start, now, grain, self.config.max_segment_hours)
        unsupported = grain > 60 and metric.endswith('_p95') and not self.config.histogram_aggregation_supported
        if unsupported:
            return {
                'metric': metric,
                'unit': '秒',
                'points': [],
                'granularity_seconds': grain,
                'data_status': 'not_supported',
                'error_code': 'histogram_aggregation_not_supported',
                'failed_intervals': [
                    {
                        'start_at': start.isoformat(),
                        'end_at': now.isoformat(),
                        'error_code': 'histogram_aggregation_not_supported',
                    }
                ],
            }

        async def read(part):
            begin, end, step = part
            try:
                value = await self._segment(metric, begin, end, step, filters, now=now, hours=hours)
                return value, None
            except Exception as exc:
                code = (
                    'history_segment_timeout'
                    if isinstance(exc, TimeoutError)
                    else exc.code
                    if isinstance(exc, HistorySourceError)
                    else 'history_segment_failed'
                )
                logger.warning(
                    'Metric history read failed metric=%s reason=%s http_status=%s upstream_code=%s',
                    metric,
                    code,
                    getattr(exc, 'http_status', None),
                    getattr(exc, 'upstream_code', None),
                )
                return {'unit': '', 'points': []}, {
                    'start_at': begin.isoformat(),
                    'end_at': end.isoformat(),
                    'error_code': code,
                    **(
                        {'http_status': exc.http_status}
                        if isinstance(exc, HistorySourceError) and exc.http_status
                        else {}
                    ),
                    **(
                        {'upstream_code': exc.upstream_code}
                        if isinstance(exc, HistorySourceError) and exc.upstream_code is not None
                        else {}
                    ),
                }

        results = await asyncio.gather(*(read(part) for part in parts))
        points, failures, unit = [], [], ''
        for (begin, end, step), (value, error) in zip(parts, results, strict=True):
            unit = value['unit'] or unit
            if error:
                failures.append(error)
            by_time = {point['time']: point['value'] for point in value['points']}
            stamp = begin
            while stamp < end:
                stop = min(stamp + timedelta(seconds=step), end)
                points.append({
                    'time': stamp.isoformat(),
                    'value': by_time.get(stamp.isoformat()),
                    'bucket_end': stop.isoformat(),
                    'partial_period': step != grain,
                })
                stamp = stop
        # A boundary marker closes the axis; upstream's inclusive end bucket is never counted twice.
        points.append({'time': now.isoformat(), 'value': None, 'boundary': True})
        observed = [point for point in points if point['value'] is not None]
        return {
            'metric': metric,
            'unit': unit,
            'points': points,
            'granularity_seconds': grain,
            'data_status': ('partial' if observed else 'unavailable')
            if failures
            else ('ready' if observed else 'no_samples'),
            'failed_intervals': failures,
            'coverage': {
                'observed_buckets': len(observed),
                'expected_buckets': len(points) - 1,
                'first_sample_at': observed[0]['time'] if observed else None,
                'last_sample_at': observed[-1]['time'] if observed else None,
                'retention_days': self.config.retention_days,
                'retention_verified': self.config.retention_days is not None,
            },
        }

    async def _segment(self, metric, start, end, grain, filters, *, now, hours):
        key = (metric, start, end, grain, tuple(sorted(filters.items())))
        entry = self._cache.get(key)
        if entry and time.monotonic() - entry[0] < entry[1]:
            self._cache.move_to_end(key)
            return copy.deepcopy(entry[2])
        task = self._pending.get(key)
        if task is None:
            if len(self._pending) >= 256:
                raise RuntimeError('history_segment_capacity')
            task = asyncio.create_task(self._read(key, metric, start, end, grain, filters, now, hours))
            self._pending[key] = task
            task.add_done_callback(lambda done: self._finish(key, done))
        self._waiters[task] = self._waiters.get(task, 0) + 1
        try:
            return copy.deepcopy(await asyncio.shield(task))
        finally:
            self._waiters[task] -= 1
            if not self._waiters[task]:
                self._waiters.pop(task)
                if not task.done():
                    if self._pending.get(key) is task:
                        self._pending.pop(key, None)
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)

    async def _read(self, key, metric, start, end, grain, filters, now, hours):
        async with self._slots:
            if self.client is None:
                self.client = aiohttp.ClientSession(trust_env=False, connector=aiohttp.TCPConnector(limit=12))
            body = {
                'proj_id': self.config.project_id,
                'app_mark': self.config.app_mark,
                'env': self.config.environment,
                'sec_lvl_name': self.config.group,
                'metric_name': metric,
                'is_english': 'yes',
                'is_together': True,
                'gap': grain // 60,
                'limit': 1,
                'tag_set': [{'key': key, 'value': [value], 'op': '='} for key, value in filters.items()],
                'begin_time': (start + timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S'),
                'end_time': (end + timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S'),
            }
            document = await self._request(body)
            charts = document.get('data', {}).get('chart_info', [])
            if len(charts) > 1:
                raise RuntimeError('history_response_scope_mismatch')
            chart = charts[0] if charts else {}
            if chart:
                condition = chart.get('cond', {})
                prefix = f'{self.config.app_mark}_{self.config.environment}_{self.config.group}_'
                if condition.get('metric') != [metric] or not condition.get('app_mark', '').startswith(prefix):
                    raise RuntimeError('history_response_scope_mismatch')
            rows = chart.get('detail_data_list', [])
            if len(rows) > math.ceil((end - start).total_seconds() / grain) + 1:
                raise RuntimeError('history_response_too_large')
            points = []
            previous = None
            for row in rows:
                stamp = datetime.strptime(row['time'], '%Y-%m-%d %H:%M') - timedelta(hours=8)
                if stamp < start or stamp > end or (previous is not None and stamp <= previous):
                    raise RuntimeError('history_response_time_mismatch')
                if (stamp - start).total_seconds() % grain:
                    raise RuntimeError('history_response_granularity_mismatch')
                previous = stamp
                if stamp == end:
                    continue
                value = row.get('current')
                if value is not None and (
                    isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                ):
                    raise RuntimeError('history_invalid_sample')
                points.append({'time': stamp.isoformat(), 'value': value})
            result = {'unit': chart.get('unit', ''), 'points': points}
            observed = any(point['value'] is not None for point in points)
            ttl = 1800 if observed and end <= now - timedelta(hours=2) else (55 if hours <= 24 else 240)
            self._cache[key] = (time.monotonic(), ttl, result)
            self._cache.move_to_end(key)
            while len(self._cache) > 512:
                self._cache.popitem(last=False)
            return result

    async def _request(self, body):
        assert self.client is not None
        staffname, token = self.credential.read()
        attempt = 0
        while True:
            # Bound the fan-out when several month-long panels are opened together.
            async with self._pacing_lock:
                await asyncio.sleep(max(0, self._next_request_at - time.monotonic()))
                self._next_request_at = time.monotonic() + self._request_interval
            try:
                async with self.client.post(
                    self.config.endpoint,
                    json=body,
                    headers={'staffname': staffname, 'token': token, 'projectname': self.config.project_name},
                    timeout=aiohttp.ClientTimeout(total=UPSTREAM_TIMEOUT_SECONDS),
                    allow_redirects=False,
                ) as response:
                    if response.status != 200:
                        raise HistorySourceError('history_upstream_http_error', http_status=response.status)
                    content = bytearray()
                    async for chunk in response.content.iter_chunked(65536):
                        content.extend(chunk)
                        if len(content) > 512 * 1024:
                            raise RuntimeError('history_response_too_large')
                    document = json.loads(content)
                if document.get('code') != 0:
                    code = document.get('code')
                    raise HistorySourceError(
                        'history_upstream_rejected', upstream_code=code if isinstance(code, int) else None
                    )
                return document
            except (aiohttp.ClientError, TimeoutError, HistorySourceError) as exc:
                if attempt or (isinstance(exc, HistorySourceError) and not exc.retryable):
                    raise
                attempt += 1
                await asyncio.sleep(0.1)

    def _finish(self, key, task):
        if self._pending.get(key) is task:
            self._pending.pop(key, None)
        if not task.cancelled():
            task.exception()

    async def close(self):
        tasks = list(self._pending.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._cache.clear()
        if self.client is not None:
            await self.client.close()
