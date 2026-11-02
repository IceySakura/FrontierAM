from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from nexus.runner.trace.tdsql.sink_source import TDSQLBoundlessTraceSink, TDSQLTraceSink


ENV_NAME = 'NEXUS_TRAJ_TDSQL_CLOSE_FLUSH_TIMEOUT_SECONDS'


@pytest.fixture(params=[TDSQLTraceSink, TDSQLBoundlessTraceSink])
def sink_class(request, monkeypatch):
    monkeypatch.delenv(ENV_NAME, raising=False)
    return request.param


@pytest.mark.parametrize(
    ('env', 'options', 'expected'),
    [
        (None, {}, 1200),
        ('1800', {}, 1800),
        ('1800', {'close_flush_timeout_seconds': 90}, 90),
        ('invalid', {'close_flush_timeout_seconds': 90}, 90),
        ('0.5', {}, 0.5),
    ],
)
@pytest.mark.asyncio
async def test_close_uses_resolved_flush_timeout(sink_class, monkeypatch, env, options, expected):
    if env is not None:
        monkeypatch.setenv(ENV_NAME, env)
    sink = sink_class(host='mysql.example', user='runner', db='traces', **options)
    writer = Mock()
    writer.wait_until_settled = AsyncMock()
    writer.consume_run_failure.return_value = None
    sink._writer = writer
    sink._last_seq = 17
    sink._run_span_id = 'span-1'

    await sink.close()

    assert sink.close_flush_timeout_seconds == expected
    writer.wait_until_settled.assert_awaited_once_with(17, timeout=expected)
    writer.consume_run_failure.assert_called_once_with('span-1')
    assert sink.metadata()['status'] == 'succeeded'


@pytest.mark.parametrize('value', [0, -1, float('inf'), float('-inf'), float('nan'), True, '600', None])
def test_sink_rejects_invalid_flush_timeout(sink_class, value):
    with pytest.raises(ValueError, match='close_flush_timeout_seconds must be a positive finite number'):
        sink_class(host='mysql.example', user='runner', db='traces', close_flush_timeout_seconds=value)


@pytest.mark.parametrize('value', ['', 'invalid', '0', '-1', 'inf', 'nan'])
def test_sink_rejects_invalid_flush_timeout_environment(sink_class, monkeypatch, value):
    monkeypatch.setenv(ENV_NAME, value)
    with pytest.raises(ValueError, match='must be a positive finite number'):
        sink_class(host='mysql.example', user='runner', db='traces')


@pytest.mark.asyncio
async def test_close_timeout_reports_effective_duration_and_remains_a_failure(sink_class):
    sink = sink_class(host='mysql.example', user='runner', db='traces', close_flush_timeout_seconds=12.5)
    writer = Mock()
    writer.wait_until_settled = AsyncMock(side_effect=TimeoutError)
    writer.settled_through = 4
    sink._writer = writer
    sink._last_seq = 17
    sink._run_id = 'run-1'

    with pytest.raises(RuntimeError, match=r'did not settle within 12\.5s for run run-1'):
        await sink.close()

    writer.wait_until_settled.assert_awaited_once_with(17, timeout=12.5)
    writer.consume_run_failure.assert_not_called()
    assert sink.metadata()['status'] == 'upload_timeout'
