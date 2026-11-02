"""End-to-end tests for the 方案 D streaming transform (transcode path)."""

from __future__ import annotations

import json
import tracemalloc
from typing import Any, cast

import pytest

from nexus.distributed.export import results
from nexus.distributed.export.client import CloudApi
from nexus.distributed.export.types import ExportError, TraceResult
from nexus.distributed.export.writer import AtomicJsonlOutput, ResultSpillPool
from tests.unit.distributed.test_export_spill import Api, _record_plan


def _varied_outputs() -> dict[str, Any]:
    return {
        'exe-plain': {'success': True, 'task_output': {'v': 1}, 'processor_results': {'score': 2.5}},
        'exe-unicode': {'success': True, 'task_output': {'note': '中文\U0001f600 ":," \\ ok'}},
        'exe-missing-members': {'success': True},
        'exe-empty-payloads': {'success': True, 'task_output': {}, 'agent_output': [], 'processor_results': ''},
        'exe-falsey-payloads': {'success': True, 'task_output': 0, 'agent_output': False},
        'exe-error-object': {'success': True, 'error': {'type': 'Boom', 'detail': 'x'}},
        'exe-error-null': {'success': True, 'error': None, 'task_output': {'v': 3}},
        'exe-extra-truthy': {'success': True, 'task_output': {'v': 4}, 'extra': {'k': 1}},
        'exe-extra-falsy': {'success': True, 'task_output': {'v': 5}, 'extra': ''},
        'exe-long-error': {'success': False, 'error': 'E' * 10000},
        'exe-cache-ids': {
            'success': True,
            'task_output': {'v': 6},
            'cache_task_id': 'cache-1',
            'cache_task_ids': {'in': 'out'},
        },
    }


def _varied_plans(outputs: dict[str, Any]) -> list[results._RecordPlan]:
    plans = []
    for execution_id, output in outputs.items():
        status = 'failed' if execution_id == 'exe-long-error' else 'succeeded'
        single = results._SingleExport(
            execution_id=execution_id, run_id=execution_id, status=status, retry_count=0, context={}
        )
        plans.append(results._RecordPlan(single=single))
    return plans


async def _export_rows(tmp_path, api, plans, label: str) -> bytes:
    writer = AtomicJsonlOutput(tmp_path / f'out-{label}.jsonl', force=True)
    await writer.open()
    written = await results._write_records(
        writer, cast(Any, api), plans, spill=ResultSpillPool(tmp_path / f'out-{label}.jsonl')
    )
    await writer.close_and_validate(expected_count=written, require_sorted=False)
    await writer.publish()
    return (tmp_path / f'out-{label}.jsonl').read_bytes()


async def test_transcode_rows_match_dict_path_byte_for_byte(tmp_path, monkeypatch):
    outputs = _varied_outputs()
    plans = _varied_plans(outputs)
    monkeypatch.setenv('NEXUS_EXPORT_RESULT_TRANSCODE', '0')
    legacy = await _export_rows(tmp_path, Api(outputs), plans, 'legacy')
    monkeypatch.delenv('NEXUS_EXPORT_RESULT_TRANSCODE', raising=False)
    transcoded = await _export_rows(tmp_path, Api(outputs), plans, 'transcode')
    assert transcoded == legacy
    parsed = [json.loads(line) for line in transcoded.splitlines()]
    assert [row['run_id'] for row in parsed] == list(outputs)


async def test_transcode_member_chunks_keep_transcoder_state_across_boundaries(tmp_path, monkeypatch):
    """Regression: member blocks were transcoded with a fresh transcoder per block.

    A ``_RESULT_SCAN_CHUNK_BYTES`` boundary inside a string value made the next
    block resume in the outside-string state, so every ':' / ',' in the string
    content gained a space — data URLs, paths and code were silently corrupted
    in production exports. The whole member span must go through one
    transcoder so its state machine survives chunk boundaries.
    """
    monkeypatch.delenv('NEXUS_EXPORT_RESULT_TRANSCODE', raising=False)
    monkeypatch.setattr(results, '_RESULT_SCAN_CHUNK_BYTES', 16)
    blob = 'data:image/jpeg;base64,/9j/4AAQSkZJRg' + ' D:\\nexus, EState::Running:' * 4
    outputs = {
        'exe-boundary': {
            'success': True,
            'task_output': {'blob': blob},
            'processor_results': {'url': 'data:image/jpeg;base64,/9j/:,test'},
            'extra': {'tail': blob},
        }
    }
    plans = _varied_plans(outputs)
    monkeypatch.setenv('NEXUS_EXPORT_RESULT_TRANSCODE', '0')
    legacy = await _export_rows(tmp_path, Api(outputs), plans, 'boundary-legacy')
    monkeypatch.delenv('NEXUS_EXPORT_RESULT_TRANSCODE', raising=False)
    transcoded = await _export_rows(tmp_path, Api(outputs), plans, 'boundary-transcode')
    assert transcoded == legacy
    row = json.loads(transcoded)
    assert row['task_output']['blob'] == blob
    assert row['processor_results']['url'] == 'data:image/jpeg;base64,/9j/:,test'
    assert row['extra']['tail'] == blob


async def test_transcode_semantics_match_make_jsonl_record(tmp_path, monkeypatch):
    monkeypatch.delenv('NEXUS_EXPORT_RESULT_TRANSCODE', raising=False)
    outputs = _varied_outputs()
    raw = await _export_rows(tmp_path, Api(outputs), _varied_plans(outputs), 'x')
    rows = [json.loads(line) for line in raw.splitlines()]
    by_id = {row['run_id']: row for row in rows}
    assert by_id['exe-error-object']['error'] == {'type': 'Boom', 'detail': 'x'}
    assert by_id['exe-error-null']['error'] is None
    failed = by_id['exe-long-error']
    assert failed['status'] == 'failed' and failed['error'] == 'E' * 10000
    cache_row = by_id['exe-cache-ids']
    assert cache_row['metadata']['_nexus']['cache_task_id'] == 'cache-1'
    assert cache_row['metadata']['_nexus']['cache_task_ids'] == {'in': 'out'}
    assert by_id['exe-extra-truthy']['extra'] == {'k': 1}
    assert 'extra' not in by_id['exe-extra-falsy']
    for name in ('task_output', 'agent_output', 'processor_results'):
        assert by_id['exe-missing-members'][name] is None
    assert by_id['exe-falsey-payloads']['task_output'] == 0
    assert by_id['exe-falsey-payloads']['agent_output'] is False


async def test_transcode_nan_body_aborts(tmp_path, monkeypatch):
    monkeypatch.delenv('NEXUS_EXPORT_RESULT_TRANSCODE', raising=False)
    outputs = {'exe-nan': {'success': True, 'task_output': {'score': float('nan')}}}
    api = Api(outputs)
    writer = AtomicJsonlOutput(tmp_path / 'out.jsonl', force=True)
    await writer.open()
    plans = [_record_plan('exe-nan', outputs['exe-nan'])]
    with pytest.raises(ExportError, match='NaN or Infinity'):
        await results._write_records(writer, cast(Any, api), plans, spill=ResultSpillPool(tmp_path / 'out.jsonl'))


@pytest.mark.parametrize(
    'label, body',
    [
        ('deep-nesting', b'{"success":true,"task_output":' + b'[' * 11 + b']' * 11 + b'}'),
        ('duplicate-member', b'{"success":true,"task_output":{"v":1},"task_output":{"v":2}}'),
        ('paired-surrogate', b'{"success":true,"task_output":{"note":"a\\ud83d\\ude00b"}}'),
    ],
)
async def test_transcode_divergent_bodies_fall_back_to_identical_rows(tmp_path, monkeypatch, label, body):
    if label == 'deep-nesting':
        # The legacy dict path cannot encode very deep nesting either; shrink the
        # scan limit so the fallback triggers at a depth the legacy path handles.
        monkeypatch.setattr(results, '_RESULT_SCAN_MAX_DEPTH', 10)

    class _RawBodyApi(Api):
        async def get_result_to_file(self, execution_id, fileobj, chunk_size=1 << 20, scan=None):
            fileobj.write(body)
            if scan is not None:
                scan.feed(body)
            return len(body)

    outputs = {'exe-x': {'success': True, 'task_output': {'v': 1}}}
    plans = [_record_plan('exe-x', outputs['exe-x'])]
    # Legacy run: the same divergent body consumed by the orjson dict path.
    monkeypatch.setenv('NEXUS_EXPORT_RESULT_TRANSCODE', '0')
    legacy = await _export_rows(tmp_path, _RawBodyApi(outputs), plans, f'legacy-{label}')
    # Transcode run: scan detects the divergence and falls back to that same path.
    monkeypatch.delenv('NEXUS_EXPORT_RESULT_TRANSCODE', raising=False)
    transcoded = await _export_rows(tmp_path, _RawBodyApi(outputs), plans, f'transcode-{label}')
    assert transcoded == legacy


async def test_transcode_memory_stays_bounded_on_giant_body(tmp_path, monkeypatch):
    monkeypatch.delenv('NEXUS_EXPORT_RESULT_TRANSCODE', raising=False)
    blob_mb = 100
    body_header = b'{"success": true, "task_output": {"blob": "'
    body_tail = b'"}}'

    class _GiantApi(Api):
        async def get_result_to_file(self, execution_id, fileobj, chunk_size=1 << 20, scan=None):
            if scan is not None:
                scan.reset()
            fileobj.write(body_header)
            if scan is not None:
                scan.feed(body_header)
            block = b'x' * (1 << 20)
            for _ in range(blob_mb):
                fileobj.write(block)
                if scan is not None:
                    scan.feed(block)
            fileobj.write(body_tail)
            if scan is not None:
                scan.feed(body_tail)
            return len(body_header) + blob_mb * (1 << 20) + len(body_tail)

    api = _GiantApi({})
    writer = AtomicJsonlOutput(tmp_path / 'out.jsonl', force=True)
    await writer.open()
    single = results._SingleExport(
        execution_id='exe-giant', run_id='exe-giant', status='succeeded', retry_count=0, context={}
    )
    plans = [results._RecordPlan(single=single)]
    tracemalloc.start()
    try:
        written = await results._write_records(
            writer, cast(Any, api), plans, spill=ResultSpillPool(tmp_path / 'out.jsonl')
        )
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert written == 1
    assert peak < 50 << 20
    await writer.close_and_validate(expected_count=1, require_sorted=False)
    await writer.publish()
    row = json.loads((tmp_path / 'out.jsonl').read_text())
    assert row['task_output']['blob'] == 'x' * (blob_mb << 20)


def test_result_transcode_default_on_with_rollback_override(monkeypatch):
    from nexus.distributed.export.results import _result_transcode_enabled

    monkeypatch.delenv('NEXUS_EXPORT_RESULT_TRANSCODE', raising=False)
    assert _result_transcode_enabled()
    monkeypatch.setenv('NEXUS_EXPORT_RESULT_TRANSCODE', '0')
    assert not _result_transcode_enabled()
    monkeypatch.setenv('NEXUS_EXPORT_RESULT_TRANSCODE', '1')
    assert _result_transcode_enabled()


def test_result_concurrency_env_override_and_clamp(monkeypatch):
    from nexus.distributed.export.results import _result_concurrency

    monkeypatch.delenv('NEXUS_EXPORT_RESULT_CONCURRENCY', raising=False)
    assert _result_concurrency() == 4
    for env, want in (('6', 6), ('1', 1), ('8', 8), ('99', 8), ('0', 1), ('-3', 1), ('junk', 4)):
        monkeypatch.setenv('NEXUS_EXPORT_RESULT_CONCURRENCY', env)
        assert _result_concurrency() == want, env


# ---------------------------------------------------------------------------
# TDSQL-fallback 管线（trace_reader 配置时）也走流式转码
# ---------------------------------------------------------------------------


class _FakeTraceReader:
    """Duck-typed TdsqlTraceReader：output 缺失 → API fallback。"""

    def __init__(self, outputs=None):
        self._outputs = outputs or {}

    async def read_many(self, execution_ids, parent_execution_id=None, include_timestamps=False):
        return {eid: TraceResult(output=self._outputs.get(eid)) for eid in execution_ids}


def _tdsql_outputs() -> dict[str, Any]:
    return {
        'exe-t1': {'success': True, 'task_output': {'v': 1}, 'processor_results': {'score': 2}},
        'exe-t2': {'success': True, 'task_output': {'note': '中文\U0001f600 ":,"'}},
        'exe-t3': {'success': True, 'error': {'type': 'E'}, 'task_output': {'v': 3}},
    }


async def test_tdsql_fallback_streaming_matches_dict_path_byte_for_byte(tmp_path, monkeypatch):
    from nexus.distributed.export.types import TraceResult

    outputs = _tdsql_outputs()
    plans_ids = list(outputs)

    async def run(env):
        if env == 'off':
            monkeypatch.setenv('NEXUS_EXPORT_RESULT_TRANSCODE', '0')
        else:
            monkeypatch.delenv('NEXUS_EXPORT_RESULT_TRANSCODE', raising=False)
        reader = _FakeTraceReader()
        api = Api(outputs)
        path = tmp_path / f'tdsql-{env}.jsonl'
        await results.export_execution(
            api=api,
            execution_id=plans_ids,
            output=path,
            force=True,
            trace_reader=reader,
        )
        return path.read_bytes()

    legacy = await run('off')
    transcoded = await run('on')
    assert transcoded == legacy
    parsed = [json.loads(line) for line in transcoded.splitlines()]
    assert [row['run_id'] for row in parsed] == plans_ids


async def test_tdsql_hit_streams_from_dict_path_unchanged(tmp_path, monkeypatch):
    from nexus.distributed.export.types import TraceResult

    monkeypatch.delenv('NEXUS_EXPORT_RESULT_TRANSCODE', raising=False)
    outputs = _tdsql_outputs()
    tdsql_output = {'success': True, 'task_output': {'from': 'tdsql'}}
    reader = _FakeTraceReader({'exe-t1': tdsql_output})

    async def export(label):
        api = Api(outputs)
        path = tmp_path / f'out-{label}.jsonl'
        await results.export_execution(
            api=api,
            execution_id=list(outputs),
            output=path,
            force=True,
            trace_reader=reader,
        )
        return [json.loads(line) for line in path.read_text().splitlines()]

    monkeypatch.setenv('NEXUS_EXPORT_RESULT_TRANSCODE', '0')
    legacy = await export('legacy')
    monkeypatch.delenv('NEXUS_EXPORT_RESULT_TRANSCODE', raising=False)
    current = await export('current')
    by_id_current = {row['run_id']: row for row in current}
    # TDSQL 命中的记录来自 DB dict（dict 路径，两环境一致）；其余走流式转码
    assert by_id_current['exe-t1']['task_output'] == {'from': 'tdsql'}
    del legacy


async def test_oversized_cache_id_member_streams_identically(tmp_path, monkeypatch):
    # >1MiB 的 cache_task_id 字符串成员：曾经回退 orjson，现在按 span 解析投影，
    # 输出与 dict 路径逐字节一致且不再触发记录级回退。
    big_id = 'cache-' + 'x' * (1 << 20)
    outputs = {'exe-big': {'success': True, 'input_cache_task_id': big_id, 'task_output': {'v': 1}}}

    async def run(env):
        if env == 'off':
            monkeypatch.setenv('NEXUS_EXPORT_RESULT_TRANSCODE', '0')
        else:
            monkeypatch.delenv('NEXUS_EXPORT_RESULT_TRANSCODE', raising=False)
        api = Api(outputs)
        path = tmp_path / f'big-cache-{env}.jsonl'
        await results.export_execution(api=api, execution_id=list(outputs), output=path, force=True)
        return path.read_bytes()

    legacy = await run('off')
    transcoded = await run('on')
    assert transcoded == legacy
    row = json.loads(transcoded)
    assert row['metadata']['_nexus']['input_cache_task_id'] == big_id


# ---------------------------------------------------------------------------
# #5：TDSQL-hit 走字节管线（event_data 不整文档 loads）
# ---------------------------------------------------------------------------


class _FakeTdsqlConnection:
    """按 SQL 前缀路由结果集的最小 aiomysql 替身。"""

    def __init__(self, event_data: str):
        self._event_data = event_data

    class Cursor:
        def __init__(self, event_data):
            self._event_data = event_data
            self._rows: list[tuple] = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def execute(self, sql, params):
            if sql.startswith('SELECT 1'):
                self._rows = []  # 无分区列 → 关闭分区裁剪
            elif sql.startswith('SELECT DISTINCT run_span_id'):
                self._rows = [('s1',)]
            elif sql.startswith('SELECT run_id, MIN'):
                self._rows = [('exe-t1', 1_000_000, 2_000_000)]
            elif sql.startswith('SELECT record_id, time_unix_nano'):
                self._rows = [(7, 1_000_000)]
            elif sql.startswith('SELECT event_data'):
                self._rows = [(self._event_data,)]
            else:
                self._rows = []

        def fetchall(self):
            return self._rows

    def cursor(self):
        return self.Cursor(self._event_data)

    def close(self):
        return None


async def test_tdsql_hit_streams_from_event_data_without_full_parse(tmp_path, monkeypatch):
    from nexus.distributed.export.results import TdsqlTraceReader

    outputs = {'exe-t1': {'success': True, 'task_output': {'note': 'a\ud83d\ude00b中文'}, 'v': [1, None]}}
    event_data = json.dumps(
        {
            'events': [
                {'name': 'run.event', 'attributes': {'event_type': 'update'}},
                {
                    'name': 'run.event',
                    'attributes': {
                        'event_type': 'run.output',
                        'event_data': {'output': outputs['exe-t1'], 'attempt': 2},
                    },
                },
                {'name': 'other.event'},
            ]
        },
        default=str,  # 写入端真实格式：宽松分隔符 + ensure_ascii + default=str
    )

    class _TdsqlApi(Api, CloudApi):
        """isinstance(api, CloudApi) enables the spill pipeline."""

    async def run(env):
        if env == 'off':
            monkeypatch.setenv('NEXUS_EXPORT_RESULT_TRANSCODE', '0')
        else:
            monkeypatch.delenv('NEXUS_EXPORT_RESULT_TRANSCODE', raising=False)
        if env == 'on':
            # 哨兵：字节管线下不得整文档 loads
            def _no_decode(value):
                raise AssertionError('event_data must not be fully parsed in streaming mode')

            monkeypatch.setattr(results, '_decode_event_data', _no_decode)
        connection = _FakeTdsqlConnection(event_data)

        async def connect(**kwargs):
            return connection

        reader = TdsqlTraceReader(
            host='db.test',
            port=3306,
            user='reader',
            password='unused',
            database='trace',
            table='trajectory_span_events_pr_' + 'a' * 32,
            connect=connect,
            transient_errors=(),
        )
        api = _TdsqlApi(outputs)
        path = tmp_path / f'tdsql-{env}.jsonl'
        await results.export_execution(api=api, execution_id='exe-t1', output=path, force=True, trace_reader=reader)
        return path.read_bytes()

    legacy = await run('off')
    transcoded = await run('on')
    assert transcoded == legacy
    row = json.loads(transcoded)
    assert row['run_id'] == 'exe-t1'
    assert row['task_output'] == {'note': 'a😀b中文'}
    assert row['start_time'] is not None  # 时间戳来自 DB MIN/MAX
