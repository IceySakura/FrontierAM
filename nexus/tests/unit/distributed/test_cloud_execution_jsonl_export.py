from __future__ import annotations

import asyncio
import copy
import io
import json

from pathlib import Path
from typing import Any, cast

import pytest

from aiohttp import ClientSession, web
from aiohttp.test_utils import TestServer

from nexus.distributed import export as module
from nexus.distributed.export import cli as export_cli
from nexus.distributed.export import results as export_results
from nexus.distributed.export_types import TraceStreamKey


def test_legacy_main_sanitizes_unexpected_exception_text(monkeypatch, capsys, tmp_path: Path) -> None:
    def fail_with_secret(_args) -> int:
        raise RuntimeError('request included sentinel-secret')

    monkeypatch.setattr(export_cli, 'run_cloud_export', fail_with_secret)

    assert (
        module.main([
            '--execution-id',
            'exe-1',
            '--output',
            str(tmp_path / 'results.jsonl'),
        ])
        == 1
    )
    captured = capsys.readouterr()
    assert captured.out == ''
    assert captured.err == 'Export failed: RuntimeError\n'
    assert 'sentinel-secret' not in captured.err


def test_trace_stream_key_normalizes_legacy_scope_and_exposes_tuple_identity() -> None:
    key = TraceStreamKey(
        storage_scope='batch_principal_v1',
        table='trajectory_span_events_pr_0123456789abcdef0123456789abcdef',
        run_id='run:one',
        run_span_id='span:two',
    )

    assert key.storage_scope == 'principal_v1'
    assert key.as_tuple() == (
        'principal_v1',
        'trajectory_span_events_pr_0123456789abcdef0123456789abcdef',
        'run:one',
        'span:two',
    )
    assert {key} == {key}


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('storage_scope', 'shared_v1'),
        ('table', 'trajectory_span_events_pr_deadbeef; DROP TABLE events'),
        ('run_id', ''),
        ('run_span_id', ''),
    ],
)
def test_trace_stream_key_rejects_invalid_binding(field: str, value: str) -> None:
    values = {
        'storage_scope': 'principal_v1',
        'table': 'trajectory_span_events_pr_0123456789abcdef0123456789abcdef',
        'run_id': 'run-1',
        'run_span_id': 'span-1',
    }
    values[field] = value

    with pytest.raises(ValueError, match=field.replace('_', ' ')):
        TraceStreamKey(**values)


@pytest.mark.parametrize(
    ('distributed', 'local'),
    [
        ('succeeded', 'success'),
        ('failed', 'failed'),
        ('lost', 'failed'),
        ('expired', 'timeout'),
        ('killed', 'cancelled'),
        ('cancelled', 'cancelled'),
    ],
)
def test_normalize_status_uses_existing_jsonl_vocabulary(distributed: str, local: str) -> None:

    assert module.normalize_status(distributed) == local


def test_make_jsonl_record_preserves_run_output_and_item_metadata() -> None:

    record = module.make_jsonl_record(
        run_id='item-002',
        distributed_status='succeeded',
        start_time='2026-08-07T01:02:03+00:00',
        end_time='2026-08-07T01:02:09+00:00',
        retry_count=2,
        fallback_error={'type': 'ignored'},
        output={
            'success': True,
            'error': None,
            'task_output': {'answer': 42},
            'agent_output': {'text': 'done'},
            'processor_results': {'score': {'value': 1}},
            'cache_stats': {'hits': 3},
        },
        metadata={'dataset_row': 9},
    )

    assert record == {
        'run_id': 'item-002',
        'status': 'success',
        'start_time': '2026-08-07T01:02:03+00:00',
        'end_time': '2026-08-07T01:02:09+00:00',
        'retry_count': 2,
        'error': None,
        'task_output': {'answer': 42},
        'agent_output': {'text': 'done'},
        'processor_results': {'score': {'value': 1}},
        'metadata': {'dataset_row': 9},
    }


@pytest.mark.parametrize('status', ['succeeded', 'failed'])
@pytest.mark.parametrize(
    'ids',
    [
        {},
        {'input_cache_task_id': 'input-id', 'cache_task_id': 'effective-id'},
        {
            'input_cache_task_ids': {'main': 'main-input', 'judge': 'judge-input'},
            'cache_task_ids': {'main': 'main-effective', 'judge': 'judge-effective'},
        },
        {'input_cache_task_id': 'only-input'},
        {'cache_task_id': 'only-effective'},
        {'input_cache_task_ids': {'judge': 'judge-input'}, 'cache_task_id': 'only-effective'},
        {'input_cache_task_id': 'only-input', 'cache_task_ids': {'main': 'main-effective'}},
    ],
)
def test_make_jsonl_record_exports_recorded_cache_ids_without_changing_shape(status, ids) -> None:
    metadata = {'cache_task_id': 'legacy-generator-id', '_nexus': {'original_run_id': 'source-1'}}
    output = {
        'success': status == 'succeeded',
        'task_output': {'answer': 42},
        'agent_output': {'text': 'done'},
        'processor_results': {'score': 1},
        **ids,
    }
    original_metadata = copy.deepcopy(metadata)
    original_output = copy.deepcopy(output)
    record = module.make_jsonl_record(
        run_id='source-1',
        distributed_status=status,
        start_time=None,
        end_time=None,
        retry_count=0,
        fallback_error=None,
        output=output,
        metadata=metadata,
    )

    assert list(record) == [
        'run_id',
        'status',
        'start_time',
        'end_time',
        'retry_count',
        'error',
        'task_output',
        'agent_output',
        'processor_results',
        'metadata',
    ]
    assert record['metadata'] == {
        'cache_task_id': 'legacy-generator-id',
        '_nexus': {'original_run_id': 'source-1', **ids},
    }
    for name in ('task_output', 'agent_output', 'processor_results'):
        assert record[name] == output[name]
    assert metadata == original_metadata
    assert output == original_output
    if 'cache_task_ids' in ids:
        record['metadata']['_nexus']['cache_task_ids']['new-client'] = 'new-id'
        assert output == original_output


@pytest.mark.parametrize('metadata', [None, {}, {'_nexus': {'source': 'generator'}}])
def test_missing_cache_ids_do_not_add_empty_metadata(metadata) -> None:
    projected = export_results._project_cache_task_ids(
        output={
            'input_cache_task_id': None,
            'input_cache_task_ids': {},
            'cache_task_id': '',
            'cache_task_ids': {},
        },
        metadata=metadata,
        run_id='exe-1',
    )

    assert projected == metadata


def test_cache_id_metadata_conflicts_keep_existing_values_and_merge_client_keys(caplog) -> None:
    metadata = {
        'cache_task_id': 'legacy-unclassified',
        '_nexus': {
            'child_execution_id': 'child-1',
            'source': 'generator',
            'input_cache_task_id': 'existing-input',
            'input_cache_task_ids': {'main': 'same-input', 'old-client': 'old-input'},
            'cache_task_id': None,
            'cache_task_ids': {'main': 'existing-effective', 'old-client': 'old-effective'},
        },
    }
    original = copy.deepcopy(metadata)
    projected = export_results._project_cache_task_ids(
        output={
            'input_cache_task_id': 'new-input',
            'input_cache_task_ids': {'main': 'same-input', 'judge': 'judge-input'},
            'cache_task_id': 'new-effective',
            'cache_task_ids': {'main': 'new-effective', 'judge': 'judge-effective'},
        },
        metadata=metadata,
        run_id='dataset-run-id',
    )

    assert projected == {
        'cache_task_id': 'legacy-unclassified',
        '_nexus': {
            **cast(dict[str, Any], original['_nexus']),
            'input_cache_task_ids': {'main': 'same-input', 'old-client': 'old-input', 'judge': 'judge-input'},
            'cache_task_ids': {
                'main': 'existing-effective',
                'old-client': 'old-effective',
                'judge': 'judge-effective',
            },
        },
    }
    assert metadata == original
    assert len(caplog.records) == 3
    assert all(
        'execution=child-1' in record.message and 'reason=metadata_conflict' in record.message
        for record in caplog.records
    )
    assert 'client=main' in caplog.text
    assert 'existing-input' not in caplog.text
    assert 'new-effective' not in caplog.text
    projected['_nexus']['cache_task_ids']['old-client'] = 'changed'
    assert metadata == original


@pytest.mark.parametrize('field', ['input_cache_task_id', 'cache_task_id', 'input_cache_task_ids', 'cache_task_ids'])
def test_matching_cache_id_metadata_does_not_warn(field, caplog) -> None:
    value = {'main': 'recorded'} if field.endswith('_ids') else 'recorded'
    metadata = {'_nexus': {field: value}}

    assert (
        export_results._project_cache_task_ids(
            output={field: value},
            metadata=metadata,
            run_id='exe-1',
        )
        == metadata
    )
    assert not caplog.records


@pytest.mark.parametrize(
    ('metadata', 'output', 'reason'),
    [
        ('sentinel-secret', {'cache_task_id': 'id'}, 'invalid_metadata'),
        ({'_nexus': 'sentinel-secret'}, {'cache_task_id': 'id'}, 'invalid_nexus_metadata'),
        (None, {'cache_task_id': {'api_key': 'sentinel-secret'}}, 'invalid_cache_task_id'),
        (None, {'cache_task_ids': ['sentinel-secret']}, 'invalid_cache_task_id'),
        (None, {'cache_task_ids': {'main': {'api_key': 'sentinel-secret'}}}, 'invalid_cache_task_id'),
        ({'_nexus': {'cache_task_ids': 'sentinel-secret'}}, {'cache_task_ids': {'main': 'id'}}, 'metadata_conflict'),
    ],
)
def test_invalid_cache_metadata_is_preserved_without_exposing_values(metadata, output, reason, caplog) -> None:
    assert export_results._project_cache_task_ids(output=output, metadata=metadata, run_id='exe-1') == metadata
    assert f'reason={reason}' in caplog.text
    assert 'sentinel-secret' not in caplog.text


def test_make_jsonl_record_allows_failed_execution_without_output() -> None:

    record = module.make_jsonl_record(
        run_id='item-failed',
        distributed_status='failed',
        start_time=None,
        end_time='2026-08-07T01:02:09+00:00',
        retry_count=0,
        fallback_error={'type': 'StartupFailed', 'message': 'runtime unavailable'},
        output=None,
        metadata={},
    )

    assert record['error'] == {'type': 'StartupFailed', 'message': 'runtime unavailable'}
    assert record['task_output'] is None
    assert record['agent_output'] is None
    assert record['processor_results'] is None


def test_make_jsonl_record_rejects_success_without_output() -> None:

    with pytest.raises(module.ExportError, match='successful execution has no RunOutput'):
        module.make_jsonl_record(
            run_id='item-missing',
            distributed_status='succeeded',
            start_time=None,
            end_time=None,
            retry_count=0,
            fallback_error=None,
            output=None,
            metadata={},
        )


def test_extract_run_output_selects_last_output_in_trace_record() -> None:
    event_data = {
        'events': [
            {
                'name': 'run.event',
                'attributes': {'event_type': 'run.output', 'event_data': {'output': {'success': True, 'value': 1}}},
            },
            {'name': 'chat.completion.delta', 'attributes': {'content': 'ignored'}},
            {
                'name': 'run.event',
                'attributes': {'event_type': 'run.output', 'event_data': {'output': {'success': True, 'value': 2}}},
            },
        ]
    }

    assert module.extract_run_output(event_data) == {'success': True, 'value': 2}


def test_project_lifecycle_times_uses_status_events_not_execution_update_time() -> None:
    events = [
        {
            'seq': 1,
            'type': 'execution.created',
            'payload': {'status': 'queued'},
            'created_at': '2026-08-07T01:00:00+00:00',
        },
        {
            'seq': 2,
            'type': 'execution.status.changed',
            'payload': {'from': 'assigned', 'to': 'starting'},
            'created_at': '2026-08-07T01:01:00+00:00',
        },
        {
            'seq': 3,
            'type': 'execution.status.changed',
            'payload': {'from': 'starting', 'to': 'running'},
            'created_at': '2026-08-07T01:01:05+00:00',
        },
        {
            'seq': 4,
            'type': 'execution.status.changed',
            'payload': {'from': 'running', 'to': 'succeeded'},
            'created_at': '2026-08-07T01:02:00+00:00',
        },
    ]

    assert module.project_lifecycle_times(events) == (
        '2026-08-07T01:01:00+00:00',
        '2026-08-07T01:02:00+00:00',
    )


def test_validate_trace_table_rejects_sql_identifier_injection() -> None:

    assert module.validate_trace_table('trajectory_span_events_pr_0123456789abcdef0123456789abcdef') == (
        'trajectory_span_events_pr_0123456789abcdef0123456789abcdef'
    )
    with pytest.raises(module.ExportError, match='invalid Trace Database table'):
        module.validate_trace_table('trajectory_span_events_pr_deadbeef; DROP TABLE runs')


def test_resolve_secret_enforces_mutual_exclusion_and_private_file(tmp_path: Path) -> None:
    secret_file = tmp_path / 'secret'
    secret_file.write_text('from-file\n', encoding='utf-8')
    secret_file.chmod(0o600)

    assert (
        module.resolve_secret(
            inline=None,
            path=secret_file,
            inline_option='--db-password',
            file_option='--db-password-file',
            stderr=io.StringIO(),
        )
        == 'from-file'
    )
    with pytest.raises(module.ExportError, match='mutually exclusive'):
        module.resolve_secret(
            inline='inline',
            path=secret_file,
            inline_option='--db-password',
            file_option='--db-password-file',
            stderr=io.StringIO(),
        )

    secret_file.chmod(0o640)
    with pytest.raises(module.ExportError, match='must have mode 0600'):
        module.resolve_secret(
            inline=None,
            path=secret_file,
            inline_option='--db-password',
            file_option='--db-password-file',
            stderr=io.StringIO(),
        )


def test_resolve_secret_warns_when_inline_value_is_used() -> None:
    stderr = io.StringIO()

    assert (
        module.resolve_secret(
            inline='inline-secret',
            path=None,
            inline_option='--taihu-token',
            file_option='--taihu-token-file',
            stderr=stderr,
        )
        == 'inline-secret'
    )
    warning = stderr.getvalue()
    assert '--taihu-token' in warning
    assert 'shell history' in warning
    assert 'inline-secret' not in warning


def test_validate_batch_summary_requires_complete_equal_terminal_counts() -> None:
    summary = {
        'parent_execution_id': 'exe-batch',
        'generation_status': 'complete',
        'materialized_count': 2,
        'total_items': 2,
        'terminal_items': 2,
        'submission_closed': True,
        'summary_version': 7,
        'item_status_counts': {'succeeded': 1, 'failed': 1},
    }

    assert module.validate_batch_summary(summary) == 2
    assert module.batch_snapshot(summary) == {
        'summary_version': 7,
        'generation_status': 'complete',
        'materialized_count': 2,
        'total_items': 2,
        'terminal_items': 2,
        'submission_closed': True,
        'item_status_counts': {'failed': 1, 'succeeded': 1},
    }

    with pytest.raises(module.ExportError, match='Batch is not fully terminal'):
        module.validate_batch_summary({**summary, 'terminal_items': 1})


async def test_atomic_jsonl_output_publishes_valid_file(tmp_path: Path) -> None:
    output = tmp_path / 'results.jsonl'
    writer = module.AtomicJsonlOutput(output, force=False)
    await writer.open()
    await writer.write_records([
        {
            'run_id': 'item-001',
            'status': 'success',
            'start_time': None,
            'end_time': None,
            'retry_count': 0,
            'error': None,
            'task_output': {'value': 1},
            'agent_output': None,
            'processor_results': {},
            'metadata': {},
        }
    ])
    await writer.close_and_validate(expected_count=1, require_sorted=True)
    await writer.publish()

    assert not output.with_name('results.jsonl.partial').exists()
    assert output.read_text(encoding='utf-8') == (
        '{"run_id": "item-001", "status": "success", "start_time": null, "end_time": null, '
        '"retry_count": 0, "error": null, "task_output": {"value": 1}, "agent_output": null, '
        '"processor_results": {}, "metadata": {}}\n'
    )


async def test_atomic_jsonl_output_preserves_partial_after_validation_failure(tmp_path: Path) -> None:
    output = tmp_path / 'results.jsonl'
    writer = module.AtomicJsonlOutput(output, force=False)
    await writer.open()
    await writer.write_records([{'run_id': 'incomplete'}])

    with pytest.raises(module.ExportError, match='unexpected JSONL fields'):
        await writer.close_and_validate(expected_count=1, require_sorted=True)
    await writer.abort()

    assert not output.exists()
    assert output.with_name('results.jsonl.partial').is_file()


async def test_batch_export_writes_small_groups_before_fetching_more_result_payloads():
    pending_payloads = 0
    max_pending = 0
    fetched_groups = []
    written = []
    total = 25

    class Api:
        async def get_batch_items(self, execution_id, cursor):
            assert cursor is None
            return {
                'items': [
                    {
                        'item_id': f'item-{i:03}',
                        'status': 'succeeded',
                        'current_attempt_no': 0,
                        'current_child_execution_id': f'child-{i:03}',
                    }
                    for i in range(total)
                ],
                'next_cursor': None,
            }

        async def get_control_events(self, execution_id):
            return []

        async def get_result(self, execution_id):
            nonlocal pending_payloads, max_pending
            pending_payloads += 1
            max_pending = max(max_pending, pending_payloads)
            return {'success': True, 'task_output': {'id': execution_id}, 'processor_results': {}}

    class Reader:
        async def read_many(self, ids):
            fetched_groups.append(len(ids))
            return {i: module.TraceResult(start_time='start', end_time='end') for i in ids}

    class Writer:
        async def write_records(self, records):
            nonlocal pending_payloads
            written.extend(records)
            pending_payloads -= len(records)

    api = Api()
    parent = {'execution_id': 'parent', 'spec_type': 'nexus_batch'}
    singles = [
        export_results._batch_export(parent, item)
        async for item in export_results._batch_items(
            cast(Any, api), 'parent', allow_partial=False, expected_count=total
        )
    ]
    plans = [export_results._RecordPlan(single=single, trace_reader=Reader()) for single in singles]
    await export_results._write_records(cast(Any, Writer()), cast(Any, api), plans)
    assert len(written) == total
    assert max_pending <= 2
    assert max(fetched_groups) <= 20
    assert pending_payloads == 0


async def test_result_jsonl_cancellation_waits_for_file_writer(tmp_path, monkeypatch):
    import threading

    started = threading.Event()
    release = threading.Event()
    writer = module.AtomicJsonlOutput(tmp_path / 'result.jsonl', force=False)
    await writer.open()
    original = writer._write_records

    def blocked(records):
        started.set()
        assert release.wait(2)
        original(records)

    monkeypatch.setattr(writer, '_write_records', blocked)
    task = asyncio.create_task(writer.write_records([{'run_id': 'one', 'payload': 'a' * 100000}]))
    try:
        assert await asyncio.to_thread(started.wait, 1)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await writer.abort()
    assert json.loads(writer.partial.read_text())['run_id'] == 'one'


async def test_cloud_api_retries_transient_server_error() -> None:
    attempts = 0

    async def execution_handler(request: web.Request) -> web.Response:
        nonlocal attempts
        assert request.headers['Authorization'] == 'Bearer test-pat'
        attempts += 1
        if attempts < 3:
            return web.json_response({'error': 'busy'}, status=503)
        return web.json_response({'execution_id': 'exe-retry'})

    app = web.Application()
    app.router.add_get('/v1/executions/exe-retry', execution_handler)
    server = TestServer(app)
    await server.start_server()
    sleeps: list[float] = []

    async def no_sleep(delay: float) -> None:
        sleeps.append(delay)

    try:
        async with ClientSession() as session:
            api = module.CloudApi(
                session,
                control_url=str(server.make_url('/')).rstrip('/'),
                application_url=str(server.make_url('/')).rstrip('/'),
                token='test-pat',
                sleep=no_sleep,
            )
            assert await api.get_execution('exe-retry') == {'execution_id': 'exe-retry'}
    finally:
        await server.close()

    assert attempts == 3
    assert len(sleeps) == 2


async def test_export_execution_writes_single_result_from_public_apis(tmp_path: Path) -> None:
    execution = {
        'execution_id': 'exe-single',
        'spec_type': 'nexus_single',
        'status': 'succeeded',
        'attempt_count': 1,
        'current_attempt_id': 'att-1',
        'result_artifact_id': 'art-1',
        'error_json': None,
    }

    async def execution_handler(_request: web.Request) -> web.Response:
        return web.json_response(execution)

    async def result_handler(_request: web.Request) -> web.Response:
        return web.json_response({
            'success': True,
            'task_output': {'answer': 42},
            'agent_output': None,
            'processor_results': {'score': 1},
            'error': None,
        })

    async def events_handler(_request: web.Request) -> web.Response:
        return web.json_response({
            'items': [
                {
                    'seq': 1,
                    'type': 'execution.status.changed',
                    'payload': {'from': 'assigned', 'to': 'starting'},
                    'created_at': '2026-08-07T01:01:00+00:00',
                },
                {
                    'seq': 2,
                    'type': 'execution.status.changed',
                    'payload': {'from': 'running', 'to': 'succeeded'},
                    'created_at': '2026-08-07T01:02:00+00:00',
                },
            ]
        })

    app = web.Application()
    app.router.add_get('/v1/executions/exe-single', execution_handler)
    app.router.add_get('/v1/executions/exe-single/result', result_handler)
    app.router.add_get('/v1/executions/exe-single/control-events', events_handler)
    server = TestServer(app)
    await server.start_server()
    output = tmp_path / 'single.jsonl'
    stale_metadata = output.with_name('single.jsonl.batch_metadata.json')
    stale_metadata.write_text('{"stale": true}\n', encoding='utf-8')
    try:
        async with ClientSession() as session:
            api = module.CloudApi(
                session,
                control_url=str(server.make_url('/')).rstrip('/'),
                application_url=str(server.make_url('/')).rstrip('/'),
                token='test-pat',
            )
            await module.export_execution(
                api=api,
                execution_id='exe-single',
                output=output,
                force=True,
                trace_reader=None,
            )
    finally:
        await server.close()

    assert json_lines(output) == [
        {
            'run_id': 'exe-single',
            'status': 'success',
            'start_time': '2026-08-07T01:01:00+00:00',
            'end_time': '2026-08-07T01:02:00+00:00',
            'retry_count': 0,
            'error': None,
            'task_output': {'answer': 42},
            'agent_output': None,
            'processor_results': {'score': 1},
            'metadata': None,
        }
    ]
    assert not stale_metadata.exists()


@pytest.mark.parametrize('with_cache_ids', [False, True])
async def test_export_execution_pages_batch_and_joins_current_child_outputs(tmp_path: Path, with_cache_ids) -> None:
    first_ids = (
        {
            'input_cache_task_ids': {'main': 'main-input', 'judge': 'judge-input'},
            'cache_task_ids': {'main': 'main-effective', 'judge': 'judge-effective'},
        }
        if with_cache_ids
        else {}
    )
    second_ids = {'cache_task_id': 'failed-effective'} if with_cache_ids else {}
    execution = {
        'execution_id': 'exe-batch',
        'spec_type': 'nexus_batch',
        'status': 'failed',
        'attempt_count': 1,
        'current_attempt_id': 'att-parent',
        'result_artifact_id': None,
        'error_json': {'type': 'BatchFailed'},
    }
    summary = {
        'parent_execution_id': 'exe-batch',
        'generation_status': 'complete',
        'materialized_count': 2,
        'total_items': 2,
        'terminal_items': 2,
        'submission_closed': True,
        'summary_version': 4,
        'item_status_counts': {'succeeded': 1, 'failed': 1},
        'task_list_manifest': {
            'batch_metadata': {
                'task_name': 'swe_task',
                'experiment': 'hyway-mini-black',
            }
        },
    }
    pages = {
        None: {
            'items': [
                {
                    'item_id': 'item-001',
                    'status': 'succeeded',
                    'child_execution_status': 'succeeded',
                    'current_attempt_no': 0,
                    'current_child_execution_id': 'child-current-1',
                    'metadata': {
                        'row': 1,
                        '_nexus': {'original_run_id': 'source-z', 'source': 'generator'},
                    },
                }
            ],
            'next_cursor': 'item-001',
        },
        'item-001': {
            'items': [
                {
                    'item_id': 'item-002',
                    'status': 'failed',
                    'child_execution_status': 'failed',
                    'current_attempt_no': 1,
                    'current_child_execution_id': 'child-current-2',
                    'metadata': {'row': 2, '_nexus': {'original_run_id': 'source-a'}},
                }
            ],
            'next_cursor': None,
        },
    }

    async def execution_handler(_request: web.Request) -> web.Response:
        return web.json_response(execution)

    async def summary_handler(_request: web.Request) -> web.Response:
        return web.json_response(summary)

    async def items_handler(request: web.Request) -> web.Response:
        return web.json_response(pages[request.query.get('cursor')])

    app = web.Application()
    app.router.add_get('/v1/executions/exe-batch', execution_handler)
    app.router.add_get('/v1/executions/exe-batch/batch', summary_handler)
    app.router.add_get('/v1/executions/exe-batch/batch/items', items_handler)

    async def empty_events(_request):
        return web.json_response({'items': []})

    app.router.add_get('/v1/executions/{child}/control-events', empty_events)
    server = TestServer(app)
    await server.start_server()

    class TraceReader:
        def __init__(self) -> None:
            self.calls: list[tuple[str, ...]] = []
            # Mirror the attributes TdsqlTraceReader exposes: export_execution backfills the
            # Boundless partition key onto the reader, and a stub without them silently
            # skipped that path instead of exercising it.
            self.table = 'trajectory_span_events_pr_' + '0' * 32
            self.parent_created_at: str | None = None

        async def read_many(self, run_ids: list[str]) -> dict[str, Any]:
            self.calls.append(tuple(run_ids))
            outputs = {
                'child-current-1': module.TraceResult(
                    output={'success': True, 'task_output': {'value': 'one'}, 'processor_results': {}, **first_ids},
                    start_time='2026-08-07T01:00:00+00:00',
                    end_time='2026-08-07T01:00:10+00:00',
                ),
                'child-current-2': module.TraceResult(
                    output={
                        'success': False,
                        'error': {'type': 'TaskFailed', 'message': 'bad item'},
                        'task_output': None,
                        'processor_results': {},
                        **second_ids,
                    },
                    start_time='2026-08-07T01:01:00+00:00',
                    end_time='2026-08-07T01:01:10+00:00',
                ),
            }
            return {run_id: outputs[run_id] for run_id in run_ids}

    trace_reader = TraceReader()
    output = tmp_path / 'batch.jsonl'
    try:
        async with ClientSession() as session:
            api = module.CloudApi(
                session,
                control_url=str(server.make_url('/')).rstrip('/'),
                application_url=str(server.make_url('/')).rstrip('/'),
                token='test-pat',
            )
            await module.export_execution(
                api=api,
                execution_id='exe-batch',
                output=output,
                force=False,
                trace_reader=trace_reader,
            )
    finally:
        await server.close()

    records = json_lines(output)
    assert [record['run_id'] for record in records] == ['source-z', 'source-a']
    assert records[0]['task_output'] == {'value': 'one'}
    assert records[0]['metadata'] == {
        'row': 1,
        '_nexus': {
            'original_run_id': 'source-z',
            'source': 'generator',
            'parent_execution_id': 'exe-batch',
            'child_execution_id': 'child-current-1',
            **first_ids,
        },
    }
    assert records[1]['status'] == 'failed'
    assert records[1]['retry_count'] == 1
    assert records[1]['error'] == {'type': 'TaskFailed', 'message': 'bad item'}
    assert records[1]['metadata']['_nexus'] == {
        'original_run_id': 'source-a',
        'parent_execution_id': 'exe-batch',
        'child_execution_id': 'child-current-2',
        **second_ids,
    }
    assert trace_reader.calls == [('child-current-1',), ('child-current-2',)]
    assert json.loads(output.with_name('batch.jsonl.batch_metadata.json').read_text()) == {
        'task_name': 'swe_task',
        'experiment': 'hyway-mini-black',
    }


async def test_export_execution_tolerates_trace_reader_without_partition_attributes(tmp_path: Path) -> None:
    """A duck-typed reader lacking ``table`` must still export.

    Backfilling the Boundless partition key is a pruning optimisation: skipping it widens the
    scan but keeps the rows correct. Reaching for the attribute unguarded turned that into an
    AttributeError that aborted the whole export.
    """
    execution = {
        'execution_id': 'exe-batch',
        'spec_type': 'nexus_batch',
        'status': 'succeeded',
        'attempt_count': 1,
        'current_attempt_id': 'att-parent',
        'result_artifact_id': None,
        'error_json': None,
    }
    summary = {
        'parent_execution_id': 'exe-batch',
        'generation_status': 'complete',
        'materialized_count': 1,
        'total_items': 1,
        'terminal_items': 1,
        'submission_closed': True,
        'summary_version': 1,
        'item_status_counts': {'succeeded': 1},
    }
    page = {
        'items': [
            {
                'item_id': 'item-001',
                'status': 'succeeded',
                'child_execution_status': 'succeeded',
                'current_attempt_no': 0,
                'current_child_execution_id': 'child-current-1',
                'metadata': {},
            }
        ],
        'next_cursor': None,
    }

    async def execution_handler(_request: web.Request) -> web.Response:
        return web.json_response(execution)

    async def summary_handler(_request: web.Request) -> web.Response:
        return web.json_response(summary)

    async def items_handler(_request: web.Request) -> web.Response:
        return web.json_response(page)

    async def child_events_handler(_request: web.Request) -> web.Response:
        return web.json_response({'items': []})

    app = web.Application()
    app.router.add_get('/v1/executions/exe-batch', execution_handler)
    app.router.add_get('/v1/executions/exe-batch/batch', summary_handler)
    app.router.add_get('/v1/executions/exe-batch/batch/items', items_handler)
    app.router.add_get('/v1/executions/child-current-1/control-events', child_events_handler)
    server = TestServer(app)
    await server.start_server()

    class ReaderWithoutTable:
        async def read_many(self, run_ids: list[str]) -> dict[str, Any]:
            return {
                run_id: module.TraceResult(
                    output={'success': True, 'task_output': {'value': 'one'}, 'processor_results': {}},
                    start_time='2026-08-07T01:00:00+00:00',
                    end_time='2026-08-07T01:00:10+00:00',
                )
                for run_id in run_ids
            }

    output = tmp_path / 'batch.jsonl'
    try:
        async with ClientSession() as session:
            api = module.CloudApi(
                session,
                control_url=str(server.make_url('/')).rstrip('/'),
                application_url=str(server.make_url('/')).rstrip('/'),
                token='test-pat',
            )
            await module.export_execution(
                api=api,
                execution_id='exe-batch',
                output=output,
                force=False,
                trace_reader=ReaderWithoutTable(),
            )
    finally:
        await server.close()

    assert [record['task_output'] for record in json_lines(output)] == [{'value': 'one'}]


async def test_export_execution_does_not_publish_when_snapshot_changes(tmp_path: Path) -> None:
    reads = 0

    async def execution_handler(_request: web.Request) -> web.Response:
        nonlocal reads
        reads += 1
        return web.json_response({
            'execution_id': 'exe-changing',
            'spec_type': 'nexus_single',
            'status': 'succeeded',
            'attempt_count': reads,
            'current_attempt_id': f'att-{reads}',
            'result_artifact_id': 'art-1',
            'error_json': None,
        })

    async def result_handler(_request: web.Request) -> web.Response:
        return web.json_response({'success': True, 'task_output': {'ok': True}, 'processor_results': {}})

    async def events_handler(_request: web.Request) -> web.Response:
        return web.json_response({'items': []})

    app = web.Application()
    app.router.add_get('/v1/executions/exe-changing', execution_handler)
    app.router.add_get('/v1/executions/exe-changing/result', result_handler)
    app.router.add_get('/v1/executions/exe-changing/control-events', events_handler)
    server = TestServer(app)
    await server.start_server()
    output = tmp_path / 'changing.jsonl'
    try:
        async with ClientSession() as session:
            api = module.CloudApi(
                session,
                control_url=str(server.make_url('/')).rstrip('/'),
                application_url=str(server.make_url('/')).rstrip('/'),
                token='test-pat',
            )
            with pytest.raises(module.ExportError, match='Execution changed during export'):
                await module.export_execution(
                    api=api,
                    execution_id='exe-changing',
                    output=output,
                    force=False,
                    trace_reader=None,
                )
    finally:
        await server.close()

    assert not output.exists()
    assert output.with_name('changing.jsonl.partial').is_file()


async def test_tdsql_trace_reader_chunks_ids_and_limits_concurrency() -> None:
    active = 0
    max_active = 0
    timestamp_chunk_sizes: list[int] = []

    class Cursor:
        def __init__(self) -> None:
            self.rows: list[tuple[Any, ...]] = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, _exc_type, _exc, _tb) -> None:
            return None

        async def execute(self, sql: str, params: tuple[Any, ...]) -> None:
            await asyncio.sleep(0.005)
            if 'MIN(time_unix_nano)' in sql:
                run_ids = list(params)
                timestamp_chunk_sizes.append(len(run_ids))
                self.rows = [(run_id, 1_700_000_000_000_000_000, 1_700_000_001_000_000_000) for run_id in run_ids]
            elif sql.startswith('SELECT DISTINCT run_span_id'):
                self.rows = [(f'span-{params[0]}',)]
            elif sql.startswith('SELECT record_id, time_unix_nano'):
                assert 'span_id = %s' in sql
                assert 'event_data' not in sql
                self.rows = [(7, 1_700_000_001_000_000_000)]
            else:
                assert sql.startswith('SELECT event_data')
                assert 'LIKE' not in sql
                assert 'ORDER BY' not in sql
                self.rows = [
                    (
                        json.dumps({
                            'events': [
                                {
                                    'name': 'run.event',
                                    'attributes': {
                                        'event_type': 'run.output',
                                        'event_data': {
                                            'output': {'success': True, 'task_output': {'run_id': params[0]}}
                                        },
                                    },
                                }
                            ]
                        }),
                    )
                ]

        async def fetchall(self) -> list[tuple[Any, ...]]:
            return self.rows

    class Connection:
        def cursor(self) -> Cursor:
            return Cursor()

        def close(self) -> None:
            nonlocal active
            active -= 1

    async def connect(**_kwargs: Any) -> Connection:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0)
        return Connection()

    reader = module.TdsqlTraceReader(
        host='db.test',
        port=3306,
        user='readonly',
        password='secret',
        database='trace',
        table='trajectory_span_events_pr_0123456789abcdef0123456789abcdef',
        connect=connect,
        transient_errors=(),
    )
    run_ids = [f'child-{index:04d}' for index in range(1000)]

    results = await reader.read_many(run_ids)

    assert timestamp_chunk_sizes == [1] * 1000
    assert max_active == 4
    assert results['child-0001'] == module.TraceResult(
        output={'success': True, 'task_output': {'run_id': 'child-0001'}},
        start_time='2023-11-14T22:13:20+00:00',
        end_time='2023-11-14T22:13:21+00:00',
    )


@pytest.mark.parametrize('omit_latest', [False, True])
async def test_tdsql_trace_reader_selects_latest_attempt_output_by_event_time(omit_latest) -> None:

    def output_row(run_span_id: str, record_id: int, time_unix_nano: int, value: str | None) -> tuple[Any, ...]:
        return (
            'child-1',
            run_span_id,
            record_id,
            time_unix_nano,
            json.dumps({
                'events': [
                    {
                        'name': 'run.event',
                        'attributes': {
                            'event_type': 'run.output',
                            'event_data': {
                                'output': None if value is None else {'success': True, 'task_output': {'value': value}}
                            },
                        },
                    }
                ]
            }),
        )

    class Cursor:
        def __init__(self) -> None:
            self.rows: list[tuple[Any, ...]] = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, _exc_type, _exc, _tb) -> None:
            return None

        async def execute(self, sql: str, params: tuple[Any, ...]) -> None:
            if 'MIN(time_unix_nano)' in sql:
                self.rows = [('child-1', 100, 400)]
            elif sql.startswith('SELECT DISTINCT run_span_id'):
                self.rows = [('attempt-new',), ('attempt-old',)]
            elif sql.startswith('SELECT record_id, time_unix_nano'):
                self.rows = [(1, 400)] if params[1] == 'attempt-new' else [(99, 200)]
            else:
                assert params[1] == 'attempt-new'
                self.rows = [(output_row('attempt-new', 1, 400, None if omit_latest else 'new')[4],)]

        def fetchall(self) -> list[tuple[Any, ...]]:
            return self.rows

    class Connection:
        def cursor(self) -> Cursor:
            return Cursor()

        def close(self) -> None:
            return None

    async def connect(**_kwargs: Any) -> Connection:
        return Connection()

    reader = module.TdsqlTraceReader(
        host='db.test',
        port=3306,
        user='readonly',
        password='secret',
        database='trace',
        table='trajectory_span_events_pr_0123456789abcdef0123456789abcdef',
        connect=connect,
        transient_errors=(),
    )

    result = (await reader.read_many(['child-1']))['child-1']

    assert result.output == (None if omit_latest else {'success': True, 'task_output': {'value': 'new'}})


async def test_tdsql_trace_reader_retries_transient_connect_failure() -> None:

    class TemporaryDatabaseError(Exception):
        pass

    attempts = 0
    sleeps: list[float] = []

    class Cursor:
        async def __aenter__(self):
            return self

        async def __aexit__(self, _exc_type, _exc, _tb) -> None:
            return None

        async def execute(self, _sql: str, _params: tuple[Any, ...]) -> None:
            return None

        def fetchall(self) -> list[tuple[Any, ...]]:
            return []

    class Connection:
        def cursor(self) -> Cursor:
            return Cursor()

        def close(self) -> None:
            return None

    async def connect(**_kwargs: Any) -> Connection:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise TemporaryDatabaseError('try again')
        return Connection()

    async def no_sleep(delay: float) -> None:
        sleeps.append(delay)

    reader = module.TdsqlTraceReader(
        host='db.test',
        port=3306,
        user='readonly',
        password='secret',
        database='trace',
        table='trajectory_span_events_pr_0123456789abcdef0123456789abcdef',
        connect=connect,
        transient_errors=(TemporaryDatabaseError,),
        sleep=no_sleep,
    )

    assert await reader.read_many(['child-1']) == {'child-1': module.TraceResult()}
    assert attempts == 3
    assert len(sleeps) == 2


def test_parse_args_supports_inline_api_and_database_credentials(tmp_path: Path) -> None:

    args = module.parse_args([
        '--control-url',
        'https://control.test',
        '--application-url',
        'https://application.test',
        '--execution-id',
        'exe-1',
        '--taihu-token',
        'pat-inline',
        '--db-host',
        'db.test',
        '--db-port',
        '3307',
        '--db-name',
        'trace',
        '--db-user',
        'readonly',
        '--db-password',
        'db-inline',
        '--trace-table',
        'trajectory_span_events_pr_0123456789abcdef0123456789abcdef',
        '--output',
        str(tmp_path / 'result.jsonl'),
        '--force',
    ])

    assert args.control_url == 'https://control.test'
    assert args.application_url == 'https://application.test'
    assert args.execution_id == 'exe-1'
    assert args.taihu_token == 'pat-inline'
    assert args.db_port == 3307
    assert args.db_password == 'db-inline'
    assert args.output == tmp_path / 'result.jsonl'
    assert args.force is True


def test_parse_args_reads_application_url_from_environment(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv('NEXUS_APPLICATION_URL', 'https://application.env')

    args = module.parse_args([
        '--execution-id',
        'exe-1',
        '--taihu-token',
        'pat-inline',
        '--output',
        str(tmp_path / 'result.jsonl'),
    ])

    assert args.application_url == 'https://application.env'


async def test_run_requires_application_url_instead_of_falling_back_to_control(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv('NEXUS_APPLICATION_URL', raising=False)
    args = module.parse_args([
        '--control-url',
        'https://control.test',
        '--execution-id',
        'exe-1',
        '--taihu-token',
        'pat-inline',
        '--output',
        str(tmp_path / 'result.jsonl'),
    ])

    async def unexpected_export(**_kwargs) -> None:
        pytest.fail('export must not start without an Application URL')

    monkeypatch.setattr(export_cli, 'export_execution', unexpected_export)

    with pytest.raises(module.ExportError, match='--application-url.*NEXUS_APPLICATION_URL'):
        await export_cli._run(args)


def test_create_trace_reader_rejects_partial_database_configuration(tmp_path: Path) -> None:
    args = module.parse_args([
        '--execution-id',
        'exe-1',
        '--taihu-token',
        'pat-inline',
        '--db-host',
        'db.test',
        '--output',
        str(tmp_path / 'result.jsonl'),
    ])

    with pytest.raises(module.ExportError, match='incomplete Trace Database arguments'):
        module.create_trace_reader(args, stderr=io.StringIO())


async def test_batch_item_prefers_result_api_even_when_database_is_available() -> None:
    metadata = {
        'row': 1,
        '_nexus': {'original_run_id': 'source-item-1', 'source': 'generator'},
    }

    class Api:
        async def get_result(self, execution_id: str) -> dict[str, Any]:
            assert execution_id == 'child-1'
            return {'success': True, 'task_output': {'source': 'api'}, 'processor_results': {}}

        async def get_execution(self, _execution_id: str) -> dict[str, Any]:
            raise AssertionError('execution detail should not be requested when the result and trace times exist')

        async def get_control_events(self, _execution_id: str) -> list[dict[str, Any]]:
            return [
                {
                    'type': 'execution.status.changed',
                    'payload': {'to': 'succeeded'},
                    'created_at': '2026-08-07T01:00:01+00:00',
                }
            ]

    record = await export_results._single_record(
        api=cast(Any, Api()),
        single=export_results._batch_export(
            {'execution_id': 'batch-1', 'spec_type': 'nexus_batch'},
            {
                'item_id': 'item-1',
                'status': 'succeeded',
                'current_attempt_no': 0,
                'current_child_execution_id': 'child-1',
                'metadata': metadata,
            },
        ),
        trace_reader=None,
    )

    assert record['task_output'] == {'source': 'api'}
    assert record['run_id'] == 'source-item-1'
    assert record['metadata'] == {
        'row': 1,
        '_nexus': {
            'original_run_id': 'source-item-1',
            'source': 'generator',
            'parent_execution_id': 'batch-1',
            'child_execution_id': 'child-1',
        },
    }
    assert metadata == {
        'row': 1,
        '_nexus': {'original_run_id': 'source-item-1', 'source': 'generator'},
    }


async def test_batch_item_without_child_uses_item_id_and_records_null_child_identity() -> None:

    class Api:
        async def get_result(self, _execution_id: str) -> None:
            raise AssertionError('result should not be requested without a child execution')

        async def get_execution(self, _execution_id: str) -> dict[str, Any]:
            raise AssertionError('execution should not be requested without a child execution')

        async def get_control_events(self, _execution_id: str) -> list[dict[str, Any]]:
            raise AssertionError('control events should not be requested without a child execution')

    metadata = {'row': 2}
    record = await export_results._single_record(
        api=cast(Any, Api()),
        single=export_results._batch_export(
            {'execution_id': 'batch-1', 'spec_type': 'nexus_batch'},
            {
                'item_id': 'item-no-child',
                'status': 'failed',
                'current_attempt_no': 0,
                'current_child_execution_id': None,
                'metadata': metadata,
            },
        ),
        trace_reader=None,
    )

    assert record['run_id'] == 'item-no-child'
    assert record['metadata'] == {
        'row': 2,
        '_nexus': {
            'original_run_id': 'item-no-child',
            'parent_execution_id': 'batch-1',
            'child_execution_id': None,
        },
    }
    assert metadata == {'row': 2}


async def test_batch_item_keeps_api_error_when_child_failed_before_output() -> None:

    class Api:
        async def get_result(self, _execution_id: str) -> None:
            return None

        async def get_execution(self, execution_id: str) -> dict[str, Any]:
            assert execution_id == 'child-failed'
            return {
                'execution_id': execution_id,
                'parent_execution_id': 'batch-1',
                'error_json': {'type': 'StartupFailed', 'message': 'runtime unavailable'},
            }

        async def get_control_events(self, _execution_id: str) -> list[dict[str, Any]]:
            return [
                {
                    'seq': 1,
                    'type': 'execution.status.changed',
                    'payload': {'from': 'starting', 'to': 'failed'},
                    'created_at': '2026-08-07T01:00:02+00:00',
                }
            ]

    record = await export_results._single_record(
        api=cast(Any, Api()),
        single=export_results._batch_export(
            {'execution_id': 'batch-1', 'spec_type': 'nexus_batch'},
            {
                'item_id': 'item-failed',
                'status': 'failed',
                'current_attempt_no': 0,
                'current_child_execution_id': 'child-failed',
                'metadata': {},
            },
        ),
        trace_reader=None,
    )

    assert record['status'] == 'failed'
    assert record['error'] == {'type': 'StartupFailed', 'message': 'runtime unavailable'}
    assert record['task_output'] is None
    assert record['end_time'] == '2026-08-07T01:00:02+00:00'


def test_decode_event_data_reports_malformed_tdsql_json_as_export_error() -> None:

    with pytest.raises(module.ExportError, match='malformed TDSQL event_data'):
        export_results._decode_event_data('{not-json')


def json_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


@pytest.mark.parametrize('force', [False, True])
async def test_result_force_recovers_interrupted_jsonl_and_metadata(tmp_path, force):
    for writer_type in (module.AtomicJsonlOutput, module.AtomicJsonOutput):
        path = tmp_path / writer_type.__name__
        path.write_text('previous published output')
        partial = path.with_name(path.name + '.partial')
        partial.write_text('interrupted write')
        writer = writer_type(path, force=force)
        if not force:
            operation = (
                writer.open() if isinstance(writer, module.AtomicJsonlOutput) else writer.prepare({'dataset': 'new'})
            )
            with pytest.raises(module.ExportError, match='output already exists'):
                await operation
            assert partial.read_text() == 'interrupted write'
            continue
        if isinstance(writer, module.AtomicJsonlOutput):
            await writer.open()
            await writer.close_and_validate(expected_count=0, require_sorted=False)
        else:
            await writer.prepare({'dataset': 'new'})
        assert path.read_text() == 'previous published output'
        await writer.publish()
        assert not partial.exists()
        assert path.read_text() != 'previous published output'


# ---------------------------------------------------------------------------
# TDSQL 字节管线的候选选择语义（对齐 legacy latest/latest_key）与线程生命周期
# ---------------------------------------------------------------------------


def _event_data(rows_events: list[dict[str, Any]]) -> str:
    return json.dumps({'events': rows_events}, default=str)


def _run_output_event(output: Any) -> dict[str, Any]:
    return {
        'name': 'run.event',
        'attributes': {'event_type': 'run.output', 'event_data': {'output': output}},
    }


def _reader_for(records: dict[int, str], timestamps: dict[int, int]) -> module.TdsqlTraceReader:
    """单 span、按 record_id 给 event_data 的最小 TDSQL 替身。"""

    class Cursor:
        def __init__(self) -> None:
            self.rows: list[tuple[Any, ...]] = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def execute(self, sql: str, params: tuple[Any, ...]) -> None:
            if 'MIN(time_unix_nano)' in sql:
                self.rows = [('child-1', 100, 500)]
            elif sql.startswith('SELECT DISTINCT run_span_id'):
                self.rows = [('s1',)]
            elif sql.startswith('SELECT record_id, time_unix_nano'):
                self.rows = [(rid, timestamps[rid]) for rid in sorted(records)]
            else:
                self.rows = [(records[params[-1]],)]

        def fetchall(self) -> list[tuple[Any, ...]]:
            return self.rows

    class Connection:
        def cursor(self) -> Cursor:
            return Cursor()

        def close(self) -> None:
            return None

    async def connect(**_kwargs: Any) -> Connection:
        return Connection()

    return module.TdsqlTraceReader(
        host='db.test',
        port=3306,
        user='readonly',
        password='secret',
        database='trace',
        table='trajectory_span_events_pr_0123456789abcdef0123456789abcdef',
        connect=connect,
        transient_errors=(),
    )


async def test_tdsql_null_output_on_newer_record_falls_back_to_api(tmp_path) -> None:
    """旧记录 output=A、新记录 run.output 的 output=null：以新记录为准 → None → API 回退。

    回归：字节管线曾在此导出旧 attempt 的 A（candidate_path 未被 null 命中清空）。
    """
    reader = _reader_for(
        records={
            1: _event_data([_run_output_event({'success': True, 'task_output': {'value': 'A'}})]),
            2: _event_data([_run_output_event(None)]),
        },
        timestamps={1: 200, 2: 400},
    )
    pool = export_results.ResultSpillPool(tmp_path / 'spill')

    legacy = (await reader.read_many(['child-1']))['child-1']
    scan_out: dict[str, Any] = {}
    streamed = (await reader.read_many(['child-1'], scan_out=scan_out, spill_pool=pool))['child-1']

    assert legacy.output is None  # legacy：最新记录 output=null → None → API 回退
    assert streamed.output is None  # 字节管线：必须与 legacy 一致（修复前导出 'A'）
    assert scan_out.get('prepared') is None  # 不携带任何候选 body


async def test_tdsql_record_without_run_output_does_not_block_watermark(tmp_path) -> None:
    """无 run.output 的记录不得推进水位：其后的有效 run.output 记录必须仍然生效。

    回归：字节管线曾让无 run.output 的记录推进 candidate_key，导致按 key 排序更早
    的有效记录被跳过（导出回退到 API，丢失 TDSQL 数据）。
    """
    reader = _reader_for(
        records={
            1: _event_data([{'name': 'other.event', 'attributes': {'k': 1}}]),
            2: _event_data([_run_output_event({'success': True, 'task_output': {'value': 'B'}})]),
        },
        timestamps={1: 500, 2: 300},  # record 1 先处理但事件时间更大
    )
    pool = export_results.ResultSpillPool(tmp_path / 'spill')

    legacy = (await reader.read_many(['child-1']))['child-1']
    scan_out: dict[str, Any] = {}
    streamed = (await reader.read_many(['child-1'], scan_out=scan_out, spill_pool=pool))['child-1']

    assert legacy.output == {'success': True, 'task_output': {'value': 'B'}}
    # 字节路径命中：output 为流式标记，body 经 scan_out 交接（生产由 _read_result 消费）
    assert streamed.output is export_results._STREAMING_PREPARED
    assert scan_out.get('prepared') is not None


async def test_tdsql_fallback_download_failure_aborts_scan_thread(tmp_path) -> None:
    """TDSQL 未命中 → API 下载失败：扫描线程必须被 abort，不得按条泄漏。"""
    import threading

    import aiohttp

    reader = _reader_for(
        records={1: _event_data([{'name': 'other.event'}])},  # 无 run.output → fallback
        timestamps={1: 100},
    )
    pool = export_results.ResultSpillPool(tmp_path / 'spill')

    class FailingApi:
        async def get_control_events(self, execution_id):
            return []

        async def get_result_to_file(self, execution_id, fileobj, chunk_size=1 << 20, scan=None):
            raise aiohttp.ClientPayloadError('connection reset')

    def scan_threads() -> int:
        # 只统计扫描线程：to_thread 的默认线程池会保留空闲 worker，
        # threading.active_count 对它们不免疫。
        return sum(1 for t in threading.enumerate() if t.name == 'nexus-result-scan')

    before = threading.active_count()
    before_scan = scan_threads()
    for _ in range(5):
        with pytest.raises(aiohttp.ClientPayloadError):
            await export_results._read_result(
                api=FailingApi(),
                execution_id='child-1',
                trace_reader=reader,
                parent_execution_id=None,
                fallback_spill=pool,
                scan_factory=export_results._ScanSlot,
                scan_out={},
            )
    assert threading.active_count() - before <= 1  # 至多剩 to_thread 空闲 worker
    assert scan_threads() == before_scan  # 扫描线程零残留


# ---------------------------------------------------------------------------
# 第二轮审查回归：output span 内的重复键/NaN/非关键事件异常 → legacy 语义回退
# ---------------------------------------------------------------------------


def _dup_keys_output_event_data() -> str:
    # 手写根级重复键（json.dumps 产不出）——dict 路径语义：最后值胜出。
    envelope = json.dumps(
        {
            'events': [
                {
                    'name': 'run.event',
                    'attributes': {
                        'event_type': 'run.output',
                        'event_data': {'output': 'PLACEHOLDER'},
                    },
                }
            ]
        },
        default=str,
    )
    return envelope.replace('"PLACEHOLDER"', '{"success": true, "task_output": {"v": 1}, "task_output": {"v": 2}}')


def _nan_output_event_data() -> str:
    return json.dumps(
        {
            'events': [
                {
                    'name': 'run.event',
                    'attributes': {
                        'event_type': 'run.output',
                        'event_data': {'output': {'success': True, 'score': float('nan')}},
                    },
                }
            ]
        },
        default=str,
    )


async def test_tdsql_duplicate_keys_in_output_matches_dict_path(tmp_path) -> None:
    reader = _reader_for(records={1: _dup_keys_output_event_data()}, timestamps={1: 100})
    pool = export_results.ResultSpillPool(tmp_path / 'spill')
    legacy = (await reader.read_many(['child-1']))['child-1']
    scan_out: dict[str, Any] = {}
    streamed = (await reader.read_many(['child-1'], scan_out=scan_out, spill_pool=pool))['child-1']
    assert legacy.output == {'success': True, 'task_output': {'v': 2}}
    assert streamed.output == legacy.output


async def test_tdsql_nan_in_output_matches_dict_path(tmp_path) -> None:
    reader = _reader_for(records={1: _nan_output_event_data()}, timestamps={1: 100})
    pool = export_results.ResultSpillPool(tmp_path / 'spill')
    legacy = (await reader.read_many(['child-1']))['child-1']
    scan_out: dict[str, Any] = {}
    streamed = (await reader.read_many(['child-1'], scan_out=scan_out, spill_pool=pool))['child-1']
    # NaN 不可用 == 比较，用序列化形态对照（两边都是 json.loads 语义的 NaN）
    assert json.dumps(legacy.output) == '{"success": true, "score": NaN}'
    assert json.dumps(streamed.output) == json.dumps(legacy.output)


async def test_tdsql_broken_other_event_falls_back_to_legacy_decode(tmp_path) -> None:
    # 非 output 事件里的重复键会让嵌套扫描失败 → 整条 event_data 回退 legacy
    # decode；正常（更早出现）的 run.output 事件必须仍然生效。
    text = (
        '{"events": ['
        '{"name": "run.event", "attributes": {"event_type": "run.output", '
        '"event_data": {"output": {"success": true, "task_output": {"v": 7}}}}}, '
        '{"name": "broken.event", "name": "broken.event.2", "payload": {"a": 1}}'
        ']}'
    )
    reader = _reader_for(records={1: text}, timestamps={1: 100})
    pool = export_results.ResultSpillPool(tmp_path / 'spill')
    legacy = (await reader.read_many(['child-1']))['child-1']
    scan_out: dict[str, Any] = {}
    streamed = (await reader.read_many(['child-1'], scan_out=scan_out, spill_pool=pool))['child-1']
    assert legacy.output == {'success': True, 'task_output': {'v': 7}}
    assert streamed.output == legacy.output


# ---------------------------------------------------------------------------
# 鲁棒性：结果下载中途断连（ClientPayloadError）必须重试
# ---------------------------------------------------------------------------


async def test_result_download_retries_mid_body_connection_reset(tmp_path) -> None:
    """正文流式读取阶段的瞬时断连（ClientPayloadError）必须走重试。

    回归：aiohttp 的 ClientPayloadError 不在 ClientConnectionError 继承链上，
    曾不在可重试集合里——一次中途断连直接终止整个导出（生产 part43 的失败形态）。
    用显式抛 ClientPayloadError 的 mock 会话构造，保证对旧代码判别失败。
    """
    import aiohttp

    from nexus.distributed.export.client import CloudApi

    body = json.dumps({'success': True, 'task_output': {'v': 1}}).encode()

    from aiohttp.http_exceptions import TransferEncodingError

    class _Response:
        def __init__(self, payload: bytes, fail_first_read: bool, exc=aiohttp.ClientPayloadError):
            self.status = 200
            self.content = self
            self._payload = payload
            self._fail = fail_first_read
            self._exc = exc
            self._reads = 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def read(self, size: int) -> bytes:
            self._reads += 1
            if self._fail and self._reads == 1:
                raise self._exc('connection reset mid-body')
            chunk = self._payload[:size]
            self._payload = self._payload[size:]
            return chunk

    class _Session:
        """第一次 GET 的正文在读取中途抛 ClientPayloadError，第二次完整返回。"""

        def __init__(self, exc=aiohttp.ClientPayloadError) -> None:
            self.get_urls: list[str] = []
            self._exc = exc

        def get(self, url, headers=None, timeout=None):
            # 第一次 GET：正文读取中途抛瞬时异常；第二次：完整返回。
            self.get_urls.append(url)
            first = len(self.get_urls) == 1
            payload = b'\x00CORRUPT' if first else body
            return _Response(payload, fail_first_read=first, exc=self._exc)

    session = _Session()
    api = CloudApi(session, control_url='http://control', application_url='http://app', token='t')
    out = tmp_path / 'body.bin'
    with out.open('wb') as f:
        length = await api.get_result_to_file('exe-x', f)

    assert length == len(body)
    assert out.read_bytes() == body
    assert len(session.get_urls) == 2  # 首次断连重试 + 二次成功

    # part43 的实际异常类型：服务端掐断 chunked 流 → TransferEncodingError，同样必须重试
    session2 = _Session(exc=TransferEncodingError)
    api2 = CloudApi(session2, control_url='http://control', application_url='http://app', token='t')
    out2 = tmp_path / 'body2.bin'
    with out2.open('wb') as f:
        length2 = await api2.get_result_to_file('exe-x', f)
    assert length2 == len(body)
    assert out2.read_bytes() == body
    assert len(session2.get_urls) == 2


async def test_tdsql_heavy_work_keeps_event_loop_responsive(tmp_path) -> None:
    """TDSQL 的同步重活（整列定位/扫描/拷贝）不得阻塞事件循环。

    回归：并发导出时，一道题的 TDSQL 同步处理会饿死另一道题的流式下载
    （生产 part43 形态：服务端掐断停顿超过 ~40s 的连接）。定位/拷贝/解码
    现已移入工作线程——处理期间事件循环必须持续可调度。
    """
    import time

    small = json.dumps({
        'events': [
            {'name': 'run.event', 'attributes': {'event_type': 'run.output', 'event_data': {'output': {'v': 1}}}}
        ]
    })
    reader = _reader_for(records={1: small}, timestamps={1: 100})

    orig_locate = export_results.TdsqlTraceReader._locate_run_output_span

    def slow_locate(self, text, path):
        time.sleep(0.3)  # 同步阻塞（模拟 GB 级扫描耗时）
        return orig_locate(self, text, path)

    export_results.TdsqlTraceReader._locate_run_output_span = slow_locate
    ticks = 0

    async def heartbeat() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.02)
            ticks += 1

    try:
        pool = export_results.ResultSpillPool(tmp_path / 'spill')
        scan_out: dict[str, Any] = {}
        hb = asyncio.create_task(heartbeat())
        res = (await reader.read_many(['child-1'], scan_out=scan_out, spill_pool=pool))['child-1']
        hb.cancel()
        assert res.output is export_results._STREAMING_PREPARED
        # to_thread 下循环保持可调度（0.3s / 0.02s ≈ 15 tick）；旧同步实现为 0
        assert ticks >= 5, f'事件循环被阻塞：0.3s 内仅 {ticks} 次调度'
    finally:
        export_results.TdsqlTraceReader._locate_run_output_span = orig_locate


def test_in_ram_concurrency_env_override(monkeypatch) -> None:
    """in-RAM 管线并发：env 可覆盖，非法值回退默认，越界夹取。"""
    monkeypatch.delenv('NEXUS_EXPORT_IN_RAM_CONCURRENCY', raising=False)
    assert export_results._in_ram_concurrency() == 2  # 默认（legacy 值）
    monkeypatch.setenv('NEXUS_EXPORT_IN_RAM_CONCURRENCY', '1')
    assert export_results._in_ram_concurrency() == 1  # 运维降并发开关
    monkeypatch.setenv('NEXUS_EXPORT_IN_RAM_CONCURRENCY', '8')
    assert export_results._in_ram_concurrency() == 4  # 上限夹取（每槽 GB 级内存）
    monkeypatch.setenv('NEXUS_EXPORT_IN_RAM_CONCURRENCY', 'not-a-number')
    assert export_results._in_ram_concurrency() == 2  # 非法值回退
    monkeypatch.setenv('NEXUS_EXPORT_IN_RAM_CONCURRENCY', '0')
    assert export_results._in_ram_concurrency() == 1  # 下限夹取


async def test_result_download_empty_json_body_means_missing(tmp_path) -> None:
    """200 + 空 JSON body（{}）必须按"无结果"处理，不得撞 success 校验。

    回归：lost 题（无 artifact）的 /result 返回 200 + {}。旧 in-RAM 路径的
    get_result 有 `not value → None` 守卫（占位行）；!1597 引入的下载管线
    没有等价守卫，2 字节 body 被当成真实 RunOutput，`invalid RunOutput
    success field` 终止整批导出（生产 6 分片失败的原因）。
    """
    import aiohttp

    from nexus.distributed.export.client import CloudApi

    class _Response:
        def __init__(self, payload: bytes):
            self.status = 200
            self.content = self
            self._payload = payload

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def read(self, size: int) -> bytes:
            chunk = self._payload[:size]
            self._payload = self._payload[size:]
            return chunk

    class _Session:
        def __init__(self) -> None:
            self.get_urls: list[str] = []

        def get(self, url, headers=None, timeout=None):
            self.get_urls.append(url)
            return _Response(b'{}')

    api = CloudApi(_Session(), control_url='http://control', application_url='http://app', token='t')
    out = tmp_path / 'body.bin'
    with out.open('wb') as f:
        length = await api.get_result_to_file('exe-x', f)

    assert length is None  # 空 body → missing 语义（与 404 同路）
    assert out.read_bytes() == b'{}'


async def test_read_result_empty_json_body_yields_placeholder_row(tmp_path) -> None:
    """端到端：TDSQL 未命中 + API 返回 200/{} → 占位行，而不是 invalid success field。

    用真实 CloudApi（mock 底层 session 按 URL 路由），确保修复点
    （get_result_to_file 的 falsy 守卫）真实参与。
    """
    from nexus.distributed.export.client import CloudApi
    from nexus.distributed.export.results import _read_result

    class _Response:
        def __init__(self, payload: bytes):
            self.status = 200
            self.content = self
            self._payload = payload

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def read(self, size: int | None = None) -> bytes:
            if size is None:
                chunk, self._payload = self._payload, b''
                return chunk
            chunk = self._payload[:size]
            self._payload = self._payload[size:]
            return chunk

    class _Session:
        def get(self, url, headers=None, timeout=None):
            if 'control-events' in url:
                return _Response(b'{"items": []}')
            return _Response(b'{}')  # lost 题：/result 返回 200 + 空 JSON

    reader = _reader_for(records={1: json.dumps({'events': [{'name': 'other.event'}]})}, timestamps={1: 100})
    pool = export_results.ResultSpillPool(tmp_path / 'spill')
    api = CloudApi(_Session(), control_url='http://control', application_url='http://app', token='t')
    res = await _read_result(
        api=api,
        execution_id='child-1',
        trace_reader=reader,
        parent_execution_id=None,
        fallback_spill=pool,
        scan_factory=export_results._ScanSlot,
        scan_out={},
    )
    assert res.output is None  # 占位行（missing 语义），而非 ExportError
