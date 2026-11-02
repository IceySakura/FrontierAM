from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import sys

from collections import Counter
from pathlib import Path
from types import ModuleType

import pytest

from aiohttp import web
from aiohttp.test_utils import TestServer

from nexus.cli.commands.export import export_batch, register_export_parser
from nexus.distributed import export as cloud_export
from nexus.runner.batch import BatchDatabase


_TRACE_TABLE = 'trajectory_span_events_pr_0123456789abcdef0123456789abcdef'
_TRACE_DB_ARGS = (
    '--db-host',
    'db.test',
    '--db-name',
    'trace',
    '--db-user',
    'readonly',
    '--db-password',
    'password',
    '--trace-table',
    _TRACE_TABLE,
)
_RESULT_MODES = [
    pytest.param(False, None, id='full'),
    pytest.param(True, None, id='partial-default'),
    pytest.param(True, 'success', id='partial-success'),
    pytest.param(True, 'failed', id='partial-failed'),
    pytest.param(True, 'all', id='partial-all'),
]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='nexus')
    register_export_parser(parser.add_subparsers(dest='command'))
    return parser


def _parse(*argv: str) -> argparse.Namespace:
    return _parser().parse_args(['export', *argv])


def _create_local_database(path: Path) -> None:
    async def create() -> None:
        async with BatchDatabase(path) as database:
            await database.insert_run('local-1', {'run_id': 'local-1'}, {'source': 'fixture'})
            await database.update_run_status(
                'local-1',
                'success',
                task_output={'answer': 7},
                processor_results={},
            )

    asyncio.run(create())


def test_local_sqlite_export_keeps_existing_jsonl_behavior(tmp_path: Path) -> None:
    database = tmp_path / 'batch.db'
    output = tmp_path / 'results.jsonl'
    _create_local_database(database)

    assert export_batch(_parse(str(database), '-o', str(output))) == 0
    assert json.loads(output.read_text(encoding='utf-8'))['run_id'] == 'local-1'


def test_local_sqlite_export_allows_explicit_results_content(tmp_path: Path) -> None:
    database = tmp_path / 'batch.db'
    output = tmp_path / 'results.jsonl'
    _create_local_database(database)

    assert export_batch(_parse(str(database), '--content', 'results', '-o', str(output))) == 0
    assert json.loads(output.read_text(encoding='utf-8'))['run_id'] == 'local-1'


@pytest.mark.asyncio
@pytest.mark.parametrize('spec_type', ['nexus_single', 'nexus_live'])
@pytest.mark.parametrize('status', ['succeeded', 'failed'])
async def test_cloud_single_export_writes_public_api_result(tmp_path: Path, capsys, spec_type, status) -> None:
    cache_ids = {'input_cache_task_id': 'input-id', 'cache_task_id': 'd41d8cd98f00b204e9800998ecf8427e'}
    execution = {
        'execution_id': 'exe-single',
        'spec_type': spec_type,
        'status': status,
        'attempt_count': 1,
        'current_attempt_id': 'att-1',
        'result_artifact_id': 'art-1',
        'error_json': None,
    }

    async def execution_handler(_request: web.Request) -> web.Response:
        return web.json_response(execution)

    async def result_handler(_request: web.Request) -> web.Response:
        return web.json_response(
            {
                'success': status == 'succeeded',
                'task_output': {'answer': 42},
                'agent_output': None,
                'processor_results': {},
                'error': None,
                **cache_ids,
            }
        )

    async def events_handler(_request: web.Request) -> web.Response:
        return web.json_response({'items': []})

    app = web.Application()
    app.router.add_get('/v1/executions/exe-single', execution_handler)
    app.router.add_get('/v1/executions/exe-single/result', result_handler)
    app.router.add_get('/v1/executions/exe-single/control-events', events_handler)
    server = TestServer(app)
    await server.start_server()
    output = tmp_path / 'cloud.jsonl'
    base_url = str(server.make_url('/')).rstrip('/')
    try:
        result = await asyncio.to_thread(
            export_batch,
            _parse(
                '--execution-id',
                'exe-single',
                '--control-url',
                base_url,
                '--application-url',
                base_url,
                '--taihu-token',
                'test-pat',
                '-o',
                str(output),
            ),
        )
    finally:
        await server.close()

    assert result == 0
    record = json.loads(output.read_text(encoding='utf-8'))
    assert record['status'] == ('success' if status == 'succeeded' else 'failed')
    assert record['task_output'] == {'answer': 42}
    assert record['metadata'] == {'_nexus': cache_ids}
    captured = capsys.readouterr()
    success_message = f'Exported JSONL to {output.resolve()}\n'
    assert success_message not in captured.out
    assert success_message in captured.err


@pytest.mark.parametrize('failure', [False, True])
def test_result_export_warnings_use_stderr_and_restore_logging(tmp_path: Path, monkeypatch, capsys, failure) -> None:
    from nexus.distributed.export import cli as export_cli

    root = logging.getLogger()
    original_handlers = root.handlers[:]
    original_level = root.level
    stdout_handler = logging.StreamHandler(sys.stdout)
    root.addHandler(stdout_handler)
    original_filters = {handler: handler.filters[:] for handler in root.handlers}

    async def export_with_conflict(_args) -> None:
        cloud_export.make_jsonl_record(
            run_id='exe-conflict',
            distributed_status='succeeded',
            start_time=None,
            end_time=None,
            retry_count=0,
            fallback_error=None,
            output={'success': True, 'cache_task_id': 'recorded-id'},
            metadata={'_nexus': {'cache_task_id': 'existing-id'}},
        )
        if failure:
            raise cloud_export.ExportError('fixture failure')

    monkeypatch.setattr(export_cli, '_run', export_with_conflict)
    try:
        args = argparse.Namespace(output=tmp_path / 'result.jsonl')
        if failure:
            with pytest.raises(cloud_export.ExportError, match='fixture failure'):
                export_cli.run_cloud_export(args)
        else:
            assert export_cli.run_cloud_export(args) == 0
        assert root.handlers == [*original_handlers, stdout_handler]
        assert root.level == original_level
        assert {handler: handler.filters for handler in root.handlers} == original_filters
    finally:
        root.removeHandler(stdout_handler)
        stdout_handler.close()

    captured = capsys.readouterr()
    assert captured.out == ''
    assert 'execution=exe-conflict field=cache_task_id reason=metadata_conflict' in captured.err
    assert 'recorded-id' not in captured.err
    assert 'existing-id' not in captured.err


@pytest.mark.parametrize(
    'argv',
    [
        ('batch.db', '--execution-id', 'exe-1'),
        (),
        ('batch.db', '--content', 'traces'),
        ('batch.db', '--follow'),
        ('batch.db', '--control-url', 'https://control.test'),
        ('--execution-id', 'exe-1', '--format', 'csv'),
        ('--execution-id', 'exe-1', '--follow'),
        ('--execution-id', 'exe-1', '--resume'),
        ('--execution-id', 'exe-1', '--content', 'traces', '--resume'),
        ('--execution-id', 'exe-1', '--content', 'traces', '--force'),
        ('--execution-id', 'exe-1', '-o', '-'),
    ],
)
def test_invalid_combinations_exit_two_without_creating_output(tmp_path: Path, argv: tuple[str, ...]) -> None:
    output = tmp_path / 'nested' / 'results.jsonl'
    args = _parse(*argv, '-o', str(output)) if '-o' not in argv else _parse(*argv)

    assert export_batch(args) == 2
    assert not output.exists()
    assert not output.parent.exists()


def test_cloud_export_requires_output() -> None:
    assert export_batch(_parse('--execution-id', 'exe-1', '--taihu-token', 'test-pat')) == 2


@pytest.mark.parametrize(
    'override',
    [
        ('--execution-id', '   '),
        ('--control-url', ''),
        ('--control-url', '   '),
        ('--application-url', ''),
        ('--application-url', '   '),
        ('--taihu-token', '   '),
        ('--db-host', '   '),
        ('--db-name', '   '),
        ('--db-user', '   '),
        ('--db-password', '   '),
        ('--trace-table', '   '),
        ('--db-port', '0'),
        ('--db-port', '65536'),
        ('--trace-table', 'trajectory_span_events_pr_deadbeef'),
    ],
)
def test_invalid_cloud_source_values_stop_before_dispatch_or_output(
    monkeypatch,
    tmp_path: Path,
    override: tuple[str, ...],
) -> None:
    dispatched = False

    def dispatch(_args: argparse.Namespace) -> int:
        nonlocal dispatched
        dispatched = True
        return 0

    monkeypatch.setattr(cloud_export, 'run_cloud_export', dispatch)
    monkeypatch.setenv('NEXUS_CONTROL_URL', 'https://environment-control.test')
    monkeypatch.setenv('NEXUS_APPLICATION_URL', 'https://environment-application.test')
    output = tmp_path / 'nested' / 'results.jsonl'
    args = _parse(
        '--execution-id',
        'exe-1',
        '--control-url',
        'https://control.test',
        '--application-url',
        'https://application.test',
        '--taihu-token',
        'test-pat',
        *_TRACE_DB_ARGS,
        *override,
        '-o',
        str(output),
    )

    assert export_batch(args) == 2
    assert dispatched is False
    assert not output.exists()
    assert not output.parent.exists()


@pytest.mark.parametrize('content', ['results', 'traces'])
def test_explicit_poll_interval_requires_followed_traces(monkeypatch, tmp_path: Path, content: str) -> None:
    dispatched = False

    def dispatch(_args: argparse.Namespace) -> int:
        nonlocal dispatched
        dispatched = True
        return 0

    monkeypatch.setattr(cloud_export, 'run_cloud_export', dispatch)
    event_export = ModuleType('nexus.distributed.export_events')
    event_export.run_trace_export = dispatch  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, event_export.__name__, event_export)
    output = tmp_path / 'nested' / 'results.jsonl'
    argv = [
        '--execution-id',
        'exe-1',
        '--content',
        content,
        '--control-url',
        'https://control.test',
        '--application-url',
        'https://application.test',
        '--taihu-token',
        'test-pat',
        '--poll-interval',
        '2',
        '-o',
        str(output),
    ]
    if content == 'traces':
        argv.extend(_TRACE_DB_ARGS)

    assert export_batch(_parse(*argv)) == 2
    assert dispatched is False
    assert not output.exists()
    assert not output.parent.exists()


def test_cloud_export_errors_use_stderr_without_exposing_credentials(monkeypatch, capsys, tmp_path: Path) -> None:
    def fail_with_secret(args: argparse.Namespace) -> int:
        raise RuntimeError(f'upstream request included {args.taihu_token}')

    monkeypatch.setattr(cloud_export, 'run_cloud_export', fail_with_secret)
    args = _parse(
        '--execution-id',
        'exe-1',
        '--application-url',
        'https://application.test',
        '--taihu-token',
        'secret-pat',
        '-o',
        str(tmp_path / 'results.jsonl'),
    )

    assert export_batch(args) == 1
    captured = capsys.readouterr()
    assert captured.out == ''
    assert captured.err == 'Export failed: RuntimeError\n'
    assert 'secret-pat' not in captured.err


@pytest.mark.parametrize('value', ['0', '-1', 'nan', 'inf', '-inf'])
def test_poll_interval_requires_a_positive_finite_number(value: str) -> None:
    with pytest.raises(SystemExit) as raised:
        _parse('--execution-id', 'exe-1', '--poll-interval', value, '-o', 'result.jsonl')

    assert raised.value.code == 2


def test_parser_accepts_trace_follow_source_options() -> None:
    args = _parse(
        '--execution-id',
        'exe-1',
        '--content',
        'traces',
        '--follow',
        '--resume',
        '--poll-interval',
        '0.25',
        '--control-url',
        'https://control.test',
        '--taihu-token-file',
        '/tmp/token',
        '--db-host',
        'db.test',
        '--db-port',
        '3307',
        '--db-name',
        'trace',
        '--db-user',
        'readonly',
        '--db-password-file',
        '/tmp/password',
        '--trace-table',
        'trajectory_span_events_pr_0123456789abcdef0123456789abcdef',
        '-o',
        'events.jsonl',
    )

    assert args.database is None
    assert args.content == 'traces'
    assert args.resume is True
    assert math.isclose(args.poll_interval, 0.25)
    assert args.db_port == 3307


def test_trace_command_restores_logging_and_writes_snapshot_without_application(monkeypatch, tmp_path, capsys):
    import logging

    from tests.unit.distributed.test_cloud_event_export import Source

    from nexus.distributed.export import cli as export_events

    async def execution(self, execution_id):
        return {'execution_id': execution_id, 'spec_type': 'nexus_batch', 'status': 'running'}

    class LoggedSource(Source):
        async def discover_raw_streams(self, *args, **kwargs):
            logging.getLogger('nexus.runner.trace.tdsql.writer').warning('source fixture warning')
            return await super().discover_raw_streams(*args, **kwargs)

    async def classic_table(self):
        return False

    monkeypatch.setattr(export_events.TdsqlTraceReader, 'has_partition_column', classic_table)
    monkeypatch.setattr(export_events.CloudApi, 'get_execution', execution)
    monkeypatch.setattr('nexus.runner.trace.tdsql.sink_source.TDSQLTraceSource', lambda **kwargs: LoggedSource())
    handlers = logging.getLogger().handlers[:]
    output = tmp_path / 'events'
    args = _parse(
        '--execution-id',
        'batch-1',
        '--content',
        'traces',
        '--taihu-token',
        'secret',
        *_TRACE_DB_ARGS,
        '-o',
        str(output),
    )
    assert export_batch(args) == 0
    assert len(output.read_text().splitlines()) == 2
    assert logging.getLogger().handlers == handlers
    captured = capsys.readouterr()
    assert captured.out == ''
    assert 'source fixture warning' in captured.err
    assert 'secret' not in captured.err


@pytest.mark.parametrize('signum', [2, 15])
def test_trace_command_preserves_signal_exit_and_restores_handler(monkeypatch, tmp_path, signum, capsys):
    import os
    import signal

    from tests.unit.distributed.test_cloud_event_export import Source

    from nexus.distributed.export import cli as export_events

    async def execution(self, execution_id):
        return {'execution_id': execution_id, 'spec_type': 'nexus_batch', 'status': 'running'}

    class InterruptedSource(Source):
        async def read_raw_events_after(self, *args, **kwargs):
            os.kill(os.getpid(), signum)
            await asyncio.sleep(0)
            return await super().read_raw_events_after(*args, **kwargs)

    async def classic_table(self):
        return False

    monkeypatch.setattr(export_events.TdsqlTraceReader, 'has_partition_column', classic_table)
    monkeypatch.setattr(export_events.CloudApi, 'get_execution', execution)
    monkeypatch.setattr('nexus.runner.trace.tdsql.sink_source.TDSQLTraceSource', lambda **kwargs: InterruptedSource())
    before = signal.getsignal(signum)
    output = tmp_path / 'events'
    args = _parse(
        '--execution-id',
        'batch-1',
        '--content',
        'traces',
        '--follow',
        '--taihu-token',
        'secret',
        *_TRACE_DB_ARGS,
        '-o',
        str(output),
    )
    assert export_batch(args) == 128 + signum
    assert signal.getsignal(signum) == before
    assert json.loads(Path(f'{output}.manifest.json').read_text())['status'] == 'interrupted'
    assert capsys.readouterr().out == ''


def test_trace_resume_mismatched_source_exits_usage_without_reading_database(monkeypatch, tmp_path):
    from tests.unit.distributed.test_cloud_event_export_output import _source

    from nexus.distributed.export import cli as export_events
    from nexus.distributed.export_output import JsonlEventWriter

    output = tmp_path / 'events'

    async def prepare():
        writer = JsonlEventWriter(output, source=_source())
        await writer.open()
        await writer.close(reason='interrupted')

    asyncio.run(prepare())
    before = output.read_bytes()

    async def execution(self, execution_id):
        return {'execution_id': execution_id, 'spec_type': 'nexus_batch', 'status': 'running'}

    class Unreadable:
        async def discover_raw_streams(self, *args, **kwargs):
            raise AssertionError('database must not be queried before resume identity validation')

        async def plan_raw_event_read(self, *args, **kwargs):
            raise AssertionError('database must not be queried before resume identity validation')

    async def classic_table(self):
        raise AssertionError('schema must not be queried before resume identity validation')

    monkeypatch.setattr(export_events.TdsqlTraceReader, 'has_partition_column', classic_table)
    monkeypatch.setattr(export_events.CloudApi, 'get_execution', execution)
    monkeypatch.setattr('nexus.runner.trace.tdsql.sink_source.TDSQLTraceSource', lambda **kwargs: Unreadable())
    args = _parse(
        '--execution-id',
        'batch-1',
        '--content',
        'traces',
        '--follow',
        '--resume',
        '--taihu-token',
        'secret',
        *_TRACE_DB_ARGS,
        '-o',
        str(output),
    )
    assert export_batch(args) == 2
    assert output.read_bytes() == before


def test_partition_key_requires_the_rest_of_the_trace_database_arguments(tmp_path: Path, capsys) -> None:
    args = _parse(
        '--execution-id',
        'exe-1',
        '--content',
        'traces',
        '--taihu-token',
        'secret',
        '--parent-created-at',
        '2026-09-20T10:11:12.131415',
        '-o',
        str(tmp_path / 'events.jsonl'),
    )

    assert export_batch(args) == 2
    assert 'incomplete Trace Database arguments' in capsys.readouterr().err


def test_partition_key_must_be_an_iso_datetime(tmp_path: Path, capsys) -> None:
    args = _parse(
        '--execution-id',
        'exe-1',
        '--content',
        'traces',
        '--taihu-token',
        'secret',
        *_TRACE_DB_ARGS,
        '--parent-created-at',
        'yesterday',
        '-o',
        str(tmp_path / 'events.jsonl'),
    )

    assert export_batch(args) == 2
    assert '--parent-created-at must be an ISO 8601 datetime' in capsys.readouterr().err


def test_partition_key_is_rejected_for_local_database_exports(tmp_path: Path, capsys) -> None:
    database = tmp_path / 'batch.db'
    _create_local_database(database)
    args = _parse(
        str(database),
        '--parent-created-at',
        '2026-09-20T10:11:12.131415',
        '-o',
        str(tmp_path / 'results.jsonl'),
    )

    assert export_batch(args) == 2
    assert 'Cloud export options cannot be used with a local database' in capsys.readouterr().err


def test_trace_backend_requires_the_rest_of_the_trace_database_arguments(tmp_path: Path, capsys) -> None:
    """It names which backend the credentials belong to, so it is meaningless alone."""
    args = _parse(
        '--execution-id',
        'exe-1',
        '--content',
        'traces',
        '--taihu-token',
        'secret',
        '--trace-backend',
        'tdsql-boundless',
        '-o',
        str(tmp_path / 'events.jsonl'),
    )

    assert export_batch(args) == 2
    assert 'incomplete Trace Database arguments' in capsys.readouterr().err


def test_trace_backend_must_not_be_blank(tmp_path: Path, capsys) -> None:
    args = _parse(
        '--execution-id',
        'exe-1',
        '--content',
        'traces',
        '--taihu-token',
        'secret',
        *_TRACE_DB_ARGS,
        '--trace-backend',
        '   ',
        '-o',
        str(tmp_path / 'events.jsonl'),
    )

    assert export_batch(args) == 2
    assert '--trace-backend must not be empty' in capsys.readouterr().err


def test_trace_backend_is_rejected_for_local_database_exports(tmp_path: Path, capsys) -> None:
    database = tmp_path / 'batch.db'
    _create_local_database(database)
    args = _parse(
        str(database),
        '--trace-backend',
        'tdsql',
        '-o',
        str(tmp_path / 'results.jsonl'),
    )

    assert export_batch(args) == 2
    assert 'Cloud export options cannot be used with a local database' in capsys.readouterr().err


@pytest.mark.parametrize(('target', 'parent_id'), [('nexus_batch', None), ('nexus_single', 'batch-parent')])
def test_trace_export_probes_boundless_and_uses_parent_partition(monkeypatch, tmp_path, target, parent_id):
    from tests.unit.distributed.test_cloud_event_export import Source

    from nexus.distributed.export import cli
    from nexus.runner.trace.tdsql.schema import BOUNDLESS_DIALECT

    selected = []
    scopes = []

    class PartitionedSource(Source):
        dialect = 'tdsql'

        async def discover_raw_streams(self, metadata, **kwargs):
            assert self.dialect == BOUNDLESS_DIALECT
            scopes.append(metadata.copy())
            return []

    def source(**kwargs):
        selected.append(kwargs)
        return PartitionedSource()

    async def execution(self, execution_id):
        if execution_id == 'batch-parent':
            return {'execution_id': execution_id, 'created_at': '2026-09-25T01:02:03', 'spec_type': 'nexus_batch'}
        return {
            'execution_id': execution_id,
            'spec_type': target,
            'status': 'succeeded',
            'created_at': '2026-09-25T01:02:03' if parent_id is None else '2026-09-26T02:03:04',
            'parent_execution_id': parent_id,
        }

    async def partitioned(self):
        return True

    monkeypatch.setattr(cli.CloudApi, 'get_execution', execution)
    monkeypatch.setattr(cli.TdsqlTraceReader, 'has_partition_column', partitioned)
    monkeypatch.setattr('nexus.runner.trace.tdsql.sink_source.TDSQLTraceSource', source)
    args = _parse(
        '--execution-id',
        'target',
        '--content',
        'traces',
        '--taihu-token',
        'unused',
        *_TRACE_DB_ARGS,
        '-o',
        str(tmp_path / 'events.jsonl'),
    )
    assert export_batch(args) == 0
    assert selected[0]['read_driver'] == 'aiomysql'
    assert scopes
    assert all(scope['parent_created_at'] == '2026-09-25 01:02:03.000000' for scope in scopes)


@pytest.mark.parametrize('content', ['results', 'traces'])
def test_allow_partial_dispatches_only_cloud_results(monkeypatch, tmp_path, capsys, content):
    from nexus.distributed.export import cli

    calls = []

    async def capture_export(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(cli, 'export_execution', capture_export)
    args = _parse(
        '--execution-id',
        'batch',
        '--content',
        content,
        '--allow-partial',
        '--application-url',
        'https://application.test',
        '--taihu-token',
        'test-token',
        '-o',
        str(tmp_path / 'results.jsonl'),
    )
    assert export_batch(args) == (0 if content == 'results' else 2)
    if content == 'results':
        assert len(calls) == 1
        assert calls[0]['allow_partial'] is True
    else:
        assert calls == []
        assert '--allow-partial is available only with --content results' in capsys.readouterr().err


@pytest.mark.parametrize('standalone', [False, True])
@pytest.mark.parametrize(('allow_partial', 'result_status'), _RESULT_MODES)
def test_result_status_parsers_preserve_raw_value(standalone, allow_partial, result_status):
    argv = ['--execution-id', 'batch', '--output', 'results.jsonl']
    if allow_partial:
        argv.append('--allow-partial')
    if result_status is not None:
        argv.extend(['--result-status', result_status])
    args = cloud_export.parse_args(argv) if standalone else _parse(*argv)
    assert args.allow_partial is allow_partial
    assert args.result_status == result_status


@pytest.mark.parametrize('standalone', [False, True])
@pytest.mark.parametrize('value', ['', ' ', ',item-001', 'item-001,', 'item-001,,item-002'])
def test_invalid_item_ids_fail_before_control_reads(monkeypatch, tmp_path, capsys, standalone, value):
    from nexus.distributed.export.client import CloudApi

    async def unexpected(*args, **kwargs):
        pytest.fail('invalid item IDs must fail before Control reads')

    monkeypatch.setattr(CloudApi, 'get_execution', unexpected)
    output = tmp_path / 'results.jsonl'
    argv = [
        '--execution-id',
        'batch',
        '--item-ids',
        value,
        '--application-url',
        'https://application.test',
        '--taihu-token',
        'test-token',
        '--output',
        str(output),
    ]
    code = cloud_export.main(argv) if standalone else export_batch(_parse(*argv))
    assert code == 1
    assert '--item-ids must contain non-empty item IDs' in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize(
    ('source', 'expected_code'),
    [
        (['batch.db'], 2),
        (['--execution-id', 'batch', '--content', 'traces'], 2),
        (['--execution-id', 'batch', '--execution-id', 'another-batch'], 1),
    ],
)
def test_item_ids_require_one_cloud_results_target(tmp_path, capsys, source, expected_code):
    output = tmp_path / 'results.jsonl'
    code = export_batch(
        _parse(
            *source,
            '--item-ids',
            'item-001',
            '--application-url',
            'https://application.test',
            '--taihu-token',
            'test-token',
            '--output',
            str(output),
        )
    )
    assert code == expected_code
    assert not output.exists()
    assert ('Invalid export arguments:' if expected_code == 2 else 'Export failed:') in capsys.readouterr().err


@pytest.mark.parametrize('standalone', [False, True])
@pytest.mark.parametrize('result_status', ['', 'succeeded', 'lost', 'SUCCESS', ' all '])
def test_result_status_parsers_reject_unknown_values(standalone, result_status):
    argv = ['--execution-id', 'batch', '--output', 'results.jsonl', '--allow-partial', '--result-status', result_status]
    with pytest.raises(SystemExit) as raised:
        cloud_export.parse_args(argv) if standalone else _parse(*argv)
    assert raised.value.code == 2


@pytest.mark.parametrize('standalone', [False, True])
@pytest.mark.parametrize('result_status', ['success', 'failed', 'all'])
def test_result_status_requires_partial_before_resources(monkeypatch, tmp_path, capsys, standalone, result_status):
    from nexus.distributed.export import cli

    def unexpected(*args, **kwargs):
        pytest.fail('invalid result selection must not read secrets, create readers or open HTTP sessions')

    monkeypatch.setattr(cli, 'resolve_secret', unexpected)
    monkeypatch.setattr(cli, 'create_trace_reader', unexpected)
    monkeypatch.setattr(cli.aiohttp, 'ClientSession', unexpected)
    output = tmp_path / 'nested' / 'results.jsonl'
    argv = [
        '--execution-id',
        'batch',
        '--application-url',
        'https://application.test',
        '--taihu-token-file',
        str(tmp_path / 'missing-token'),
        '--result-status',
        result_status,
        '--output',
        str(output),
    ]
    assert (cli.main(argv) if standalone else export_batch(_parse(*argv))) == (1 if standalone else 2)
    assert '--result-status requires --allow-partial' in capsys.readouterr().err
    assert not output.parent.exists()


@pytest.mark.parametrize('source', ['local', 'traces'])
@pytest.mark.parametrize('allow_partial', [False, True])
@pytest.mark.parametrize('result_status', ['success', 'failed', 'all'])
def test_result_status_rejects_local_and_traces_before_dispatch(
    monkeypatch, tmp_path, capsys, source, allow_partial, result_status
):
    from nexus.distributed.export import cli

    def unexpected(*args, **kwargs):
        pytest.fail('invalid export mode must not dispatch')

    monkeypatch.setattr(cli, 'run_trace_export', unexpected)
    monkeypatch.setattr(cloud_export, 'run_cloud_export', unexpected)
    output = tmp_path / 'nested' / 'results.jsonl'
    argv = ['batch.db'] if source == 'local' else ['--execution-id', 'batch', '--content', 'traces']
    argv.extend(['--taihu-token', 'test-token'] if source == 'traces' else [])
    argv.extend(['--result-status', result_status, '--output', str(output)])
    if allow_partial:
        argv.append('--allow-partial')
    assert export_batch(_parse(*argv)) == 2
    message = (
        'Cloud export options cannot be used with a local database'
        if source == 'local'
        else '--result-status is available only with --content results'
    )
    assert message in capsys.readouterr().err
    assert not output.parent.exists()


@pytest.mark.parametrize('standalone', [False, True])
@pytest.mark.parametrize('with_database', [False, True])
@pytest.mark.parametrize(('allow_partial', 'result_status'), _RESULT_MODES)
def test_repeated_execution_ids_reach_the_same_exporter_with_database(
    monkeypatch, tmp_path, standalone, with_database, allow_partial, result_status
):
    from nexus.distributed.export import cli

    calls = []
    reader = object()

    async def capture(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(cli, 'export_execution', capture)
    monkeypatch.setattr(cli, 'create_trace_reader', lambda args: reader if args.db_host else None)
    argv = [
        '--execution-id',
        'batch-a',
        '--execution-id',
        'child-a',
        '--execution-id',
        'batch-a',
        '--application-url',
        'https://application.test',
        '--taihu-token',
        'test-token',
        '--output',
        str(tmp_path / 'results.jsonl'),
    ]
    if allow_partial:
        argv.append('--allow-partial')
    if result_status is not None:
        argv.extend(['--result-status', result_status])
    if with_database:
        argv.extend(_TRACE_DB_ARGS)
    assert (cli.main(argv) if standalone else export_batch(_parse(*argv))) == 0
    assert len(calls) == 1
    assert list(dict.fromkeys(calls[0]['execution_id'])) == ['batch-a', 'child-a']
    assert calls[0]['trace_reader'] is (reader if with_database else None)
    assert calls[0]['allow_partial'] is allow_partial
    assert calls[0]['result_status'] == result_status


def test_repeated_execution_ids_are_not_accepted_for_traces(tmp_path, capsys):
    output = tmp_path / 'traces.jsonl'
    args = _parse('--execution-id', 'a', '--execution-id', 'b', '--content', 'traces', '-o', str(output))
    assert export_batch(args) == 2
    assert 'only with --content results' in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize('child_first', [False, True])
async def test_cli_parent_and_redundant_child_preserve_batch_records(tmp_path, child_first):
    from tests.unit.distributed.test_cloud_multi_execution_export import MixedApi

    api = MixedApi()

    async def execution(request):
        return web.json_response(await api.get_execution(request.match_info['id']))

    async def summary(request):
        return web.json_response(await api.get_batch_summary(request.match_info['id']))

    async def items(request):
        return web.json_response(await api.get_batch_items(request.match_info['id'], request.query.get('cursor')))

    async def result(request):
        return web.json_response(await api.get_result(request.match_info['id']))

    async def events(request):
        return web.json_response({'items': []})

    app = web.Application()
    app.router.add_get('/v1/executions/{id}', execution)
    app.router.add_get('/v1/executions/{id}/batch', summary)
    app.router.add_get('/v1/executions/{id}/batch/items', items)
    app.router.add_get('/v1/executions/{id}/result', result)
    app.router.add_get('/v1/executions/{id}/control-events', events)
    server = TestServer(app)
    await server.start_server()
    base_url = str(server.make_url('/')).rstrip('/')
    path = tmp_path / 'results.jsonl'
    ids = ['exe-a', 'batch-a'] if child_first else ['batch-a', 'exe-a']
    try:
        code = await asyncio.to_thread(
            export_batch,
            _parse(
                '--execution-id',
                ids[0],
                '--execution-id',
                ids[1],
                '--control-url',
                base_url,
                '--application-url',
                base_url,
                '--taihu-token',
                'test-token',
                '-o',
                str(path),
            ),
        )
    finally:
        await server.close()
    assert code == 0
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row['run_id'] for row in rows] == ['shared-run', 'a-failed', 'a-cancelled']
    assert rows[-1]['error']['type'] == 'BatchItemNoChildExecution'
    assert rows[0]['metadata']['_nexus']['child_execution_id'] == 'exe-a'
    assert json.loads(path.with_suffix('.jsonl.batch_metadata.json').read_text()) == {'dataset': 'a'}


@pytest.fixture
async def cloud_results_http():
    from tests.unit.distributed.test_cloud_multi_execution_export import MixedApi

    api = MixedApi()
    for index, (status, run_id) in enumerate([('expired', 'a-timeout'), ('killed', 'a-killed')], start=4):
        child_id = f'exe-{status}'
        api.items['batch-a'].append(api.item(f'item-{index:03}', child_id, status, run_id))
        api.executions[child_id] = {
            **api.executions['exe-z'],
            'execution_id': child_id,
            'status': status,
        }
        api.outputs[child_id] = {'success': False, 'agent_output': {'saved': status}}
    requests = []

    async def execution(request):
        requests.append(('execution', request.match_info['id']))
        return web.json_response(await api.get_execution(request.match_info['id']))

    async def summary(request):
        requests.append(('summary', request.match_info['id']))
        return web.json_response(await api.get_batch_summary(request.match_info['id']))

    async def items(request):
        requests.append(('items', request.match_info['id']))
        return web.json_response(await api.get_batch_items(request.match_info['id'], request.query.get('cursor')))

    async def result(request):
        requests.append(('result', request.match_info['id']))
        output = await api.get_result(request.match_info['id'])
        if output is None:
            raise web.HTTPNotFound()
        return web.json_response(output)

    async def events(request):
        requests.append(('events', request.match_info['id']))
        return web.json_response({'items': []})

    app = web.Application()
    app.router.add_get('/v1/executions/{id}', execution)
    app.router.add_get('/v1/executions/{id}/batch', summary)
    app.router.add_get('/v1/executions/{id}/batch/items', items)
    app.router.add_get('/v1/executions/{id}/result', result)
    app.router.add_get('/v1/executions/{id}/control-events', events)
    async with TestServer(app) as server:
        yield api, str(server.make_url('/')).rstrip('/'), requests


@pytest.mark.parametrize('standalone', [False, True])
@pytest.mark.parametrize(('allow_partial', 'result_status'), _RESULT_MODES)
async def test_item_ids_http_export_preserves_selected_batch_rows(
    tmp_path, cloud_results_http, standalone, allow_partial, result_status
):
    api, base_url, requests = cloud_results_http
    output = tmp_path / 'items.jsonl'
    argv = [
        '--execution-id',
        'batch-a',
        '--item-ids',
        'item-005, item-001',
        '--item-ids',
        'item-001',
        '--control-url',
        base_url,
        '--application-url',
        base_url,
        '--taihu-token',
        'test-token',
        '--output',
        str(output),
    ]
    if allow_partial:
        argv.append('--allow-partial')
    if result_status is not None:
        argv.extend(['--result-status', result_status])
    code = (
        await asyncio.to_thread(cloud_export.main, argv)
        if standalone
        else await asyncio.to_thread(export_batch, _parse(*argv))
    )
    assert code == 0
    expected = {
        'all': [('shared-run', 'exe-a'), ('a-killed', 'exe-killed')],
        'success': [('shared-run', 'exe-a')],
        'failed': [('a-killed', 'exe-killed')],
    }[result_status or ('success' if allow_partial else 'all')]
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row['run_id'] for row in rows] == [run_id for run_id, _ in expected]
    assert Counter(api.result_calls) == Counter(child_id for _, child_id in expected)
    assert Counter(child for kind, child in requests if kind == 'events') == Counter(
        child_id for _, child_id in expected
    )
    assert all(row['metadata']['_nexus']['parent_execution_id'] == 'batch-a' for row in rows)
    for row, (_, child_id) in zip(rows, expected, strict=True):
        assert row['metadata']['_nexus']['child_execution_id'] == child_id
        assert 'item_id' not in row['metadata']['_nexus']
        assert row['agent_output'] == api.outputs[child_id].get('agent_output')
    assert json.loads(output.with_suffix('.jsonl.batch_metadata.json').read_text()) == {'dataset': 'a'}


@pytest.mark.parametrize('standalone', [False, True])
@pytest.mark.parametrize(('allow_partial', 'result_status'), _RESULT_MODES)
@pytest.mark.parametrize('targets', ['batch', 'batches', 'mixed'])
async def test_result_status_http_modes_preserve_pipeline(
    tmp_path, capsys, cloud_results_http, standalone, allow_partial, result_status, targets
):
    api, base_url, requests = cloud_results_http
    if allow_partial:
        api.executions['batch-a']['status'] = 'running'
        api.items['batch-a'].append(api.item('item-006', None, 'pending', 'a-pending'))
    ids = {
        'batch': ['batch-a'],
        'batches': ['batch-a', 'batch-b', 'batch-a'],
        'mixed': ['exe-z', 'exe-a', 'batch-a', 'batch-b', 'exe-b', 'batch-a', 'exe-z'],
    }[targets]
    output = tmp_path / 'results.jsonl'
    argv = [
        '--control-url',
        base_url,
        '--application-url',
        base_url,
        '--taihu-token',
        'test-token',
        '--output',
        str(output),
    ]
    for execution_id in ids:
        argv.extend(['--execution-id', execution_id])
    if allow_partial:
        argv.append('--allow-partial')
    if result_status is not None:
        argv.extend(['--result-status', result_status])
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    code = (
        await asyncio.to_thread(cloud_export.main, argv)
        if standalone
        else await asyncio.to_thread(export_batch, _parse(*argv))
    )
    assert code == 0
    if standalone:
        assert root.handlers == handlers
        assert root.level == level

    mode = result_status or ('success' if allow_partial else 'all')
    expected = {
        'all': ['shared-run', 'a-failed', 'a-cancelled', 'a-timeout', 'a-killed'],
        'success': ['shared-run'],
        'failed': ['a-failed', 'a-cancelled', 'a-timeout', 'a-killed'],
    }[mode]
    expected_children = {
        'all': ['exe-a', 'exe-z', 'exe-expired', 'exe-killed'],
        'success': ['exe-a'],
        'failed': ['exe-z', 'exe-expired', 'exe-killed'],
    }[mode]
    if targets != 'batch' and mode != 'success':
        expected += ['shared-run', 'b-empty']
        expected_children += ['exe-b', 'exe-empty']
    if targets == 'mixed':
        if mode == 'success':
            expected += ['exe-z', 'exe-b']
            expected_children += ['exe-z', 'exe-b']
        elif mode == 'failed':
            expected += ['exe-a']
            expected_children += ['exe-a']
    if allow_partial and mode in {'failed', 'all'}:
        expected = [run_id for run_id in expected if run_id not in {'a-cancelled', 'b-empty'}]

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row['run_id'] for row in rows] == expected
    assert Counter(api.result_calls) == Counter(expected_children)
    assert Counter(child for kind, child in requests if kind == 'events') == Counter(expected_children)
    assert all(row['status'] in {'success', 'failed', 'timeout', 'cancelled'} for row in rows)
    for row in rows:
        if row['run_id'].startswith('exe-'):
            assert row['metadata'] is None
            child_id = row['run_id']
        else:
            assert row['metadata']['_nexus']['original_run_id'] == row['run_id']
            child_id = row['metadata']['_nexus']['child_execution_id']
            assert row['metadata']['_nexus']['parent_execution_id'] in ids
            assert 'item_id' not in row['metadata']['_nexus']
        if child_id is None:
            assert row['error']['type'] == 'BatchItemNoChildExecution'
        for field in ('task_output', 'agent_output', 'processor_results'):
            assert row[field] == (api.outputs.get(child_id) or {}).get(field)
    if mode != 'success':
        failed = next(row for row in rows if row['run_id'] == 'a-failed')
        assert failed['status'] == 'failed'
        assert failed['error'] == {'type': 'TraceDrainFailed'}
        assert next(row for row in rows if row['run_id'] == 'a-timeout')['status'] == 'timeout'
        assert next(row for row in rows if row['run_id'] == 'a-killed')['status'] == 'cancelled'
    sidecar = output.with_suffix('.jsonl.batch_metadata.json')
    assert json.loads(sidecar.read_text()) == (
        api.metadata['batch-a'] if targets == 'batch' else {'batches': api.metadata}
    )
    captured = capsys.readouterr()
    assert f'Exported JSONL to {output.resolve()}' in captured.err
    if standalone:
        assert captured.out == ''
        assert [line for line in captured.err.splitlines() if 'Results export:' in line] == [
            f'Results export: written_records={len(rows)}'
        ]


@pytest.mark.parametrize('standalone', [False, True])
@pytest.mark.parametrize('mode', ['failed', 'all'])
async def test_failed_http_omits_error_only_results_after_reading_without_losing_batch_metadata(
    tmp_path, capsys, cloud_results_http, standalone, mode
):
    api, base_url, requests = cloud_results_http
    api.items['batch-a'] = api.items['batch-a'][1:]
    api.outputs.clear()
    # An explicitly selected successful execution is retained even with empty artifact fields.
    api.outputs['exe-a'] = {'success': True}
    # All three empty artifact fields count as error-only, despite a persisted RunOutput.
    api.outputs['exe-z'] = {
        'success': False,
        'error': {'type': 'TaskFailed'},
        'task_output': {},
        'agent_output': [],
        'processor_results': '',
    }
    output = tmp_path / 'filtered.jsonl'
    argv = [
        '--execution-id',
        'batch-a',
        '--execution-id',
        'exe-a',
        '--execution-id',
        'exe-empty',
        '--application-url',
        base_url,
        '--control-url',
        base_url,
        '--taihu-token',
        'test-token',
        '--allow-partial',
        '--result-status',
        mode,
        '--output',
        str(output),
    ]
    code = (
        await asyncio.to_thread(cloud_export.main, argv)
        if standalone
        else await asyncio.to_thread(export_batch, _parse(*argv))
    )
    assert code == 0
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row['run_id'] for row in rows] == ['exe-a']
    assert next(row for row in rows if row['run_id'] == 'exe-a')['status'] == 'success'
    assert Counter(child for kind, child in requests if kind == 'result') == Counter(
        [
            'exe-z',
            'exe-expired',
            'exe-killed',
            'exe-a',
            'exe-empty',
        ]
    )
    assert json.loads(output.with_suffix('.jsonl.batch_metadata.json').read_text()) == api.metadata['batch-a']
    captured = capsys.readouterr()
    if standalone:
        assert [line for line in captured.err.splitlines() if 'Results export:' in line] == [
            f'Results export: written_records={len(rows)}'
        ]


@pytest.mark.parametrize('standalone', [False, True])
@pytest.mark.parametrize('result_status', [None, 'success', 'failed', 'all'])
@pytest.mark.parametrize('ids', [['exe-a'], ['exe-z', 'exe-b', 'exe-z']])
async def test_partial_http_rejects_single_live_only_before_results(
    tmp_path, capsys, cloud_results_http, standalone, result_status, ids
):
    api, base_url, requests = cloud_results_http
    output = tmp_path / 'nested' / 'results.jsonl'
    argv = [
        '--control-url',
        base_url,
        '--application-url',
        base_url,
        '--taihu-token',
        'test-token',
        '--output',
        str(output),
        '--allow-partial',
    ]
    for execution_id in ids:
        argv.extend(['--execution-id', execution_id])
    if result_status is not None:
        argv.extend(['--result-status', result_status])
    code = (
        await asyncio.to_thread(cloud_export.main, argv)
        if standalone
        else await asyncio.to_thread(export_batch, _parse(*argv))
    )
    assert code == 1
    assert '--allow-partial is supported only for Batch results' in capsys.readouterr().err
    assert api.result_calls == []
    assert all(kind == 'execution' for kind, _ in requests)
    assert not output.parent.exists()


@pytest.mark.parametrize('verbose', [False, True])
def test_integrated_results_logging_preserves_verbose(monkeypatch, tmp_path, capsys, verbose):
    from nexus.distributed.export import cli

    async def logged_export(**kwargs):
        logging.getLogger(cli.__name__).debug('export debug detail')
        logging.getLogger(cli.__name__).info('export stats detail')
        logging.getLogger(cli.__name__).warning('export warning detail')

    monkeypatch.setattr(cli, 'export_execution', logged_export)
    argv = [
        '--execution-id',
        'batch',
        '--application-url',
        'https://application.test',
        '--taihu-token',
        'test-token',
        '--output',
        str(tmp_path / 'results.jsonl'),
    ]
    if verbose:
        argv.append('--verbose')
    assert export_batch(_parse(*argv)) == 0
    captured = capsys.readouterr()
    assert ('export debug detail' in captured.out) is verbose
    assert 'export stats detail' in captured.out
    assert 'export stats detail' not in captured.err
    assert 'export debug detail' not in captured.err
    assert 'export warning detail' not in captured.out
    assert 'export warning detail' in captured.err


@pytest.mark.parametrize('failure', [cloud_export.ExportError('fixture failure'), KeyboardInterrupt()])
def test_standalone_results_restores_logging_on_failure(monkeypatch, tmp_path, capsys, failure):
    from nexus.distributed.export import cli

    async def interrupted_export(**kwargs):
        logging.getLogger(cli.__name__).info('export progress')
        raise failure

    monkeypatch.setattr(cli, 'export_execution', interrupted_export)
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    code = cli.main(
        [
            '--execution-id',
            'batch',
            '--application-url',
            'https://application.test',
            '--taihu-token',
            'test-token',
            '--output',
            str(tmp_path / 'results.jsonl'),
        ]
    )
    assert code == (130 if isinstance(failure, KeyboardInterrupt) else 1)
    assert root.handlers == handlers
    assert root.level == level
    captured = capsys.readouterr()
    assert captured.out == ''
    assert 'export progress' in captured.err
