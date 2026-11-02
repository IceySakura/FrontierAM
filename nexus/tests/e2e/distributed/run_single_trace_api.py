"""Runner-test HTTP/Trace DB acceptance, executed inside the target cluster Job.

This program submits only its finite fixtures and cleans up every accepted
Execution. Database reads require the target namespace context; it never inserts
trace rows. The Runtime probe and normal Single driver own all trace writes.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import shutil
import signal
import sys
import time
import traceback
import uuid

from contextlib import suppress
from pathlib import Path
from urllib.parse import quote, urlsplit

import aiohttp

from nexus.runner.trace import principal_trace_scope_id
from nexus.runner.trace.tdsql.schema import principal_trace_table_name


CONTROL = 'https://test-control-runner-nexus-agent.mcp.woa.com'
APPLICATION = 'https://test-application-runner-nexus-agent.mcp.woa.com'
CLUSTER = 'cls-bb9678ho'
NAMESPACE = 'ns-prjftdbn-300007534-production'
TERMINAL = {'succeeded', 'failed', 'lost', 'expired', 'killed'}
COLUMNS = (
    'id',
    'run_id',
    'run_span_id',
    'parent_execution_id',
    'record_id',
    'event_type',
    'time_unix_nano',
    'trace_id',
    'span_id',
    'parent_span_id',
    'event_data',
    'created_at',
)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    ).hexdigest()


def save(path, value):
    fd = os.open(path, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())


async def wait_export_checkpoint(process, checkpoint, count, *, timeout_seconds=120):
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.returncode is not None:
            raise AssertionError(f'local export exited before checkpoint: {process.returncode}')
        if checkpoint.exists():
            current = json.loads(checkpoint.read_text())
            if current['committed_records'] >= count:
                return current
        await asyncio.sleep(0.1)
    raise TimeoutError('local exporter did not commit the expected prefix')


def accepted_execution_ids(stdout):
    decoder = json.JSONDecoder()
    ids = []
    for index, character in enumerate(stdout):
        if character != '{':
            continue
        try:
            value, _ = decoder.raw_decode(stdout[index:])
        except ValueError:
            continue
        if isinstance(value, dict) and isinstance(value.get('execution_id'), str):
            identity = value['execution_id']
            if re.fullmatch(r'exe_[0-9a-f]{32}', identity) and identity not in ids:
                ids.append(identity)
    return ids


def submission_log_tail(path, secret_values):
    text = path.read_text(errors='replace') if path.exists() else ''
    for value in secret_values:
        if value:
            text = text.replace(value, '<redacted>')
    text = re.sub(r'\x1b\[[0-9;]*m', '', text)
    text = re.sub(r'https?://\S+', '<url>', text)
    text = re.sub(r'[A-Za-z0-9_+/=-]{24,}', '<opaque-value>', text)
    return text[-4000:]


def normalized_page(page, *, execution_id, run_span_id, committed, preview=False):
    validated = []
    previous = 0
    for raw in page['items']:
        assert set(raw) == set(COLUMNS) | ({'event_data_bytes', 'event_data_omitted'} if preview else set())
        assert raw['run_id'] == execution_id
        assert raw['run_span_id'] == run_span_id
        assert all(
            isinstance(raw[name], str) and raw[name].isascii() and raw[name].isdigit()
            for name in ('id', 'record_id', 'time_unix_nano')
        )
        if raw['event_data'] is not None:
            assert isinstance(json.loads(raw['event_data']), dict)
        if preview:
            assert type(raw['event_data_omitted']) is bool
            assert type(raw['event_data_bytes']) is int
            assert raw['event_data_bytes'] >= 0
            assert (raw['event_data'] is None) == raw['event_data_omitted']
            if not raw['event_data_omitted']:
                assert len(raw['event_data'].encode()) == raw['event_data_bytes']
        row = {name: raw[name] for name in COLUMNS}
        row.update({name: int(raw[name]) for name in ('id', 'record_id', 'time_unix_nano')})
        assert row['record_id'] > previous
        previous = row['record_id']
        if previous in committed:
            assert committed[previous] == row
        validated.append(row)
    return validated


class Acceptance:
    def __init__(self, args):
        self.args = args
        self.marker = 'single-trace-api-' + uuid.uuid4().hex[:12]
        self.root = args.output_dir / self.marker
        self.root.mkdir(parents=True, mode=0o700)
        self.source = self.root / 'source'
        self.source.mkdir(mode=0o700)
        self.owned = []
        self.batches = []
        self.processes = []
        self.submissions = {}
        self.rows = {}
        self.streams = {}
        self.cursors = {}
        self.requests = []
        self.running = {}
        self.bindings = {}
        self.disconnect_replayed = False
        self.saw_large_page = False
        self.api = None
        self.owner = None
        self.report = {'marker': self.marker, 'passed': False}
        self.preview = getattr(args, 'preview', False)
        self.omitted = {}
        self.local_task = None
        self.local_finish = asyncio.Event()
        self.report['api_mode'] = 'preview' if self.preview else 'raw'

    def journal(self):
        save(
            self.root / 'ownership.json',
            {'owned': self.owned, 'batches': self.batches, 'submissions': self.submissions},
        )

    def own(self, execution):
        if execution not in self.owned:
            self.owned.append(execution)
            self.journal()
            print(
                'TRACE_API_E2E_PROGRESS='
                + json.dumps({'phase': 'owned', 'execution_id': execution, 'marker': self.marker}),
                flush=True,
            )

    async def request(self, path, *, application=False, params=None, method='GET'):
        base = APPLICATION if application else CONTROL
        for retry in range(6):
            async with self.api.request(method, base + path, params=params, allow_redirects=False) as response:
                body = await response.read()
                payload = json.loads(body)
                entry = {
                    'path': path,
                    'method': method,
                    'status': response.status,
                    'response_sha256': hashlib.sha256(body).hexdigest(),
                }
                if params and params.get('cursor'):
                    entry['cursor_sha256'] = hashlib.sha256(params['cursor'].encode()).hexdigest()
                if isinstance(payload, dict):
                    entry['items'] = len(payload.get('items', []))
                    if response.status >= 400:
                        error_code = payload.get('error', '')
                        entry['error'] = (
                            error_code
                            if isinstance(error_code, str) and re.fullmatch(r'[a-z][a-z0-9_:.]{0,127}', error_code)
                            else 'unknown_error'
                        )
                    if payload.get('next_cursor'):
                        entry['next_cursor_sha256'] = hashlib.sha256(payload['next_cursor'].encode()).hexdigest()
                self.requests.append(entry)
                if response.status < 400:
                    assert response.status == 200, 'unexpected redirect or response'
                    return payload, len(body)
                if response.status in {429, 503} and payload.get('retryable') is True and retry < 5:
                    delay = max(float(response.headers.get('Retry-After', '0')), min(2**retry, 8))
                    await asyncio.sleep(delay)
                    continue
                raise RuntimeError(f'HTTP {response.status}: {payload.get("error", "unknown_error")}')
        raise AssertionError('unreachable')

    async def execution(self, execution_id):
        value, _ = await self.request(f'/v1/executions/{quote(execution_id, safe="")}')
        return value

    def prepare(self):
        for name in ('pyproject.toml', 'uv.lock', 'README.md'):
            shutil.copyfile(self.args.project_root / name, self.source / name)
        for metadata in self.args.project_root.glob('services/*/*/pyproject.toml'):
            destination = self.source / metadata.relative_to(self.args.project_root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(metadata, destination)
        shutil.copyfile(self.args.probe, self.source / 'trace_probe.py')
        single = {
            'imports': ['trace_probe'],
            'task_name': 'single_trace_api_probe',
            'task_config': {'label': 'single', 'marker': self.marker, 'seconds': 35, 'large_event': True},
        }
        if self.preview:
            single['task_config'].update(seconds=60, oversized_event_bytes=20 * 1024 * 1024, oversized_event_count=2)
        save(self.source / 'single.json', single)
        generator = """from nexus.runner.batch import BatchItem
from nexus.runner.single import RunInput

def get_total_count():
    return 3

def generate_configs():
    for label, seconds, fail_first in [('long', 30, False), ('retry', 12, True), ('late', 10, False)]:
        yield BatchItem(run_id=f'MARKER-{label}', config=RunInput(imports=['trace_probe'],
            task_name='single_trace_api_probe', task_config={'label': label, 'marker': 'MARKER',
            'seconds': seconds, 'fail_first': fail_first}), metadata={'fixture': 'single-trace-api', 'label': label})
"""
        (self.source / 'generator.py').write_text(generator.replace('MARKER', self.marker))

    async def submit(self, mode):
        key = self.marker + '-' + mode
        self.submissions[mode] = {'idempotency_key': key, 'status': 'unresolved'}
        self.journal()
        print(
            'TRACE_API_E2E_PROGRESS='
            + json.dumps({'phase': 'submitting', 'mode': mode, 'idempotency_key': key, 'marker': self.marker}),
            flush=True,
        )
        args = [
            'submit',
            mode,
            '--environment',
            'tkex-nanjing-test',
            '--source',
            str(self.source),
            '--project-root',
            str(self.args.project_root),
            '--control-url',
            CONTROL,
            '--taihu-token-file',
            str(self.args.token_file),
            '--wheel-extra',
            'distributed',
            '--manager-slots',
            '1',
            '--manager-cpu-cores',
            '2',
            '--manager-memory-gib',
            '4',
            '--execution-ttl-seconds',
            '1800',
            '--idempotency-key',
            key,
            '--json',
        ]
        if mode == 'single':
            args += [str(self.source / 'single.json')]
        else:
            args += [
                '--generator',
                'generator.py',
                '--generation',
                'local',
                '--run-trace-sink',
                'tdsql',
                '--max-inflight',
                '1',
                '--manager-count',
                '1',
                '--batch-manager-cpu-cores',
                '1',
                '--batch-manager-memory-gib',
                '4',
                '--retry-failed',
                '1',
                '--timeout-per-run',
                '180',
            ]
        stdout = self.root / (mode + '.stdout')
        stderr = self.root / (mode + '.stderr')
        process = None
        try:
            with stdout.open('wb') as out, stderr.open('wb') as err:
                stdout.chmod(0o600)
                stderr.chmod(0o600)
                spawning = asyncio.create_task(
                    asyncio.create_subprocess_exec(
                        sys.executable,
                        '-m',
                        'nexus.cli.main',
                        *args,
                        cwd=self.args.project_root,
                        stdout=out,
                        stderr=err,
                    )
                )
                try:
                    process = await asyncio.shield(spawning)
                except asyncio.CancelledError:
                    process = await spawning
                    self.processes.append(process)
                    raise
                self.processes.append(process)
                await asyncio.wait_for(process.wait(), 1200)
        finally:
            for execution_id in accepted_execution_ids(stdout.read_text(errors='replace') if stdout.exists() else ''):
                self.own(execution_id)
                self.submissions[mode].update(status='accepted', execution_id=execution_id)
                if mode == 'batch' and execution_id not in self.batches:
                    self.batches.append(execution_id)
                self.journal()
        if process.returncode != 0 or self.submissions[mode]['status'] != 'accepted':
            secret_values = [self.args.token_file.read_text().strip(), self.args.db_password_file.read_text().strip()]
            print(
                'TRACE_API_SUBMISSION_FAILURE='
                + json.dumps({
                    'mode': mode,
                    'idempotency_key': key,
                    'returncode': process.returncode,
                    'stderr_tail': submission_log_tail(stderr, secret_values),
                    'stdout_tail': submission_log_tail(stdout, secret_values),
                }),
                flush=True,
            )
        assert process.returncode == 0, 'submission failed; inspect sanitized Job diagnostics or private stderr'
        assert self.submissions[mode]['status'] == 'accepted', 'submission outcome unresolved'
        return self.submissions[mode]['execution_id']

    async def discover_children(self, parent):
        value, _ = await self.request(f'/v1/executions/{parent}/batch/items', params={'limit': 100})
        assert value.get('next_cursor') is None
        details = []
        for item in value['items']:
            detail, _ = await self.request(f'/v1/executions/{parent}/batch/items/{quote(item["item_id"], safe="")}')
            details.append(detail)
            for attempt in detail['attempts']:
                self.own(attempt['child_execution_id'])
        return details

    async def follow(self, execution_id):
        execution = await self.execution(execution_id)
        assignment = execution.get('current_assignment') or {}
        binding = (assignment.get('runtime_info') or {}).get('run_trace', {}).get('source_binding')
        if binding is not None:
            assert binding['run_id'] == execution_id
            assert binding['sources'], 'running Manager omitted configured source identity'
            assert binding['sources'][0]['backend'] == 'tdsql'
            self.bindings[execution_id] = binding
        after = None
        while True:
            params = {'limit': 100}
            if after:
                params['cursor'] = after
            discovery, _ = await self.request(
                f'/v1/executions/{execution_id}/trace-streams', application=True, params=params
            )
            assert discovery['availability'] in {'waiting_source', 'readable'}
            for item in discovery['items']:
                key = (execution_id, item['stream_id'])
                self.streams[key] = item
            after = discovery['next_cursor']
            if after is None:
                break
        for key in [key for key in self.streams if key[0] == execution_id]:
            path = f'/v1/executions/{execution_id}/trace-streams/{self.streams[key]["stream_id"]}/events'
            while True:
                params = {'limit': 200}
                if self.preview:
                    params['mode'] = 'preview'
                if key in self.cursors:
                    params['cursor'] = self.cursors[key]
                disconnected_now = False
                if not self.disconnect_replayed:
                    async with self.api.get(APPLICATION + path, params=params, allow_redirects=False) as response:
                        assert response.status == 200
                        partial = await response.content.read(256)
                        response.close()
                        self.report['disconnect'] = {
                            'path': path,
                            'cursor_sha256': hashlib.sha256(params.get('cursor', '').encode()).hexdigest(),
                            'partial_bytes': len(partial),
                        }
                        disconnected_now = True
                    # Resume from the same uncommitted cursor after HTTP disconnect.
                    self.disconnect_replayed = True
                page, response_bytes = await self.request(path, application=True, params=params)
                if disconnected_now:
                    self.report['disconnect']['retried_response_sha256'] = self.requests[-1]['response_sha256']
                self.saw_large_page |= response_bytes > 2 * 1024 * 1024
                rows = self.rows.setdefault(key, {})
                normalized = normalized_page(
                    page,
                    execution_id=execution_id,
                    run_span_id=self.streams[key]['run_span_id'],
                    committed=rows,
                    preview=self.preview,
                )
                if self.preview:
                    omissions = self.omitted.setdefault(key, {})
                    for row in page['items']:
                        if row['event_data_omitted']:
                            omissions[int(row['record_id'])] = row['event_data_bytes']
                    if omissions and 'raw_guard' not in self.report:
                        raw_params = {key: value for key, value in params.items() if key != 'mode'}
                        for _ in range(6):
                            async with self.api.get(APPLICATION + path, params=raw_params) as raw_response:
                                error = await raw_response.json()
                                if raw_response.status == 413:
                                    assert error['error'] == 'trace_event_too_large'
                                    self.report['raw_guard'] = {
                                        'status': 413,
                                        'error': error['error'],
                                        'detail': error['detail'],
                                    }
                                    break
                                assert raw_response.status == 200
                                assert error['has_more']
                                raw_params['cursor'] = error['next_cursor']
                        assert 'raw_guard' in self.report
                        continuation, _ = await self.request(
                            path, application=True, params={'cursor': page['next_cursor'], 'mode': 'raw', 'limit': 1}
                        )
                        assert continuation['mode'] == 'raw'
                        self.report['cross_mode_cursor'] = True
                rows.update((row['record_id'], row) for row in normalized)
                self.cursors[key] = page['next_cursor']
                probe_visible = any(
                    'probe.step' in (row['event_data'] or '') or 'cloud.export.probe' in (row['event_data'] or '')
                    for row in rows.values()
                )
                if probe_visible and execution['status'] == 'running' and page['execution_status'] == 'running':
                    self.running[execution_id] = {
                        'execution_status': execution['status'],
                        'api_status': page['execution_status'],
                        'events': len(rows),
                        'response_sha256': digest(page),
                    }
                if not page['has_more']:
                    break
        return execution

    async def wait_terminal(self, parent, *, batch=False):
        deadline = time.monotonic() + 1800
        last_progress = 0
        while time.monotonic() < deadline:
            if batch:
                await self.discover_children(parent)
                for execution in list(self.owned):
                    if execution != parent and execution not in self.batches:
                        await self.follow(execution)
                state = await self.execution(parent)
            else:
                state = await self.follow(parent)
            if time.monotonic() - last_progress >= 15:
                print(
                    'TRACE_API_E2E_PROGRESS='
                    + json.dumps({
                        'phase': 'following',
                        'execution_id': parent,
                        'status': state['status'],
                        'streams': len(self.streams),
                        'events': sum(len(rows) for rows in self.rows.values()),
                    }),
                    flush=True,
                )
                last_progress = time.monotonic()
            if state['status'] in TERMINAL:
                assert state['status'] == 'succeeded', 'fixture did not succeed'
                return state
            await asyncio.sleep(0.5)
        raise TimeoutError('fixture did not terminate')

    async def released(self, execution_id, *, require_allocations):
        for _ in range(240):
            execution = await self.execution(execution_id)
            allocations, _ = await self.request(f'/v1/executions/{execution_id}/allocations')
            items = allocations.get('items', allocations.get('allocations', []))
            if (
                execution['status'] in TERMINAL
                and (items or not require_allocations)
                and all(row['status'] == 'released' for row in items)
            ):
                return {'execution_status': execution['status'], 'allocations': items}
            await asyncio.sleep(1)
        raise TimeoutError('allocation release unverified')

    async def check_local_export(self, execution_id):
        """Exercise the real local CLI while the same cloud fixture is still running."""
        db = json.loads(self.args.trace_db_config.read_text())
        output = self.root / 'local-events.jsonl'
        checkpoint = output.with_name(output.name + '.checkpoint.json')
        peak_rss_bytes = 0
        monitors = []

        async def monitor_memory(process):
            nonlocal peak_rss_bytes
            while process.returncode is None:
                with suppress(FileNotFoundError, ProcessLookupError):
                    status = Path(f'/proc/{process.pid}/status').read_text()
                    match = re.search(r'^VmHWM:\s+(\d+) kB$', status, re.MULTILINE)
                    if match:
                        peak_rss_bytes = max(peak_rss_bytes, int(match[1]) * 1024)
                await asyncio.sleep(0.1)

        command = [
            sys.executable,
            '-m',
            'nexus.cli.main',
            'export',
            '--execution-id',
            execution_id,
            '--content',
            'traces',
            '--follow',
            '--poll-interval',
            '0.2',
            '--control-url',
            CONTROL,
            '--taihu-token-file',
            str(self.args.token_file),
            '--db-host',
            db['read_host'],
            '--db-port',
            str(db['read_port']),
            '--db-name',
            db['database'],
            '--db-user',
            db['username'],
            '--db-password-file',
            str(self.args.db_password_file),
            '--trace-table',
            db['trace_table_name'],
            '-o',
            str(output),
        ]

        async def launch(resume=False):
            suffix = 'resume' if resume else 'initial'
            with (
                (self.root / f'local-{suffix}.stdout').open('wb') as stdout,
                (self.root / f'local-{suffix}.stderr').open('wb') as stderr,
            ):
                process = await asyncio.create_subprocess_exec(
                    *command,
                    *(['--resume'] if resume else []),
                    cwd=self.args.project_root,
                    stdout=stdout,
                    stderr=stderr,
                )
            self.processes.append(process)
            monitors.append(asyncio.create_task(monitor_memory(process)))
            save(self.root / 'local-processes.json', {'pids': [p.pid for p in self.processes]})
            return process

        process = await launch()
        try:
            # The exporter starts while queued; cold dependency builds share the fixture's startup deadline.
            initial = await wait_export_checkpoint(process, checkpoint, 1, timeout_seconds=1800)
            state = await self.execution(execution_id)
            assert state['status'] == 'running', 'local exporter missed the running fixture'
            process.kill()
            assert await process.wait() == -signal.SIGKILL
            initial = json.loads(checkpoint.read_text())
            with output.open('rb') as stream:
                prefix_hash = hashlib.sha256(stream.read(initial['committed_bytes'])).hexdigest()
            assert prefix_hash == initial['committed_sha256']
            with output.open('ab') as stream:
                stream.write(b'{"interrupted_tail":')
            process = await launch(resume=True)
            await self.local_finish.wait()
            oracle = await self.oracle(execution_id)
            final = await wait_export_checkpoint(process, checkpoint, len(oracle))
            process.send_signal(signal.SIGINT)
            await asyncio.wait_for(process.wait(), 30)
            assert process.returncode == 130
            rows = [json.loads(line) for line in output.read_text().splitlines()]

            def order(row):
                return row['run_span_id'], row['record_id']

            assert sorted(rows, key=order) == sorted(oracle, key=order)
            assert all(set(row) == set(COLUMNS) for row in rows)
            assert len({(row['run_span_id'], row['record_id']) for row in rows}) == len(rows)
            with output.open('rb') as stream:
                assert hashlib.sha256(stream.read(initial['committed_bytes'])).hexdigest() == prefix_hash
            large = [row for row in rows if len(row['event_data'].encode()) > 8 * 1024 * 1024]
            assert len(large) >= 2
            assert final['version'] == 1
            assert final['committed_records'] == len(oracle)
            self.report['local_export'] = {
                'running_verified': True,
                'sigkill_resume_verified': True,
                'records': len(rows),
                'bytes': output.stat().st_size,
                'oversized_records': len(large),
                'max_event_data_bytes': max(len(row['event_data'].encode()) for row in rows),
                'db_sha256': digest(sorted(oracle, key=order)),
                'export_sha256': digest(sorted(rows, key=order)),
                'checkpoint_version': final['version'],
                'unchanged_prefix_sha256': prefix_hash,
                'peak_rss_bytes': peak_rss_bytes,
                'peak_rss_measurement': 'maximum sampled /proc/<export-pid>/status VmHWM across initial and resume',
            }
        finally:
            if process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), 15)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
            await asyncio.gather(*monitors)

    async def oracle(self, execution_id):
        from nexus.runner.trace.tdsql._driver import connect

        db = json.loads(self.args.trace_db_config.read_text())
        table = principal_trace_table_name(principal_trace_scope_id(self.owner))
        assert table == db['trace_table_name']
        connection = await connect(
            host=db['read_host'],
            port=db['read_port'],
            db=db['database'],
            user=db['username'],
            password=self.args.db_password_file.read_text().strip(),
            autocommit=True,
            charset='utf8mb4',
            connect_timeout=10,
        )
        try:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    f'SELECT {", ".join(COLUMNS)} FROM {table} WHERE run_id=%s ORDER BY run_span_id,record_id',
                    (execution_id,),
                )
                result = [dict(zip(COLUMNS, row, strict=True)) for row in await cursor.fetchall()]
        finally:
            connection.close()
        for row in result:
            if row['created_at'] is not None:
                row['created_at'] = row['created_at'].isoformat()
        return result

    async def reconcile(self, execution_id):
        oracle = await self.oracle(execution_id)
        for _ in range(60):
            await self.follow(execution_id)
            observed = [row for key, rows in self.rows.items() if key[0] == execution_id for row in rows.values()]

            def key(row):
                return row['run_span_id'], row['record_id']

            omissions = {
                (self.streams[stream]['run_span_id'], record): size
                for stream, records in self.omitted.items()
                if stream[0] == execution_id
                for record, size in records.items()
            }
            expected = []
            for row in oracle:
                if key(row) in omissions:
                    assert len(row['event_data'].encode()) == omissions[key(row)]
                    expected.append({**row, 'event_data': None})
                else:
                    expected.append(row)
            if sorted(expected, key=key) == sorted(observed, key=key):
                assert oracle, 'finite fixture emitted no persisted events'
                return {
                    'events': len(oracle),
                    'streams': len({row['run_span_id'] for row in oracle}),
                    'api_sha256': digest(sorted(observed, key=key)),
                    'db_sha256': digest(sorted(oracle, key=key)),
                    'all_columns': list(COLUMNS),
                    'omitted_records': len(omissions),
                }
            await asyncio.sleep(0.5)
        raise AssertionError('API does not match post-release database oracle')

    async def cleanup(self):
        if self.local_task is not None and not self.local_task.done():
            self.local_task.cancel()
            await asyncio.gather(self.local_task, return_exceptions=True)
        for process in self.processes:
            if process.returncode is None:
                with suppress(ProcessLookupError):
                    process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), 10)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
        # Re-read accepted IDs after draining submitters, before stopping producers.
        for mode, submission in self.submissions.items():
            path = self.root / f'{mode}.stdout'
            for execution in accepted_execution_ids(path.read_text(errors='replace') if path.exists() else ''):
                self.own(execution)
                submission.update(status='accepted', execution_id=execution)
                if mode == 'batch' and execution not in self.batches:
                    self.batches.append(execution)
        self.journal()
        result = {}
        for execution in list(self.batches) + [value for value in self.owned if value not in self.batches]:
            try:
                state = await self.execution(execution)
                if state['status'] not in TERMINAL:
                    await self.request(f'/v1/executions/{execution}/stop', method='POST')
                result[execution] = await self.released(execution, require_allocations=execution in self.batches)
                if execution in self.batches:
                    await self.discover_children(execution)
            except Exception as error:
                result[execution] = {'cleanup_error': type(error).__name__}
        for child in [value for value in self.owned if value not in result]:
            try:
                state = await self.execution(child)
                if state['status'] not in TERMINAL:
                    await self.request(f'/v1/executions/{child}/stop', method='POST')
                result[child] = await self.released(child, require_allocations=False)
            except Exception as error:
                result[child] = {'cleanup_error': type(error).__name__}
        self.report['cleanup'] = result
        self.report['cleanup_verified'] = all('cleanup_error' not in value for value in result.values()) and all(
            value['status'] == 'accepted' for value in self.submissions.values()
        )
        save(self.root / 'requests.json', self.requests)

    async def run(self):
        self.prepare()
        async with aiohttp.ClientSession(
            headers={'Authorization': 'Bearer ' + self.args.token_file.read_text().strip()},
            timeout=aiohttp.ClientTimeout(total=30, connect=10),
        ) as session:
            self.api = session
            try:
                me, _ = await self.request('/v1/me')
                self.owner = me['principal_id']
                single = await self.submit('single')
                if self.preview:
                    self.local_task = asyncio.create_task(self.check_local_export(single))
                single_state = await self.wait_terminal(single)
                assert single_state['driver_config'] == {}, (
                    'default-source scenario requires no frozen explicit selection'
                )
                assert single in self.running
                assert single in self.bindings
                result, _ = await self.request(f'/v1/executions/{single}/result', application=True)
                assert result['success'] is True
                from nexus.distributed.execution.drivers import nexus_single

                expected_driver = hashlib.sha256(Path(nexus_single.__file__).read_bytes()).hexdigest()
                assert result['task_output']['driver_sha256'] == expected_driver
                await self.released(single, require_allocations=True)
                single_comparison = await self.reconcile(single)
                if self.preview:
                    assert single_comparison['omitted_records'] >= 2
                    assert self.report['cross_mode_cursor']
                    self.local_finish.set()
                    await self.local_task
                parent = await self.submit('batch')
                await self.wait_terminal(parent, batch=True)
                details = await self.discover_children(parent)
                children = {attempt['child_execution_id'] for detail in details for attempt in detail['attempts']}
                assert len(details) == 3
                assert len(children) == 4
                assert all(detail['status'] == 'succeeded' for detail in details)
                await self.released(parent, require_allocations=True)
                comparisons = {}
                for child in children:
                    await self.released(child, require_allocations=False)
                    comparisons[child] = await self.reconcile(child)
                assert children.intersection(self.running), 'no child had events while running'
                assert single_comparison['events'] > 500
                assert self.disconnect_replayed
                assert self.saw_large_page
                self.report.update(
                    single_execution_id=single,
                    batch_execution_id=parent,
                    child_execution_ids=sorted(children),
                    single=single_comparison,
                    children=comparisons,
                    running=self.running,
                    source_bindings=self.bindings,
                    disconnect_replayed=True,
                    large_singleton=True,
                    manager_driver_sha256=expected_driver,
                )
            except BaseException as error:
                frame = traceback.extract_tb(error.__traceback__)[-1]
                self.report['failure'] = {
                    'error_type': type(error).__name__,
                    'last_request': self.requests[-1] if self.requests else None,
                    'file': Path(frame.filename).name,
                    'line': frame.lineno,
                    'function': frame.name,
                }
                raise
            finally:
                cleanup = asyncio.create_task(self.cleanup())
                while not cleanup.done():
                    try:
                        await asyncio.shield(cleanup)
                    except asyncio.CancelledError:
                        continue
                cleanup.result()
                save(self.root / 'report.json', self.report)
            assert self.report['cleanup_verified']
            final_checks = {}
            for execution in [single, *children]:
                fresh = await self.oracle(execution)
                expected = single_comparison if execution == single else comparisons[execution]
                final_hash = digest(sorted(fresh, key=lambda row: (row['run_span_id'], row['record_id'])))
                assert final_hash == expected['db_sha256']
                final_checks[execution] = final_hash
            self.report['post_cleanup_db_sha256'] = final_checks
            self.report['request_count'] = len(self.requests)
            self.report['requests_sha256'] = digest(self.requests)
            self.report['passed'] = True
            save(self.root / 'report.json', self.report)
            print(
                'TRACE_API_E2E_REPORT=' + json.dumps(self.report, ensure_ascii=False, separators=(',', ':')), flush=True
            )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path('/app'))
    parser.add_argument('--probe', type=Path, default=Path(__file__).with_name('single_trace_api_probe.py'))
    parser.add_argument('--token-file', type=Path, required=True)
    parser.add_argument('--db-password-file', type=Path, required=True)
    parser.add_argument('--trace-db-config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('/tmp/single-trace-api-e2e'))
    parser.add_argument(
        '--preview', action='store_true', help='Verify preview omission and complete local export of 20 MiB events'
    )
    return parser.parse_args()


async def run_cancellable(acceptance):
    task = asyncio.create_task(asyncio.wait_for(acceptance.run(), 3600))
    loop = asyncio.get_running_loop()
    cancelled = False

    def stop():
        nonlocal cancelled
        if not cancelled:
            cancelled = True
            task.cancel()

    for signum in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(signum, stop)
    try:
        await task
    finally:
        for signum in (signal.SIGTERM, signal.SIGINT):
            loop.remove_signal_handler(signum)


def validate_target_job():
    if os.environ.get('NEXUS_TRACE_API_E2E_CLUSTER_ID') != CLUSTER or os.environ.get('POD_NAMESPACE') != NAMESPACE:
        raise RuntimeError('Trace DB E2E must run in the authorized runner-test cluster Job')
    assert urlsplit(CONTROL).hostname == 'test-control-runner-nexus-agent.mcp.woa.com'
    assert urlsplit(APPLICATION).hostname == 'test-application-runner-nexus-agent.mcp.woa.com'


def main():
    args = parse_args()
    validate_target_job()
    acceptance = Acceptance(args)
    try:
        asyncio.run(run_cancellable(acceptance))
    except BaseException as error:
        print('TRACE_API_E2E_REPORT=' + json.dumps(acceptance.report, ensure_ascii=False, separators=(',', ':')), flush=True)
        print(
            'TRACE_API_E2E_FAILURE='
            + json.dumps({
                'error_type': type(error).__name__,
                'marker': acceptance.marker,
                'owned': acceptance.owned,
                'cleanup_verified': acceptance.report.get('cleanup_verified', False),
                'failure': acceptance.report.get('failure'),
                'last_request': acceptance.requests[-1] if acceptance.requests else None,
            }),
            flush=True,
        )
        raise SystemExit(1) from None


if __name__ == '__main__':
    main()
