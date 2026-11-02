"""Opt-in runner-test proof for running exports, retries, resume and terminal tails."""

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
import uuid

from collections import Counter
from contextlib import suppress
from pathlib import Path

import aiohttp
import asyncmy


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'outputs/cloud-event-export-e2e'
CONTROL = 'https://test-control-runner-nexus-agent.mcp.woa.com'
APPLICATION = 'https://test-application-runner-nexus-agent.mcp.woa.com'
TOKEN_FILE = Path.home() / '.nexus/auth/taihu-pat'
DB_PASSWORD_FILE = Path()
CONFIG = {}
SOURCE = Path()
MARKER = ''
RUN = Path()
ENVIRONMENT = 'tkex-nanjing-test'
NEXUS = Path(sys.executable).with_name('nexus')
PROCESSES = []
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


def save(name, value):
    path = RUN / name
    fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())


async def launch(name, args):
    stdout = RUN / f'{name}.stdout'
    stderr = RUN / f'{name}.stderr'
    with stdout.open('wb') as out, stderr.open('wb') as err:
        stdout.chmod(0o600)
        stderr.chmod(0o600)
        spawning = asyncio.create_task(
            asyncio.create_subprocess_exec(str(NEXUS), *args, cwd=ROOT, stdout=out, stderr=err)
        )
        try:
            process = await asyncio.shield(spawning)
        except asyncio.CancelledError:
            process = await spawning
            PROCESSES.append(process)
            raise
        PROCESSES.append(process)
        return process


def export_args(execution_id, output, *, content='traces', follow=False, resume=False):
    args = [
        'export',
        '--execution-id',
        execution_id,
        '--content',
        content,
        '--control-url',
        CONTROL,
        '--application-url',
        APPLICATION,
        '--taihu-token-file',
        str(TOKEN_FILE),
        '--db-host',
        CONFIG['read_host'],
        '--db-port',
        str(CONFIG['read_port']),
        '--db-name',
        CONFIG['database'],
        '--db-user',
        CONFIG['username'],
        '--db-password-file',
        str(DB_PASSWORD_FILE),
        '--trace-table',
        CONFIG['trace_table_name'],
        '-o',
        str(output),
    ]
    if follow:
        args += ['--follow', '--poll-interval', '0.2']
    if resume:
        args += ['--resume']
    return args


async def read_oracle(parent, *, batch=True):
    table = CONFIG['trace_table_name']
    assert re.fullmatch(r'trajectory_span_events_pr_[0-9a-f]{32}', table)
    connection = await asyncmy.connect(
        host=CONFIG['read_host'],
        port=CONFIG['read_port'],
        user=CONFIG['username'],
        password=DB_PASSWORD_FILE.read_text(),
        db=CONFIG['database'],
        autocommit=True,
        charset='utf8mb4',
        connect_timeout=10,
    )
    try:
        async with connection.cursor() as cursor:
            filter_column = 'parent_execution_id' if batch else 'run_id'
            await cursor.execute(
                f'SELECT {", ".join(COLUMNS)} FROM {table} WHERE {filter_column}=%s '
                'ORDER BY run_id, run_span_id, record_id',
                (parent,),
            )
            result = [dict(zip(COLUMNS, row, strict=True)) for row in await cursor.fetchall()]
        for row in result:
            if row['created_at'] is not None:
                row['created_at'] = row['created_at'].isoformat()
        return result
    finally:
        connection.close()


async def request_api(session, control, path, *, method='GET'):
    async with session.request(method, control + path) as response:
        if response.status >= 400:
            raise RuntimeError(f'API {path} failed with HTTP {response.status}')
        return await response.json()


async def run_cancellable(work):
    task = asyncio.create_task(work)
    loop = asyncio.get_running_loop()
    cancelled = False

    def cancel_once():
        nonlocal cancelled
        if not cancelled:
            cancelled = True
            task.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, cancel_once)
    try:
        return await task
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(sig)


def record_submission(name, owned, submissions):
    path = RUN / f'{name}.stdout'
    decoder = json.JSONDecoder()
    accepted = None
    changed = False
    stdout = path.read_text(errors='replace') if path.exists() else ''
    for index, char in enumerate(stdout):
        if char != '{':
            continue
        try:
            value, _ = decoder.raw_decode(stdout[index:])
        except ValueError:
            continue
        if isinstance(value, dict) and isinstance(value.get('execution_id'), str) and value['execution_id']:
            accepted = value
            target = value['execution_id']
            if target not in owned:
                owned.append(target)
                changed = True
            changed = changed or submissions[name].get('execution_id') != target
            submissions[name]['execution_id'] = target
            submissions[name]['status'] = 'accepted'
    if changed:
        save('ownership.json', {'owned': owned, 'submissions': submissions})
    return accepted


async def await_submission(name, process, owned, submissions, *, timeout=1200):
    deadline = time.monotonic() + timeout
    while process.returncode is None:
        record_submission(name, owned, submissions)
        if time.monotonic() >= deadline:
            raise TimeoutError(f'{name} timed out; ownership requires cleanup verification')
        await asyncio.sleep(0.1)
    accepted = record_submission(name, owned, submissions)
    assert process.returncode == 0, f'{name} failed; inspect its private stderr artifact'
    assert accepted is not None, f'{name} outcome unresolved; inspect ownership.json'
    return accepted


async def drain_process(process, *, timeout=20):
    if process.returncode is None:
        with suppress(ProcessLookupError):
            process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout)
        except asyncio.TimeoutError:
            with suppress(ProcessLookupError):
                process.kill()
            await asyncio.wait_for(process.wait(), 5)
    else:
        await process.wait()


async def wait_released(api, target, *, timeout=360, interval=2, require_allocations=True):
    async def poll():
        while True:
            execution = await api(f'/v1/executions/{target}')
            allocations = await api(f'/v1/executions/{target}/allocations')
            raw = allocations.get('items', allocations.get('allocations', []))
            if (
                execution['status'] in TERMINAL
                and (raw or not require_allocations)
                and all(item.get('status') == 'released' for item in raw)
            ):
                return {'execution': execution, 'allocations': allocations}
            await asyncio.sleep(interval)

    try:
        return await asyncio.wait_for(poll(), timeout)
    except asyncio.TimeoutError as exc:
        raise TimeoutError(f'{target} terminal/resource release not verified') from exc


def event_key(row):
    return row['run_id'], row['run_span_id'], row['record_id']


def committed_rows(output):
    checkpoint = output.with_name(output.name + '.checkpoint.json')
    if not checkpoint.exists():
        return None
    state = json.loads(checkpoint.read_text())
    data = output.read_bytes()[: state['committed_bytes']]
    assert hashlib.sha256(data).hexdigest() == state['committed_sha256']
    rows = [json.loads(line) for line in data.splitlines()]
    assert len(rows) == state['committed_records']
    assert len({event_key(row) for row in rows}) == len(rows)
    return rows


async def reconcile_tail(api, targets, oracle, follower, output, *, timeout=420, interval=1, stop_signal=signal.SIGINT):
    async def reconcile():
        released = await asyncio.gather(
            *(
                wait_released(api, target, timeout=timeout, interval=interval, require_allocations=index == 0)
                for index, target in enumerate(targets)
            )
        )
        save(output.name + '.producers-released.json', dict(zip(targets, released, strict=True)))
        while True:
            assert follower.returncode is None, 'follower stopped before tail verification'
            expected = await oracle()
            rows = committed_rows(output)
            if rows is not None and sorted(rows, key=event_key) == sorted(expected, key=event_key):
                break
            await asyncio.sleep(interval)
        follower.send_signal(stop_signal)
        await asyncio.wait_for(follower.wait(), 20)
        assert follower.returncode in {128 + stop_signal, -stop_signal}, 'follower shutdown failed'
        # A new independent read catches any commit after the pre-shutdown equality.
        final = await oracle()
        save(output.name + '.final-oracle.json', final)
        rows = committed_rows(output)
        assert rows is not None
        assert sorted(rows, key=event_key) == sorted(final, key=event_key), 'database tail changed after exporter stop'
        assert (
            output.stat().st_size
            == json.loads(output.with_name(output.name + '.checkpoint.json').read_text())['committed_bytes']
        )
        return final, rows

    return await asyncio.wait_for(reconcile(), timeout)


async def finish_cleanup(work):
    task = asyncio.create_task(work)
    cancellation = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            cancellation = exc
    result = task.result()
    if cancellation is not None:
        raise cancellation
    return result


async def cleanup_resources(api, processes, owned, submissions, *, timeout=120, interval=2):
    local = await asyncio.gather(*(drain_process(process) for process in processes), return_exceptions=True)
    # A cancelled/failed submit can still have printed an accepted response while draining.
    ownership_errors = []
    for name in submissions:
        try:
            record_submission(name, owned, submissions)
        except Exception:
            ownership_errors.append(name)
    unresolved = [name for name, value in submissions.items() if value.get('status') != 'accepted']
    deadline = time.monotonic() + timeout

    def remaining():
        return max(0.001, deadline - time.monotonic())

    async def clean(target):
        try:

            async def stop_and_verify():
                execution = await api(f'/v1/executions/{target}')
                if execution['status'] not in TERMINAL:
                    await api(f'/v1/executions/{target}/stop', method='POST')
                return await wait_released(
                    api, target, timeout=remaining(), interval=interval, require_allocations=False
                )

            facts = await asyncio.wait_for(stop_and_verify(), remaining())
            return {'verified': True, **facts}
        except Exception as exc:
            return {'verified': False, 'error_type': type(exc).__name__}

    initial = list(owned)
    results = dict(zip(initial, await asyncio.gather(*(clean(target) for target in initial)), strict=True))

    async def discover_children(target):
        async def discover():
            # Discover after stopping the parent, so it cannot schedule a new attempt after this read.
            data = await api(f'/v1/executions/{target}/batch/items?limit=500')
            assert data.get('next_cursor') is None, 'bounded three-item fixture unexpectedly paginated'
            for item in data['items']:
                detail = await api(f'/v1/executions/{target}/batch/items/{item["item_id"]}')
                for attempt in detail['attempts']:
                    child = attempt['child_execution_id']
                    if child not in owned:
                        owned.append(child)
                        save('ownership.json', {'owned': owned, 'submissions': submissions})

        try:
            await asyncio.wait_for(discover(), remaining())
        except Exception as exc:
            results[target] = {'verified': False, 'child_discovery_error_type': type(exc).__name__}

    batches = {
        entry['execution_id']
        for entry in submissions.values()
        if entry.get('spec_type') == 'nexus_batch' and entry.get('execution_id') in results
    }
    await asyncio.gather(*(discover_children(target) for target in batches))
    children = [target for target in owned if target not in results]
    results.update(zip(children, await asyncio.gather(*(clean(target) for target in children)), strict=True))
    report = {
        'local_processes': [
            {'pid': process.pid, 'verified': not isinstance(result, BaseException), 'returncode': process.returncode}
            for process, result in zip(processes, local, strict=True)
        ],
        'executions': results,
        'unresolved_submissions': sorted(set(unresolved + ownership_errors)),
    }
    save('ownership.json', {'owned': owned, 'submissions': submissions})
    save('cleanup.json', report)
    return report


async def main():
    RUN.mkdir(mode=0o700, exist_ok=True)
    save(
        'test-scope.json',
        {
            'control_url': CONTROL,
            'application_url': APPLICATION,
            'environment_id': ENVIRONMENT,
            'marker': MARKER,
            'items': 3,
            'max_inflight': 1,
            'retry_failed': 1,
        },
    )
    session = aiohttp.ClientSession(
        headers={'Authorization': f'Bearer {TOKEN_FILE.read_text().strip()}'}, timeout=aiohttp.ClientTimeout(total=20)
    )

    async def api(path, *, method='GET'):
        return await request_api(session, CONTROL, path, method=method)

    async def items(parent):
        data = await api(f'/v1/executions/{parent}/batch/items?limit=500')
        assert data.get('next_cursor') is None
        return data['items']

    parent = None
    owned = []
    submissions = {}
    follower = None
    success = False
    try:
        catalog = await api('/v1/my-environments')
        environment = next(item for item in catalog['items'] if item['environment_id'] == ENVIRONMENT)
        assert environment['status'] == 'enabled'
        assert environment['grant']['enabled'] is True
        identity_fields = ('read_host', 'read_port', 'database', 'username', 'trace_table_name')
        account = next(
            item
            for item in catalog['trace_database_accounts']
            if all(item.get(key) == CONFIG.get(key) for key in identity_fields)
        )
        assert account['status'] == 'ready', 'current Trace DB account is not ready'
        assert await read_oracle(f'preflight-{MARKER}') == []
        save(
            'preflight.json',
            {
                'environment_id': ENVIRONMENT,
                'environment_status': environment['status'],
                'backend_name': account['backend_name'],
                'account_status': account['status'],
                'trace_read_verified': True,
            },
        )
        submissions['submit'] = {'idempotency_key': MARKER, 'status': 'unresolved', 'spec_type': 'nexus_batch'}
        save('ownership.json', {'owned': owned, 'submissions': submissions})
        submitted = await launch(
            'submit',
            [
                'submit',
                'batch',
                '--generator',
                'generator.py',
                '--generation',
                'local',
                '--environment',
                ENVIRONMENT,
                '--source',
                str(SOURCE),
                '--project-root',
                str(ROOT),
                '--control-url',
                CONTROL,
                '--taihu-token-file',
                str(TOKEN_FILE),
                '--run-trace-sink',
                'tdsql',
                '--wheel-extra',
                'distributed',
                '--max-inflight',
                '1',
                '--manager-count',
                '1',
                '--manager-slots',
                '1',
                '--manager-cpu-cores',
                '2',
                '--manager-memory-gib',
                '4',
                '--batch-manager-cpu-cores',
                '1',
                '--batch-manager-memory-gib',
                '4',
                '--retry-failed',
                '1',
                '--timeout-per-run',
                '180',
                '--execution-ttl-seconds',
                '1800',
                '--idempotency-key',
                MARKER,
                '--json',
            ],
        )
        print('Submitting bounded source-package Batch to runner-test', flush=True)
        accepted = await await_submission('submit', submitted, owned, submissions)
        parent = accepted['execution_id']
        save('accepted.json', accepted)
        print('Accepted', parent, flush=True)
        output = RUN / 'events.jsonl'
        checkpoint_path = RUN / 'events.jsonl.checkpoint.json'
        follower = await launch('follow-initial', export_args(parent, output, follow=True))
        deadline = time.monotonic() + 1800
        running_evidence = None
        last_note = 0.0
        while time.monotonic() < deadline:
            assert follower.returncode is None, 'follower stopped unexpectedly'
            execution = await api(f'/v1/executions/{parent}')
            checkpoint = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else {}
            if time.monotonic() - last_note > 15:
                print(
                    'Following', execution['status'], 'committed=', checkpoint.get('committed_records', 0), flush=True
                )
                last_note = time.monotonic()
            if checkpoint.get('committed_records', 0) >= 5 and execution['status'] not in TERMINAL:
                children = await items(parent)
                active = [row for row in children if row['status'] == 'running']
                if active:
                    running_evidence = {'execution': execution, 'items': children, 'checkpoint': checkpoint}
                    save('running.json', running_evidence)
                    save(
                        'resume-prefix.json',
                        {
                            'bytes': checkpoint['committed_bytes'],
                            'sha256': hashlib.sha256(output.read_bytes()[: checkpoint['committed_bytes']]).hexdigest(),
                        },
                    )
                    follower.kill()
                    assert await follower.wait() == -signal.SIGKILL
                    follower = await launch('follow-resumed', export_args(parent, output, follow=True, resume=True))
                    print('Killed and resumed exporter while child was running', flush=True)
                    break
            if execution['status'] in TERMINAL:
                raise AssertionError('Batch ended without proving running-state export')
            await asyncio.sleep(1)
        assert running_evidence is not None, 'running export timed out'

        while time.monotonic() < deadline:
            assert follower.returncode is None, 'resumed follower stopped unexpectedly'
            execution = await api(f'/v1/executions/{parent}')
            if time.monotonic() - last_note > 15:
                print('Waiting for Batch terminal:', execution['status'], flush=True)
                last_note = time.monotonic()
            if execution['status'] in TERMINAL:
                break
            await asyncio.sleep(1)
        else:
            raise TimeoutError('Batch did not terminate')
        save('terminal.json', execution)
        assert execution['status'] == 'succeeded', f'Batch ended as {execution["status"]}'
        child_items = await items(parent)
        save('terminal-items.json', child_items)
        details = [await api(f'/v1/executions/{parent}/batch/items/{item["item_id"]}') for item in child_items]
        save('attempts.json', details)
        child_ids = {attempt['child_execution_id'] for item in details for attempt in item['attempts']}
        owned.extend(child for child in sorted(child_ids) if child not in owned)
        save('ownership.json', {'owned': owned, 'submissions': submissions})
        assert len(child_items) == 3, 'expected three tasks'
        assert len(child_ids) == 4, 'expected one child retry'
        child_facts = {child: await api(f'/v1/executions/{child}') for child in child_ids}
        save('children.json', child_facts)

        oracle, rows = await reconcile_tail(
            api, [parent, *sorted(child_ids)], lambda: read_oracle(parent), follower, output
        )
        save('oracle.json', oracle)
        assert {row['run_id'] for row in oracle} == child_ids
        assert follower.returncode == 130
        key = event_key
        prefix = json.loads((RUN / 'resume-prefix.json').read_text())
        assert hashlib.sha256(output.read_bytes()[: prefix['bytes']]).hexdigest() == prefix['sha256']

        results = await launch('results', export_args(parent, RUN / 'results.jsonl', content='results'))
        await asyncio.wait_for(results.wait(), 120)
        assert results.returncode == 0
        result_rows = [json.loads(line) for line in (RUN / 'results.jsonl').read_text().splitlines()]
        assert len(result_rows) == 3
        assert all(row['status'] == 'success' for row in result_rows)
        assert sorted(row['retry_count'] for row in result_rows) == [0, 0, 1]
        batch_snapshot = await launch('batch-snapshot', export_args(parent, RUN / 'batch-snapshot.jsonl'))
        await asyncio.wait_for(batch_snapshot.wait(), 120)
        assert batch_snapshot.returncode == 0
        snapshot_rows = [json.loads(line) for line in (RUN / 'batch-snapshot.jsonl').read_text().splitlines()]
        assert sorted(snapshot_rows, key=key) == sorted(oracle, key=key)
        counts = Counter(row['run_id'] for row in rows)
        child = counts.most_common(1)[0][0]
        snapshot = await launch('single-snapshot', export_args(child, RUN / 'single-events.jsonl'))
        await asyncio.wait_for(snapshot.wait(), 120)
        assert snapshot.returncode == 0
        single_rows = [json.loads(line) for line in (RUN / 'single-events.jsonl').read_text().splitlines()]
        assert sorted(single_rows, key=key) == sorted([row for row in oracle if row['run_id'] == child], key=key)
        assert len(single_rows) > 500, 'snapshot must exercise multiple SQL pages'

        released = json.loads((RUN / 'events.jsonl.producers-released.json').read_text())
        save('allocations.json', released[parent]['allocations'])
        submissions['live-submit'] = {'idempotency_key': MARKER + '-live', 'status': 'unresolved'}
        save('ownership.json', {'owned': owned, 'submissions': submissions})

        live_submit = await launch(
            'live-submit',
            [
                'submit',
                'live',
                str(SOURCE / 'live.yaml'),
                '--environment',
                ENVIRONMENT,
                '--source',
                str(SOURCE),
                '--project-root',
                str(ROOT),
                '--control-url',
                CONTROL,
                '--taihu-token-file',
                str(TOKEN_FILE),
                '--run-trace-sink',
                'tdsql',
                '--wheel-extra',
                'distributed',
                '--manager-slots',
                '1',
                '--manager-cpu-cores',
                '2',
                '--manager-memory-gib',
                '4',
                '--execution-ttl-seconds',
                '900',
                '--idempotency-key',
                MARKER + '-live',
                '--json',
            ],
        )
        print('Submitting Live probe', flush=True)
        live_accepted = await await_submission('live-submit', live_submit, owned, submissions)
        live_id = live_accepted['execution_id']
        save('live-accepted.json', live_accepted)
        live_output = RUN / 'live-events.jsonl'
        live_checkpoint = RUN / 'live-events.jsonl.checkpoint.json'
        follower = await launch('live-follow', export_args(live_id, live_output, follow=True))
        live_running = False
        live_deadline = time.monotonic() + 1200
        while time.monotonic() < live_deadline:
            assert follower.returncode is None
            live_execution = await api(f'/v1/executions/{live_id}')
            checkpoint = json.loads(live_checkpoint.read_text()) if live_checkpoint.exists() else {}
            if checkpoint.get('committed_records', 0) >= 5 and live_execution['status'] not in TERMINAL:
                if not live_running:
                    save('live-running.json', {'execution': live_execution, 'checkpoint': checkpoint})
                    print('Live running-state export verified', flush=True)
                live_running = True
            if live_execution['status'] in TERMINAL:
                break
            if time.monotonic() - last_note > 15:
                print(
                    'Live', live_execution['status'], 'committed=', checkpoint.get('committed_records', 0), flush=True
                )
                last_note = time.monotonic()
            await asyncio.sleep(1)
        else:
            raise TimeoutError('Live did not terminate')
        assert live_running
        assert live_execution['status'] == 'succeeded'
        save('live-terminal.json', live_execution)
        live_oracle, live_rows = await reconcile_tail(
            api,
            [live_id],
            lambda: read_oracle(live_id, batch=False),
            follower,
            live_output,
            stop_signal=signal.SIGTERM,
        )
        save('live-oracle.json', live_oracle)
        assert follower.returncode == 143
        live_result = await launch('live-results', export_args(live_id, RUN / 'live-results.jsonl', content='results'))
        await asyncio.wait_for(live_result.wait(), 120)
        assert live_result.returncode == 0
        live_released = json.loads((RUN / 'live-events.jsonl.producers-released.json').read_text())
        save('live-allocations.json', live_released[live_id]['allocations'])
        report = {
            'passed': True,
            'execution_id': parent,
            'child_count': len(child_ids),
            'items': 3,
            'raw_events': len(rows),
            'streams': len({(r['run_id'], r['run_span_id']) for r in rows}),
            'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
            'running_export': True,
            'sigkill_resume': True,
            'all_fields_equal': True,
            'results_export': True,
            'batch_snapshot': True,
            'single_snapshot': True,
            'single_snapshot_events': len(single_rows),
            'allocations_released': True,
            'live_execution_id': live_id,
            'live_running_export': True,
            'live_raw_events': len(live_rows),
            'live_sigterm': True,
            'live_allocations_released': True,
        }
        success = True
    finally:
        try:
            cleanup = await finish_cleanup(cleanup_resources(api, PROCESSES, owned, submissions))
            verified = (
                not cleanup['unresolved_submissions']
                and all(item['verified'] for item in cleanup['local_processes'])
                and all(item['verified'] for item in cleanup['executions'].values())
            )
            if not verified:
                save('report.json', {'passed': False, 'cleanup_verified': False})
                print('Cleanup incomplete; inspect ownership.json and cleanup.json', flush=True)
                if success:
                    raise AssertionError('resource cleanup not verified')
            elif success:
                save('report.json', report)
                print(json.dumps(report), flush=True)
        except Exception as exc:
            if success:
                raise
            print('Cleanup evidence requires inspection:', type(exc).__name__, flush=True)
        finally:
            await session.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Run opt-in Cloud event export E2E against runner-test.')
    parser.add_argument(
        '--trace-db-config',
        type=Path,
        required=True,
        help='Private JSON with read_host/read_port/database/username/trace_table_name',
    )
    parser.add_argument('--db-password-file', type=Path, required=True)
    parser.add_argument('--taihu-token-file', type=Path, default=TOKEN_FILE)
    parser.add_argument('--control-url', default=CONTROL)
    parser.add_argument('--application-url', default=APPLICATION)
    parser.add_argument('--environment', default=ENVIRONMENT)
    parser.add_argument('--output-dir', type=Path, default=OUT)
    return parser.parse_args()


def prepare(args):
    global OUT, CONTROL, APPLICATION, TOKEN_FILE, DB_PASSWORD_FILE, CONFIG, SOURCE, MARKER, RUN, ENVIRONMENT
    from urllib.parse import urlsplit

    if urlsplit(args.control_url).hostname != 'test-control-runner-nexus-agent.mcp.woa.com':
        raise ValueError('this write-capable E2E is restricted to runner-test Control')
    if urlsplit(args.application_url).hostname != 'test-application-runner-nexus-agent.mcp.woa.com':
        raise ValueError('this E2E is restricted to runner-test Application')
    OUT = args.output_dir.expanduser().resolve()
    OUT.mkdir(parents=True, exist_ok=True, mode=0o700)
    CONTROL, APPLICATION = args.control_url.rstrip('/'), args.application_url.rstrip('/')
    ENVIRONMENT = args.environment
    TOKEN_FILE = args.taihu_token_file.expanduser().resolve()
    DB_PASSWORD_FILE = args.db_password_file.expanduser().resolve()
    CONFIG = json.loads(args.trace_db_config.expanduser().read_text())
    MARKER = 'export-e2e-' + uuid.uuid4().hex[:12]
    RUN = OUT / MARKER
    SOURCE = RUN / 'source'
    SOURCE.mkdir(parents=True, mode=0o700)
    RUN.chmod(0o700)
    for name in ('pyproject.toml', 'uv.lock', 'README.md'):
        shutil.copyfile(ROOT / name, SOURCE / name)
    shutil.copyfile(Path(__file__).with_name('cloud_export_probe.py'), SOURCE / 'export_probe.py')
    config = {
        'marker': MARKER,
        'tasks': [
            {'label': 'long', 'seconds': 35, 'fail_first': False},
            {'label': 'retry', 'seconds': 12, 'fail_first': True},
            {'label': 'late', 'seconds': 15, 'fail_first': False},
        ],
    }
    (SOURCE / 'probe-config.json').write_text(json.dumps(config, indent=2))
    generator = """import json
from pathlib import Path
from nexus.runner.batch import BatchItem
from nexus.runner.single import RunInput

def get_total_count():
    return 3

def generate_configs():
    config = json.loads(Path(__file__).with_name('probe-config.json').read_text())
    for item in config['tasks']:
        yield BatchItem(
            run_id=f"{config['marker']}-{item['label']}",
            config=RunInput(imports=['export_probe'], task_name='cloud_export_probe',
                            task_config={'marker': config['marker'], **item}),
            metadata={'test': 'cloud-event-export', 'label': item['label']},
        )
"""
    (SOURCE / 'generator.py').write_text(generator)
    # JSON is a YAML subset; avoid another serializer in this test helper.
    (SOURCE / 'live.yaml').write_text(
        json.dumps({
            'imports': ['export_probe'],
            'task_name': 'cloud_export_probe',
            'task_config': {'label': 'live', 'marker': MARKER, 'seconds': 25, 'fail_first': False},
        })
    )
    print('E2E evidence directory:', RUN, flush=True)


if __name__ == '__main__':
    prepare(parse_args())
    asyncio.run(run_cancellable(main()))
