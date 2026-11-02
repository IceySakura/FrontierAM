from __future__ import annotations

import asyncio
import json
import signal
import subprocess
import sys

from pathlib import Path

import aiohttp
import pytest

from aiohttp import web
from aiohttp.test_utils import TestServer

from nexus.distributed.export_output import JsonlEventWriter
from tests.e2e import test_cloud_event_export as wrapper
from tests.e2e.distributed import run_cloud_event_export as harness
from tests.unit.distributed.test_cloud_event_export_output import _event, _source, _stream


def test_outer_timeout_kills_owned_descendants_and_records_unresolved_cleanup(tmp_path):
    pidfile = tmp_path / 'child-pid'
    script = """
import subprocess, sys, signal, time
from pathlib import Path
signal.signal(signal.SIGTERM, signal.SIG_IGN)
child = subprocess.Popen([sys.executable, '-c', 'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)'])
Path(sys.argv[1]).write_text(str(child.pid))
time.sleep(60)
"""
    with pytest.raises(subprocess.TimeoutExpired):
        wrapper.run_owned_command(
            [sys.executable, '-c', script, str(pidfile)],
            timeout=0.5,
            cleanup_grace=0.2,
            evidence=tmp_path / 'timeout.json',
        )
    pid = int(pidfile.read_text())
    stat = Path(f'/proc/{pid}/stat')
    assert not stat.exists() or stat.read_text().split()[2] == 'Z'
    evidence = json.loads((tmp_path / 'timeout.json').read_text())
    assert evidence['forced_group_kill'] is True
    assert evidence['remote_cleanup'] == 'unverified; inspect ownership.json and cleanup.json in the run directory'


async def test_sigterm_cancels_harness_and_finishes_async_cleanup(tmp_path):
    marker = tmp_path / 'ready'
    done = tmp_path / 'done'
    script = """
import asyncio, sys
from pathlib import Path
from tests.e2e.distributed.run_cloud_event_export import run_cancellable
async def work():
    Path(sys.argv[1]).write_text('ready')
    try:
        await asyncio.Event().wait()
    finally:
        await asyncio.sleep(.05)
        Path(sys.argv[2]).write_text('cleaned')
asyncio.run(run_cancellable(work()))
"""
    process = await asyncio.create_subprocess_exec(sys.executable, '-c', script, str(marker), str(done))
    try:
        for _ in range(200):
            if marker.exists() or process.returncode is not None:
                break
            await asyncio.sleep(0.01)
        assert marker.exists()
        process.send_signal(signal.SIGTERM)
        await asyncio.wait_for(process.wait(), 3)
        assert done.read_text() == 'cleaned'
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()


async def test_failed_submit_preserves_accepted_ownership_before_assertion(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, 'RUN', tmp_path)
    owned, submissions = [], {'submit': {'idempotency_key': 'local-marker'}}
    stdout = tmp_path / 'submit.stdout'
    with stdout.open('wb') as stream:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            '-c',
            'print(\'{"execution_id":"exe-owned"}\', flush=True); raise SystemExit(1)',
            stdout=stream,
        )
    with pytest.raises(AssertionError, match='submit failed'):
        await harness.await_submission('submit', process, owned, submissions, timeout=3)
    assert owned == ['exe-owned']
    assert json.loads((tmp_path / 'ownership.json').read_text())['submissions']['submit']['execution_id'] == 'exe-owned'


async def test_release_wait_includes_late_oracle_rows_and_independent_final_read(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, 'RUN', tmp_path)
    writer = JsonlEventWriter(tmp_path / 'events.jsonl', source=_source())
    await writer.open()
    await writer.commit_page(_stream(), expected_cursor=0, rows=[_event(1)])
    rows = [_event(1)]
    released = False
    reads = 0

    async def endpoint(request):
        if request.path.endswith('/allocations'):
            return web.json_response({'items': [{'status': 'released' if released else 'running'}]})
        return web.json_response({'execution_id': 'batch-1', 'status': 'succeeded'})

    async def oracle():
        nonlocal reads
        reads += 1
        return list(rows)

    async def late_commit():
        nonlocal released
        await asyncio.sleep(0.05)
        rows.append(_event(7))
        released = True
        await asyncio.sleep(0.05)
        await writer.commit_page(_stream(), expected_cursor=1, rows=[_event(7)])

    process = await asyncio.create_subprocess_exec(sys.executable, '-c', 'import time; time.sleep(60)')
    producer = asyncio.create_task(late_commit())
    app = web.Application()
    app.router.add_get('/{path:.*}', endpoint)
    try:
        async with TestServer(app) as server, aiohttp.ClientSession() as session:

            async def api(path, *, method='GET'):
                return await harness.request_api(session, str(server.make_url('')).rstrip('/'), path, method=method)

            final, actual = await harness.reconcile_tail(
                api, ['batch-1'], oracle, process, writer.output, timeout=3, interval=0.01, stop_signal=signal.SIGTERM
            )
        assert released
        assert reads >= 2
        assert final == actual == [_event(1), _event(7)]
    finally:
        await producer
        await writer.close(reason='interrupted')
        if process.returncode is None:
            process.kill()
            await process.wait()


async def test_release_timeout_is_failure_and_cleanup_attempts_every_owned_target(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, 'RUN', tmp_path)
    stopped = set()

    async def endpoint(request):
        target = request.match_info['target']
        if request.method == 'POST':
            stopped.add(target)
            return web.json_response({})
        if target == 'broken':
            return web.json_response({'error': 'private-secret'}, status=503)
        if request.path.endswith('/allocations'):
            return web.json_response({'items': [{'status': 'released' if target in stopped else 'running'}]})
        return web.json_response({'execution_id': target, 'status': 'killed' if target in stopped else 'running'})

    app = web.Application()
    app.router.add_route('*', '/v1/executions/{target}{suffix:.*}', endpoint)
    process = await asyncio.create_subprocess_exec(sys.executable, '-c', 'import time; time.sleep(60)')
    async with TestServer(app) as server, aiohttp.ClientSession() as session:

        async def api(path, *, method='GET'):
            return await harness.request_api(session, str(server.make_url('')).rstrip('/'), path, method=method)

        report = {}

        async def attempt():
            nonlocal report
            try:
                await harness.wait_released(api, 'healthy', timeout=0.05, interval=0.01)
            finally:
                report = await harness.cleanup_resources(
                    api, [process], ['broken', 'healthy'], {}, timeout=0.2, interval=0.01
                )

        with pytest.raises(TimeoutError, match='release'):
            await attempt()
        assert process.returncode is not None
        assert report['executions']['healthy']['verified'] is True
        assert report['executions']['broken']['verified'] is False
        assert 'private-secret' not in json.dumps(report)
        assert stopped == {'healthy'}


async def test_post_stop_oracle_growth_fails_instead_of_publishing_success(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, 'RUN', tmp_path)
    writer = JsonlEventWriter(tmp_path / 'events.jsonl', source=_source())
    await writer.open()
    await writer.commit_page(_stream(), expected_cursor=0, rows=[_event(1)])
    reads = 0

    async def oracle():
        nonlocal reads
        reads += 1
        return [_event(1)] if reads == 1 else [_event(1), _event(7)]

    async def endpoint(request):
        if request.path.endswith('/allocations'):
            return web.json_response({'items': [{'status': 'released'}]})
        return web.json_response({'execution_id': 'batch-1', 'status': 'succeeded'})

    app = web.Application()
    app.router.add_get('/{path:.*}', endpoint)
    process = await asyncio.create_subprocess_exec(sys.executable, '-c', 'import time; time.sleep(60)')
    try:
        async with TestServer(app) as server, aiohttp.ClientSession() as session:

            async def api(path, *, method='GET'):
                return await harness.request_api(session, str(server.make_url('')).rstrip('/'), path, method=method)

            with pytest.raises(AssertionError, match='tail changed'):
                await harness.reconcile_tail(
                    api, ['batch-1'], oracle, process, writer.output, timeout=3, stop_signal=signal.SIGTERM
                )
        assert reads == 2
        assert not (tmp_path / 'report.json').exists()
    finally:
        await writer.close(reason='interrupted')
        if process.returncode is None:
            process.kill()
            await process.wait()


async def test_cancel_during_child_launch_retains_process_for_draining(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, 'RUN', tmp_path)
    monkeypatch.setattr(harness, 'NEXUS', Path(sys.executable))
    processes = []
    monkeypatch.setattr(harness, 'PROCESSES', processes)
    entered, release = asyncio.Event(), asyncio.Event()
    original = asyncio.create_subprocess_exec
    actual = None

    async def delayed_return(*args, **kwargs):
        nonlocal actual
        actual = await original(*args, **kwargs)
        entered.set()
        await release.wait()
        return actual

    monkeypatch.setattr(asyncio, 'create_subprocess_exec', delayed_return)
    task = asyncio.create_task(harness.launch('child', ['-c', 'import time; time.sleep(60)']))
    try:
        await asyncio.wait_for(entered.wait(), 3)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert processes == [actual]
        assert actual is not None
        await harness.drain_process(actual, timeout=0.1)
        assert actual.returncode is not None
    finally:
        release.set()
        if actual is not None and actual.returncode is None:
            actual.kill()
            await actual.wait()


async def test_unknown_submit_outcome_remains_unresolved_after_local_cleanup(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, 'RUN', tmp_path)
    process = await asyncio.create_subprocess_exec(sys.executable, '-c', 'raise SystemExit(1)')
    await process.wait()
    submissions = {'submit': {'idempotency_key': 'local-marker', 'status': 'unresolved'}}

    async def api(path, *, method='GET'):
        raise AssertionError('no remote identity exists to clean')

    report = await harness.cleanup_resources(api, [process], [], submissions)
    assert report['unresolved_submissions'] == ['submit']
    assert json.loads((tmp_path / 'ownership.json').read_text())['submissions'] == submissions


async def test_failure_cleanup_discovers_and_verifies_batch_attempt_children(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, 'RUN', tmp_path)
    stopped = set()

    async def endpoint(request):
        if request.path.endswith('/batch/items'):
            return web.json_response({'items': [{'item_id': 'item-1'}], 'next_cursor': None})
        if request.path.endswith('/batch/items/item-1'):
            return web.json_response({'attempts': [{'child_execution_id': 'child-1'}]})
        target = request.match_info['target']
        if request.method == 'POST':
            stopped.add(target)
            return web.json_response({})
        if request.path.endswith('/allocations'):
            return web.json_response({'items': [{'status': 'released'}]})
        return web.json_response({'execution_id': target, 'status': 'killed' if target in stopped else 'running'})

    app = web.Application()
    app.router.add_route('*', '/v1/executions/{target}{suffix:.*}', endpoint)
    owned = ['batch-1']
    submissions = {'submit': {'execution_id': 'batch-1', 'status': 'accepted', 'spec_type': 'nexus_batch'}}
    async with TestServer(app) as server, aiohttp.ClientSession() as session:

        async def api(path, *, method='GET'):
            return await harness.request_api(session, str(server.make_url('')).rstrip('/'), path, method=method)

        report = await harness.cleanup_resources(api, [], owned, submissions, timeout=2, interval=0.01)
    assert stopped == {'batch-1', 'child-1'}
    assert report['executions']['child-1']['verified'] is True
    assert 'child-1' in json.loads((tmp_path / 'ownership.json').read_text())['owned']


async def test_cancellation_during_cleanup_waits_for_resource_verification(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, 'RUN', tmp_path)
    entered, release = asyncio.Event(), asyncio.Event()
    stopped = False

    async def endpoint(request):
        nonlocal stopped
        if request.method == 'POST':
            entered.set()
            await release.wait()
            stopped = True
            return web.json_response({})
        if request.path.endswith('/allocations'):
            return web.json_response({'items': [{'status': 'released'}]})
        return web.json_response({'execution_id': 'owned', 'status': 'killed' if stopped else 'running'})

    app = web.Application()
    app.router.add_route('*', '/{path:.*}', endpoint)
    async with TestServer(app) as server, aiohttp.ClientSession() as session:

        async def api(path, *, method='GET'):
            return await harness.request_api(session, str(server.make_url('')).rstrip('/'), path, method=method)

        task = asyncio.create_task(harness.finish_cleanup(harness.cleanup_resources(api, [], ['owned'], {}, timeout=2)))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            task.cancel()
            await asyncio.sleep(0.01)
            assert not task.done()
        finally:
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
    assert json.loads((tmp_path / 'cleanup.json').read_text())['executions']['owned']['verified'] is True
