from aiohttp.test_utils import TestClient, TestServer


async def test_builder_serves_bounded_local_file_with_auth(tmp_path):
    from nexus.distributed.dependency_build.logs import create_build_log_app

    path = tmp_path / 'build.log'
    path.write_text('first\nsecond\n')
    async with TestClient(TestServer(create_build_log_app(path, token='read-secret'))) as client:
        assert (await client.get('/v1/internal/logs')).status == 401
        response = await client.get('/v1/internal/logs?lines=1', headers={'Authorization': 'Bearer read-secret'})
        assert await response.text() == 'second\n'
        with path.open('a') as output:
            output.write('third\n')
        response = await client.get('/v1/internal/logs?lines=1', headers={'Authorization': 'Bearer read-secret'})
        assert await response.text() == 'third\n'
        path.write_text(('x' * 8192 + '\n') * 1000)
        response = await client.get('/v1/internal/logs?lines=999999', headers={'Authorization': 'Bearer read-secret'})
        assert len(await response.read()) <= 256 * 1024
        invalid = await client.get('/v1/internal/logs?lines=oops', headers={'Authorization': 'Bearer read-secret'})
        assert invalid.status == 400


def test_build_log_token_is_scoped_to_build_attempt():
    from nexus.distributed.dependency_build.logs import build_log_token

    token = build_log_token('key', 'build-a', 1)
    assert token != build_log_token('key', 'build-a', 2)
    assert token != build_log_token('key', 'build-b', 1)
    assert token != build_log_token('other-key', 'build-a', 1)


def test_unicode_logs_rotate_within_byte_budget(tmp_path, monkeypatch):
    from nexus.distributed.dependency_build import logs

    monkeypatch.setattr(logs, 'BUILD_LOG_MAX_BYTES', 65536)
    path = tmp_path / 'build.log'
    handler = logs.configure_build_log(path, build_id='bld-1', attempt='1')
    try:
        for index in range(200):
            logs.output_logger.info('%s %s', index, '日志' * 300)
        handler.close()
        assert sum(file.stat().st_size for file in tmp_path.iterdir()) <= 65536
        assert '199 ' in path.read_text()
        assert len(path.read_text().splitlines()) < 200
    finally:
        logs.output_logger.removeHandler(handler)
        handler.close()


def test_file_keeps_latest_output_after_console_cap(tmp_path, capfd):
    import sys

    from nexus.distributed.dependency_build import logs
    from nexus.distributed.dependency_build.worker import _run_streaming

    path = tmp_path / 'build.log'
    handler = logs.configure_build_log(path, build_id='bld-1', attempt='1')
    try:
        _run_streaming([sys.executable, '-c', "print(('x' * 4096 + '\\n') * 600); print('final-marker')"], cwd=tmp_path)
        handler.close()
        assert 'final-marker' not in capfd.readouterr().out
        assert 'final-marker' in path.read_text()
    finally:
        logs.output_logger.removeHandler(handler)
        handler.close()


def test_collected_file_uses_shared_json_formatter_and_build_scope(tmp_path):
    import json

    from nexus.distributed.dependency_build import logs

    path = tmp_path / 'build.jsonl'
    handler = logs.configure_build_log(path, build_id='bld-1', attempt='2')
    try:
        logs.output_logger.info('build output')
        handler.close()
        record = json.loads(path.read_text())
        assert record['dependency_build_id'] == 'bld-1'
        assert record['build_attempt'] == '2'
        assert record['message'] == 'build output'
        assert record['component'] == 'dependency_builder'
    finally:
        logs.output_logger.removeHandler(handler)
        handler.close()


async def test_blocked_log_io_does_not_block_build_or_shutdown(tmp_path):
    import asyncio
    import os
    import sys
    import textwrap

    script = textwrap.dedent("""
        import pathlib, sys, threading, time
        from nexus.distributed.dependency_build import logs, worker
        directory = pathlib.Path(sys.argv[1])
        target = sys.argv[2]
        stuck = threading.Event()
        entered = threading.Event()
        def blocked(*args):
            entered.set()
            stuck.wait()
        if target == 'file':
            logs._BuildLogFile.emit = blocked
        elif target == 'flush':
            original_open = logs._BuildLogFile._open
            class SlowFlush:
                def __init__(self, stream): self.stream = stream
                def __getattr__(self, name): return getattr(self.stream, name)
                def flush(self): blocked()
            def slow_open(self): return SlowFlush(original_open(self))
            logs._BuildLogFile._open = slow_open
        else:
            logs._write_stdout = blocked
        handler = logs.configure_build_log(directory / 'build.jsonl', build_id='build', attempt='1')
        worker._emit('start\\n')
        assert entered.wait(3)
        command = 'import pathlib; [print("x"*1024) for _ in range(6000)]; pathlib.Path('+repr(str(directory/'completed'))+').touch()'
        started = time.monotonic()
        worker._run_streaming([sys.executable, '-c', command], cwd=directory)
        assert (directory/'completed').exists()
        assert handler.dropped > 0
        assert handler.queue.qsize() <= logs.BUILD_LOG_QUEUE_SIZE
        handler.close()
        assert time.monotonic()-started < 8
        (directory/'closed').touch()
    """)
    for target in ('file', 'flush', 'stdout'):
        directory = tmp_path / target
        directory.mkdir()
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            '-c',
            script,
            str(directory),
            target,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ},
        )
        try:
            _, error = await asyncio.wait_for(process.communicate(), timeout=12)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise AssertionError(f'{target} blocked Builder process exit') from None
        assert process.returncode == 0, error.decode()
        assert (directory / 'closed').exists()


def test_async_log_queue_preserves_latest_records_and_reports_drops(tmp_path, monkeypatch):
    import json
    import threading

    from nexus.distributed.dependency_build import logs

    entered, release = threading.Event(), threading.Event()
    monkeypatch.setattr(logs, 'BUILD_LOG_QUEUE_SIZE', 3)
    write_file = logs._BuildLogFile.emit

    def slow_write(self, record):
        entered.set()
        release.wait(timeout=5)
        write_file(self, record)

    monkeypatch.setattr(logs._BuildLogFile, 'emit', slow_write)
    path = tmp_path / 'build.jsonl'
    handler = logs.configure_build_log(path, build_id='build', attempt='1')
    try:
        logs.output_logger.info('start')
        assert entered.wait(3)
        for i in range(10):
            logs.output_logger.info('line %d', i)
        assert handler.queue.qsize() == 3
        assert handler.dropped == 7
        release.set()
        handler.close()
        messages = [json.loads(line)['message'] for line in path.read_text().splitlines()]
        assert messages[:4] == ['start', 'line 7', 'line 8', 'line 9']
        assert '7 log records dropped' in messages[-1]
    finally:
        release.set()
        logs.output_logger.removeHandler(handler)
        handler.close()


def test_async_log_is_visible_while_command_is_running(tmp_path):
    import sys

    from nexus.distributed.dependency_build import logs, worker

    path = tmp_path / 'build.jsonl'
    handler = logs.configure_build_log(path, build_id='build', attempt='1')
    script = (
        'import pathlib,time,sys; print("first-line", flush=True); '
        f'p=pathlib.Path({str(path)!r}); deadline=time.monotonic()+5; '
        '\nwhile time.monotonic()<deadline:'
        '\n if p.exists() and "first-line" in p.read_text(): break'
        '\n time.sleep(.01)'
        '\nelse: sys.exit(1)'
        '\nprint("Build complete",flush=True)'
    )
    try:
        worker._run_streaming([sys.executable, '-c', script], cwd=tmp_path)
        handler.close()
        assert 'Build complete' in path.read_text()
    finally:
        logs.output_logger.removeHandler(handler)
        handler.close()
