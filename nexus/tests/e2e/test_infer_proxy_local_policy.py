"""Real local CLI single/batch flows against a controlled HTTP capacity/inference service."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import threading
import time

from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from tests.e2e.distributed import request_budget_signal_server as signal


pytestmark = pytest.mark.e2e
ROOT = Path(__file__).resolve().parents[2]
PROBE = """from nexus.core import TaskRunInput, TaskRunOutput, require_context, task_registry

class ProbeInput(TaskRunInput):
    requests: int = 4

class ProbeOutput(TaskRunOutput):
    responses: list[str]

@task_registry.register('local_admission_probe')
async def probe(value: ProbeInput) -> ProbeOutput:
    client = require_context().get_openai_client('main')
    responses = []
    for _ in range(value.requests):
        response = await client.create_completion([{'role': 'user', 'content': 'hello'}])
        responses.append(response.choices[0].message.content)
    return ProbeOutput(responses=responses)
"""


@pytest.fixture
def server():
    with signal.lock:
        signal.state.update(factor=0.5, remaining=600, mode='ok', delay=0, client_calls=0, completions=0)
        signal.completion_times.clear()
    service = ThreadingHTTPServer(('127.0.0.1', 0), signal.Handler)
    thread = threading.Thread(target=service.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{service.server_port}'
    finally:
        service.shutdown()
        service.server_close()
        thread.join(timeout=5)


def prepare(tmp_path, url, *, enabled=True):
    (tmp_path / 'probe.py').write_text(PROBE)
    client_args = {'api_key': signal.client_key, 'base_url': f'{url}/v1', 'max_retries': 1}
    if enabled:
        client_args['local_request_policy'] = {'quota_qpm': 600, 'capacity_url': f'{url}/capacity'}
    config = {
        'imports': ['probe'],
        'task_name': 'local_admission_probe',
        'openai_clients': {
            'main': {
                'name': 'infer_proxy',
                'config': {'client_args': client_args, 'request_args': {'model': 'hy3-offline'}},
            }
        },
    }
    (tmp_path / 'single.json').write_text(json.dumps(config))
    (tmp_path / 'generator.py').write_text(
        'import json\nfrom pathlib import Path\nfrom nexus.runner.batch import BatchItem\n'
        'from nexus.runner.single import RunInput\n'
        'def generate_configs():\n'
        '    config = json.loads(Path(__file__).with_name("single.json").read_text())\n'
        '    for i in range(3):\n'
        '        yield BatchItem(run_id=f"probe-{i}", config=RunInput.model_validate(config))\n'
    )
    return {**os.environ, 'PYTHONPATH': os.pathsep.join([str(ROOT), str(tmp_path)])}


@pytest.mark.parametrize(('batch', 'enabled'), [(False, True), (True, True), (False, False)])
def test_local_cli_single_and_batch_share_capacity(tmp_path, server, batch, enabled):
    env = prepare(tmp_path, server, enabled=enabled)
    command = [sys.executable, '-m', 'nexus.cli.main', 'run']
    if batch:
        command += ['batch', '-g', 'generator.py', '-d', 'batch.db', '--max-concurrent', '3']
    else:
        command += ['single', 'single.json', '-o', 'result.json']
    completed = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)
    (tmp_path / 'cli.log').write_text(completed.stdout + completed.stderr)
    assert completed.returncode == 0, completed.stderr[-3000:]
    if batch:
        with sqlite3.connect(tmp_path / 'batch.db') as db:
            rows = db.execute('SELECT status FROM runs').fetchall()
        assert rows == [('success',)] * 3
    else:
        result = json.loads((tmp_path / 'result.json').read_text())
        assert result['success'] is True
    (tmp_path / 'evidence.json').write_text(json.dumps({**signal.state, 'completion_times': signal.completion_times}))
    assert signal.state['completions'] == (12 if batch else 4)
    assert signal.state['client_calls'] == int(enabled)
    if enabled:
        times = signal.completion_times
        # The first SDK call may initialize lazily after its permit, compressing the first arrival gap.
        assert times[-1] - times[0] >= (len(times) - 2) * 0.2 - 0.03


def test_local_cli_pause_polls_after_sixty_seconds_and_recovers(tmp_path, server):
    env = prepare(tmp_path, server)
    with signal.lock:
        signal.state.update(remaining=0, factor=1.0)
    with (tmp_path / 'cli.log').open('w') as output:
        process = subprocess.Popen(
            [sys.executable, '-m', 'nexus.cli.main', 'run', 'single', 'single.json', '-o', 'result.json'],
            cwd=tmp_path,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 20
            while signal.state['client_calls'] == 0 and time.monotonic() < deadline:
                time.sleep(0.1)
            assert signal.state['client_calls'] == 1
            started = time.monotonic()
            time.sleep(1)
            assert signal.state['completions'] == 0
            with signal.lock:
                signal.state['remaining'] = 600
            assert process.wait(timeout=80) == 0
            assert time.monotonic() - started >= 59
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
    assert signal.state['client_calls'] == 2
    assert signal.state['completions'] == 4
    assert json.loads((tmp_path / 'result.json').read_text())['success'] is True
    (tmp_path / 'evidence.json').write_text(json.dumps({**signal.state, 'completion_times': signal.completion_times}))
