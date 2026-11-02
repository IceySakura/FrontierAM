import asyncio
import json
import os
import sqlite3

from pathlib import Path
from typing import Any

import pytest

from aiohttp.test_utils import TestClient, TestServer

from nexus.runner.live.server import MANAGER_KEY, create_app
from nexus.runner.single import RunOutput


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, '').strip().lower() in {'1', 'true', 'yes', 'on'}


def _load_dotenv_if_present() -> None:
    env_file = Path('.env')
    if not env_file.exists():
        return
    for raw_line in env_file.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def _openai_clients(answer: str) -> dict[str, Any]:
    if not _truthy_env('NEXUS_LIVE_SQLITE_REAL_OPENAI'):
        return {
            'main': {
                'name': 'mock',
                'config': {'client_args': {'responses': [answer]}},
            }
        }

    _load_dotenv_if_present()
    app_id = os.environ.get('MODEL_SERVICE_API_APP_ID', '').strip()
    app_key = os.environ.get('MODEL_SERVICE_API_APP_KEY', '').strip()
    base_url = os.environ.get('MODEL_SERVICE_API_BASE_URL', '').strip()
    if not app_id or not app_key or not base_url:
        pytest.skip('real OpenAI live SQLite integration requires MODEL_SERVICE_API_APP_ID/KEY/BASE_URL')

    provider = os.environ.get('MODEL_SERVICE_API_PROVIDER', 'azure_openai').strip()
    timeout = os.environ.get('MODEL_SERVICE_API_TIMEOUT', '60').strip()
    model = os.environ.get('NEXUS_LIVE_SQLITE_REAL_OPENAI_MODEL') or os.environ.get('NEXUS_MODEL') or 'gpt-5.5'

    return {
        'main': {
            'name': os.environ.get('NEXUS_LIVE_SQLITE_OPENAI_CLIENT_NAME', 'litellm'),
            'config': {
                'client_args': {
                    'api_key': f'{app_id}:{app_key}?provider={provider}&model={model}&timeout={timeout}',
                    'base_url': base_url,
                },
                'request_args': {'model': model},
            },
        }
    }


def _main_agent_payload(run_id: str, answer: str) -> dict[str, Any]:
    imports = [
        'nexus.extensions.tasks.identity.task',
        'nexus.extensions.agents.main.agent',
    ]
    if _truthy_env('NEXUS_LIVE_SQLITE_REAL_OPENAI'):
        client_name = os.environ.get('NEXUS_LIVE_SQLITE_OPENAI_CLIENT_NAME', 'litellm')
        imports.append(f'nexus.extensions.openai_clients.{client_name}')
    else:
        imports.append('nexus.extensions.openai_clients.mock')

    return {
        'run_id': run_id,
        'imports': imports,
        'task_name': 'identity',
        'agent_name': 'main_agent',
        'agent_config': {
            'user_prompt': f'Reply with only this exact text: {answer}',
            'tools': [],
            'max_iterations': 1,
        },
        'openai_clients': _openai_clients(answer),
    }


def _live_cache_config(*key_paths: str) -> dict[str, Any]:
    return {'cache': {'namespace': 'integration-test', 'key_paths': list(key_paths)}}


async def _wait_for_terminal(
    client: TestClient,
    run_id: str,
    *,
    attempts: int = 100,
    interval_seconds: float = 0.05,
) -> dict[str, Any]:
    for _ in range(attempts):
        response = await client.get(f'/v1/runs/{run_id}')
        assert response.status == 200
        payload = await response.json()
        if payload['status'] in {'closed', 'failed', 'killed'}:
            return payload
        await asyncio.sleep(interval_seconds)
    raise AssertionError(f'run did not finish: {run_id}')


async def _wait_for_status(client: TestClient, run_id: str, status: str) -> dict[str, Any]:
    for _ in range(100):
        response = await client.get(f'/v1/runs/{run_id}')
        assert response.status == 200
        payload = await response.json()
        if payload['status'] == status:
            return payload
        await asyncio.sleep(0.05)
    raise AssertionError(f'run {run_id} did not reach {status}')


async def _runs_by_id(client: TestClient) -> dict[str, dict[str, Any]]:
    response = await client.get('/v1/runs?limit=50')
    assert response.status == 200
    payload = await response.json()
    return {item['run_id']: item for item in payload['runs']}


async def _runs_by_cache_key(
    client: TestClient, cache_key: str, *, status: str | None = None
) -> dict[str, dict[str, Any]]:
    params = {'limit': '50', 'cache_key': cache_key}
    if status:
        params['status'] = status
    response = await client.get('/v1/runs', params=params)
    assert response.status == 200
    payload = await response.json()
    return {item['run_id']: item for item in payload['runs']}


async def _runs_by_query(client: TestClient, query: str) -> dict[str, dict[str, Any]]:
    response = await client.get('/v1/runs', params={'limit': '50', 'q': query})
    assert response.status == 200
    payload = await response.json()
    return {item['run_id']: item for item in payload['runs']}


async def _start_client(db_path: Path, *, terminal_run_retention_seconds: float = 60.0) -> TestClient:
    client = TestClient(
        TestServer(create_app(storage_path=db_path, terminal_run_retention_seconds=terminal_run_retention_seconds))
    )
    await client.start_server()
    return client


async def _wait_for_evicted(client: TestClient, run_id: str) -> None:
    manager = client.server.app[MANAGER_KEY]
    for _ in range(100):
        if run_id not in manager.runs:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f'run was not evicted from memory: {run_id}')


def _stored_trace_metadata(db_path: Path, run_id: str) -> dict[str, Any] | None:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute('SELECT trace_metadata_json FROM live_runs WHERE run_id = ?', (run_id,)).fetchone()
    if row is None or row[0] is None:
        return None
    return json.loads(row[0])


async def _wait_for_stored_trace_metadata(db_path: Path, run_id: str) -> dict[str, Any]:
    for _ in range(100):
        metadata = _stored_trace_metadata(db_path, run_id)
        if metadata is not None:
            return metadata
        await asyncio.sleep(0.01)
    raise AssertionError(f'run trace metadata was not stored: {run_id}')


@pytest.mark.asyncio
async def test_live_service_page_exposes_metadata_copy_controls(tmp_path) -> None:
    client = await _start_client(tmp_path / 'live.sqlite')
    try:
        response = await client.get('/')
        assert response.status == 200
        text = await response.text()
        assert 'metadata-copy-button' in text
        assert 'copyMetadataValue' in text
        assert 'navigator.clipboard.writeText' in text
        assert 'fallbackCopyText' in text
        assert 'document.execCommand("copy")' in text
        assert 'key_paths:' in text
        assert 'agent_config.version' in text
        assert 'runtime_configs.main.image' in text
        assert 'openai_clients.main.config.request_args.model' in text
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_live_service_persists_main_agent_run_list_across_restart(tmp_path) -> None:
    db_path = tmp_path / 'live.sqlite'
    run_ids = ['run_service_history_1', 'run_service_history_2']
    real_openai = _truthy_env('NEXUS_LIVE_SQLITE_REAL_OPENAI')

    client = await _start_client(db_path)
    try:
        for index, run_id in enumerate(run_ids, start=1):
            response = await client.post('/v1/runs', json=_main_agent_payload(run_id, f'answer {index}'))
            assert response.status == 201

        for index, run_id in enumerate(run_ids, start=1):
            payload = await _wait_for_terminal(client, run_id)
            assert payload['status'] == 'closed'
            final_answer = payload['output']['agent_output']['final_answer']
            if real_openai:
                assert final_answer
            else:
                assert final_answer == f'answer {index}'
    finally:
        await client.close()

    restarted = await _start_client(db_path)
    try:
        response = await restarted.get('/v1/runs')
        assert response.status == 200
        payload = await response.json()
        runs = {item['run_id']: item for item in payload['runs']}

        assert set(run_ids) <= set(runs)
        assert all(runs[run_id]['status'] == 'closed' for run_id in run_ids)

        response = await restarted.get('/v1/runs/run_service_history_1')
        assert response.status == 200
        detail = await response.json()
        final_answer = detail['output']['agent_output']['final_answer']
        if real_openai:
            assert final_answer
        else:
            assert final_answer == 'answer 1'
    finally:
        await restarted.close()


@pytest.mark.asyncio
async def test_live_service_summary_run_list_omits_heavy_fields_but_detail_keeps_them(monkeypatch, tmp_path) -> None:
    async def fake_run_single(run_input, *, hooks=None):
        await hooks.on_context_ready(type('Ctx', (), {'runtime_provider': None, 'get_runtime': None})())
        output = RunOutput(task_output={'archive_b64': 'x' * 8192, 'run_id': run_input.run_id})
        await hooks.on_run_output(None, output)
        return output

    monkeypatch.setattr('nexus.runner.live.manager.run_single', fake_run_single)

    run_id = 'run_service_summary'
    client = await _start_client(tmp_path / 'live.sqlite', terminal_run_retention_seconds=-1)
    try:
        response = await client.post('/v1/runs', json={'run_id': run_id, 'task_name': 'fake_task'})
        assert response.status == 201
        await _wait_for_terminal(client, run_id)

        response = await client.get('/v1/runs', params={'summary': 'true', 'limit': '50'})
        assert response.status == 200
        summary_runs = {item['run_id']: item for item in (await response.json())['runs']}
        assert summary_runs[run_id]['status'] == 'closed'
        assert 'output' not in summary_runs[run_id]
        assert 'runtimes' not in summary_runs[run_id]

        response = await client.get(f'/v1/runs/{run_id}')
        assert response.status == 200
        detail = await response.json()
        assert detail['output']['task_output']['archive_b64'] == 'x' * 8192
        assert detail['runtimes'] == {}
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_live_service_persists_raw_trace_metadata_only_after_terminal(monkeypatch, tmp_path) -> None:
    context_ready = asyncio.Event()
    release = asyncio.Event()

    async def fake_run_single(run_input, *, hooks=None):
        await hooks.on_context_ready(type('Ctx', (), {'runtime_provider': None, 'get_runtime': None})())
        context_ready.set()
        await release.wait()
        output = RunOutput(task_output={'run_id': run_input.run_id})
        await hooks.on_run_output(None, output)
        return output

    monkeypatch.setattr('nexus.runner.live.manager.run_single', fake_run_single)

    db_path = tmp_path / 'live.sqlite'
    run_id = 'run_service_snapshot_flush'
    client = await _start_client(db_path, terminal_run_retention_seconds=-1)
    try:
        response = await client.post('/v1/runs', json={'run_id': run_id, 'task_name': 'fake_task'})
        assert response.status == 201
        await asyncio.wait_for(context_ready.wait(), timeout=1)

        running_detail = await _wait_for_status(client, run_id, 'running')
        assert running_detail['events_cursor'] > 0
        assert _stored_trace_metadata(db_path, run_id) is None

        release.set()
        closed_detail = await _wait_for_terminal(client, run_id)

        assert closed_detail['status'] == 'closed'
        metadata = await _wait_for_stored_trace_metadata(db_path, run_id)
        assert metadata['version'] == 4
        assert metadata['run_id'] == run_id
        local_sink = next(item for item in metadata['sinks'] if item['name'] == 'local')
        trace_path = Path(local_sink['metadata']['trace_events_path'])
        assert trace_path.is_file()
        assert sum(1 for line in trace_path.read_text(encoding='utf-8').splitlines() if line) > 0
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_live_service_replays_events_after_terminal_run_eviction(tmp_path) -> None:
    db_path = tmp_path / 'live.sqlite'
    run_id = 'run_service_evicted_events'
    client = await _start_client(db_path, terminal_run_retention_seconds=0.01)
    try:
        response = await client.post('/v1/runs', json=_main_agent_payload(run_id, 'evicted answer'))
        assert response.status == 201

        detail = await _wait_for_terminal(client, run_id, attempts=1200, interval_seconds=0.25)
        assert detail['status'] == 'closed'
        await _wait_for_evicted(client, run_id)

        response = await client.get(f'/v1/runs/{run_id}/events?cursor=0')
        assert response.status == 200
        text = await response.text()

        assert 'event: run.output' in text
        assert 'event: run.status' in text
        assert '"status": "closed"' in text
        assert 'evicted answer' in text
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_live_service_real_openai_run_can_be_filtered_by_cache_key(tmp_path) -> None:
    if not _truthy_env('NEXUS_LIVE_SQLITE_REAL_OPENAI'):
        pytest.skip('set NEXUS_LIVE_SQLITE_REAL_OPENAI=1 to run real OpenAI cache-key integration')

    db_path = tmp_path / 'live.sqlite'
    run_id = 'run_real_openai_cache_key'
    client = await _start_client(db_path)
    try:
        payload = _main_agent_payload(run_id, 'cache key real answer')
        payload['live'] = _live_cache_config('task_name', 'agent_name')
        response = await client.post('/v1/runs', json=payload)
        assert response.status == 201
        created = await response.json()
        assert created['run_id'] == run_id
        assert created['cache_key']

        detail = await _wait_for_terminal(client, run_id, attempts=1200, interval_seconds=0.25)
        assert detail['status'] == 'closed'
        assert detail['cache_key'] == created['cache_key']
        assert detail['output']['agent_output']['final_answer']

        runs_by_key = await _runs_by_cache_key(client, created['cache_key'])
        assert list(runs_by_key) == [run_id]
        assert runs_by_key[run_id]['status'] == 'closed'
        assert runs_by_key[run_id]['cache_key'] == created['cache_key']

        response = await client.get(f'/v1/runs/{run_id}/config')
        assert response.status == 200
        config = await response.json()
        client_args = config['request']['openai_clients']['main']['config']['client_args']
        assert client_args['api_key'] == '<redacted>'
        assert client_args['base_url']
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_live_service_consumes_one_of_multiple_cached_runs(monkeypatch, tmp_path) -> None:
    started_run_ids: list[str] = []

    async def fake_run_single(run_input, *, hooks=None):
        started_run_ids.append(run_input.run_id)
        await hooks.on_context_ready(type('Ctx', (), {'runtime_provider': None, 'get_runtime': None})())
        prompt = await hooks.agent_input_overrides['get_user_input']('ready')
        output = type(
            'Output',
            (),
            {
                'success': True,
                'model_dump': lambda self, mode='json': {'task_output': {'prompt': prompt}},
            },
        )()
        await hooks.on_run_output(None, output)
        return output

    monkeypatch.setattr('nexus.runner.live.manager.run_single', fake_run_single)

    db_path = tmp_path / 'live.sqlite'
    cached_payload = {
        'task_name': 'fake_task',
        'task_config': {'user_prompt': 'same', 'volatile_trace_id': 'trace-a'},
        'live': _live_cache_config('task_name', 'task_config.user_prompt'),
    }
    reuse_payload = {
        'task_name': 'fake_task',
        'task_config': {'user_prompt': 'same', 'volatile_trace_id': 'trace-b'},
        'live': _live_cache_config('task_name', 'task_config.user_prompt'),
    }
    different_payload = {
        'task_name': 'fake_task',
        'task_config': {'user_prompt': 'different'},
        'live': _live_cache_config('task_name', 'task_config.user_prompt'),
    }
    client = await _start_client(db_path)
    try:
        cache_run_ids = []
        cache_key = None
        for _ in range(3):
            response = await client.post('/v1/runs/cache', json=cached_payload)
            assert response.status == 201
            payload = await response.json()
            cache_run_ids.append(payload['run_id'])
            assert payload['reused_cache'] is False
            assert payload['cache_key']
            cache_key = cache_key or payload['cache_key']
            assert payload['cache_key'] == cache_key
        assert cache_key is not None

        for run_id in cache_run_ids:
            await _wait_for_status(client, run_id, 'cached')

        runs = await _runs_by_id(client)
        assert set(cache_run_ids) <= set(runs)
        assert [runs[run_id]['status'] for run_id in cache_run_ids] == ['cached', 'cached', 'cached']
        assert [runs[run_id]['cache_key'] for run_id in cache_run_ids] == [cache_key, cache_key, cache_key]
        assert sum(1 for item in runs.values() if item['status'] == 'cached') == 3
        assert started_run_ids == cache_run_ids

        cached_by_key = await _runs_by_cache_key(client, cache_key, status='cached')
        assert set(cached_by_key) == set(cache_run_ids)
        assert all(item['status'] == 'cached' for item in cached_by_key.values())
        assert all(item['cache_key'] == cache_key for item in cached_by_key.values())

        response = await client.post('/v1/runs', json=reuse_payload)
        assert response.status == 201
        reused = await response.json()

        assert reused['run_id'] == cache_run_ids[0]
        assert reused['status'] == 'idle'
        assert reused['reused_cache'] is True
        assert reused['cache_key'] == cache_key
        assert started_run_ids == cache_run_ids

        runs = await _runs_by_id(client)
        assert [runs[run_id]['status'] for run_id in cache_run_ids] == ['idle', 'cached', 'cached']
        assert sum(1 for item in runs.values() if item['status'] == 'cached') == 2
        assert sum(1 for item in runs.values() if item['status'] == 'idle') == 1

        cached_by_key = await _runs_by_cache_key(client, cache_key, status='cached')
        assert set(cached_by_key) == set(cache_run_ids[1:])
        assert all(item['cache_key'] == cache_key for item in cached_by_key.values())

        response = await client.post('/v1/runs', json=different_payload)
        assert response.status == 201
        fresh = await response.json()

        assert fresh['run_id'] not in cache_run_ids
        assert fresh['reused_cache'] is False
        assert fresh['cache_key'] != cache_key
        await _wait_for_status(client, fresh['run_id'], 'idle')
        assert started_run_ids == [*cache_run_ids, fresh['run_id']]

        runs = await _runs_by_id(client)
        assert [runs[run_id]['status'] for run_id in cache_run_ids] == ['idle', 'cached', 'cached']
        assert runs[fresh['run_id']]['status'] == 'idle'
        assert sum(1 for item in runs.values() if item['status'] == 'cached') == 2
        assert sum(1 for item in runs.values() if item['status'] == 'idle') == 2
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_live_service_rejects_cache_api_without_key_paths(tmp_path) -> None:
    client = await _start_client(tmp_path / 'live.sqlite')
    try:
        response = await client.post('/v1/runs/cache', json={'task_name': 'fake_task'})
        payload = await response.json()

        assert response.status == 400
        assert 'live.cache.key_paths' in payload['error']
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_live_service_run_list_searches_run_id_substrings(monkeypatch, tmp_path) -> None:
    async def fake_run_single(run_input, *, hooks=None):
        await hooks.on_context_ready(type('Ctx', (), {'runtime_provider': None, 'get_runtime': None})())
        return type(
            'Output',
            (),
            {
                'success': True,
                'model_dump': lambda self, mode='json': {'task_output': {'run_id': run_input.run_id}},
            },
        )()

    monkeypatch.setattr('nexus.runner.live.manager.run_single', fake_run_single)

    client = await _start_client(tmp_path / 'live.sqlite')
    try:
        for run_id in ('run_f80e79759ce1', 'run_f80e79759ce2', 'run_other'):
            response = await client.post('/v1/runs', json={'run_id': run_id, 'task_name': 'fake_task'})
            assert response.status == 201
            await _wait_for_terminal(client, run_id)

        runs = await _runs_by_query(client, 'f80e79759ce')
        assert set(runs) == {'run_f80e79759ce1', 'run_f80e79759ce2'}
        assert all(item['run_id'].startswith('run_f80e79759ce') for item in runs.values())
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_live_service_reuses_cached_run_from_cache_api(monkeypatch, tmp_path) -> None:
    async def fake_run_single(run_input, *, hooks=None):
        await hooks.on_context_ready(type('Ctx', (), {'runtime_provider': None, 'get_runtime': None})())
        prompt = await hooks.agent_input_overrides['get_user_input']('ready')
        output = type(
            'Output',
            (),
            {
                'success': True,
                'model_dump': lambda self, mode='json': {'task_output': {'prompt': prompt}},
            },
        )()
        await hooks.on_run_output(None, output)
        return output

    monkeypatch.setattr('nexus.runner.live.manager.run_single', fake_run_single)

    db_path = tmp_path / 'live.sqlite'
    payload = {
        'task_name': 'fake_task',
        'task_config': {'user_prompt': 'same'},
        'live': _live_cache_config('task_name', 'task_config.user_prompt'),
    }
    client = await _start_client(db_path)
    try:
        response = await client.post('/v1/runs/cache', json=payload)
        assert response.status == 201
        cache_response = await response.json()
        cache_run_id = cache_response['run_id']
        assert cache_response['reused_cache'] is False

        cached_info = await _wait_for_status(client, cache_run_id, 'cached')
        assert cached_info['events_cursor'] > 0

        response = await client.get('/v1/runs')
        assert response.status == 200
        runs = {item['run_id']: item for item in (await response.json())['runs']}
        assert runs[cache_run_id]['status'] == 'cached'

        response = await client.post('/v1/runs', json=payload)
        assert response.status == 201
        reused = await response.json()
        assert reused['run_id'] == cache_run_id
        assert reused['status'] == 'idle'
        assert reused['reused_cache'] is True

        response = await client.post(f'/v1/runs/{cache_run_id}/turns', json={'text': 'from user'})
        assert response.status == 202
        final_info = await _wait_for_terminal(client, cache_run_id)
        assert final_info['status'] == 'closed'
        assert final_info['output']['task_output']['prompt'] == 'from user'
    finally:
        await client.close()
