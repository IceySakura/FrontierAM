import argparse
import asyncio
import json
import sqlite3

from types import SimpleNamespace

import aiohttp
import pytest

from nexus.cli.commands.live import register_live_parser, serve_live
from nexus.cli.commands.run import register_run_parser
from nexus.runner.live.manager import LiveRunManager
from nexus.runner.live.models import CreateRunRequest, RunInfo, RuntimeInfo
from nexus.runner.live.server import (
    BASE_PATH_KEY,
    MANAGER_KEY,
    _proxy_runtime_http,
    _run_websocket_bridge,
    create_app,
    create_cache_run,
    get_run_config,
    list_runs,
    live_ui,
)


@pytest.mark.asyncio
async def test_live_ui_serves_frontend() -> None:
    app = create_app(base_path='/nexus/live')
    response = await live_ui(SimpleNamespace(app=app))

    assert response.content_type == 'text/html'
    assert response.headers['Cache-Control'] == 'no-store'
    assert 'Nexus Live Runner' in response.text
    assert 'RunInput YAML' in response.text
    assert 'Conversation' in response.text
    assert 'Events' in response.text
    assert 'metadataPanel' in response.text
    assert 'nexus_runtime_id' in response.text
    assert 'metadata-copy-button' in response.text
    assert 'copyMetadataValue' in response.text
    assert 'navigator.clipboard.writeText' in response.text
    assert 'fallbackCopyText' in response.text
    assert 'document.execCommand("copy")' in response.text
    assert 'submittedConfigPanel' in response.text
    assert 'Submitted Config' in response.text
    assert 'metadataCollapseButton' in response.text
    assert 'submittedConfigCollapseButton' in response.text
    assert 'metadataCollapsed' in response.text
    assert 'submittedConfigCollapsed' in response.text
    assert 'Runtime Debug' in response.text
    assert 'runtimeProxyAvailable' in response.text
    assert 'Runtime debug is unavailable for finished or historical runs.' in response.text
    assert 'statusFilterButtons' in response.text
    assert 'statusFilterStatuses' in response.text
    assert 'grid-template-rows: auto auto auto auto minmax(0, 1fr);' in response.text
    assert 'cacheKeyFilter' in response.text
    assert 'placeholder="run id / cache key"' in response.text
    assert 'params.set("q", state.cacheKeyFilter)' in response.text
    assert 'live:' in response.text
    assert 'key_paths:' in response.text
    assert 'agent_config.version' in response.text
    assert 'runtime_configs.main.image' in response.text
    assert 'openai_clients.main.config.request_args.model' in response.text
    assert 'window.__NEXUS_LIVE_BASE_PATH__ = "/nexus/live"' in response.text
    assert 'livePath("/v1/runs")' in response.text
    assert 'new URLSearchParams({ limit: "500", summary: "true" })' in response.text
    assert ':sessions' not in response.text
    assert 'sessionKey' not in response.text
    assert (
        'livePath(`/v1/runs/${encodeURIComponent(runId)}/runtimes/${encodeURIComponent(runtimeName)}/proxy`)'
        in response.text
    )
    assert '/debug/ports/8080/' not in response.text


def test_create_app_registers_frontend_routes() -> None:
    app = create_app()
    paths = {resource.canonical for resource in app.router.resources()}
    run_routes = next(resource for resource in app.router.resources() if resource.canonical == '/v1/runs')
    run_methods = {route.method for route in run_routes}

    expected_paths = {
        '/',
        '/ui',
        '/v1/runs',
        '/v1/runs/cache',
        '/v1/runs/yaml',
        '/v1/runs/{run_id}',
        '/v1/runs/{run_id}/config',
        '/v1/runs/{run_id}/kill',
        '/v1/runs/{run_id}/events',
        '/v1/runs/{run_id}/turns',
        '/v1/runs/{run_id}/runtimes',
        '/v1/runs/{run_id}/runtimes/{runtime_name}/proxy',
        '/v1/runs/{run_id}/runtimes/{runtime_name}/proxy/{runtime_path}',
    }
    assert expected_paths <= paths
    assert {'GET', 'POST'} <= run_methods
    assert '/debug/runs/{run_id}/ports/{port}' not in paths
    assert '/v1/runs/{run_id}/artifacts' not in paths
    assert '/v1/runs/{run_id}/files' not in paths


def test_create_app_registers_base_path_routes() -> None:
    app = create_app(base_path='/nexus/live/')
    paths = {resource.canonical for resource in app.router.resources()}

    expected_paths = {
        '/nexus/live',
        '/nexus/live/',
        '/nexus/live/ui',
        '/nexus/live/v1/runs',
        '/nexus/live/v1/runs/cache',
        '/nexus/live/v1/runs/{run_id}',
        '/nexus/live/v1/runs/{run_id}/config',
        '/nexus/live/v1/runs/{run_id}/events',
        '/nexus/live/v1/runs/{run_id}/runtimes/{runtime_name}/proxy',
        '/nexus/live/v1/runs/{run_id}/runtimes/{runtime_name}/proxy/{runtime_path}',
    }
    assert expected_paths <= paths


def test_live_cli_defaults_to_ipv4_and_ipv6_all_interfaces(monkeypatch) -> None:
    captured: dict = {}
    created: dict = {}

    def fake_create_app(**kwargs):
        created.update(kwargs)
        return object()

    def fake_run_app(app, **kwargs):
        captured['app'] = app
        captured.update(kwargs)

    monkeypatch.setattr('nexus.cli.commands.live.create_app', fake_create_app)
    monkeypatch.setattr('nexus.cli.commands.live.web.run_app', fake_run_app)
    monkeypatch.setattr(
        'nexus.cli.commands.live.force_flush_best_effort',
        lambda: (_ for _ in ()).throw(AssertionError('live shutdown must rely on the SDK atexit handler')),
        raising=False,
    )

    assert (
        serve_live(
            argparse.Namespace(
                host=None,
                port=8765,
                verbose=False,
                base_path='',
                database=None,
                terminal_retention_seconds=60.0,
            )
        )
        == 0
    )
    assert created == {'base_path': '', 'storage_path': None, 'terminal_run_retention_seconds': 60.0}
    assert captured['app'] is not None
    assert captured['host'] == ['0.0.0.0', '::']
    assert captured['port'] == 8765


def test_live_cli_accepts_repeated_host_overrides() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')
    register_live_parser(subparsers)

    args = parser.parse_args(['live-runner', '--host', '127.0.0.1', '--host', '::1', '--base-path', '/nexus/live'])

    assert args.host == ['127.0.0.1', '::1']
    assert args.base_path == '/nexus/live'


def test_live_cli_accepts_database_path() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')
    register_live_parser(subparsers)

    args = parser.parse_args(['live-runner', '--database', '/tmp/nexus-live.sqlite'])

    assert args.database == '/tmp/nexus-live.sqlite'


def test_live_cli_accepts_terminal_retention_override() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')
    register_live_parser(subparsers)

    args = parser.parse_args(['live-runner', '--terminal-retention-seconds', '0'])

    assert args.terminal_retention_seconds == 0


@pytest.mark.asyncio
async def test_list_runs_returns_persisted_history(tmp_path) -> None:
    manager = LiveRunManager(storage_path=tmp_path / 'live.sqlite')
    assert manager.store is not None
    await manager.store.create_run('run_history', CreateRunRequest(run_id='run_history', task_name='fake_task'))

    response = await list_runs(SimpleNamespace(app={MANAGER_KEY: manager, BASE_PATH_KEY: ''}, query={}))

    assert response.status == 200
    assert response.text is not None
    assert [item['run_id'] for item in json.loads(response.text)['runs']] == ['run_history']
    await manager.close()


@pytest.mark.asyncio
async def test_list_runs_defaults_to_full_response_and_summary_true_is_lightweight(tmp_path) -> None:
    manager = LiveRunManager(storage_path=tmp_path / 'live.sqlite')
    assert manager.store is not None
    await manager.store.create_run('run_full', CreateRunRequest(run_id='run_full', task_name='fake_task'))
    await manager.store.save_run_info(
        RunInfo(
            run_id='run_full',
            status='closed',
            events_cursor=7,
            cache_key='cache-key',
            runtime_base_url='http://runtime.example',
            runtimes={
                'main': RuntimeInfo(name='main', runtime_id='runtime-id', base_url='http://runtime.example'),
            },
            output={'task_output': {'archive_b64': 'x' * 1024}},
        )
    )

    full_response = await list_runs(SimpleNamespace(app={MANAGER_KEY: manager, BASE_PATH_KEY: ''}, query={}))
    summary_response = await list_runs(
        SimpleNamespace(app={MANAGER_KEY: manager, BASE_PATH_KEY: ''}, query={'summary': 'true'})
    )

    assert full_response.status == 200
    full_run = json.loads(full_response.text)['runs'][0]
    assert full_run['run_id'] == 'run_full'
    assert full_run['runtimes']['main']['runtime_id'] == 'runtime-id'
    assert full_run['output']['task_output']['archive_b64'] == 'x' * 1024

    assert summary_response.status == 200
    summary_run = json.loads(summary_response.text)['runs'][0]
    assert summary_run == {
        'run_id': 'run_full',
        'status': 'closed',
        'events_cursor': 7,
        'cache_key': 'cache-key',
        'current_turn_id': None,
        'runtime_base_url': 'http://runtime.example',
        'error': None,
    }
    assert 'runtimes' not in summary_run
    assert 'output' not in summary_run
    await manager.close()


@pytest.mark.asyncio
async def test_live_ui_requests_summary_run_list() -> None:
    app = create_app()
    response = await live_ui(SimpleNamespace(app=app))

    assert 'new URLSearchParams({ limit: "500", summary: "true" })' in response.text


@pytest.mark.asyncio
async def test_list_runs_filters_persisted_history_by_status(tmp_path) -> None:
    manager = LiveRunManager(storage_path=tmp_path / 'live.sqlite')
    assert manager.store is not None
    for run_id, status in (('run_idle', 'idle'), ('run_failed', 'failed'), ('run_killed', 'killed')):
        await manager.store.create_run(run_id, CreateRunRequest(run_id=run_id, task_name='fake_task'))
        await manager.store.save_run_info(RunInfo(run_id=run_id, status=status, events_cursor=0))

    response = await list_runs(
        SimpleNamespace(app={MANAGER_KEY: manager, BASE_PATH_KEY: ''}, query={'status': 'failed,killed'})
    )

    assert response.status == 200
    assert [item['run_id'] for item in json.loads(response.text)['runs']] == ['run_killed', 'run_failed']
    await manager.close()


@pytest.mark.asyncio
async def test_sqlite_store_persists_cache_key_column_and_uses_list_indexes(tmp_path) -> None:
    db_path = tmp_path / 'live.sqlite'
    manager = LiveRunManager(storage_path=db_path)
    assert manager.store is not None
    live = {'cache': {'namespace': 'unit-test', 'key_paths': ['task_name', 'task_config.user_prompt']}}
    request = CreateRunRequest(
        run_id='run_indexed', task_name='fake_task', task_config={'user_prompt': 'same'}, live=live
    )

    await manager.store.create_run('run_indexed', request)
    cache_key = (await manager.get_run_info('run_indexed')).cache_key
    await manager.close()

    assert cache_key
    with sqlite3.connect(db_path) as conn:
        columns = {row[1] for row in conn.execute('PRAGMA table_info(live_runs)')}
        indexes = {row[1] for row in conn.execute('PRAGMA index_list(live_runs)')}
        stored_cache_key = conn.execute(
            'SELECT cache_key FROM live_runs WHERE run_id = ?', ('run_indexed',)
        ).fetchone()[0]
        status_plan = [
            row[-1]
            for row in conn.execute(
                """
                EXPLAIN QUERY PLAN
                SELECT run_id, status, events_cursor, cache_key, current_turn_id, runtime_base_url, error
                FROM live_runs
                WHERE status IN ('starting', 'idle')
                ORDER BY updated_at DESC, created_at DESC
                LIMIT 100
                """
            )
        ]
        cache_plan = [
            row[-1]
            for row in conn.execute(
                """
                EXPLAIN QUERY PLAN
                SELECT run_id, status, events_cursor, cache_key, current_turn_id, runtime_base_url, error
                FROM live_runs
                WHERE cache_key = ?
                ORDER BY updated_at DESC, created_at DESC
                LIMIT 100
                """,
                (cache_key,),
            )
        ]

    assert 'cache_key' in columns
    assert 'idx_live_runs_updated_created' in indexes
    assert 'idx_live_runs_status_updated_created' in indexes
    assert 'idx_live_runs_cache_key_updated_created' in indexes
    assert 'idx_live_runs_closed_at' in indexes
    assert stored_cache_key == cache_key
    assert any('idx_live_runs_status_updated_created' in item for item in status_plan)
    assert any('idx_live_runs_cache_key_updated_created' in item for item in cache_plan)


@pytest.mark.asyncio
async def test_sqlite_store_drops_live_event_projection_table(tmp_path) -> None:
    db_path = tmp_path / 'live.sqlite'
    manager = LiveRunManager(storage_path=db_path)
    await manager.start()
    await manager.close()

    with sqlite3.connect(db_path) as connection:
        table = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'live_events'"
        ).fetchone()
        columns = {row[1] for row in connection.execute('PRAGMA table_info(live_runs)').fetchall()}

    assert table is None
    assert 'trace_metadata_json' in columns


@pytest.mark.asyncio
async def test_sqlite_store_migrates_existing_run_table_with_cache_key_backfill(tmp_path) -> None:
    db_path = tmp_path / 'old_live.sqlite'
    live = {'cache': {'namespace': 'unit-test', 'key_paths': ['task_name', 'task_config.user_prompt']}}
    request = CreateRunRequest(
        run_id='run_old_schema', task_name='fake_task', task_config={'user_prompt': 'same'}, live=live
    )
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE live_runs (
                run_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                request_json TEXT NOT NULL,
                current_turn_id TEXT,
                runtime_base_url TEXT,
                runtimes_json TEXT NOT NULL DEFAULT '{}',
                error TEXT,
                output_json TEXT,
                events_cursor INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                closed_at TEXT
            );
            CREATE TABLE live_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                event_seq INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                event_json TEXT NOT NULL,
                time TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(run_id, event_seq),
                FOREIGN KEY (run_id) REFERENCES live_runs(run_id) ON DELETE CASCADE
            );
            """
        )
        conn.execute(
            """
            INSERT INTO live_runs (
                run_id, status, request_json, runtimes_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                'run_old_schema',
                'closed',
                request.model_dump_json(),
                '{}',
                '2026-01-01T00:00:00+00:00',
                '2026-01-01T00:00:00+00:00',
            ),
        )

    manager = LiveRunManager(storage_path=db_path)
    await manager.start()
    info = await manager.get_run_info('run_old_schema')
    await manager.close()

    with sqlite3.connect(db_path) as conn:
        columns = {row[1] for row in conn.execute('PRAGMA table_info(live_runs)')}
        indexes = {row[1] for row in conn.execute('PRAGMA index_list(live_runs)')}
        stored_cache_key = conn.execute(
            'SELECT cache_key FROM live_runs WHERE run_id = ?', ('run_old_schema',)
        ).fetchone()[0]

    assert 'cache_key' in columns
    assert 'idx_live_runs_cache_key_updated_created' in indexes
    assert info.cache_key == stored_cache_key
    assert stored_cache_key


@pytest.mark.asyncio
async def test_list_runs_filters_by_cache_key(tmp_path) -> None:
    manager = LiveRunManager(storage_path=tmp_path / 'live.sqlite')
    assert manager.store is not None
    live = {'cache': {'namespace': 'unit-test', 'key_paths': ['task_name', 'task_config.user_prompt']}}
    matching = CreateRunRequest(
        run_id='run_matching', task_name='fake_task', task_config={'user_prompt': 'same'}, live=live
    )
    other = CreateRunRequest(run_id='run_other', task_name='fake_task', task_config={'user_prompt': 'other'}, live=live)
    await manager.store.create_run('run_matching', matching)
    await manager.store.create_run('run_other', other)
    cache_key = (await manager.get_run_info('run_matching')).cache_key

    response = await list_runs(
        SimpleNamespace(app={MANAGER_KEY: manager, BASE_PATH_KEY: ''}, query={'cache_key': cache_key})
    )

    assert response.status == 200
    payload = json.loads(response.text)
    assert [item['run_id'] for item in payload['runs']] == ['run_matching']
    assert payload['runs'][0]['cache_key'] == cache_key
    await manager.close()


@pytest.mark.asyncio
async def test_list_runs_keeps_history_when_cache_key_cannot_be_parsed(tmp_path) -> None:
    manager = LiveRunManager(storage_path=tmp_path / 'live.sqlite')
    assert manager.store is not None
    await manager.store.create_run(
        'run_bad_cache_key',
        CreateRunRequest(
            run_id='run_bad_cache_key',
            task_name='fake_task',
            live={'cache': {'namespace': 'unit-test', 'key_paths': ['task_config.missing']}},
        ),
    )

    response = await list_runs(SimpleNamespace(app={MANAGER_KEY: manager, BASE_PATH_KEY: ''}, query={}))

    assert response.status == 200
    payload = json.loads(response.text)
    assert payload['runs'][0]['run_id'] == 'run_bad_cache_key'
    assert payload['runs'][0]['cache_key'] is None
    await manager.close()


@pytest.mark.asyncio
async def test_list_runs_searches_by_run_id(tmp_path) -> None:
    manager = LiveRunManager(storage_path=tmp_path / 'live.sqlite')
    assert manager.store is not None
    await manager.store.create_run(
        'run_f80e79759ce1', CreateRunRequest(run_id='run_f80e79759ce1', task_name='fake_task')
    )
    await manager.store.create_run('run_other', CreateRunRequest(run_id='run_other', task_name='fake_task'))

    response = await list_runs(SimpleNamespace(app={MANAGER_KEY: manager, BASE_PATH_KEY: ''}, query={'q': 'f80e'}))

    assert response.status == 200
    payload = json.loads(response.text)
    assert [item['run_id'] for item in payload['runs']] == ['run_f80e79759ce1']
    await manager.close()


@pytest.mark.asyncio
async def test_list_runs_rejects_unknown_status_filter() -> None:
    manager = LiveRunManager()

    with pytest.raises(Exception, match='invalid run status filter'):
        await list_runs(SimpleNamespace(app={MANAGER_KEY: manager, BASE_PATH_KEY: ''}, query={'status': 'missing'}))


@pytest.mark.asyncio
async def test_create_cache_run_handler_returns_created_response() -> None:
    class FakeManager:
        async def create_cache_run(self, request: CreateRunRequest):
            assert request.task_name == 'fake_task'
            return {
                'run_id': 'run_cache',
                'status': 'starting',
                'events_cursor': 1,
                'reused_cache': False,
                'cache_key': 'cache-key',
            }

    async def read_json():
        return {'task_name': 'fake_task'}

    response = await create_cache_run(
        SimpleNamespace(
            app={MANAGER_KEY: FakeManager(), BASE_PATH_KEY: ''},
            json=read_json,
        )
    )

    assert response.status == 201
    assert json.loads(response.text) == {
        'run_id': 'run_cache',
        'status': 'starting',
        'events_cursor': 1,
        'reused_cache': False,
        'cache_key': 'cache-key',
    }


@pytest.mark.asyncio
async def test_get_run_config_returns_redacted_submitted_config() -> None:
    manager = LiveRunManager()
    await manager.create_run(
        CreateRunRequest(
            run_id='run_config',
            task_name='fake_task',
            runtime_provider_config={'token': 'gongfeng-secret'},
            openai_clients={
                'main': {
                    'name': 'litellm',
                    'config': {
                        'client_args': {
                            'api_key': 'app-id:app-key?provider=aws_third',
                            'base_url': 'http://model-service.test',
                        }
                    },
                }
            },
        )
    )

    response = await get_run_config(
        SimpleNamespace(app={MANAGER_KEY: manager, BASE_PATH_KEY: ''}, match_info={'run_id': 'run_config'})
    )

    assert response.status == 200
    payload = json.loads(response.text)
    assert payload['run_id'] == 'run_config'
    assert payload['request']['runtime_provider_config']['token'] == '<redacted>'
    assert payload['request']['openai_clients']['main']['config']['client_args']['api_key'] == '<redacted>'
    assert (
        payload['request']['openai_clients']['main']['config']['client_args']['base_url'] == 'http://model-service.test'
    )
    await manager.close()


@pytest.mark.asyncio
async def test_get_run_config_reads_persisted_request(tmp_path) -> None:
    manager = LiveRunManager(storage_path=tmp_path / 'live.sqlite')
    await manager.store.create_run(
        'run_history_config',
        CreateRunRequest(
            run_id='run_history_config',
            task_name='fake_task',
            agent_config={'user_prompt': 'hello'},
            runtime_provider_config={'token': 'stored-secret'},
        ),
    )

    response = await get_run_config(
        SimpleNamespace(app={MANAGER_KEY: manager, BASE_PATH_KEY: ''}, match_info={'run_id': 'run_history_config'})
    )

    payload = json.loads(response.text)
    assert payload['request']['agent_config']['user_prompt'] == 'hello'
    assert payload['request']['runtime_provider_config']['token'] == '<redacted>'
    await manager.close()


@pytest.mark.asyncio
async def test_sqlite_store_marks_cached_runs_failed_after_restart(tmp_path) -> None:
    from nexus.runner.live.models import RunInfo

    db_path = tmp_path / 'live.sqlite'
    manager = LiveRunManager(storage_path=db_path)
    assert manager.store is not None
    await manager.store.create_run('run_cached', CreateRunRequest(run_id='run_cached', task_name='fake_task'))
    await manager.store.save_run_info(RunInfo(run_id='run_cached', status='cached', events_cursor=0))
    await manager.close()

    restored = LiveRunManager(storage_path=db_path)
    info = await restored.get_run_info('run_cached')

    assert info.status == 'failed'
    assert info.error == 'Live server restarted before this run completed'
    await restored.close()


def test_grouped_run_live_cli_accepts_repeated_host_overrides() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')
    register_run_parser(subparsers)

    args = parser.parse_args(['run', 'live', '--host', '127.0.0.1', '--host', '::1', '--base-path', '/nexus/live'])

    assert args.command == 'run'
    assert args.run_command == 'live'
    assert args.host == ['127.0.0.1', '::1']
    assert args.base_path == '/nexus/live'

    args = parser.parse_args(['run', 'live', '--database', '/tmp/nexus-live.sqlite'])
    assert args.database == '/tmp/nexus-live.sqlite'


@pytest.mark.asyncio
async def test_runtime_debug_proxy_keeps_encoded_response_bytes(monkeypatch) -> None:
    payload = b'gzip payload bytes'
    captured = {}

    class FakeRequest:
        method = 'GET'
        headers = {'Accept-Encoding': 'gzip'}

        async def read(self) -> bytes:
            return b''

    class FakeUpstream:
        status = 200
        headers = {
            'Content-Encoding': 'gzip',
            'Content-Length': '999',
            'Transfer-Encoding': 'chunked',
        }

        async def read(self) -> bytes:
            return payload

    class FakeRequestContext:
        async def __aenter__(self) -> FakeUpstream:
            return FakeUpstream()

        async def __aexit__(self, exc_type, exc, traceback) -> None:
            return None

    class FakeSession:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def __aenter__(self) -> 'FakeSession':
            return self

        async def __aexit__(self, exc_type, exc, traceback) -> None:
            return None

        def request(self, *args, **kwargs) -> FakeRequestContext:
            captured['request_headers'] = kwargs.get('headers')
            return FakeRequestContext()

    monkeypatch.setattr('nexus.runner.live.server.aiohttp.ClientSession', FakeSession)

    response = await _proxy_runtime_http(FakeRequest(), 'http://runtime/debug/ports/8080/')

    assert captured['auto_decompress'] is False
    assert captured['request_headers'] == {'Accept-Encoding': 'gzip'}
    assert response.body == payload
    assert response.headers['Content-Encoding'] == 'gzip'
    assert 'Content-Length' not in response.headers
    assert 'Transfer-Encoding' not in response.headers


@pytest.mark.asyncio
async def test_runtime_debug_proxy_rewrites_root_relative_location(monkeypatch) -> None:
    captured = {}

    class FakeRequest:
        method = 'GET'
        headers = {}
        app = create_app(base_path='/nexus/live')
        match_info = {'run_id': 'run_123', 'runtime_name': 'main'}

        async def read(self) -> bytes:
            return b''

    class FakeUpstream:
        status = 302
        headers = {'Location': '/debug/ports/8080/'}

        async def read(self) -> bytes:
            return b''

    class FakeRequestContext:
        async def __aenter__(self) -> FakeUpstream:
            return FakeUpstream()

        async def __aexit__(self, exc_type, exc, traceback) -> None:
            return None

    class FakeSession:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def __aenter__(self) -> 'FakeSession':
            return self

        async def __aexit__(self, exc_type, exc, traceback) -> None:
            return None

        def request(self, *args, **kwargs) -> FakeRequestContext:
            return FakeRequestContext()

    monkeypatch.setattr('nexus.runner.live.server.aiohttp.ClientSession', FakeSession)

    response = await _proxy_runtime_http(FakeRequest(), 'http://runtime/debug/ports/8080')

    assert captured['auto_decompress'] is False
    assert response.status == 302
    assert response.headers['Location'] == '/nexus/live/v1/runs/run_123/runtimes/main/proxy/debug/ports/8080/'


@pytest.mark.asyncio
async def test_websocket_bridge_treats_closing_transport_as_normal() -> None:
    cancelled = False

    async def closing_transport() -> None:
        raise aiohttp.ClientConnectionResetError('Cannot write to closing transport')

    async def pending_side() -> None:
        nonlocal cancelled
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled = True
            raise

    await _run_websocket_bridge(closing_transport(), pending_side())

    assert cancelled
