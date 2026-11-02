from __future__ import annotations

import asyncio
import os

from datetime import timedelta
from urllib.parse import urlsplit

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from nexus.distributed.config import DistributedConfig
from nexus.distributed.instance_manager.internal import create_instance_manager_internal_app
from nexus.distributed.instance_manager.service import InstanceManagerService
from nexus.distributed.persistence.orm import create_schema
from nexus.runner.trace import RunTraceStreamEvent


async def test_manager_internal_api_requires_bearer_token():
    engine = create_async_engine('sqlite+aiosqlite:///:memory:')
    await create_schema(engine)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    config = DistributedConfig(secrets={'internal_token': 'secret'}, manager_healthy_ttl=timedelta(seconds=15))
    service = InstanceManagerService(
        session_factory=session_factory,
        config=config,
        manager_id='im-1',
        endpoint_url='http://127.0.0.1:9001',
        capacity_total=1,
        driver_factories={},
    )
    app = create_instance_manager_internal_app(service=service, config=config)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.get('/v1/internal/health')
        assert response.status == 401

        response = await client.get('/v1/internal/health', headers={'Authorization': 'Bearer wrong'})
        assert response.status == 401

        response = await client.get('/v1/internal/health', headers={'Authorization': 'Bearer secret'})
        assert response.status == 200
        payload = await response.json()
        assert payload['manager_id'] == 'im-1'

        response = await client.get('/internal/health', headers={'Authorization': 'Bearer secret'})
        assert response.status == 404
    finally:
        await client.close()
        await engine.dispose()


class FakeExecutor:
    def __init__(self, *, runtime_base_url: str | None = None, event_batches=None) -> None:
        self.runtime_base_url = runtime_base_url
        self.event_batches = list(event_batches or [])
        self.calls: list[tuple[str, str]] = []

    async def get_runtime_base_url(self, assignment_id, runtime_name):
        self.calls.append((assignment_id, 'get_runtime_base_url'))
        return self.runtime_base_url

    async def collect_events_after(self, assignment_id, cursor, *, timeout):
        self.calls.append((assignment_id, 'collect_events_after'))
        if self.event_batches:
            item = self.event_batches.pop(0)
            if isinstance(item, BaseException):
                raise item
            return item
        await asyncio.sleep(0.01)
        return []

    async def get_assignment_info(self, assignment_id):
        self.calls.append((assignment_id, 'get_assignment_info'))
        return {
            'assignment_id': assignment_id,
            'execution_id': 'live-1',
            'attempt_id': 'live-1-attempt-1',
            'driver': {'spec_type': 'nexus_live', 'instance_mode': 'asyncio'},
            'runtime_info': {'runtimes': {'main': {'base_url': 'http://runtime'}}},
            'live': {'status': 'idle', 'events_cursor': 7, 'current_turn_id': 'turn-1'},
        }


class FakeService:
    manager_id = 'fake-im'
    session_id = 'fake-session'

    def __init__(self, executor: FakeExecutor | None = None) -> None:
        self.executor = executor or FakeExecutor()
        self.polled = 0
        self.claimed = 0
        self.draining = False
        self.draining_updates: list[bool] = []
        self.artifact_result: dict | None = None
        self.artifact_events: list[RunTraceStreamEvent] = []

    async def poll_assignments_once(self):
        self.polled += 1
        return 0

    async def claim_commands_once(self):
        self.claimed += 1
        return 0

    def request_wakeup(self):
        self.polled += 1

    async def set_draining(self, draining: bool) -> bool:
        self.draining = draining
        self.draining_updates.append(draining)
        return True

    async def read_artifact_result(self, assignment_id):
        assert assignment_id == 'asn-1'
        if self.artifact_result is None:
            raise FileNotFoundError('missing result')
        return self.artifact_result

    async def read_live_trace_events_after(self, assignment_id, cursor):
        assert assignment_id == 'asn-1'
        return [event for event in self.artifact_events if event.id > cursor]

    async def get_assignment_info(self, assignment_id):
        return await self.executor.get_assignment_info(assignment_id)


async def test_manager_internal_wakeup_sets_signal_without_running_poll_and_claim():
    service = FakeService()
    app = create_instance_manager_internal_app(service=service, config=DistributedConfig(secrets={'internal_token': 'secret'}))
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.post('/v1/internal/wakeup', headers={'Authorization': 'Bearer secret'})
        assert response.status == 200
        assert service.polled == 1
        assert service.claimed == 0
    finally:
        await client.close()


async def test_manager_internal_api_sets_session_draining_state():
    service = FakeService()
    app = create_instance_manager_internal_app(service=service, config=DistributedConfig(secrets={'internal_token': 'secret'}))
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.patch(
            '/v1/internal/draining',
            headers={'Authorization': 'Bearer secret'},
            json={'draining': True},
        )

        assert response.status == 200
        assert await response.json() == {
            'manager_id': service.manager_id,
            'session_id': service.session_id,
            'draining': True,
        }
        assert service.draining_updates == [True]
    finally:
        await client.close()


async def test_manager_internal_api_rejects_non_boolean_draining_state():
    service = FakeService()
    app = create_instance_manager_internal_app(service=service, config=DistributedConfig(secrets={'internal_token': 'secret'}))
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.patch(
            '/v1/internal/draining',
            headers={'Authorization': 'Bearer secret'},
            json={'draining': 'true'},
        )

        assert response.status == 400
        assert service.draining_updates == []
    finally:
        await client.close()


async def test_manager_internal_events_streams_keepalive_and_driver_events():
    service = FakeService(FakeExecutor(event_batches=[[], [(1, b'event: test.event\ndata: {"ok": true}\n\n')]]))
    app = create_instance_manager_internal_app(service=service, config=DistributedConfig())
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.get('/v1/internal/executions/asn-1/events?cursor=0')
        assert response.status == 200
        keepalive = await asyncio.wait_for(response.content.readline(), timeout=2)
        assert keepalive == b': keep-alive\n'
        await asyncio.wait_for(response.content.readline(), timeout=2)
        event_line = await asyncio.wait_for(response.content.readline(), timeout=2)
        assert event_line == b'event: test.event\n'
        data_line = await asyncio.wait_for(response.content.readline(), timeout=2)
        assert data_line == b'data: {"ok": true}\n'
        response.close()
    finally:
        await client.close()


async def test_manager_internal_events_keepalive_on_collect_timeout(caplog):
    executor = FakeExecutor(
        event_batches=[
            TimeoutError('executor loop busy'),
            [(9, b'event: test.event\ndata: {"ok": true}\n\n')],
        ],
    )
    service = FakeService(executor)
    app = create_instance_manager_internal_app(service=service, config=DistributedConfig())
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        with caplog.at_level('ERROR', logger='nexus.distributed.instance_manager.internal'):
            response = await client.get('/v1/internal/executions/asn-1/events?cursor=5')
            assert response.status == 200
            # The collect TimeoutError degrades to a keep-alive instead of a 500 abort.
            keepalive = await asyncio.wait_for(response.content.readline(), timeout=2)
            assert keepalive == b': keep-alive\n'
            assert await asyncio.wait_for(response.content.readline(), timeout=2) == b'\n'
            # The same cursor is retried and the next batch streams normally.
            event_line = await asyncio.wait_for(response.content.readline(), timeout=2)
            assert event_line == b'event: test.event\n'
            data_line = await asyncio.wait_for(response.content.readline(), timeout=2)
            assert data_line == b'data: {"ok": true}\n'
            response.close()

        assert not any(
            'Unhandled instance manager internal API error' in record.getMessage() for record in caplog.records
        )
    finally:
        await client.close()


async def test_manager_internal_serves_assignment_info():
    service = FakeService(FakeExecutor())
    app = create_instance_manager_internal_app(service=service, config=DistributedConfig())
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.get('/v1/internal/executions/asn-1/info')
        assert response.status == 200
        assert await response.json() == {
            'assignment_id': 'asn-1',
            'execution_id': 'live-1',
            'attempt_id': 'live-1-attempt-1',
            'driver': {'spec_type': 'nexus_live', 'instance_mode': 'asyncio'},
            'runtime_info': {'runtimes': {'main': {'base_url': 'http://runtime'}}},
            'live': {'status': 'idle', 'events_cursor': 7, 'current_turn_id': 'turn-1'},
        }
        assert service.executor.calls == [('asn-1', 'get_assignment_info')]
    finally:
        await client.close()


async def test_manager_internal_serves_live_artifact_result_and_events():
    service = FakeService()
    service.artifact_result = {'success': True, 'large': 'payload'}
    service.artifact_events = [
        RunTraceStreamEvent(
            id=4,
            type='runtime.ready',
            time_unix_nano=1_783_468_800_000_000_000,
            data={'ok': True},
        ),
        RunTraceStreamEvent(
            id=5,
            type='run.output',
            time_unix_nano=1_783_468_800_000_000_000,
            data={'success': True},
        ),
    ]
    app = create_instance_manager_internal_app(service=service, config=DistributedConfig())
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.get('/v1/internal/executions/asn-1/artifacts/result')
        assert response.status == 200
        assert await response.json() == {'success': True, 'large': 'payload'}

        response = await client.get('/v1/internal/executions/asn-1/artifacts/events?cursor=4')
        assert response.status == 200
        assert await response.text() == (
            'id: 5\n'
            'event: run.output\n'
            'data: {"id": 5, "type": "run.output", "time": "2026-07-08T00:00:00+00:00", '
            '"data": {"success": true}}\n\n'
        )

        response = await client.get(
            '/v1/internal/executions/asn-1/artifacts/events?cursor=0',
            headers={'Last-Event-ID': '4'},
        )
        assert response.status == 200
        assert await response.text() == (
            'id: 5\n'
            'event: run.output\n'
            'data: {"id": 5, "type": "run.output", "time": "2026-07-08T00:00:00+00:00", '
            '"data": {"success": true}}\n\n'
        )
    finally:
        await client.close()


async def test_manager_internal_runtime_proxy_success_and_unavailable():
    async def ping(request):
        return web.json_response(
            {
                'path': request.path,
                'query': dict(request.query),
                'body': await request.text(),
                'custom': request.headers.get('X-Custom'),
            }
        )

    runtime_app = web.Application()
    runtime_app.router.add_post('/ping', ping)
    runtime_client = TestClient(TestServer(runtime_app))
    await runtime_client.start_server()
    try:
        service = FakeService(FakeExecutor(runtime_base_url=str(runtime_client.make_url('/')).rstrip('/')))
        app = create_instance_manager_internal_app(service=service, config=DistributedConfig())
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            response = await client.post(
                '/v1/internal/executions/asn-1/runtimes/main/proxy/ping?x=1',
                data='payload',
                headers={'X-Custom': 'kept'},
            )
            assert response.status == 200
            payload = await response.json()
            assert payload == {'path': '/ping', 'query': {'x': '1'}, 'body': 'payload', 'custom': 'kept'}
        finally:
            await client.close()
    finally:
        await runtime_client.close()

    service = FakeService(FakeExecutor(runtime_base_url='http://127.0.0.1:9'))
    app = create_instance_manager_internal_app(service=service, config=DistributedConfig())
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.get('/v1/internal/executions/asn-1/runtimes/main/proxy/ping')
        assert response.status == 502
        assert (await response.json())['error'] == 'runtime_proxy_unavailable'
    finally:
        await client.close()


async def test_manager_internal_runtime_proxy_streams_without_buffering_full_response():
    async def slow_stream(request):
        response = web.StreamResponse(status=200, headers={'Content-Type': 'text/plain'})
        await response.prepare(request)
        await response.write(b'first\n')
        await asyncio.sleep(1.0)
        await response.write(b'second\n')
        await response.write_eof()
        return response

    runtime_app = web.Application()
    runtime_app.router.add_get('/slow', slow_stream)
    runtime_client = TestClient(TestServer(runtime_app))
    await runtime_client.start_server()
    try:
        service = FakeService(FakeExecutor(runtime_base_url=str(runtime_client.make_url('/')).rstrip('/')))
        app = create_instance_manager_internal_app(service=service, config=DistributedConfig())
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            response = await asyncio.wait_for(
                client.get('/v1/internal/executions/asn-1/runtimes/main/proxy/slow'),
                timeout=0.2,
            )
            assert response.status == 200
            first = await asyncio.wait_for(response.content.readline(), timeout=0.2)
            assert first == b'first\n'
            response.close()
        finally:
            await client.close()
    finally:
        await runtime_client.close()


async def test_manager_internal_proxies_aiomonitor_webui_loop():
    async def aiomonitor_asset(request):
        return web.json_response(
            {
                'path': request.path,
                'query': dict(request.query),
                'custom': request.headers.get('X-Custom'),
            }
        )

    aiomonitor_app = web.Application()
    aiomonitor_app.router.add_get('/static/app.js', aiomonitor_asset)
    aiomonitor_client = TestClient(TestServer(aiomonitor_app))
    await aiomonitor_client.start_server()
    try:
        webui_port = urlsplit(str(aiomonitor_client.make_url('/'))).port
        assert webui_port is not None
        service = FakeService()
        app = create_instance_manager_internal_app(
            service=service,
            config=DistributedConfig(
                secrets={'internal_token': 'secret'},
                aiomonitor_webui_base_port=webui_port,
            ),
        )
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            response = await client.get(
                '/v1/internal/debug/aiomonitor/main/static/app.js?theme=dark',
                headers={'Authorization': 'Bearer secret', 'X-Custom': 'kept'},
            )
            assert response.status == 200
            assert await response.json() == {
                'path': '/static/app.js',
                'query': {'theme': 'dark'},
                'custom': 'kept',
            }

            response = await client.get(
                '/v1/internal/debug/aiomonitor/unknown/',
                headers={'Authorization': 'Bearer secret'},
            )
            assert response.status == 404
            assert (await response.json())['error'] == 'aiomonitor_loop_not_found'
        finally:
            await client.close()
    finally:
        await aiomonitor_client.close()


async def test_manager_internal_py_spy_dump_profiles_manager_process(monkeypatch):
    calls: list[tuple[str, ...]] = []

    class FakeProcess:
        returncode = 0

        async def communicate(self):
            return b'Process 123: python\nThread 1\n', b''

    async def create_subprocess_exec(*args, **kwargs):
        calls.append(args)
        assert kwargs['stdout'] is asyncio.subprocess.PIPE
        assert kwargs['stderr'] is asyncio.subprocess.PIPE
        return FakeProcess()

    monkeypatch.setattr(asyncio, 'create_subprocess_exec', create_subprocess_exec)
    app = create_instance_manager_internal_app(service=FakeService(), config=DistributedConfig())
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.get('/v1/internal/debug/py-spy/dump')

        assert response.status == 200
        assert response.content_type == 'text/plain'
        assert await response.text() == 'Process 123: python\nThread 1\n'
        assert calls == [('py-spy', 'dump', '--pid', str(os.getpid()))]
    finally:
        await client.close()


async def test_manager_internal_py_spy_record_returns_svg_file(monkeypatch):
    calls: list[tuple[str, ...]] = []

    class FakeProcess:
        returncode = 0

        async def communicate(self):
            output_path = calls[-1][calls[-1].index('--output') + 1]
            with open(output_path, 'w', encoding='utf-8') as output:
                output.write('<svg>profile</svg>')
            return b'', b''

    async def create_subprocess_exec(*args, **kwargs):
        calls.append(args)
        return FakeProcess()

    monkeypatch.setattr(asyncio, 'create_subprocess_exec', create_subprocess_exec)
    app = create_instance_manager_internal_app(service=FakeService(), config=DistributedConfig())
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.get('/v1/internal/debug/py-spy/record?duration=3&format=svg')

        assert response.status == 200
        assert response.content_type == 'image/svg+xml'
        assert await response.text() == '<svg>profile</svg>'
        assert calls
        assert calls[0][:8] == (
            'py-spy',
            'record',
            '--pid',
            str(os.getpid()),
            '--duration',
            '3',
            '--format',
            'flamegraph',
        )
        assert '--output' in calls[0]
    finally:
        await client.close()


async def test_manager_internal_runtime_proxy_forwards_aiomonitor_debug_base_path():
    async def runtime_aiomonitor_asset(request):
        return web.json_response(
            {
                'debug_base_path': request.headers.get('X-Nexus-Debug-Base-Path'),
                'path': request.path,
            }
        )

    runtime_app = web.Application()
    runtime_app.router.add_get('/debug/aiomonitor/main/static/app.js', runtime_aiomonitor_asset)
    runtime_client = TestClient(TestServer(runtime_app))
    await runtime_client.start_server()
    try:
        service = FakeService(FakeExecutor(runtime_base_url=str(runtime_client.make_url('/')).rstrip('/')))
        app = create_instance_manager_internal_app(service=service, config=DistributedConfig(secrets={'internal_token': 'secret'}))
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            response = await client.get(
                '/v1/internal/executions/asn-1/runtimes/main/proxy/debug/aiomonitor/main/static/app.js',
                headers={
                    'Authorization': 'Bearer secret',
                    'X-Nexus-Debug-Base-Path': '/v1/executions/live-1/runtimes/main/proxy/debug/aiomonitor/main',
                },
            )

            assert response.status == 200
            assert await response.json() == {
                'debug_base_path': '/v1/executions/live-1/runtimes/main/proxy/debug/aiomonitor/main',
                'path': '/debug/aiomonitor/main/static/app.js',
            }
        finally:
            await client.close()
    finally:
        await runtime_client.close()
