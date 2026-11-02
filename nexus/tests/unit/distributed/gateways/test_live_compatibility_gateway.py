from __future__ import annotations

import json

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from aiohttp.web_app import Application as AiohttpApplication

from nexus.distributed.execution.models import (
    ExecutionMode,
    ExecutionStatus,
    RouteStatus,
)
from nexus.distributed.gateways.application.routes.live_compatibility import create_live_compatibility_app


def test_nested_compatibility_app_reuses_parent_http_instrumentation(monkeypatch):
    class InstrumentedApplication(AiohttpApplication):
        pass

    monkeypatch.setattr(web, 'Application', InstrumentedApplication)

    standalone = create_live_compatibility_app(
        control_base_url='http://control.invalid',
        application_base_url='http://application.invalid',
        default_placement={'environment': 'predefined-local', 'pool': 'interactive'},
    )
    nested = create_live_compatibility_app(
        control_base_url='http://control.invalid',
        application_base_url='http://application.invalid',
        default_placement={'environment': 'predefined-local', 'pool': 'interactive'},
        instrument_http_server=False,
    )

    assert type(standalone) is InstrumentedApplication
    assert type(nested) is AiohttpApplication


def _execution_payload(execution_id: str, *, route_status: str = RouteStatus.ACTIVE) -> dict:
    return {
        'execution_id': execution_id,
        'mode': ExecutionMode.LIVE,
        'spec_type': 'nexus_live',
        'instance_mode': 'asyncio',
        'spec': {
            'run_input': {
                'imports': ['nexus.extensions'],
                'runtime_provider_name': 'distributed_fake_runtime',
                'runtime_provider_config': {'base_url': 'http://runtime'},
                'runtime_configs': {'main': {'image': 'fake'}},
                'openai_clients': {},
                'tool_configs': {},
                'task_name': 'identity',
                'task_config': {},
                'agent_name': 'distributed_interactive_agent',
                'agent_config': {'prompt': 'choose'},
                'processors': {},
                'runner': {},
            },
            'live': {'working_dir': '/workspace'},
        },
        'driver_config': {},
        'placement': {},
        'route_status': route_status,
        'status': ExecutionStatus.WAITING_INPUT,
        'current_assignment': {
            'runtime_info': {
                'runtimes': {
                    'main': {
                        'name': 'main',
                        'runtime_id': 'rt-main',
                        'base_url': 'http://runtime',
                        'proxy_url': '/v1/executions/live-1/runtimes/main/proxy',
                        'debug_gui_url': '/v1/executions/live-1/runtimes/main/proxy/debug/gui',
                    }
                },
                'input': {'turn_id': 'turn-1', 'request': {'hint': 'choose'}},
                '_artifacts': {
                    'live': {
                        'version': 4,
                        'run_id': execution_id,
                        'run_span_id': 'run-span-1',
                        'run_kind': 'live',
                        'sinks': [
                            {
                                'name': 'local',
                                'metadata': {
                                    'version': 2,
                                    'format': 'span_event_record.jsonl',
                                    'run_kind': 'live',
                                    'run_id': execution_id,
                                    'run_span_id': 'run-span-1',
                                    'execution_id': execution_id,
                                    'assignment_id': 'asn-1',
                                    'attempt_id': f'{execution_id}-attempt-1',
                                    'trace_events_path': (
                                        f'/data/nexus-distributed/live-artifacts/{execution_id}/trace_events.jsonl'
                                    ),
                                },
                            }
                        ],
                    }
                },
            }
        },
        'result_json': None,
        'error_json': None,
    }


async def test_live_compatibility_strips_untrusted_credentials():
    forwarded = []

    async def list_executions(request):
        forwarded.append({
            'authorization': request.headers.get('Authorization'),
            'identity': request.headers.get('x-tai-identity'),
            'internal_token': request.headers.get('X-Nexus-Internal-Token'),
        })
        return web.json_response({'items': [], 'total': 0, 'limit': 100, 'offset': 0})

    control = web.Application()
    control.router.add_get('/v1/executions', list_executions)
    control_client = TestClient(TestServer(control))
    await control_client.start_server()
    try:
        compat = create_live_compatibility_app(
            control_base_url=str(control_client.make_url('/')).rstrip('/'),
            application_base_url='http://application.invalid',
            default_placement={'environment': 'predefined-local', 'pool': 'interactive'},
        )
        compat_client = TestClient(TestServer(compat))
        await compat_client.start_server()
        try:
            response = await compat_client.get(
                '/v1/runs',
                headers={
                    'Authorization': 'Bearer attacker-token',
                    'X-Nexus-Internal-Token': 'attacker-internal-token',
                },
            )
            assert response.status == 401
            response = await compat_client.get('/v1/runs')
            assert response.status == 200
        finally:
            await compat_client.close()
    finally:
        await control_client.close()

    assert forwarded == [
        {
            'authorization': None,
            'identity': None,
            'internal_token': None,
        }
    ]


async def test_live_compatibility_does_not_forward_unvalidated_tai_identity():
    upstream_requests = 0

    async def list_executions(_request):
        nonlocal upstream_requests
        upstream_requests += 1
        return web.json_response({'items': [], 'total': 0, 'limit': 100, 'offset': 0})

    control = web.Application()
    control.router.add_get('/v1/executions', list_executions)
    control_client = TestClient(TestServer(control))
    await control_client.start_server()
    try:
        compatibility = create_live_compatibility_app(
            control_base_url=str(control_client.make_url('/')).rstrip('/'),
            application_base_url='http://application.invalid',
            default_placement={'environment': 'predefined-local', 'pool': 'interactive'},
        )
        compatibility_client = TestClient(TestServer(compatibility))
        await compatibility_client.start_server()
        try:
            response = await compatibility_client.get('/v1/runs', headers={'x-tai-identity': 'unvalidated-jwe'})
            assert response.status == 401
        finally:
            await compatibility_client.close()
    finally:
        await control_client.close()

    assert upstream_requests == 0


async def test_live_compatibility_health_is_local_and_does_not_need_user_identity():
    environment_requests = 0

    async def get_environment(_request):
        nonlocal environment_requests
        environment_requests += 1
        return web.json_response({'error': 'health_must_not_call_control'}, status=500)

    control = web.Application()
    control.router.add_get('/v1/environments/{environment_id}', get_environment)
    control_client = TestClient(TestServer(control))
    await control_client.start_server()
    try:
        compat = create_live_compatibility_app(
            control_base_url=str(control_client.make_url('/')).rstrip('/'),
            application_base_url='http://application.invalid',
            default_placement={'environment': 'predefined-local', 'pool': 'interactive'},
        )
        compat_client = TestClient(TestServer(compat))
        await compat_client.start_server()
        try:
            response = await compat_client.get('/v1/health')
            assert response.status == 200
            assert await response.json() == {'status': 'ok'}

            response = await compat_client.get('/v1/health')
            assert response.status == 200
            assert await response.json() == {'status': 'ok'}
        finally:
            await compat_client.close()
    finally:
        await control_client.close()

    assert environment_requests == 0


async def test_live_compatibility_never_exposes_tai_identity_in_upstream_errors():
    async def list_executions(request):
        return web.json_response(
            {
                'error': 'invalid_tai_identity',
                'debug': request.headers.get('x-tai-identity'),
            },
            status=401,
        )

    control = web.Application()
    control.router.add_get('/v1/executions', list_executions)
    control_client = TestClient(TestServer(control))
    await control_client.start_server()
    try:
        compatibility = create_live_compatibility_app(
            control_base_url=str(control_client.make_url('/')).rstrip('/'),
            application_base_url='http://application.invalid',
            default_placement={'environment': 'predefined-local', 'pool': 'interactive'},
        )
        compatibility_client = TestClient(TestServer(compatibility))
        await compatibility_client.start_server()
        try:
            response = await compatibility_client.get('/v1/runs', headers={'x-tai-identity': 'sensitive-jwe'})
            assert response.status == 401
            assert 'sensitive-jwe' not in await response.text()
        finally:
            await compatibility_client.close()
    finally:
        await control_client.close()


async def test_live_compatibility_creates_old_run_through_typed_submission():
    created_payloads: list[dict] = []

    async def create_submission(request):
        payload = await request.json()
        created_payloads.append(payload)
        return web.json_response(
            {'execution_id': 'exe-live-1', 'status': 'queued', 'phase': 'queued', 'reused': False},
            status=202,
        )

    control = web.Application()
    control.router.add_post('/v1/submissions/live', create_submission)
    control_client = TestClient(TestServer(control))
    await control_client.start_server()
    try:
        compat = create_live_compatibility_app(
            control_base_url=str(control_client.make_url('/')).rstrip('/'),
            application_base_url='http://application.invalid',
            default_placement={'environment': 'predefined-local', 'pool': 'interactive'},
        )
        compat_client = TestClient(TestServer(compat))
        await compat_client.start_server()
        try:
            response = await compat_client.post(
                '/v1/runs',
                json={
                    'run_id': 'live-1',
                    'imports': ['nexus.extensions'],
                    'runtime_provider_name': 'distributed_fake_runtime',
                    'runtime_provider_config': {'base_url': 'http://runtime'},
                    'runtime_configs': {'main': {'image': 'fake'}},
                    'task_name': 'identity',
                    'agent_name': 'distributed_interactive_agent',
                    'agent_config': {'prompt': 'choose'},
                    'live': {'working_dir': '/workspace'},
                },
            )
            assert response.status == 201
            assert await response.json() == {
                'run_id': 'exe-live-1',
                'status': 'starting',
                'events_cursor': 0,
                'reused_cache': False,
                'cache_key': None,
            }

            assert len(created_payloads) == 1
            submitted = created_payloads[0]
            assert submitted['submission_version'] == 1
            assert submitted['idempotency_key'].startswith('legacy-live:')
            assert len(submitted['idempotency_key']) <= 128
            assert submitted['environment_id'] == 'predefined-local'
            assert submitted['pool'] == 'interactive'
            assert submitted['code'] == {'mode': 'config_only'}
            assert submitted['run_input']['run_id'] == 'live-1'
            assert submitted['run_input']['task_name'] == 'identity'
            assert submitted['run_input']['workflow_states'] == {}
            assert submitted['live'] == {'working_dir': '/workspace', 'cache': None}
        finally:
            await compat_client.close()
    finally:
        await control_client.close()


async def test_live_compatibility_yaml_create_uses_typed_submission():
    submitted: list[dict] = []

    async def create_submission(request):
        submitted.append(await request.json())
        return web.json_response(
            {'execution_id': 'exe-yaml-1', 'status': 'queued', 'phase': 'queued', 'reused': False},
            status=202,
        )

    control = web.Application()
    control.router.add_post('/v1/submissions/live', create_submission)
    control_client = TestClient(TestServer(control))
    await control_client.start_server()
    try:
        compatibility = create_live_compatibility_app(
            control_base_url=str(control_client.make_url('/')).rstrip('/'),
            application_base_url='http://application.invalid',
            default_placement={'environment': 'predefined-local', 'pool': 'interactive'},
        )
        compatibility_client = TestClient(TestServer(compatibility))
        await compatibility_client.start_server()
        try:
            response = await compatibility_client.post(
                '/v1/runs/yaml',
                json={
                    'run_input_yaml': 'task_name: identity\nagent_name: distributed_interactive_agent\n',
                    'live': {'working_dir': '/workspace'},
                },
            )
            assert response.status == 201
            assert (await response.json())['run_id'] == 'exe-yaml-1'
            assert submitted[0]['code'] == {'mode': 'config_only'}
            assert submitted[0]['environment_id'] == 'predefined-local'
            assert submitted[0]['pool'] == 'interactive'
            assert submitted[0]['run_input']['task_name'] == 'identity'
        finally:
            await compatibility_client.close()
    finally:
        await control_client.close()


async def test_live_compatibility_uses_fixed_default_placement_and_ignores_request_override():
    created_payloads: list[dict] = []
    sequence = 0

    async def create_submission(request):
        nonlocal sequence
        sequence += 1
        payload = await request.json()
        created_payloads.append(payload)
        return web.json_response(
            {
                'execution_id': f'exe-live-{sequence}',
                'status': 'queued',
                'phase': 'queued',
                'reused': False,
            },
            status=202,
        )

    control = web.Application()
    control.router.add_post('/v1/submissions/live', create_submission)
    control_client = TestClient(TestServer(control))
    await control_client.start_server()
    try:
        compat = create_live_compatibility_app(
            control_base_url=str(control_client.make_url('/')).rstrip('/'),
            application_base_url='http://application.invalid',
            default_placement={'environment': 'predefined-local', 'pool': 'interactive', 'labels': {'region': 'local'}},
        )
        compat_client = TestClient(TestServer(compat))
        await compat_client.start_server()
        try:
            response = await compat_client.post('/v1/runs', json={'run_id': 'live-default', 'task_name': 'identity'})
            assert response.status == 201

            response = await compat_client.post(
                '/v1/runs',
                json={
                    'run_id': 'live-override',
                    'task_name': 'identity',
                    'placement': {'environment': 'predefined-other', 'pool': 'batch'},
                },
            )
            assert response.status == 201

            assert created_payloads[0]['environment_id'] == 'predefined-local'
            assert created_payloads[0]['pool'] == 'interactive'
            assert created_payloads[1]['environment_id'] == 'predefined-local'
            assert created_payloads[1]['pool'] == 'interactive'
        finally:
            await compat_client.close()
    finally:
        await control_client.close()


async def test_live_compatibility_cache_uses_stable_typed_submission_idempotency():
    submitted: list[dict] = []

    async def create_submission(request):
        payload = await request.json()
        submitted.append(payload)
        return web.json_response(
            {
                'execution_id': 'exe-cache-1',
                'status': 'queued',
                'phase': 'queued',
                'reused': len(submitted) > 1,
            },
            status=202,
        )

    control = web.Application()
    control.router.add_post('/v1/submissions/live', create_submission)
    control_client = TestClient(TestServer(control))
    await control_client.start_server()
    try:
        compat = create_live_compatibility_app(
            control_base_url=str(control_client.make_url('/')).rstrip('/'),
            application_base_url='http://application.invalid',
            default_placement={'environment': 'predefined-local', 'pool': 'interactive'},
        )
        compat_client = TestClient(TestServer(compat))
        await compat_client.start_server()
        try:
            request_body = {'run_id': 'caller-value', 'task_name': 'identity'}
            first = await compat_client.post('/v1/runs/cache', json=request_body)
            second = await compat_client.post(
                '/v1/runs/cache', json={**request_body, 'run_id': 'different-caller-value'}
            )
            assert first.status == 201
            assert second.status == 201
            first_payload = await first.json()
            second_payload = await second.json()
            assert first_payload['run_id'] == 'exe-cache-1'
            assert first_payload['reused_cache'] is False
            assert second_payload['run_id'] == 'exe-cache-1'
            assert second_payload['reused_cache'] is True
            assert first_payload['cache_key'] == second_payload['cache_key']
            assert submitted[0]['idempotency_key'] == submitted[1]['idempotency_key']
            assert submitted[0]['idempotency_key'].startswith('legacy-live-cache:')
        finally:
            await compat_client.close()
    finally:
        await control_client.close()


async def test_live_compatibility_strips_untrusted_credentials_for_sse_http_and_websocket():
    forwarded: list[tuple[str, str | None, str | None]] = []

    def record(request: web.Request, transport: str) -> None:
        forwarded.append((
            transport,
            request.headers.get('Authorization'),
            request.headers.get('x-tai-identity'),
        ))

    async def stream_events(request):
        record(request, 'sse')
        response = web.StreamResponse(status=200, headers={'Content-Type': 'text/event-stream'})
        await response.prepare(request)
        await response.write(b'id: 1\nevent: ready\ndata: {}\n\n')
        await response.write_eof()
        return response

    async def proxy(request):
        if request.headers.get('Upgrade', '').lower() == 'websocket':
            record(request, 'websocket')
            websocket = web.WebSocketResponse()
            await websocket.prepare(request)
            await websocket.send_str('ready')
            await websocket.close()
            return websocket
        record(request, 'http')
        return web.json_response({'ok': True})

    application = web.Application()
    application.router.add_get('/v1/executions/{execution_id}/live/events', stream_events)
    application.router.add_route(
        '*',
        '/v1/executions/{execution_id}/runtimes/{runtime_name}/proxy/{tail:.*}',
        proxy,
    )
    application_client = TestClient(TestServer(application))
    await application_client.start_server()
    try:
        compat = create_live_compatibility_app(
            control_base_url='http://control.invalid',
            application_base_url=str(application_client.make_url('/')).rstrip('/'),
            default_placement={'environment': 'predefined-local', 'pool': 'interactive'},
        )
        compat_client = TestClient(TestServer(compat))
        await compat_client.start_server()
        attacker_headers = {
            'Authorization': 'Bearer attacker-token',
        }
        try:
            response = await compat_client.get('/v1/runs/live-1/events', headers=attacker_headers)
            assert response.status == 200
            assert 'attacker-token' not in await response.text()

            response = await compat_client.get(
                '/v1/runs/live-1/runtimes/main/proxy/ping',
                headers=attacker_headers,
            )
            assert response.status == 200
            assert 'attacker-token' not in await response.text()

            websocket = await compat_client.ws_connect(
                '/v1/runs/live-1/runtimes/main/proxy/socket',
                headers=attacker_headers,
            )
            message = await websocket.receive()
            assert message.data == 'ready'
            await websocket.close()
        finally:
            await compat_client.close()
    finally:
        await application_client.close()

    assert forwarded == [
        ('sse', None, None),
        ('http', None, None),
        ('websocket', None, None),
    ]


async def test_live_compatibility_maps_cached_create_get_config_turns_events_and_proxy():
    commands: list[dict] = []
    stopped: list[str] = []
    execution = _execution_payload('live-1', route_status=RouteStatus.ACTIVE)

    async def create_submission(request):
        await request.json()
        execution.update(_execution_payload('live-1', route_status=RouteStatus.ACTIVE))
        return web.json_response(
            {'execution_id': 'live-1', 'status': 'queued', 'phase': 'queued', 'reused': False},
            status=202,
        )

    async def get_execution(request):
        return web.json_response(execution)

    async def list_executions(request):
        assert request.query.get('mode') == 'live'
        return web.json_response({'items': [execution]})

    async def stop_execution(request):
        stopped.append(request.match_info['execution_id'])
        execution['status'] = ExecutionStatus.KILLED
        execution['route_status'] = RouteStatus.CLOSED
        return web.json_response(execution)

    async def submit_command(request):
        commands.append(await request.json())
        return web.json_response({'command_id': 'cmd-1', 'assignment_id': 'asn-1'}, status=202)

    async def stream_events(request):
        assert request.query.get('cursor') == '5'
        response = web.StreamResponse(status=200, headers={'Content-Type': 'text/event-stream'})
        await response.prepare(request)
        data = {'data': {'runtimes': {'main': {'proxy_url': '/v1/executions/live-1/runtimes/main/proxy'}}}}
        await response.write(f'id: 6\nevent: runtime.ready\ndata: {json.dumps(data)}\n\n'.encode())
        await response.write_eof()
        return response

    async def proxy_ping(request):
        return web.json_response({'path': request.path, 'query': dict(request.query)})

    control = web.Application()
    control.router.add_post('/v1/submissions/live', create_submission)
    control.router.add_get('/v1/executions', list_executions)
    control.router.add_get('/v1/executions/{execution_id}', get_execution)
    control.router.add_post('/v1/executions/{execution_id}/stop', stop_execution)
    app_gateway = web.Application()
    app_gateway.router.add_post('/v1/executions/{execution_id}/commands', submit_command)
    app_gateway.router.add_get('/v1/executions/{execution_id}/live/events', stream_events)
    app_gateway.router.add_get('/v1/executions/{execution_id}/runtimes/{runtime_name}/proxy/{tail:.*}', proxy_ping)

    control_client = TestClient(TestServer(control))
    app_client = TestClient(TestServer(app_gateway))
    await control_client.start_server()
    await app_client.start_server()
    try:
        compat = create_live_compatibility_app(
            control_base_url=str(control_client.make_url('/')).rstrip('/'),
            application_base_url=str(app_client.make_url('/')).rstrip('/'),
            default_placement={'environment': 'predefined-local', 'pool': 'interactive'},
        )
        compat_client = TestClient(TestServer(compat))
        await compat_client.start_server()
        try:
            response = await compat_client.post('/v1/runs/cache', json={'run_id': 'live-1', 'task_name': 'identity'})
            assert response.status == 201
            assert (await response.json())['status'] == 'starting'

            response = await compat_client.get('/v1/runs?summary=1')
            assert response.status == 200
            assert (await response.json())['runs'][0]['run_id'] == 'live-1'

            response = await compat_client.get('/v1/runs/live-1')
            assert response.status == 200
            run_info = await response.json()
            assert run_info['run_id'] == 'live-1'
            assert run_info['status'] == 'idle'
            assert run_info['runtimes']['main']['proxy_url'] == '/v1/runs/live-1/runtimes/main/proxy'
            assert run_info['runtimes']['main']['debug_gui_url'] == '/v1/runs/live-1/runtimes/main/proxy/debug/gui'

            response = await compat_client.get('/v1/runs/live-1/config')
            assert response.status == 200
            config = await response.json()
            assert config['run_id'] == 'live-1'
            assert 'run_id' in config['request']
            assert config['request']['task_name'] == 'identity'

            response = await compat_client.post('/v1/runs/live-1/turns', json={'text': 'Alpha'})
            assert response.status == 202
            assert await response.json() == {'run_id': 'live-1', 'turn_id': 'cmd-1', 'status': 'idle'}
            assert commands == [{'type': 'input.submit', 'payload': {'text': 'Alpha'}, 'idempotency_key': None}]

            response = await compat_client.get('/v1/runs/live-1/events?cursor=5')
            assert response.status == 200
            events_text = await response.text()
            assert '/v1/runs/live-1/runtimes/main/proxy' in events_text
            assert '/v1/executions/live-1/runtimes/main/proxy' not in events_text

            response = await compat_client.get('/v1/runs/live-1/runtimes/main/proxy/ping?x=1')
            assert response.status == 200
            assert await response.json() == {
                'path': '/v1/executions/live-1/runtimes/main/proxy/ping',
                'query': {'x': '1'},
            }

            response = await compat_client.delete('/v1/runs/live-1')
            assert response.status == 200
            assert stopped == ['live-1']
            assert (await response.json())['status'] == 'killed'
        finally:
            await compat_client.close()
    finally:
        await app_client.close()
        await control_client.close()


async def test_live_compatibility_get_run_merges_active_execution_info():
    execution = _execution_payload('live-1')

    async def get_execution(request):
        return web.json_response(execution)

    async def get_info(request):
        return web.json_response({
            'assignment_id': 'asn-1',
            'execution_id': 'live-1',
            'attempt_id': 'live-1-attempt-1',
            'driver': {'spec_type': 'nexus_live', 'instance_mode': 'asyncio'},
            'runtime_info': {
                'runtimes': {
                    'main': {
                        'name': 'main',
                        'runtime_id': 'rt-main-active',
                        'base_url': 'http://runtime-active',
                        'proxy_url': '/v1/executions/live-1/runtimes/main/proxy',
                        'debug_gui_url': '/v1/executions/live-1/runtimes/main/proxy/debug/gui',
                    }
                }
            },
            'live': {'status': 'idle', 'events_cursor': 12, 'current_turn_id': 'turn-active'},
        })

    control = web.Application()
    control.router.add_get('/v1/executions/{execution_id}', get_execution)
    app_gateway = web.Application()
    app_gateway.router.add_get('/v1/executions/{execution_id}/info', get_info)

    control_client = TestClient(TestServer(control))
    app_client = TestClient(TestServer(app_gateway))
    await control_client.start_server()
    await app_client.start_server()
    try:
        compat = create_live_compatibility_app(
            control_base_url=str(control_client.make_url('/')).rstrip('/'),
            application_base_url=str(app_client.make_url('/')).rstrip('/'),
            default_placement={'environment': 'predefined-local', 'pool': 'interactive'},
        )
        compat_client = TestClient(TestServer(compat))
        await compat_client.start_server()
        try:
            response = await compat_client.get('/v1/runs/live-1')
            assert response.status == 200
            run_info = await response.json()
            assert run_info['status'] == 'idle'
            assert run_info['events_cursor'] == 12
            assert run_info['current_turn_id'] == 'turn-active'
            assert run_info['runtime_base_url'] == 'http://runtime-active'
            assert run_info['runtimes']['main']['proxy_url'] == '/v1/runs/live-1/runtimes/main/proxy'
        finally:
            await compat_client.close()
    finally:
        await app_client.close()
        await control_client.close()


async def test_live_compatibility_get_run_expands_external_result():
    execution = _execution_payload('live-1')
    execution['status'] = ExecutionStatus.SUCCEEDED
    execution['route_status'] = RouteStatus.CLOSED
    execution['current_assignment']['runtime_info']['_artifacts']['live'].update({
        'live_event_cursor': 7,
        'current_turn_id': 'turn-final',
    })
    artifact_meta = execution['current_assignment']['runtime_info']['_artifacts']['live']
    execution['result_json'] = {
        '_external': True,
        'kind': 'nexus_live_result',
        'version': 2,
        'metadata': artifact_meta,
    }

    async def get_execution(request):
        return web.json_response(execution)

    async def get_result(request):
        return web.json_response({'success': True, 'large': 'payload'})

    control = web.Application()
    control.router.add_get('/v1/executions/{execution_id}', get_execution)
    app_gateway = web.Application()
    app_gateway.router.add_get('/v1/executions/{execution_id}/live/result', get_result)

    control_client = TestClient(TestServer(control))
    app_client = TestClient(TestServer(app_gateway))
    await control_client.start_server()
    await app_client.start_server()
    try:
        compat = create_live_compatibility_app(
            control_base_url=str(control_client.make_url('/')).rstrip('/'),
            application_base_url=str(app_client.make_url('/')).rstrip('/'),
            default_placement={'environment': 'predefined-local', 'pool': 'interactive'},
        )
        compat_client = TestClient(TestServer(compat))
        await compat_client.start_server()
        try:
            response = await compat_client.get('/v1/runs/live-1')
            assert response.status == 200
            run_info = await response.json()
            assert run_info['events_cursor'] == 0
            assert run_info['current_turn_id'] == 'turn-1'
            assert run_info['output'] == {'success': True, 'large': 'payload'}
        finally:
            await compat_client.close()
    finally:
        await app_client.close()
        await control_client.close()


async def test_live_compatibility_lists_across_control_pages_before_filtering():
    calls: list[dict[str, str]] = []
    first = _execution_payload('other-live')
    first['spec']['run_input']['task_name'] = 'other'
    second = _execution_payload('live-1')
    second['spec']['run_input']['task_name'] = 'identity'

    async def list_executions(request):
        calls.append(dict(request.query))
        offset = int(request.query.get('offset', '0') or '0')
        return web.json_response({'items': [first] if offset == 0 else [second]})

    control = web.Application()
    control.router.add_get('/v1/executions', list_executions)
    control_client = TestClient(TestServer(control))
    await control_client.start_server()
    try:
        compat = create_live_compatibility_app(
            control_base_url=str(control_client.make_url('/')).rstrip('/'),
            application_base_url='http://application.invalid',
            default_placement={'environment': 'predefined-local', 'pool': 'interactive'},
        )
        compat_client = TestClient(TestServer(compat))
        await compat_client.start_server()
        try:
            response = await compat_client.get('/v1/runs?limit=1&q=identity')
            assert response.status == 200
            payload = await response.json()
            assert [item['run_id'] for item in payload['runs']] == ['live-1']
            assert calls[:2] == [
                {'mode': 'live', 'limit': '1', 'offset': '0'},
                {'mode': 'live', 'limit': '1', 'offset': '1'},
            ]
        finally:
            await compat_client.close()
    finally:
        await control_client.close()


async def test_live_compatibility_list_respects_upstream_page_limit():
    first = _execution_payload('other-live')
    first['spec']['run_input']['task_name'] = 'other'
    second = _execution_payload('live-1')
    second['spec']['run_input']['task_name'] = 'identity'

    async def list_executions(request):
        offset = int(request.query.get('offset', '0') or '0')
        return web.json_response({
            'items': [first] if offset == 0 else [second],
            'limit': 1,
            'offset': offset,
            'total': 2,
        })

    control = web.Application()
    control.router.add_get('/v1/executions', list_executions)
    control_client = TestClient(TestServer(control))
    await control_client.start_server()
    try:
        compat = create_live_compatibility_app(
            control_base_url=str(control_client.make_url('/')).rstrip('/'),
            application_base_url='http://application.invalid',
            default_placement={'environment': 'predefined-local', 'pool': 'interactive'},
        )
        compat_client = TestClient(TestServer(compat))
        await compat_client.start_server()
        try:
            response = await compat_client.get('/v1/runs?limit=2&q=identity')
            assert response.status == 200
            payload = await response.json()
            assert [item['run_id'] for item in payload['runs']] == ['live-1']
        finally:
            await compat_client.close()
    finally:
        await control_client.close()


async def test_live_compatibility_can_be_mounted_under_application_prefix():
    execution = _execution_payload('live-1', route_status=RouteStatus.ACTIVE)
    execution['current_assignment']['runtime_info']['runtimes']['main']['proxy_url'] = (
        '/v1/runs/live-1/runtimes/main/proxy'
    )
    execution['current_assignment']['runtime_info']['runtimes']['main']['debug_gui_url'] = (
        '/v1/runs/live-1/runtimes/main/proxy/debug/gui'
    )

    async def get_execution(request):
        return web.json_response(execution)

    async def get_environment(_request):
        return web.json_response({
            'environment_id': 'predefined-local',
            'allocation_type': 'predefined',
            'status': 'enabled',
            'pool_stats': {
                'pool_id': 'interactive',
                'healthy_manager_count': 1,
            },
        })

    control = web.Application()
    control.router.add_get('/v1/executions/{execution_id}', get_execution)
    control.router.add_get('/v1/environments/{environment_id}', get_environment)
    control_client = TestClient(TestServer(control))
    await control_client.start_server()
    try:
        compat = create_live_compatibility_app(
            control_base_url=str(control_client.make_url('/')).rstrip('/'),
            application_base_url='http://application.invalid',
            default_placement={'environment': 'predefined-local', 'pool': 'interactive'},
            route_prefix='/v1/runs',
            public_runs_base_path='/compat/live/v1/runs',
            health_path='/health',
        )
        compat_client = TestClient(TestServer(compat))
        await compat_client.start_server()
        try:
            response = await compat_client.get('/health')
            assert response.status == 200

            response = await compat_client.get('/v1/runs/live-1')
            assert response.status == 200
            run_info = await response.json()
            assert run_info['runtimes']['main']['proxy_url'] == '/compat/live/v1/runs/live-1/runtimes/main/proxy'
            assert (
                run_info['runtimes']['main']['debug_gui_url']
                == '/compat/live/v1/runs/live-1/runtimes/main/proxy/debug/gui'
            )
        finally:
            await compat_client.close()
    finally:
        await control_client.close()
