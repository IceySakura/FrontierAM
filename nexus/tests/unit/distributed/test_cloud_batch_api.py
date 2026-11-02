from aiohttp import web
from aiohttp.test_utils import TestServer
from tests.integration.distributed.cloud_batch_api import CloudBatchApi


async def test_cloud_batch_api_authenticates_and_paginates_owner_resources():
    requests = []

    async def handler(request: web.Request) -> web.Response:
        requests.append((request.method, request.path_qs, request.headers.get('Authorization')))
        if request.path == '/v1/executions/parent/batch/items':
            if request.query.get('cursor') is None:
                return web.json_response(
                    {
                        'items': [{'item_id': 'item-1', 'current_child_execution_id': 'child-1'}],
                        'next_cursor': 'item-1',
                    }
                )
            return web.json_response(
                {
                    'items': [{'item_id': 'item-2', 'current_child_execution_id': 'child-2'}],
                    'next_cursor': None,
                }
            )
        if request.path in {'/v1/executions/child-1', '/v1/executions/child-2'}:
            child_id = request.path.rsplit('/', 1)[-1]
            result_number = child_id.rsplit('-', 1)[-1]
            return web.json_response(
                {
                    'execution_id': child_id,
                    'status': 'succeeded',
                    'result_artifact_id': f'result-{result_number}',
                }
            )
        raise AssertionError(request.path_qs)

    app = web.Application()
    app.router.add_route('*', '/{tail:.*}', handler)
    server = TestServer(app)
    await server.start_server()
    base_url = str(server.make_url('/')).rstrip('/')
    try:
        async with CloudBatchApi(control_url=base_url, application_url=base_url, token='submit-secret') as api:
            items = await api.list_all_batch_items('parent')
            facts = await api.execution_facts({'child-1', 'child-2'})
    finally:
        await server.close()

    assert [item['item_id'] for item in items] == ['item-1', 'item-2']
    assert set(facts) == {'child-1', 'child-2'}
    assert {item['result_artifact_id'] for item in facts.values()} == {'result-1', 'result-2'}
    assert all(authorization == 'Bearer submit-secret' for _method, _path, authorization in requests)
    assert not any(path.startswith('/v1/executions?') for _method, path, _authorization in requests)


async def test_cloud_batch_api_resolves_external_result_with_application_gateway_auth():
    requests = []

    async def handler(request: web.Request) -> web.Response:
        requests.append((request.path, request.headers.get('Authorization')))
        return web.json_response({'task_output': {'marker': 'source-package-ok:7'}})

    app = web.Application()
    app.router.add_get('/v1/executions/child-7/result', handler)
    server = TestServer(app)
    await server.start_server()
    base_url = str(server.make_url('/')).rstrip('/')
    try:
        async with CloudBatchApi(control_url=base_url, application_url=base_url, token='submit-secret') as api:
            result = await api.resolved_result(
                {
                    'execution_id': 'child-7',
                    'result_json': {'_external': True, 'kind': 'run_result'},
                }
            )
    finally:
        await server.close()

    assert result == {'task_output': {'marker': 'source-package-ok:7'}}
    assert requests == [('/v1/executions/child-7/result', 'Bearer submit-secret')]


async def test_cloud_batch_api_posts_application_commands_and_reads_authenticated_sse():
    requests = []

    async def command(request: web.Request) -> web.Response:
        requests.append((request.method, request.path, request.headers.get('Authorization'), await request.json()))
        return web.json_response({'accepted': True})

    async def events(request: web.Request) -> web.StreamResponse:
        requests.append((request.method, request.path, request.headers.get('Authorization'), None))
        response = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
        await response.prepare(request)
        await response.write(b'event: input.required\n')
        await response.write(b'data: {"type":"input.required",\n')
        await response.write(b'data: "data":{"hint":"choose"}}\n\n')
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post('/v1/executions/live/commands', command)
    app.router.add_get('/v1/executions/live/live/events', events)
    server = TestServer(app)
    await server.start_server()
    base_url = str(server.make_url('/')).rstrip('/')
    try:
        async with CloudBatchApi(control_url=base_url, application_url=base_url, token='submit-secret') as api:
            accepted = await api.application_json(
                'POST',
                '/v1/executions/live/commands',
                body={'type': 'input.submit', 'payload': {'text': 'ok'}, 'idempotency_key': 'input-1'},
            )
            event = await api.read_application_sse_until(
                '/v1/executions/live/live/events?cursor=0',
                'input.required',
            )
    finally:
        await server.close()

    assert accepted == {'accepted': True}
    assert event == {'type': 'input.required', 'data': {'hint': 'choose'}}
    assert requests == [
        (
            'POST',
            '/v1/executions/live/commands',
            'Bearer submit-secret',
            {'type': 'input.submit', 'payload': {'text': 'ok'}, 'idempotency_key': 'input-1'},
        ),
        ('GET', '/v1/executions/live/live/events', 'Bearer submit-secret', None),
    ]


async def test_cloud_batch_api_tracks_consecutive_gateway_5xx():
    attempts = 0

    async def handler(_request: web.Request) -> web.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return web.json_response({'error': 'temporary'}, status=503)
        return web.json_response({'status': 'ok'})

    app = web.Application()
    app.router.add_get('/v1/health', handler)
    server = TestServer(app)
    await server.start_server()
    base_url = str(server.make_url('/')).rstrip('/')
    try:
        async with CloudBatchApi(control_url=base_url, application_url=base_url, token='submit-secret') as api:
            assert await api.control_json('GET', '/v1/health') == {'status': 'ok'}
            assert api.consecutive_gateway_5xx == 0
            assert api.maximum_consecutive_gateway_5xx == 2
    finally:
        await server.close()
