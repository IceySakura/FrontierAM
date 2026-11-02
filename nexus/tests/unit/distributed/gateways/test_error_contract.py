from __future__ import annotations

import logging

import pytest

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from nexus.distributed.errors import (
    DomainConflictError,
    DomainValidationError,
    FencingError,
    ResourceGoneError,
    ResourceNotFoundError,
    TemporaryUnavailableError,
)
from nexus.distributed.gateways.application.app import error_middleware as application_error_middleware
from nexus.distributed.gateways.control.app import error_middleware as control_error_middleware


@pytest.mark.parametrize(
    'middleware',
    [control_error_middleware, application_error_middleware],
)
@pytest.mark.parametrize(
    'error',
    [ValueError('private-value'), KeyError('private-key')],
)
async def test_gateway_unknown_builtin_errors_are_logged_and_hidden(middleware, error, caplog):
    async def fail(_request: web.Request) -> web.Response:
        raise error

    app = web.Application(middlewares=[middleware])
    app.router.add_get('/fail', fail)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        with caplog.at_level(logging.ERROR):
            response = await client.get('/fail')
        assert response.status == 500
        assert await response.json() == {'error': 'internal_error', 'retryable': False}
        assert str(error) not in await response.text()
        assert 'Unhandled' in caplog.text
    finally:
        await client.close()


@pytest.mark.parametrize(
    'middleware',
    [control_error_middleware, application_error_middleware],
)
async def test_gateway_http_exceptions_use_shared_json_envelope(middleware):
    async def fail(_request: web.Request) -> web.Response:
        raise web.HTTPForbidden(text='scope_required')

    app = web.Application(middlewares=[middleware])
    app.router.add_get('/fail', fail)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.get('/fail')
        assert response.status == 403
        assert await response.json() == {'error': 'scope_required', 'retryable': False}
    finally:
        await client.close()


@pytest.mark.parametrize(
    'middleware',
    [control_error_middleware, application_error_middleware],
)
@pytest.mark.parametrize(
    ('error', 'status', 'payload'),
    [
        (ResourceNotFoundError('execution_not_found'), 404, {'error': 'execution_not_found', 'retryable': False}),
        (DomainConflictError('execution_conflict'), 409, {'error': 'execution_conflict', 'retryable': False}),
        (FencingError('assignment_fenced'), 409, {'error': 'assignment_fenced', 'retryable': False}),
        (
            DomainValidationError('invalid_submission', detail='field=pool_id'),
            422,
            {'error': 'invalid_submission', 'detail': 'field=pool_id', 'retryable': False},
        ),
        (ResourceGoneError('execution_closed'), 410, {'error': 'execution_closed', 'retryable': False}),
        (
            TemporaryUnavailableError('manager_unavailable'),
            503,
            {'error': 'manager_unavailable', 'retryable': True},
        ),
    ],
)
async def test_gateway_domain_errors_use_shared_json_envelope(middleware, error, status, payload):
    async def fail(_request: web.Request) -> web.Response:
        raise error

    app = web.Application(middlewares=[middleware])
    app.router.add_get('/fail', fail)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.get('/fail')
        assert response.status == status
        assert await response.json() == payload
    finally:
        await client.close()
