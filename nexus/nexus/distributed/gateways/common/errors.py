from __future__ import annotations

import logging

from typing import Any

from aiohttp import web

from nexus.distributed.errors import DistributedDomainError, FencingError
from nexus.distributed.utils.leases import LeaseLostError


_HTTP_ERROR_CODES = {
    400: 'invalid_request',
    401: 'authentication_required',
    403: 'forbidden',
    404: 'not_found',
    405: 'method_not_allowed',
    408: 'request_timeout',
    409: 'conflict',
    410: 'gone',
    413: 'request_too_large',
    422: 'validation_error',
    425: 'too_early',
    429: 'too_many_requests',
    502: 'bad_gateway',
    503: 'service_unavailable',
    504: 'gateway_timeout',
}
_RETRYABLE_HTTP_STATUSES = frozenset({408, 425, 429, 502, 503, 504})


def gateway_error_response(error: DistributedDomainError) -> web.Response:
    payload: dict[str, Any] = {
        'error': error.code,
        'retryable': error.retryable,
    }
    if error.detail is not None:
        payload['detail'] = error.detail
    return web.json_response(payload, status=error.status)


async def handle_gateway_errors(
    request: web.Request,
    handler: Any,
    *,
    logger: logging.Logger,
    component: str,
) -> web.StreamResponse:
    try:
        return await handler(request)
    except DistributedDomainError as exc:
        return gateway_error_response(exc)
    except LeaseLostError as exc:
        return gateway_error_response(FencingError(str(exc) or None))
    except web.HTTPException as exc:
        return _http_error_response(exc)
    except Exception:
        logger.exception(
            'Unhandled %s error: method=%s path=%s',
            component,
            request.method,
            request.path,
        )
        return web.json_response({'error': 'internal_error', 'retryable': False}, status=500)


def _http_error_response(error: web.HTTPException) -> web.Response:
    raw_detail = error.text.strip() if isinstance(error.text, str) else ''
    if _is_public_error_code(raw_detail):
        code = raw_detail
        detail = None
    else:
        code = _HTTP_ERROR_CODES.get(error.status, f'http_{error.status}')
        detail = raw_detail or None
    payload: dict[str, Any] = {
        'error': code,
        'retryable': error.status in _RETRYABLE_HTTP_STATUSES,
    }
    if detail is not None:
        payload['detail'] = detail
    headers = {
        name: value for name, value in error.headers.items() if name.lower() not in {'content-length', 'content-type'}
    }
    return web.json_response(payload, status=error.status, headers=headers)


def _is_public_error_code(value: str) -> bool:
    if not value or len(value) > 128 or not value[0].islower():
        return False
    return all(character.islower() or character.isdigit() or character in '_.:-' for character in value)


__all__ = [
    'gateway_error_response',
    'handle_gateway_errors',
]
