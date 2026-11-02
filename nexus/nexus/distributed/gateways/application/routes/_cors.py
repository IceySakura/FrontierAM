from __future__ import annotations

from aiohttp import web

from nexus.distributed.artifacts.tickets import UPLOAD_TICKET_HEADER
from nexus.distributed.gateways.common.keys import CONFIG_KEY


def add_cors_headers(response: web.StreamResponse, request: web.Request) -> web.StreamResponse:
    origin = request.headers.get('Origin')
    config = request.app[CONFIG_KEY]
    # The console fetches cross-origin (page served by the control gateway, results served by this
    # application gateway) with credentials: 'include'. Browsers require the response to echo the
    # exact Origin *and* Access-Control-Allow-Credentials: true; a wildcard or a missing credentials
    # header aborts the request as a CORS error. Only fall back to '*' when the caller is not a
    # browser (no Origin header), where credentials are never echoed back.
    if origin and (config.auth_mode == 'disabled' or origin in config.cors_origins):
        response.headers['Access-Control-Allow-Origin'] = origin
        response.headers['Access-Control-Allow-Credentials'] = 'true'
    elif config.auth_mode == 'disabled':
        response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, PATCH, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = request.headers.get(
        'Access-Control-Request-Headers',
        f'authorization,content-type,{UPLOAD_TICKET_HEADER.lower()}',
    )
    response.headers['Access-Control-Max-Age'] = '600'
    if origin:
        vary = response.headers.get('Vary')
        if vary:
            if 'Origin' not in {item.strip() for item in vary.split(',')}:
                response.headers['Vary'] = f'{vary}, Origin'
        else:
            response.headers['Vary'] = 'Origin'
    return response


__all__ = ['add_cors_headers']
