"""Read-only HTTP API and bundled frontend for offline trajectories."""

from __future__ import annotations

import asyncio
import json

from pathlib import Path

from aiohttp import web

from nexus.dashboard.projection import project_trajectory
from nexus.dashboard.store import DEFAULT_MAX_BYTES, ArtifactStore


STATIC_DIR = Path(__file__).with_name('static')


def create_app(path: str | Path, *, max_bytes: int = DEFAULT_MAX_BYTES) -> web.Application:
    store = ArtifactStore(path, max_bytes=max_bytes)
    lock = asyncio.Lock()

    @web.middleware
    async def security_headers(request: web.Request, handler):
        try:
            response = await handler(request)
        except web.HTTPException as exc:
            response = exc
        response.headers.update({
            'X-Content-Type-Options': 'nosniff',
            'Cache-Control': 'no-store',
            'Referrer-Policy': 'no-referrer',
            'Content-Security-Policy': "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; object-src 'none'; base-uri 'none'; frame-ancestors 'self'",
        })
        if isinstance(response, web.HTTPException):
            raise response
        return response

    async def index(_request: web.Request) -> web.StreamResponse:
        return web.FileResponse(STATIC_DIR / 'index.html')

    async def asset(request: web.Request) -> web.StreamResponse:
        name = request.match_info['name']
        if name not in {'app.js', 'media.js', 'style.css'}:
            raise web.HTTPNotFound()
        return web.FileResponse(STATIC_DIR / name)

    async def catalog(request: web.Request) -> web.Response:
        try:
            offset = max(0, int(request.query.get('offset', '0')))
            limit = min(200, max(1, int(request.query.get('limit', '50'))))
        except ValueError as exc:
            raise web.HTTPBadRequest(text='Invalid pagination') from exc
        async with lock:
            result = await asyncio.to_thread(store.scan)
        query = request.query.get('q', '').casefold()
        items = [item for item in result['items'] if query in str(item.get('session_id', '')).casefold()]
        return web.json_response({
            'items': items[offset : offset + limit],
            'total': len(items),
            'offset': offset,
            'limit': limit,
            'errors': result['errors'],
        })

    def load(artifact_id: str) -> dict:
        result = store.load(artifact_id)
        result['graph'] = project_trajectory(result['trajectory'])
        return result

    async def detail(request: web.Request) -> web.Response:
        try:
            async with lock:
                result = await asyncio.to_thread(load, request.match_info['artifact_id'])
            return web.json_response(result, dumps=lambda value: json.dumps(value, ensure_ascii=False, allow_nan=False))
        except KeyError as exc:
            raise web.HTTPNotFound(text='Unknown or invalid artifact') from exc
        except (ValueError, TypeError, OSError) as exc:
            raise web.HTTPUnprocessableEntity(text=f'Cannot load trajectory: {exc}') from exc

    app = web.Application(middlewares=[security_headers])
    app.router.add_get('/', index)
    app.router.add_get('/assets/{name}', asset)
    app.router.add_get('/api/trajectories', catalog)
    app.router.add_get('/api/trajectories/{artifact_id}', detail)
    return app
