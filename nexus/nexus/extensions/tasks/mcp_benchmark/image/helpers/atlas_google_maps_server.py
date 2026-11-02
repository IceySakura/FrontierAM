#!/usr/bin/env python3
"""Google Maps MCP server for Toolathlon eval compatibility.

This server is intentionally separate from the agent-side Node.js server.
It exposes the unprefixed ``maps_*`` tool names that Toolathlon eval scripts
call directly, while delegating to Google Maps HTTP APIs with the real
``GOOGLE_MAPS_API_KEY``.
"""

from __future__ import annotations

import json
import os

from datetime import datetime
from typing import Any

import requests

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import McpError
from mcp.types import INTERNAL_ERROR, INVALID_PARAMS, ErrorData, TextContent, Tool


GEOCODE_URL = 'https://maps.googleapis.com/maps/api/geocode/json'
PLACE_TEXTSEARCH_URL = 'https://maps.googleapis.com/maps/api/place/textsearch/json'
PLACE_DETAILS_URL = 'https://maps.googleapis.com/maps/api/place/details/json'
DISTANCE_MATRIX_URL = 'https://maps.googleapis.com/maps/api/distancematrix/json'
DIRECTIONS_URL = 'https://maps.googleapis.com/maps/api/directions/json'
ELEVATION_URL = 'https://maps.googleapis.com/maps/api/elevation/json'
REQUEST_TIMEOUT = 30

_API_KEY = os.environ.get('GOOGLE_MAPS_API_KEY', '').strip()
_PROXY_URL = os.environ.get('HTTP_PROXY') or os.environ.get('http_proxy') or 'http://hyrl-proxy.woa.com:3128'
_PROXIES: dict[str, str] = {'http': _PROXY_URL, 'https': _PROXY_URL}
_CA_BUNDLE = (
    os.environ.get('REQUESTS_CA_BUNDLE') or os.environ.get('SSL_CERT_FILE') or '/etc/ssl/certs/ca-certificates.crt'
)
_DEBUG_LOG_PATH = os.environ.get('NEXUS_GMAPS_DEBUG_LOG', '').strip()


def _error(code: int, message: str) -> McpError:
    return McpError(ErrorData(code=code, message=message))


def _json_content(payload: Any) -> list[TextContent]:
    return [TextContent(type='text', text=json.dumps(payload, ensure_ascii=False, indent=2))]


def _debug_log(event: str, **fields: Any) -> None:
    if not _DEBUG_LOG_PATH:
        return
    try:
        with open(_DEBUG_LOG_PATH, 'a', encoding='utf-8') as fh:
            entry = {'ts': datetime.utcnow().isoformat(), 'event': event, **fields}
            fh.write(json.dumps(entry, ensure_ascii=False) + '\n')
    except Exception:
        pass


def _http_get(
    url: str,
    *,
    params: dict[str, Any],
    use_proxy: bool,
) -> requests.Response:
    if use_proxy:
        return requests.get(
            url,
            params=params,
            timeout=REQUEST_TIMEOUT,
            headers={'User-Agent': 'nexus-mcp-benchmark'},
            proxies=_PROXIES,
            verify=_CA_BUNDLE,
        )
    session = requests.Session()
    session.trust_env = False
    try:
        return session.get(
            url,
            params=params,
            timeout=REQUEST_TIMEOUT,
            headers={'User-Agent': 'nexus-mcp-benchmark'},
            verify=_CA_BUNDLE,
        )
    finally:
        session.close()


def _ensure_api_key() -> str:
    if not _API_KEY:
        raise _error(INTERNAL_ERROR, 'GOOGLE_MAPS_API_KEY is missing')
    return _API_KEY


def _request_json(url: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
    query = dict(params or {})
    query.setdefault('key', _ensure_api_key())
    _debug_log('request.start', url=url, params={k: ('<redacted>' if k == 'key' else v) for k, v in query.items()})
    attempts: list[tuple[str, bool]] = (
        # Gongfeng containers cannot reach Google directly; trying direct first
        # burns the full 30s request timeout and causes Toolathlon eval to time
        # out before the proxy fallback returns. Prefer the configured proxy.
        [('proxy', True), ('direct', False)] if 'maps.googleapis.com' in url else [('proxy', True)]
    )

    last_exc: requests.RequestException | None = None
    response = None
    for mode, use_proxy in attempts:
        try:
            _debug_log('request.try', url=url, mode=mode)
            response = _http_get(url, params=query, use_proxy=use_proxy)
            _debug_log('request.try_done', url=url, mode=mode, status_code=response.status_code)
            break
        except requests.RequestException as exc:
            last_exc = exc
            _debug_log('request.try_error', url=url, mode=mode, error=repr(exc))
    if response is None:
        assert last_exc is not None
        _debug_log('request.error', url=url, error=repr(last_exc))
        raise _error(INTERNAL_ERROR, f'Request failed: {last_exc}') from last_exc
    if response.status_code != 200:
        _debug_log('request.http_error', url=url, status_code=response.status_code, body=response.text[:300])
        raise _error(INTERNAL_ERROR, f'HTTP {response.status_code} for {response.url}: {response.text[:300]}')
    try:
        payload = response.json()
    except ValueError as exc:
        _debug_log('request.json_error', url=url, body=response.text[:300], error=repr(exc))
        raise _error(INTERNAL_ERROR, f'Invalid JSON response from Google Maps: {response.text[:300]}') from exc
    status = str(payload.get('status') or 'OK')
    if status not in {'OK', 'ZERO_RESULTS'}:
        _debug_log('request.api_error', url=url, status=status, error_message=payload.get('error_message'))
        raise _error(INTERNAL_ERROR, f'Google Maps API error {status}: {payload.get("error_message") or payload}')
    _debug_log('request.done', url=url, status=status, top_keys=list(payload)[:10])
    return payload


def _coerce_location_text(value: Any) -> str:
    if isinstance(value, dict):
        if value.get('place_id'):
            return f'place_id:{value["place_id"]}'
        lat = value.get('lat') if value.get('lat') is not None else value.get('latitude')
        lng = value.get('lng') if value.get('lng') is not None else value.get('longitude')
        if lat is not None and lng is not None:
            return f'{lat},{lng}'
        address = value.get('address') or value.get('formatted_address') or value.get('name')
        if address:
            return str(address)
    text = str(value or '').strip()
    if not text:
        raise _error(INVALID_PARAMS, 'Expected a non-empty location value')
    return text


def _place_from_result(place: dict[str, Any]) -> dict[str, Any]:
    geometry = place.get('geometry') or {}
    location = geometry.get('location') or {}
    return {
        'name': place.get('name') or '',
        'formatted_address': place.get('formatted_address') or place.get('vicinity') or '',
        'place_id': place.get('place_id') or '',
        'location': {
            'lat': location.get('lat'),
            'lng': location.get('lng'),
        },
        'rating': place.get('rating'),
        'types': place.get('types') or [],
    }


def _maps_search_places(arguments: dict[str, Any]) -> dict[str, Any]:
    query = str(arguments.get('query') or arguments.get('address') or '').strip()
    if not query:
        raise _error(INVALID_PARAMS, 'query is required')
    payload = _request_json(PLACE_TEXTSEARCH_URL, params={'query': query})
    return {'places': [_place_from_result(place) for place in payload.get('results', [])]}


def _maps_geocode(arguments: dict[str, Any]) -> dict[str, Any]:
    address = str(arguments.get('address') or arguments.get('query') or '').strip()
    if not address:
        raise _error(INVALID_PARAMS, 'address is required')
    payload = _request_json(GEOCODE_URL, params={'address': address})
    results = payload.get('results') or []
    if not results:
        raise _error(INVALID_PARAMS, f'Unable to geocode location: {address}')
    result = results[0]
    location = (result.get('geometry') or {}).get('location') or {}
    return {
        'location': {
            'lat': location.get('lat'),
            'lng': location.get('lng'),
        },
        'formatted_address': result.get('formatted_address', ''),
        'place_id': result.get('place_id', ''),
        'name': address,
    }


def _maps_reverse_geocode(arguments: dict[str, Any]) -> dict[str, Any]:
    lat = arguments.get('latitude') if arguments.get('latitude') is not None else arguments.get('lat')
    lng = arguments.get('longitude') if arguments.get('longitude') is not None else arguments.get('lng')
    if lat is None or lng is None:
        raise _error(INVALID_PARAMS, 'latitude/longitude are required')
    payload = _request_json(GEOCODE_URL, params={'latlng': f'{float(lat)},{float(lng)}'})
    return {'results': payload.get('results', [])}


def _maps_place_details(arguments: dict[str, Any]) -> dict[str, Any]:
    place_id = str(arguments.get('place_id') or '').strip()
    if not place_id:
        query = arguments.get('query') or arguments.get('name') or arguments.get('address')
        query_text = str(query or '').strip()
        if not query_text:
            raise _error(INVALID_PARAMS, 'place_id or query is required')
        search_payload = _request_json(PLACE_TEXTSEARCH_URL, params={'query': query_text})
        results = search_payload.get('results') or []
        if not results:
            raise _error(INVALID_PARAMS, f'Unable to resolve place: {query_text}')
        place_id = str(results[0].get('place_id') or '').strip()
        if not place_id:
            return results[0]

    payload = _request_json(
        PLACE_DETAILS_URL,
        params={
            'place_id': place_id,
            'fields': (
                'place_id,name,formatted_address,geometry,opening_hours,international_phone_number,'
                'formatted_phone_number,website,url,rating,types'
            ),
        },
    )
    result = payload.get('result') or {}
    if not result:
        raise _error(INTERNAL_ERROR, f'No place details returned for {place_id}')
    return result


def _maps_distance_matrix(arguments: dict[str, Any]) -> dict[str, Any]:
    origins = arguments.get('origins') or []
    destinations = arguments.get('destinations') or []
    if not origins or not destinations:
        raise _error(INVALID_PARAMS, 'origins and destinations are required')
    payload = _request_json(
        DISTANCE_MATRIX_URL,
        params={
            'origins': '|'.join(_coerce_location_text(item) for item in origins),
            'destinations': '|'.join(_coerce_location_text(item) for item in destinations),
            'mode': str(arguments.get('mode') or 'driving'),
        },
    )
    rows = payload.get('rows') or []
    return {
        'origin_addresses': payload.get('origin_addresses') or [],
        'destination_addresses': payload.get('destination_addresses') or [],
        'results': rows,
        'rows': rows,
        'status': payload.get('status', 'OK'),
    }


def _maps_directions(arguments: dict[str, Any]) -> dict[str, Any]:
    origin = _coerce_location_text(arguments.get('origin') or (arguments.get('origins') or [None])[0])
    destination = _coerce_location_text(arguments.get('destination') or (arguments.get('destinations') or [None])[0])
    payload = _request_json(
        DIRECTIONS_URL,
        params={
            'origin': origin,
            'destination': destination,
            'mode': str(arguments.get('mode') or 'driving'),
            'alternatives': 'false',
        },
    )
    routes = payload.get('routes') or []
    for route in routes:
        if not isinstance(route, dict):
            continue
        if route.get('duration') and route.get('distance'):
            continue
        legs = route.get('legs') or []
        if not isinstance(legs, list) or not legs:
            continue
        total_duration = 0
        total_distance = 0
        distance_text_parts: list[str] = []
        duration_text_parts: list[str] = []
        for leg in legs:
            if not isinstance(leg, dict):
                continue
            duration = leg.get('duration') or {}
            distance = leg.get('distance') or {}
            total_duration += int(duration.get('value') or 0)
            total_distance += int(distance.get('value') or 0)
            if distance.get('text'):
                distance_text_parts.append(str(distance['text']))
            if duration.get('text'):
                duration_text_parts.append(str(duration['text']))
        if total_duration > 0:
            route['duration'] = {
                'value': total_duration,
                'text': duration_text_parts[0] if len(duration_text_parts) == 1 else f'{total_duration // 60} mins',
            }
        if total_distance > 0:
            route['distance'] = {
                'value': total_distance,
                'text': distance_text_parts[0]
                if len(distance_text_parts) == 1
                else (f'{total_distance / 1000:.1f} km' if total_distance >= 1000 else f'{total_distance} m'),
            }
    return payload


def _maps_elevation(arguments: dict[str, Any]) -> dict[str, Any]:
    locations = arguments.get('locations') or []
    if not locations:
        raise _error(INVALID_PARAMS, 'locations is required')
    payload = _request_json(
        ELEVATION_URL,
        params={
            'locations': '|'.join(_coerce_location_text(item) for item in locations),
        },
    )
    return {'results': payload.get('results', []), 'status': payload.get('status', 'OK')}


async def main() -> None:
    server = Server('toolathlon-google-maps-eval-server')
    tools = {
        'maps_search_places': _maps_search_places,
        'maps_geocode': _maps_geocode,
        'maps_reverse_geocode': _maps_reverse_geocode,
        'maps_place_details': _maps_place_details,
        'maps_distance_matrix': _maps_distance_matrix,
        'maps_directions': _maps_directions,
        'maps_elevation': _maps_elevation,
    }

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        _debug_log('list_tools')
        return [
            Tool(
                name=name,
                description=f'Google Maps eval compatibility tool: {name}.',
                inputSchema={'type': 'object', 'properties': {}, 'additionalProperties': True},
            )
            for name in tools
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        _debug_log('call_tool.start', name=name, arguments=arguments or {})
        handler = tools.get(name)
        if handler is None:
            _debug_log('call_tool.unknown', name=name)
            raise _error(INVALID_PARAMS, f'Unknown tool: {name}')
        try:
            payload = handler(arguments or {})
            text = json.dumps(payload, ensure_ascii=False, indent=2)
            _debug_log('call_tool.done', name=name, text_len=len(text), preview=text[:400])
            return [TextContent(type='text', text=text)]
        except McpError:
            _debug_log('call_tool.mcp_error', name=name)
            raise
        except Exception as exc:  # pragma: no cover
            _debug_log('call_tool.error', name=name, error=repr(exc))
            raise _error(INTERNAL_ERROR, f'Google Maps eval server failed: {exc}') from exc

    async with stdio_server() as streams:
        await server.run(streams[0], streams[1], server.create_initialization_options())


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
