#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re

from collections.abc import Iterable
from typing import Any
from urllib.parse import parse_qs, quote_plus, urljoin, urlparse

import requests

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import McpError
from mcp.types import INTERNAL_ERROR, INVALID_PARAMS, ErrorData, TextContent, Tool


BASE_URL = 'https://www.youtube.com'
USER_AGENT = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36'
YT_API_KEY_RE = re.compile(r'"INNERTUBE_API_KEY":"([^"]+)"')
YT_CLIENT_VERSION_RE = re.compile(r'"INNERTUBE_CONTEXT_CLIENT_VERSION":"([^"]+)"')


def _error(code: int, message: str) -> McpError:
    return McpError(ErrorData(code=code, message=message))


def _json_content(payload: Any) -> list[TextContent]:
    return [TextContent(type='text', text=json.dumps(payload, ensure_ascii=False, indent=2))]


def _headers() -> dict[str, str]:
    return {
        'User-Agent': USER_AGENT,
        'Accept-Language': 'en-US,en;q=0.9',
    }


def _fetch_html(url: str) -> str:
    response = requests.get(url, headers=_headers(), timeout=60)
    response.raise_for_status()
    return response.text


def _extract_js_object(html: str, marker: str, *, label: str) -> dict[str, Any]:
    start = html.find(marker)
    if start < 0:
        raise _error(INTERNAL_ERROR, f'Could not locate {label} marker on YouTube page')
    start = html.find('{', start)
    if start < 0:
        raise _error(INTERNAL_ERROR, f'Could not locate {label} JSON payload on YouTube page')
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(html)):
        char = html[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                return json.loads(html[start : index + 1])
    raise _error(INTERNAL_ERROR, f'Could not parse complete {label} payload from YouTube page')


def _extract_api_key(html: str) -> str:
    match = YT_API_KEY_RE.search(html)
    if match is None:
        raise _error(INTERNAL_ERROR, 'Could not extract YouTube API key')
    return match.group(1)


def _extract_client_version(html: str) -> str:
    match = YT_CLIENT_VERSION_RE.search(html)
    if match is None:
        raise _error(INTERNAL_ERROR, 'Could not extract YouTube client version')
    return match.group(1)


def _text(value: Any) -> str:
    if not isinstance(value, dict):
        return ''
    simple = value.get('simpleText')
    if simple:
        return str(simple)
    runs = value.get('runs') or []
    return ''.join(str(item.get('text') or '') for item in runs)


def _thumbnail_url(renderer: dict[str, Any]) -> str | None:
    items = ((renderer.get('thumbnail') or {}).get('thumbnails')) or []
    if not items:
        return None
    return str(items[-1].get('url') or '')


def _published_text(renderer: dict[str, Any]) -> str:
    for key in ('publishedTimeText', 'publishedTimeText', 'videoInfo'):
        value = renderer.get(key)
        text = _text(value)
        if text:
            return text
    return ''


def _description_text(renderer: dict[str, Any]) -> str:
    for key in ('descriptionSnippet', 'detailedMetadataSnippets'):
        value = renderer.get(key)
        if isinstance(value, dict):
            text = _text(value)
            if text:
                return text
        if isinstance(value, list):
            for item in value:
                text = _text(item.get('snippetText') or item)
                if text:
                    return text
    return ''


def _duration_label_to_seconds(label: str) -> int:
    total = 0
    for amount, unit in re.findall(r'(\d+)\s+(hour|hours|minute|minutes|second|seconds)', label):
        value = int(amount)
        if unit.startswith('hour'):
            total += value * 3600
        elif unit.startswith('minute'):
            total += value * 60
        else:
            total += value
    return total


def _clock_to_seconds(text: str) -> int:
    parts = [int(part) for part in text.split(':') if part.isdigit()]
    if not parts:
        return 0
    total = 0
    for value in parts:
        total = total * 60 + value
    return total


def _video_from_renderer(renderer: dict[str, Any]) -> dict[str, Any]:
    video_id = str(renderer.get('videoId') or '')
    title = _text(renderer.get('title'))
    owner_text = _text(renderer.get('ownerText') or renderer.get('longBylineText') or renderer.get('shortBylineText'))
    duration_text = _text(renderer.get('lengthText'))
    return {
        'video_id': video_id,
        'title': title,
        'url': f'{BASE_URL}/watch?v={video_id}' if video_id else '',
        'duration_text': duration_text,
        'duration_seconds': _clock_to_seconds(duration_text),
        'published_text': _published_text(renderer),
        'views_text': _text(renderer.get('viewCountText')),
        'channel': owner_text,
        'description': _description_text(renderer),
        'thumbnail': _thumbnail_url(renderer),
    }


def _channel_from_renderer(renderer: dict[str, Any]) -> dict[str, Any]:
    navigation = renderer.get('navigationEndpoint') or (
        ((renderer.get('title') or {}).get('runs') or [{}])[0].get('navigationEndpoint') or {}
    )
    web_meta = (navigation.get('commandMetadata') or {}).get('webCommandMetadata') or {}
    path = str(web_meta.get('url') or '')
    return {
        'channel_id': str(renderer.get('channelId') or ''),
        'title': _text(renderer.get('title')),
        'url': urljoin(BASE_URL, path) if path else '',
        'description': _text(renderer.get('descriptionSnippet')),
        'videos_text': _text(renderer.get('videoCountText')),
        'subscribers_text': _text(renderer.get('subscriberCountText')),
    }


def _iter_values(payload: Any, key: str) -> Iterable[dict[str, Any]]:
    if isinstance(payload, dict):
        for current_key, value in payload.items():
            if current_key == key and isinstance(value, dict):
                yield value
            yield from _iter_values(value, key)
    elif isinstance(payload, list):
        for item in payload:
            yield from _iter_values(item, key)


def _continuation_tokens(payload: Any) -> list[str]:
    tokens: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key == 'continuationEndpoint' and isinstance(value, dict):
                token = ((value.get('continuationCommand') or {}).get('token')) or ''
                if token:
                    tokens.append(str(token))
            tokens.extend(_continuation_tokens(value))
    elif isinstance(payload, list):
        for item in payload:
            tokens.extend(_continuation_tokens(item))
    return tokens


def _browse_continuation(api_key: str, client_version: str, token: str) -> dict[str, Any]:
    response = requests.post(
        f'{BASE_URL}/youtubei/v1/browse?key={api_key}',
        headers={**_headers(), 'Content-Type': 'application/json'},
        json={
            'context': {
                'client': {
                    'clientName': 'WEB',
                    'clientVersion': client_version,
                    'hl': 'en',
                    'gl': 'US',
                }
            },
            'continuation': token,
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def _collect_paginated_items(
    *,
    initial_payload: dict[str, Any],
    api_key: str,
    client_version: str,
    renderer_key: str,
    limit: int,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = list(_iter_values(initial_payload, renderer_key))
    seen_tokens: set[str] = set()
    pending = list(dict.fromkeys(_continuation_tokens(initial_payload)))
    while pending and len(items) < limit:
        token = pending.pop(0)
        if token in seen_tokens:
            continue
        seen_tokens.add(token)
        payload = _browse_continuation(api_key, client_version, token)
        items.extend(_iter_values(payload, renderer_key))
        for next_token in _continuation_tokens(payload):
            if next_token not in seen_tokens:
                pending.append(next_token)
    return items[:limit]


def _first_value(payload: Any, key: str) -> dict[str, Any] | None:
    for value in _iter_values(payload, key):
        return value
    return None


def _normalize_channel_url(channel: str) -> str:
    normalized = str(channel or '').strip()
    if not normalized:
        raise _error(INVALID_PARAMS, 'channel is required')
    if normalized.startswith('http://') or normalized.startswith('https://'):
        return normalized
    if normalized.startswith('@'):
        return f'{BASE_URL}/{normalized}'
    results = search_youtube(query=normalized, result_type='channel', limit=5)
    for result in results:
        title = str(result.get('title') or '').lower()
        if title == normalized.lower():
            return str(result.get('url') or '')
    if results:
        return str(results[0].get('url') or '')
    raise _error(INVALID_PARAMS, f'Could not resolve YouTube channel: {channel}')


def search_youtube(*, query: str, result_type: str = 'video', limit: int = 10) -> list[dict[str, Any]]:
    if not query:
        raise _error(INVALID_PARAMS, 'query is required')
    filter_map = {'video': 'EgIQAQ%3D%3D', 'channel': 'EgIQAg%3D%3D', 'playlist': 'EgIQAw%3D%3D'}
    if result_type not in filter_map:
        raise _error(INVALID_PARAMS, f'Unsupported result_type: {result_type}')
    url = f'{BASE_URL}/results?search_query={quote_plus(query)}&sp={filter_map[result_type]}&hl=en&gl=US'
    html = _fetch_html(url)
    data = _extract_js_object(html, 'var ytInitialData = ', label='ytInitialData')
    results: list[dict[str, Any]] = []
    if result_type == 'channel':
        for renderer in _iter_values(data, 'channelRenderer'):
            results.append(_channel_from_renderer(renderer))
    elif result_type == 'playlist':
        for renderer in _iter_values(data, 'playlistRenderer'):
            playlist_id = str(renderer.get('playlistId') or '')
            results.append({
                'playlist_id': playlist_id,
                'title': _text(renderer.get('title')),
                'url': f'{BASE_URL}/playlist?list={playlist_id}' if playlist_id else '',
                'channel': _text(renderer.get('shortBylineText')),
                'video_count_text': _text(renderer.get('videoCount')),
            })
    else:
        for renderer in _iter_values(data, 'videoRenderer'):
            results.append(_video_from_renderer(renderer))
    return results[:limit]


def get_playlist_videos(
    *, playlist_url: str | None = None, playlist_id: str | None = None, limit: int = 200
) -> list[dict[str, Any]]:
    resolved_id = str(playlist_id or '').strip()
    if not resolved_id:
        if not playlist_url:
            raise _error(INVALID_PARAMS, 'playlist_url or playlist_id is required')
        parsed = urlparse(str(playlist_url))
        resolved_id = parse_qs(parsed.query).get('list', [''])[0]
        if not resolved_id:
            raise _error(INVALID_PARAMS, f'Could not parse playlist id from URL: {playlist_url}')
    url = f'{BASE_URL}/playlist?list={resolved_id}&hl=en&gl=US'
    html = _fetch_html(url)
    data = _extract_js_object(html, 'var ytInitialData = ', label='ytInitialData')
    api_key = _extract_api_key(html)
    client_version = _extract_client_version(html)
    renderers = _collect_paginated_items(
        initial_payload=data,
        api_key=api_key,
        client_version=client_version,
        renderer_key='playlistVideoRenderer',
        limit=limit,
    )
    items: list[dict[str, Any]] = []
    for renderer in renderers:
        video_id = str(renderer.get('videoId') or '')
        duration_text = _text(renderer.get('lengthText'))
        items.append({
            'video_id': video_id,
            'title': _text(renderer.get('title')),
            'url': f'{BASE_URL}/watch?v={video_id}&list={resolved_id}' if video_id else '',
            'duration_text': duration_text,
            'duration_seconds': _clock_to_seconds(duration_text),
            'channel': _text(renderer.get('shortBylineText')),
            'index_text': _text(renderer.get('index')),
        })
    return items


def get_channel_videos(*, channel: str, limit: int = 120) -> list[dict[str, Any]]:
    url = _normalize_channel_url(channel)
    if '/videos' not in url:
        url = url.rstrip('/') + '/videos'
    url = url + '&hl=en&gl=US' if '?' in url else url + '?hl=en&gl=US'
    html = _fetch_html(url)
    data = _extract_js_object(html, 'var ytInitialData = ', label='ytInitialData')
    api_key = _extract_api_key(html)
    client_version = _extract_client_version(html)
    renderers = _collect_paginated_items(
        initial_payload=data,
        api_key=api_key,
        client_version=client_version,
        renderer_key='videoRenderer',
        limit=limit,
    )
    return [_video_from_renderer(renderer) for renderer in renderers]


def get_video_details(*, video_id: str | None = None, video_url: str | None = None) -> dict[str, Any]:
    resolved_id = str(video_id or '').strip()
    if not resolved_id:
        if not video_url:
            raise _error(INVALID_PARAMS, 'video_id or video_url is required')
        parsed = urlparse(str(video_url))
        if parsed.netloc.endswith('youtu.be'):
            resolved_id = parsed.path.strip('/')
        else:
            resolved_id = parse_qs(parsed.query).get('v', [''])[0]
        if not resolved_id:
            raise _error(INVALID_PARAMS, f'Could not parse video id from URL: {video_url}')
    html = _fetch_html(f'{BASE_URL}/watch?v={resolved_id}&hl=en&gl=US')
    player = _extract_js_object(html, 'var ytInitialPlayerResponse = ', label='ytInitialPlayerResponse')
    initial_data = _extract_js_object(html, 'var ytInitialData = ', label='ytInitialData')
    details = player.get('videoDetails') or {}
    microformat = ((player.get('microformat') or {}).get('playerMicroformatRenderer')) or {}
    header = _first_value(initial_data, 'videoDescriptionHeaderRenderer') or {}
    duration_match = re.search(
        r'"lengthText":\{"accessibility":\{"accessibilityData":\{"label":"([^"]+)"',
        html,
    )
    duration_label = duration_match.group(1) if duration_match else ''
    author = details.get('author') or microformat.get('ownerChannelName') or _text(header.get('channel'))
    title = str(details.get('title') or _text(header.get('title')) or '')
    publish_date = str(microformat.get('publishDate') or _text(header.get('publishDate')) or '')
    view_count_text = _text(header.get('views'))
    view_count = int(details.get('viewCount') or 0)
    if not view_count and view_count_text:
        digits = re.sub(r'[^0-9]', '', view_count_text)
        if digits:
            view_count = int(digits)
    duration_seconds = int(details.get('lengthSeconds') or 0)
    if not duration_seconds and duration_label:
        duration_seconds = _duration_label_to_seconds(duration_label)
    return {
        'video_id': resolved_id,
        'title': title,
        'url': f'{BASE_URL}/watch?v={resolved_id}',
        'channel': str(author or ''),
        'channel_id': str(details.get('channelId') or microformat.get('externalChannelId') or ''),
        'description': str(details.get('shortDescription') or ''),
        'duration_seconds': duration_seconds,
        'duration_text': duration_label,
        'published_date': publish_date,
        'upload_date': str(microformat.get('uploadDate') or ''),
        'view_count': view_count,
        'keywords': list(details.get('keywords') or []),
    }


def _tool_definitions() -> list[Tool]:
    return [
        Tool(
            name='search_youtube',
            description='Search YouTube videos, channels, or playlists using public search results.',
            inputSchema={
                'type': 'object',
                'properties': {
                    'query': {'type': 'string'},
                    'result_type': {'type': 'string', 'enum': ['video', 'channel', 'playlist']},
                    'limit': {'type': 'integer'},
                },
                'required': ['query'],
                'additionalProperties': False,
            },
        ),
        Tool(
            name='get_channel_videos',
            description='List videos from a YouTube channel or handle. Accepts channel URL, @handle, or channel name.',
            inputSchema={
                'type': 'object',
                'properties': {
                    'channel': {'type': 'string'},
                    'limit': {'type': 'integer'},
                },
                'required': ['channel'],
                'additionalProperties': False,
            },
        ),
        Tool(
            name='get_playlist_videos',
            description='List videos from a YouTube playlist URL or playlist id.',
            inputSchema={
                'type': 'object',
                'properties': {
                    'playlist_url': {'type': 'string'},
                    'playlist_id': {'type': 'string'},
                    'limit': {'type': 'integer'},
                },
                'additionalProperties': False,
            },
        ),
        Tool(
            name='get_video_details',
            description='Get exact metadata for a video, including published_date and duration_seconds.',
            inputSchema={
                'type': 'object',
                'properties': {
                    'video_id': {'type': 'string'},
                    'video_url': {'type': 'string'},
                },
                'additionalProperties': False,
            },
        ),
    ]


def _handle_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    try:
        if name == 'search_youtube':
            payload = search_youtube(
                query=str(arguments.get('query') or '').strip(),
                result_type=str(arguments.get('result_type') or 'video').strip(),
                limit=int(arguments.get('limit') or 10),
            )
        elif name == 'get_channel_videos':
            payload = get_channel_videos(
                channel=str(arguments.get('channel') or '').strip(),
                limit=int(arguments.get('limit') or 120),
            )
        elif name == 'get_playlist_videos':
            payload = get_playlist_videos(
                playlist_url=arguments.get('playlist_url'),
                playlist_id=arguments.get('playlist_id'),
                limit=int(arguments.get('limit') or 200),
            )
        elif name == 'get_video_details':
            payload = get_video_details(
                video_id=arguments.get('video_id'),
                video_url=arguments.get('video_url'),
            )
        else:
            raise _error(INVALID_PARAMS, f'Unknown tool: {name}')
        return _json_content(payload)
    except McpError:
        raise
    except requests.RequestException as exc:
        raise _error(INTERNAL_ERROR, f'YouTube request failed: {exc}') from exc
    except Exception as exc:
        raise _error(INTERNAL_ERROR, str(exc)) from exc


async def _run_server() -> None:
    server = Server('toolathlon-youtube')

    @server.list_tools()
    async def _list_tools() -> list[Tool]:
        return _tool_definitions()

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        return _handle_tool(name, arguments or {})

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('mock', 'api'), default='mock')
    parser.parse_args()

    import asyncio

    asyncio.run(_run_server())


if __name__ == '__main__':
    main()
