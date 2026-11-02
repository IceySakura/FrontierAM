#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os

from datetime import datetime, timezone
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import McpError
from mcp.types import INTERNAL_ERROR, INVALID_PARAMS, ErrorData, TextContent, Tool


try:
    from .toolathlon_google_store import mutate_google_state, read_google_state
    from .toolathlon_mock_utils import utc_now_iso
except ImportError:
    from toolathlon_google_store import mutate_google_state, read_google_state
    from toolathlon_mock_utils import utc_now_iso


def _error(code: int, message: str) -> McpError:
    return McpError(ErrorData(code=code, message=message))


def _json_content(payload: Any) -> list[TextContent]:
    return [TextContent(type='text', text=json.dumps(payload, ensure_ascii=False, indent=2))]


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith('Z'):
        text = text[:-1] + '+00:00'
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise _error(INVALID_PARAMS, f'invalid datetime: {value}') from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _event_start_sort_key(event: dict[str, Any]) -> tuple[int, str]:
    start = event.get('start') or {}
    if isinstance(start, dict):
        if start.get('dateTime'):
            return (0, str(start['dateTime']))
        if start.get('date'):
            return (1, str(start['date']))
    return (2, '')


def _event_in_range(event: dict[str, Any], time_min: datetime | None, time_max: datetime | None) -> bool:
    start = event.get('start') or {}
    start_value = None
    if isinstance(start, dict):
        start_value = start.get('dateTime') or start.get('date')
    parsed = _parse_datetime(start_value) if start_value else None
    if parsed is None:
        return True
    if time_min is not None and parsed < time_min:
        return False
    return not (time_max is not None and parsed > time_max)


def _mock_list_events(arguments: dict[str, Any]) -> list[TextContent]:
    state = read_google_state()
    time_min = _parse_datetime(arguments.get('timeMin'))
    time_max = _parse_datetime(arguments.get('timeMax'))
    max_results = int(arguments.get('maxResults') or 250)
    events = list(state['calendar']['events'].values())
    events = [event for event in events if _event_in_range(event, time_min, time_max)]
    if str(arguments.get('orderBy') or '') == 'startTime':
        events.sort(key=_event_start_sort_key)
    limited = events[:max_results]
    return [
        TextContent(
            type='text', text=f'Found {len(limited)} events:\n{json.dumps(limited, ensure_ascii=False, indent=2)}'
        )
    ]


def _mock_create_event(arguments: dict[str, Any]) -> list[TextContent]:
    summary = str(arguments.get('summary') or '').strip()
    start = arguments.get('start')
    end = arguments.get('end')
    if not summary:
        raise _error(INVALID_PARAMS, 'summary is required')
    if not isinstance(start, dict) or not isinstance(end, dict):
        raise _error(INVALID_PARAMS, 'start and end are required')

    def mutator(state: dict[str, Any]) -> dict[str, Any]:
        event_id = f'evt-{state["next_ids"]["calendar_event"]}'
        state['next_ids']['calendar_event'] = int(state['next_ids']['calendar_event']) + 1
        event = {
            'id': event_id,
            'status': 'confirmed',
            'summary': summary,
            'description': arguments.get('description', ''),
            'location': arguments.get('location', ''),
            'start': start,
            'end': end,
            'created': utc_now_iso(),
            'updated': utc_now_iso(),
            'htmlLink': f'https://calendar.mock.local/events/{event_id}',
        }
        state['calendar']['events'][event_id] = event
        return event

    event = mutate_google_state(mutator)
    return _json_content({'message': 'event created', 'event': event})


def _mock_delete_event(arguments: dict[str, Any]) -> list[TextContent]:
    event_id = str(arguments.get('eventId') or '').strip()
    if not event_id:
        raise _error(INVALID_PARAMS, 'eventId is required')

    def mutator(state: dict[str, Any]) -> dict[str, Any]:
        event = state['calendar']['events'].pop(event_id, None)
        if event is None:
            raise _error(INVALID_PARAMS, f'event not found: {event_id}')
        return event

    deleted = mutate_google_state(mutator)
    return _json_content({'message': 'event deleted', 'event': deleted})


def _mock_update_event(arguments: dict[str, Any]) -> list[TextContent]:
    event_id = str(arguments.get('eventId') or '').strip()
    if not event_id:
        raise _error(INVALID_PARAMS, 'eventId is required')

    def mutator(state: dict[str, Any]) -> dict[str, Any]:
        event = state['calendar']['events'].get(event_id)
        if event is None:
            raise _error(INVALID_PARAMS, f'event not found: {event_id}')
        for key in ('summary', 'description', 'location', 'start', 'end', 'status'):
            if key in arguments:
                event[key] = arguments[key]
        event['updated'] = utc_now_iso()
        return event

    updated = mutate_google_state(mutator)
    return _json_content({'message': 'event updated', 'event': updated})


def _load_google_credentials_file() -> str:
    configured = os.getenv('GOOGLE_CREDENTIALS_PATH')
    if configured and os.path.exists(configured):
        return configured
    case_root = os.getcwd()
    if os.path.basename(case_root) == 'agent_workspace':
        case_root = os.path.dirname(case_root)
    candidates = (
        os.path.join(case_root, 'configs', 'google_credentials.json'),
        os.path.join('/opt/mcp-benchmarks/datasets/toolathlon', 'configs', 'google_credentials.json'),
    )
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise _error(INTERNAL_ERROR, 'Google OAuth credential file not found')


def _google_session():
    """Return an AuthorizedSession that automatically walks HTTP_PROXY."""
    from google.auth.transport.requests import AuthorizedSession, Request
    from google.oauth2.credentials import Credentials

    credentials_path = _load_google_credentials_file()
    with open(credentials_path, encoding='utf-8') as handle:
        data = json.load(handle)
    creds = Credentials(
        token=data.get('token'),
        refresh_token=data.get('refresh_token'),
        token_uri=data.get('token_uri'),
        client_id=data.get('client_id'),
        client_secret=data.get('client_secret'),
        scopes=data.get('scopes'),
    )
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
    return AuthorizedSession(creds)


def _api_list_events(arguments: dict[str, Any]) -> list[TextContent]:
    session = _google_session()
    params: dict[str, Any] = {
        'calendarId': 'primary',
        'singleEvents': True,
        'maxResults': int(arguments.get('maxResults') or 250),
    }
    if arguments.get('timeMin'):
        params['timeMin'] = arguments['timeMin']
    if arguments.get('timeMax'):
        params['timeMax'] = arguments['timeMax']
    if arguments.get('orderBy'):
        params['orderBy'] = arguments['orderBy']
    resp = session.get('https://www.googleapis.com/calendar/v3/calendars/primary/events', params=params)
    resp.raise_for_status()
    items = resp.json().get('items', [])
    return [
        TextContent(type='text', text=f'Found {len(items)} events:\n{json.dumps(items, ensure_ascii=False, indent=2)}')
    ]


def _api_create_event(arguments: dict[str, Any]) -> list[TextContent]:
    session = _google_session()
    resp = session.post('https://www.googleapis.com/calendar/v3/calendars/primary/events', json=arguments)
    resp.raise_for_status()
    event = resp.json()
    return _json_content({'message': 'event created', 'event': event})


def _api_delete_event(arguments: dict[str, Any]) -> list[TextContent]:
    event_id = str(arguments.get('eventId') or '').strip()
    if not event_id:
        raise _error(INVALID_PARAMS, 'eventId is required')
    session = _google_session()
    resp = session.delete(f'https://www.googleapis.com/calendar/v3/calendars/primary/events/{event_id}')
    resp.raise_for_status()
    return _json_content({'message': 'event deleted', 'eventId': event_id})


def _api_update_event(arguments: dict[str, Any]) -> list[TextContent]:
    event_id = str(arguments.get('eventId') or '').strip()
    if not event_id:
        raise _error(INVALID_PARAMS, 'eventId is required')
    session = _google_session()
    body = {key: value for key, value in arguments.items() if key != 'eventId'}
    resp = session.patch(f'https://www.googleapis.com/calendar/v3/calendars/primary/events/{event_id}', json=body)
    resp.raise_for_status()
    event = resp.json()
    return _json_content({'message': 'event updated', 'event': event})


def _tool_definitions() -> list[Tool]:
    return [
        Tool(
            name='list_events',
            description='List calendar events between timeMin and timeMax.',
            inputSchema={
                'type': 'object',
                'properties': {
                    'timeMin': {'type': 'string', 'description': 'Lower bound in ISO-8601 format.'},
                    'timeMax': {'type': 'string', 'description': 'Upper bound in ISO-8601 format.'},
                    'maxResults': {'type': 'integer', 'description': 'Maximum number of events to return.'},
                    'orderBy': {'type': 'string', 'description': 'Sort order, e.g. startTime.'},
                },
                'additionalProperties': False,
            },
        ),
        Tool(
            name='create_event',
            description='Create a calendar event with Google Calendar style payload fields.',
            inputSchema={
                'type': 'object',
                'properties': {
                    'summary': {'type': 'string'},
                    'description': {'type': 'string'},
                    'location': {'type': 'string'},
                    'start': {'type': 'object'},
                    'end': {'type': 'object'},
                },
                'required': ['summary', 'start', 'end'],
                'additionalProperties': True,
            },
        ),
        Tool(
            name='delete_event',
            description='Delete an event by eventId.',
            inputSchema={
                'type': 'object',
                'properties': {
                    'eventId': {'type': 'string'},
                },
                'required': ['eventId'],
                'additionalProperties': False,
            },
        ),
        Tool(
            name='update_event',
            description='Patch an existing event by eventId.',
            inputSchema={
                'type': 'object',
                'properties': {
                    'eventId': {'type': 'string'},
                    'summary': {'type': 'string'},
                    'description': {'type': 'string'},
                    'location': {'type': 'string'},
                    'start': {'type': 'object'},
                    'end': {'type': 'object'},
                    'status': {'type': 'string'},
                },
                'required': ['eventId'],
                'additionalProperties': True,
            },
        ),
    ]


async def _serve(mode: str) -> None:
    server = Server(f'toolathlon-google-calendar-{mode}')

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return _tool_definitions()

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        args = arguments or {}
        if mode == 'mock':
            handlers = {
                'list_events': _mock_list_events,
                'create_event': _mock_create_event,
                'delete_event': _mock_delete_event,
                'update_event': _mock_update_event,
            }
        else:
            handlers = {
                'list_events': _api_list_events,
                'create_event': _api_create_event,
                'delete_event': _api_delete_event,
                'update_event': _api_update_event,
            }
        handler = handlers.get(name)
        if handler is None:
            raise _error(INVALID_PARAMS, f'Unknown tool: {name}')
        return handler(args)

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('mock', 'api'), default='mock')
    args = parser.parse_args()

    import asyncio

    asyncio.run(_serve(args.mode))


if __name__ == '__main__':
    main()
