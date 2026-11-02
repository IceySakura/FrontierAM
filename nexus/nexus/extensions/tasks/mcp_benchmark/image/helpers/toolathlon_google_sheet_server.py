#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os

from typing import Any
from urllib.parse import quote

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import McpError
from mcp.types import INTERNAL_ERROR, INVALID_PARAMS, ErrorData, TextContent, Tool
from toolathlon_google_sheets_mock import (
    SpreadsheetNotFound,
    append_sheet_values,
    batch_update_spreadsheet,
    create_spreadsheet,
    get_sheet_values,
    get_spreadsheet,
    spreadsheet_public_url,
    update_sheet_values,
    update_spreadsheet,
)


def _error(code: int, message: str) -> McpError:
    return McpError(ErrorData(code=code, message=message))


def _json_content(payload: Any) -> list[TextContent]:
    return [TextContent(type='text', text=json.dumps(payload, ensure_ascii=False, indent=2))]


def _load_google_credentials_file() -> str:
    configured = os.getenv('GOOGLE_CREDENTIALS_PATH')
    if configured and os.path.exists(configured):
        return configured
    cwd = os.getcwd()
    if os.path.basename(cwd) == 'agent_workspace':
        cwd = os.path.dirname(cwd)
    candidates = (
        os.path.join(cwd, 'configs', 'google_credentials.json'),
        os.path.join(cwd, 'configs', 'credentials.json'),
        os.path.join('/opt/mcp-benchmarks/datasets/toolathlon', 'configs', 'google_credentials.json'),
    )
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise _error(INTERNAL_ERROR, 'Google OAuth credential file not found')


def _google_session():
    """Return an AuthorizedSession that automatically walks HTTP_PROXY.

    httplib2 (used by googleapiclient) does NOT read HTTP_PROXY and its
    ``socks`` module is None in the container image, so proxy_info never
    takes effect.  We bypass httplib2 entirely by using requests-based
    ``AuthorizedSession`` and calling the Google REST APIs directly.
    """
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


def _coerce_values(raw: Any) -> list[list[str]]:
    if raw is None:
        return []
    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith('['):
            decoded = json.loads(text)
            raw = decoded
        else:
            raise _error(INVALID_PARAMS, 'values must be a JSON array')
    if not isinstance(raw, list):
        raise _error(INVALID_PARAMS, 'values must be a list of rows')
    if raw and not isinstance(raw[0], list):
        return [[str(value) for value in raw]]
    return [[str(value) for value in row] for row in raw]


def _range(sheet_name: str | None, range_name: str | None, default: str = 'A1') -> str:
    normalized_range = str(range_name or default).strip()
    if sheet_name:
        if '!' in normalized_range:
            return normalized_range
        return f'{sheet_name}!{normalized_range}'
    return normalized_range


def _drive_folder_id(arguments: dict[str, Any]) -> str:
    folder_id = str(arguments.get('folder_id') or '').strip()
    if folder_id:
        return folder_id
    # Toolathlon's Google Sheet YAML already injects DRIVE_FOLDER_ID for tasks
    # whose eval searches inside a prepared Drive folder. If the agent does not
    # pass folder_id explicitly, keep create_spreadsheet aligned with that
    # environment-level contract instead of silently creating outside the folder.
    return str(os.getenv('DRIVE_FOLDER_ID') or os.getenv('GOOGLE_DRIVE_FOLDER_ID') or '').strip()


def _mock_create_spreadsheet(arguments: dict[str, Any]) -> list[TextContent]:
    title = str(arguments.get('title') or '').strip()
    if not title:
        raise _error(INVALID_PARAMS, 'title is required')
    sheet_names = list(arguments.get('sheet_names') or ['Sheet1'])
    folder_id = _drive_folder_id(arguments)
    parents = [folder_id] if folder_id else []
    spreadsheet = create_spreadsheet(
        title=title,
        parents=parents,
        sheets=[{'title': str(name), 'values': []} for name in sheet_names],
    )
    spreadsheet_id = spreadsheet['spreadsheetId']
    return _json_content({
        'spreadsheet_id': spreadsheet_id,
        'spreadsheet_url': spreadsheet_public_url(spreadsheet_id),
        'spreadsheet': spreadsheet,
    })


def _mock_list_spreadsheets(arguments: dict[str, Any]) -> list[TextContent]:
    from toolathlon_google_forms_mock import list_drive_files

    name_contains = str(arguments.get('name_contains') or '').strip()
    folder_id = str(arguments.get('folder_id') or '').strip()
    query = "mimeType='application/vnd.google-apps.spreadsheet' and trashed=false"
    if name_contains:
        query = f"name contains '{name_contains}' and {query}"
    if folder_id:
        query = f"'{folder_id}' in parents and {query}"
    return _json_content({'spreadsheets': list_drive_files(query=query)})


def _mock_get_spreadsheet(arguments: dict[str, Any]) -> list[TextContent]:
    spreadsheet_id = str(arguments.get('spreadsheet_id') or arguments.get('spreadsheetId') or '').strip()
    if not spreadsheet_id:
        raise _error(INVALID_PARAMS, 'spreadsheet_id is required')
    try:
        spreadsheet = get_spreadsheet(spreadsheet_id)
    except SpreadsheetNotFound as exc:
        raise _error(INVALID_PARAMS, f'Unknown spreadsheet: {spreadsheet_id}') from exc
    return _json_content({
        'spreadsheet_id': spreadsheet_id,
        'spreadsheet_url': spreadsheet_public_url(spreadsheet_id),
        'spreadsheet': spreadsheet,
    })


def _mock_read_sheet(arguments: dict[str, Any]) -> list[TextContent]:
    spreadsheet_id = str(arguments.get('spreadsheet_id') or arguments.get('spreadsheetId') or '').strip()
    if not spreadsheet_id:
        raise _error(INVALID_PARAMS, 'spreadsheet_id is required')
    range_name = _range(arguments.get('sheet_name'), arguments.get('range'), default='A:ZZ')
    try:
        values = get_sheet_values(spreadsheet_id, range_name)
    except SpreadsheetNotFound as exc:
        raise _error(INVALID_PARAMS, f'Unknown spreadsheet: {spreadsheet_id}') from exc
    return _json_content({'spreadsheet_id': spreadsheet_id, 'range': range_name, 'values': values})


def _mock_write_sheet(arguments: dict[str, Any]) -> list[TextContent]:
    spreadsheet_id = str(arguments.get('spreadsheet_id') or arguments.get('spreadsheetId') or '').strip()
    if not spreadsheet_id:
        raise _error(INVALID_PARAMS, 'spreadsheet_id is required')
    range_name = _range(arguments.get('sheet_name'), arguments.get('range'))
    values = _coerce_values(arguments.get('values'))
    result = update_sheet_values(spreadsheet_id, range_name, values)
    return _json_content({'spreadsheet_id': spreadsheet_id, **result})


def _mock_append_rows(arguments: dict[str, Any]) -> list[TextContent]:
    spreadsheet_id = str(arguments.get('spreadsheet_id') or arguments.get('spreadsheetId') or '').strip()
    if not spreadsheet_id:
        raise _error(INVALID_PARAMS, 'spreadsheet_id is required')
    range_name = _range(arguments.get('sheet_name'), arguments.get('range'), default='A:A')
    values = _coerce_values(arguments.get('values'))
    result = append_sheet_values(spreadsheet_id, range_name, values)
    return _json_content({'spreadsheet_id': spreadsheet_id, **result})


def _mock_add_sheet(arguments: dict[str, Any]) -> list[TextContent]:
    spreadsheet_id = str(arguments.get('spreadsheet_id') or arguments.get('spreadsheetId') or '').strip()
    sheet_name = str(arguments.get('sheet_name') or arguments.get('title') or '').strip()
    if not spreadsheet_id or not sheet_name:
        raise _error(INVALID_PARAMS, 'spreadsheet_id and sheet_name are required')
    result = batch_update_spreadsheet(
        spreadsheet_id,
        [{'addSheet': {'properties': {'title': sheet_name}}}],
    )
    return _json_content({'spreadsheet_id': spreadsheet_id, **result})


def _mock_rename_spreadsheet(arguments: dict[str, Any]) -> list[TextContent]:
    spreadsheet_id = str(arguments.get('spreadsheet_id') or arguments.get('spreadsheetId') or '').strip()
    title = str(arguments.get('title') or '').strip()
    if not spreadsheet_id or not title:
        raise _error(INVALID_PARAMS, 'spreadsheet_id and title are required')
    result = update_spreadsheet(spreadsheet_id, {'properties': {'title': title}})
    return _json_content({
        'spreadsheet_id': spreadsheet_id,
        'spreadsheet_url': spreadsheet_public_url(spreadsheet_id),
        'spreadsheet': result,
    })


def _api_create_spreadsheet(arguments: dict[str, Any]) -> list[TextContent]:
    title = str(arguments.get('title') or '').strip()
    if not title:
        raise _error(INVALID_PARAMS, 'title is required')
    sheet_names = list(arguments.get('sheet_names') or ['Sheet1'])
    session = _google_session()
    resp = session.post(
        'https://sheets.googleapis.com/v4/spreadsheets',
        json={
            'properties': {'title': title},
            'sheets': [{'properties': {'title': str(name)}} for name in sheet_names],
        },
    )
    resp.raise_for_status()
    spreadsheet = resp.json()
    spreadsheet_id = spreadsheet['spreadsheetId']
    folder_id = _drive_folder_id(arguments)
    if folder_id:
        move_resp = session.patch(
            f'https://www.googleapis.com/drive/v3/files/{spreadsheet_id}',
            params={'addParents': folder_id, 'fields': 'id, parents'},
        )
        move_resp.raise_for_status()
    return _json_content({
        'spreadsheet_id': spreadsheet_id,
        'spreadsheet_url': spreadsheet.get('spreadsheetUrl') or spreadsheet_public_url(spreadsheet_id),
        'spreadsheet': spreadsheet,
    })


def _api_list_spreadsheets(arguments: dict[str, Any]) -> list[TextContent]:
    session = _google_session()
    query = "mimeType='application/vnd.google-apps.spreadsheet' and trashed=false"
    name_contains = str(arguments.get('name_contains') or '').strip()
    folder_id = str(arguments.get('folder_id') or '').strip()
    if name_contains:
        query = f"name contains '{name_contains}' and {query}"
    if folder_id:
        query = f"'{folder_id}' in parents and {query}"
    resp = session.get(
        'https://www.googleapis.com/drive/v3/files',
        params={'q': query, 'fields': 'files(id, name, mimeType, webViewLink, parents)'},
    )
    resp.raise_for_status()
    files = resp.json().get('files', [])
    return _json_content({'spreadsheets': files})


def _api_get_spreadsheet(arguments: dict[str, Any]) -> list[TextContent]:
    spreadsheet_id = str(arguments.get('spreadsheet_id') or arguments.get('spreadsheetId') or '').strip()
    if not spreadsheet_id:
        raise _error(INVALID_PARAMS, 'spreadsheet_id is required')
    session = _google_session()
    resp = session.get(f'https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}')
    resp.raise_for_status()
    spreadsheet = resp.json()
    return _json_content({
        'spreadsheet_id': spreadsheet_id,
        'spreadsheet_url': spreadsheet.get('spreadsheetUrl'),
        'spreadsheet': spreadsheet,
    })


def _api_read_sheet(arguments: dict[str, Any]) -> list[TextContent]:
    spreadsheet_id = str(arguments.get('spreadsheet_id') or arguments.get('spreadsheetId') or '').strip()
    if not spreadsheet_id:
        raise _error(INVALID_PARAMS, 'spreadsheet_id is required')
    session = _google_session()
    range_name = _range(arguments.get('sheet_name'), arguments.get('range'), default='A:ZZ')
    resp = session.get(
        f'https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}/values/{quote(range_name, safe="")}'
    )
    resp.raise_for_status()
    result = resp.json()
    return _json_content({'spreadsheet_id': spreadsheet_id, 'range': range_name, 'values': result.get('values', [])})


def _api_write_sheet(arguments: dict[str, Any]) -> list[TextContent]:
    spreadsheet_id = str(arguments.get('spreadsheet_id') or arguments.get('spreadsheetId') or '').strip()
    if not spreadsheet_id:
        raise _error(INVALID_PARAMS, 'spreadsheet_id is required')
    session = _google_session()
    range_name = _range(arguments.get('sheet_name'), arguments.get('range'))
    values = _coerce_values(arguments.get('values'))
    resp = session.put(
        f'https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}/values/{quote(range_name, safe="")}',
        params={'valueInputOption': 'USER_ENTERED'},
        json={'values': values},
    )
    resp.raise_for_status()
    result = resp.json()
    return _json_content({'spreadsheet_id': spreadsheet_id, **result})


def _api_append_rows(arguments: dict[str, Any]) -> list[TextContent]:
    spreadsheet_id = str(arguments.get('spreadsheet_id') or arguments.get('spreadsheetId') or '').strip()
    if not spreadsheet_id:
        raise _error(INVALID_PARAMS, 'spreadsheet_id is required')
    session = _google_session()
    range_name = _range(arguments.get('sheet_name'), arguments.get('range'), default='A:A')
    values = _coerce_values(arguments.get('values'))
    resp = session.post(
        f'https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}/values/{quote(range_name, safe="")}:append',
        params={'valueInputOption': 'USER_ENTERED', 'insertDataOption': 'INSERT_ROWS'},
        json={'values': values},
    )
    resp.raise_for_status()
    result = resp.json()
    return _json_content({'spreadsheet_id': spreadsheet_id, **result})


def _api_add_sheet(arguments: dict[str, Any]) -> list[TextContent]:
    spreadsheet_id = str(arguments.get('spreadsheet_id') or arguments.get('spreadsheetId') or '').strip()
    sheet_name = str(arguments.get('sheet_name') or arguments.get('title') or '').strip()
    if not spreadsheet_id or not sheet_name:
        raise _error(INVALID_PARAMS, 'spreadsheet_id and sheet_name are required')
    session = _google_session()
    resp = session.post(
        f'https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}:batchUpdate',
        json={'requests': [{'addSheet': {'properties': {'title': sheet_name}}}]},
    )
    resp.raise_for_status()
    result = resp.json()
    return _json_content({'spreadsheet_id': spreadsheet_id, **result})


def _api_rename_spreadsheet(arguments: dict[str, Any]) -> list[TextContent]:
    spreadsheet_id = str(arguments.get('spreadsheet_id') or arguments.get('spreadsheetId') or '').strip()
    title = str(arguments.get('title') or '').strip()
    if not spreadsheet_id or not title:
        raise _error(INVALID_PARAMS, 'spreadsheet_id and title are required')
    session = _google_session()
    resp = session.post(
        f'https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}:batchUpdate',
        json={'requests': [{'updateSpreadsheetProperties': {'properties': {'title': title}, 'fields': 'title'}}]},
    )
    resp.raise_for_status()
    result = resp.json()
    return _json_content({'spreadsheet_id': spreadsheet_id, **result})


def _tool_definitions() -> list[Tool]:
    return [
        Tool(
            name='create_spreadsheet',
            description='Create a spreadsheet. Optionally pass sheet_names and folder_id.',
            inputSchema={
                'type': 'object',
                'properties': {
                    'title': {'type': 'string'},
                    'sheet_names': {'type': 'array', 'items': {'type': 'string'}},
                    'folder_id': {'type': 'string'},
                },
                'required': ['title'],
                'additionalProperties': False,
            },
        ),
        Tool(
            name='list_spreadsheets',
            description='List spreadsheets. Optionally filter by name_contains or folder_id.',
            inputSchema={
                'type': 'object',
                'properties': {
                    'name_contains': {'type': 'string'},
                    'folder_id': {'type': 'string'},
                },
                'additionalProperties': False,
            },
        ),
        Tool(
            name='get_spreadsheet',
            description='Get spreadsheet metadata.',
            inputSchema={
                'type': 'object',
                'properties': {'spreadsheet_id': {'type': 'string'}},
                'required': ['spreadsheet_id'],
                'additionalProperties': False,
            },
        ),
        Tool(
            name='read_sheet',
            description='Read values from a sheet. Use sheet_name and optional range such as A1:D20.',
            inputSchema={
                'type': 'object',
                'properties': {
                    'spreadsheet_id': {'type': 'string'},
                    'sheet_name': {'type': 'string'},
                    'range': {'type': 'string'},
                },
                'required': ['spreadsheet_id'],
                'additionalProperties': False,
            },
        ),
        Tool(
            name='write_sheet',
            description='Write values to a sheet starting at range or A1.',
            inputSchema={
                'type': 'object',
                'properties': {
                    'spreadsheet_id': {'type': 'string'},
                    'sheet_name': {'type': 'string'},
                    'range': {'type': 'string'},
                    'values': {'type': 'array'},
                },
                'required': ['spreadsheet_id', 'values'],
                'additionalProperties': False,
            },
        ),
        Tool(
            name='append_rows',
            description='Append one or more rows to a sheet.',
            inputSchema={
                'type': 'object',
                'properties': {
                    'spreadsheet_id': {'type': 'string'},
                    'sheet_name': {'type': 'string'},
                    'range': {'type': 'string'},
                    'values': {'type': 'array'},
                },
                'required': ['spreadsheet_id', 'values'],
                'additionalProperties': False,
            },
        ),
        Tool(
            name='add_sheet',
            description='Add a worksheet to an existing spreadsheet.',
            inputSchema={
                'type': 'object',
                'properties': {
                    'spreadsheet_id': {'type': 'string'},
                    'sheet_name': {'type': 'string'},
                },
                'required': ['spreadsheet_id', 'sheet_name'],
                'additionalProperties': False,
            },
        ),
        Tool(
            name='rename_spreadsheet',
            description='Rename an existing spreadsheet.',
            inputSchema={
                'type': 'object',
                'properties': {
                    'spreadsheet_id': {'type': 'string'},
                    'title': {'type': 'string'},
                },
                'required': ['spreadsheet_id', 'title'],
                'additionalProperties': False,
            },
        ),
    ]


def _handlers(mode: str) -> dict[str, Any]:
    if mode == 'api':
        return {
            'create_spreadsheet': _api_create_spreadsheet,
            'list_spreadsheets': _api_list_spreadsheets,
            'get_spreadsheet': _api_get_spreadsheet,
            'read_sheet': _api_read_sheet,
            'write_sheet': _api_write_sheet,
            'append_rows': _api_append_rows,
            'add_sheet': _api_add_sheet,
            'rename_spreadsheet': _api_rename_spreadsheet,
        }
    return {
        'create_spreadsheet': _mock_create_spreadsheet,
        'list_spreadsheets': _mock_list_spreadsheets,
        'get_spreadsheet': _mock_get_spreadsheet,
        'read_sheet': _mock_read_sheet,
        'write_sheet': _mock_write_sheet,
        'append_rows': _mock_append_rows,
        'add_sheet': _mock_add_sheet,
        'rename_spreadsheet': _mock_rename_spreadsheet,
    }


async def _run_server(mode: str) -> None:
    server = Server('toolathlon-google-sheet')
    handlers = _handlers(mode)

    @server.list_tools()
    async def _list_tools() -> list[Tool]:
        return _tool_definitions()

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        handler = handlers.get(name)
        if handler is None:
            raise _error(INVALID_PARAMS, f'Unknown tool: {name}')
        return handler(arguments or {})

    async with stdio_server() as streams:
        await server.run(
            streams[0],
            streams[1],
            server.create_initialization_options(),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('mock', 'api'), default='mock')
    args = parser.parse_args()

    import asyncio

    asyncio.run(_run_server(args.mode))


if __name__ == '__main__':
    main()
