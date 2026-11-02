#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os

from typing import TYPE_CHECKING, Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import McpError
from mcp.types import INTERNAL_ERROR, INVALID_PARAMS, ErrorData, TextContent, Tool
from toolathlon_google_forms_mock import (
    batch_update_form,
    create_form,
    delete_drive_file,
    form_edit_url,
    form_public_url,
    get_form,
    list_drive_files,
)


if TYPE_CHECKING:
    from google.auth.transport.requests import AuthorizedSession


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
        os.path.join('/opt/mcp-benchmarks/datasets/toolathlon', 'configs', 'google_credentials.json'),
    )
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise _error(INTERNAL_ERROR, 'Google OAuth credential file not found')


def _google_session() -> AuthorizedSession:
    """Return an AuthorizedSession that automatically walks HTTP_PROXY.

    httplib2 (used by googleapiclient) does NOT read HTTP_PROXY and its
    socks module is None in the container, so we bypass it entirely.
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


def _question_request(question: dict[str, Any], *, index: int) -> dict[str, Any]:
    title = str(question.get('title') or '').strip()
    if not title:
        raise _error(INVALID_PARAMS, 'question.title is required')
    question_type = str(question.get('type') or 'text').strip().lower()
    required = bool(question.get('required', False))
    payload: dict[str, Any] = {'required': required}
    if question_type == 'text':
        payload['textQuestion'] = {}
    elif question_type == 'paragraph':
        payload['textQuestion'] = {'paragraph': True}
    elif question_type in {'radio', 'checkbox'}:
        payload['choiceQuestion'] = {
            'type': 'CHECKBOX' if question_type == 'checkbox' else 'RADIO',
            'options': [{'value': str(value)} for value in list(question.get('options') or [])],
        }
    elif question_type == 'date':
        payload['dateQuestion'] = {'includeTime': False, 'includeYear': True}
    else:
        raise _error(INVALID_PARAMS, f'Unsupported question type: {question_type}')
    return {
        'createItem': {
            'item': {
                'title': title,
                'description': str(question.get('description') or ''),
                'questionItem': {'question': payload},
            },
            'location': {'index': index},
        }
    }


def _mock_create_form(arguments: dict[str, Any]) -> list[TextContent]:
    title = str(arguments.get('title') or '').strip()
    if not title:
        raise _error(INVALID_PARAMS, 'title is required')
    description = str(arguments.get('description') or '')
    form = create_form(title=title, description=description)
    questions = list(arguments.get('questions') or [])
    if questions:
        batch_update_form(
            form['formId'], [_question_request(question, index=index) for index, question in enumerate(questions)]
        )
        form = get_form(form['formId'])
    return _json_content({
        'form_id': form['formId'],
        'form_url': form_public_url(form['formId']),
        'edit_url': form_edit_url(form['formId']),
        'form': form,
    })


def _mock_add_question(arguments: dict[str, Any]) -> list[TextContent]:
    form_id = str(arguments.get('form_id') or arguments.get('formId') or '').strip()
    if not form_id:
        raise _error(INVALID_PARAMS, 'form_id is required')
    question = dict(arguments.get('question') or {})
    requests = [_question_request(question, index=int(arguments.get('index', len(get_form(form_id).get('items', [])))))]
    form = batch_update_form(form_id, requests)
    return _json_content({'form_id': form_id, 'form': form})


def _mock_list_forms(arguments: dict[str, Any]) -> list[TextContent]:
    name_contains = str(arguments.get('name_contains') or '').strip()
    query = "mimeType='application/vnd.google-apps.form'"
    if name_contains:
        query = f"name contains '{name_contains}' and {query}"
    files = list_drive_files(query=query)
    return _json_content({'forms': files})


def _mock_get_form(arguments: dict[str, Any]) -> list[TextContent]:
    form_id = str(arguments.get('form_id') or arguments.get('formId') or '').strip()
    if not form_id:
        raise _error(INVALID_PARAMS, 'form_id is required')
    form = get_form(form_id)
    return _json_content({
        'form_id': form_id,
        'form_url': form_public_url(form_id),
        'edit_url': form_edit_url(form_id),
        'form': form,
    })


def _mock_delete_form(arguments: dict[str, Any]) -> list[TextContent]:
    form_id = str(arguments.get('form_id') or arguments.get('formId') or '').strip()
    if not form_id:
        raise _error(INVALID_PARAMS, 'form_id is required')
    delete_drive_file(form_id)
    return _json_content({'deleted': True, 'form_id': form_id})


def _api_create_form(arguments: dict[str, Any]) -> list[TextContent]:
    title = str(arguments.get('title') or '').strip()
    if not title:
        raise _error(INVALID_PARAMS, 'title is required')
    session = _google_session()
    resp = session.post(
        'https://forms.googleapis.com/v1/forms',
        json={'info': {'title': title, 'documentTitle': title}},
    )
    resp.raise_for_status()
    form = resp.json()
    questions = list(arguments.get('questions') or [])
    if questions:
        session.post(
            f'https://forms.googleapis.com/v1/forms/{form["formId"]}:batchUpdate',
            json={'requests': [_question_request(question, index=index) for index, question in enumerate(questions)]},
        ).raise_for_status()
        form = session.get(f'https://forms.googleapis.com/v1/forms/{form["formId"]}').json()
    return _json_content({
        'form_id': form['formId'],
        'form_url': f'https://docs.google.com/forms/d/{form["formId"]}/viewform',
        'edit_url': f'https://docs.google.com/forms/d/{form["formId"]}/edit',
        'form': form,
    })


def _api_add_question(arguments: dict[str, Any]) -> list[TextContent]:
    form_id = str(arguments.get('form_id') or arguments.get('formId') or '').strip()
    if not form_id:
        raise _error(INVALID_PARAMS, 'form_id is required')
    session = _google_session()
    question = dict(arguments.get('question') or {})
    session.post(
        f'https://forms.googleapis.com/v1/forms/{form_id}:batchUpdate',
        json={'requests': [_question_request(question, index=int(arguments.get('index', 0)))]},
    ).raise_for_status()
    form = session.get(f'https://forms.googleapis.com/v1/forms/{form_id}').json()
    return _json_content({'form_id': form_id, 'form': form})


def _api_list_forms(arguments: dict[str, Any]) -> list[TextContent]:
    session = _google_session()
    query = "mimeType='application/vnd.google-apps.form'"
    name_contains = str(arguments.get('name_contains') or '').strip()
    if name_contains:
        query = f"name contains '{name_contains}' and {query}"
    resp = session.get(
        'https://www.googleapis.com/drive/v3/files',
        params={'q': query, 'fields': 'files(id, name, mimeType, webViewLink)'},
    )
    resp.raise_for_status()
    files = resp.json().get('files', [])
    return _json_content({'forms': files})


def _api_get_form(arguments: dict[str, Any]) -> list[TextContent]:
    form_id = str(arguments.get('form_id') or arguments.get('formId') or '').strip()
    if not form_id:
        raise _error(INVALID_PARAMS, 'form_id is required')
    session = _google_session()
    resp = session.get(f'https://forms.googleapis.com/v1/forms/{form_id}')
    resp.raise_for_status()
    form = resp.json()
    return _json_content({
        'form_id': form_id,
        'form_url': f'https://docs.google.com/forms/d/{form_id}/viewform',
        'edit_url': f'https://docs.google.com/forms/d/{form_id}/edit',
        'form': form,
    })


def _api_delete_form(arguments: dict[str, Any]) -> list[TextContent]:
    form_id = str(arguments.get('form_id') or arguments.get('formId') or '').strip()
    if not form_id:
        raise _error(INVALID_PARAMS, 'form_id is required')
    session = _google_session()
    resp = session.delete(f'https://www.googleapis.com/drive/v3/files/{form_id}')
    resp.raise_for_status()
    return _json_content({'deleted': True, 'form_id': form_id})


def _tool_definitions() -> list[Tool]:
    return [
        Tool(
            name='create_form',
            description='Create a Google Form. Optionally include a questions array with text/paragraph/radio/checkbox/date items.',
            inputSchema={
                'type': 'object',
                'properties': {
                    'title': {'type': 'string'},
                    'description': {'type': 'string'},
                    'questions': {'type': 'array'},
                },
                'required': ['title'],
                'additionalProperties': True,
            },
        ),
        Tool(
            name='add_question',
            description='Append a question to an existing form.',
            inputSchema={
                'type': 'object',
                'properties': {
                    'form_id': {'type': 'string'},
                    'question': {'type': 'object'},
                    'index': {'type': 'integer'},
                },
                'required': ['form_id', 'question'],
                'additionalProperties': True,
            },
        ),
        Tool(
            name='list_forms',
            description='List existing Google Forms.',
            inputSchema={
                'type': 'object',
                'properties': {
                    'name_contains': {'type': 'string'},
                },
                'additionalProperties': False,
            },
        ),
        Tool(
            name='get_form',
            description='Get the full structure and URLs of a form.',
            inputSchema={
                'type': 'object',
                'properties': {
                    'form_id': {'type': 'string'},
                },
                'required': ['form_id'],
                'additionalProperties': False,
            },
        ),
        Tool(
            name='delete_form',
            description='Delete a form by id.',
            inputSchema={
                'type': 'object',
                'properties': {
                    'form_id': {'type': 'string'},
                },
                'required': ['form_id'],
                'additionalProperties': False,
            },
        ),
    ]


async def _serve(mode: str) -> None:
    server = Server(f'toolathlon-google-forms-{mode}')

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return _tool_definitions()

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        args = arguments or {}
        if mode == 'mock':
            handlers = {
                'create_form': _mock_create_form,
                'add_question': _mock_add_question,
                'list_forms': _mock_list_forms,
                'get_form': _mock_get_form,
                'delete_form': _mock_delete_form,
            }
        else:
            handlers = {
                'create_form': _api_create_form,
                'add_question': _api_add_question,
                'list_forms': _api_list_forms,
                'get_form': _api_get_form,
                'delete_form': _api_delete_form,
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
