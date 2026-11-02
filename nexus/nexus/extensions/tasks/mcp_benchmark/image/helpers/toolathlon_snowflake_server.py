#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import McpError
from mcp.types import INVALID_PARAMS, ErrorData, TextContent, Tool
from toolathlon_snowflake_mock import execute_query as mock_execute_query
from toolathlon_snowflake_mock import fetch_all_dict as mock_fetch_all_dict


def _error(code: int, message: str) -> McpError:
    return McpError(ErrorData(code=code, message=message))


def _json_content(payload: Any) -> list[TextContent]:
    return [TextContent(type='text', text=json.dumps(payload, ensure_ascii=False, indent=2))]


def _api_helpers():
    case_root = Path.cwd()
    if case_root.name == 'agent_workspace':
        case_root = case_root.parent
    candidates = [
        case_root,
        Path('/opt/mcp-benchmarks/datasets/toolathlon'),
    ]
    for candidate in candidates:
        candidate_text = str(candidate)
        if candidate_text not in sys.path:
            sys.path.insert(0, candidate_text)

    from utils.app_specific.snowflake.client import execute_query, fetch_all_dict

    return execute_query, fetch_all_dict


def _run_query(query: str, *, mode: str, write: bool) -> dict[str, Any]:
    if not query.strip():
        raise _error(INVALID_PARAMS, 'query is required')
    execute_query, fetch_all_dict = (mock_execute_query, mock_fetch_all_dict) if mode == 'mock' else _api_helpers()
    if write:
        rowcount = execute_query(query)
        return {'ok': True, 'rowcount': rowcount, 'query': query}
    rows = fetch_all_dict(query)
    return {'ok': True, 'rowcount': len(rows), 'rows': rows, 'query': query}


def _create_databases(arguments: dict[str, Any], *, mode: str) -> list[TextContent]:
    names = [str(item).strip() for item in list(arguments.get('databases') or []) if str(item).strip()]
    if not names:
        raise _error(INVALID_PARAMS, 'databases is required')
    for name in names:
        _run_query(f'CREATE DATABASE IF NOT EXISTS {name}', mode=mode, write=True)
    return _json_content({'ok': True, 'databases': names})


def _drop_databases(arguments: dict[str, Any], *, mode: str) -> list[TextContent]:
    names = [str(item).strip() for item in list(arguments.get('databases') or []) if str(item).strip()]
    if not names:
        raise _error(INVALID_PARAMS, 'databases is required')
    for name in names:
        _run_query(f'DROP DATABASE IF EXISTS {name}', mode=mode, write=True)
    return _json_content({'ok': True, 'databases': names})


def _read_query(arguments: dict[str, Any], *, mode: str) -> list[TextContent]:
    return _json_content(_run_query(str(arguments.get('query') or ''), mode=mode, write=False))


def _write_query(arguments: dict[str, Any], *, mode: str) -> list[TextContent]:
    return _json_content(_run_query(str(arguments.get('query') or ''), mode=mode, write=True))


def _create_table(arguments: dict[str, Any], *, mode: str) -> list[TextContent]:
    return _json_content(_run_query(str(arguments.get('query') or ''), mode=mode, write=True))


def _tool_definitions() -> list[Tool]:
    return [
        Tool(
            name='create_databases',
            description='Create one or more Snowflake databases.',
            inputSchema={
                'type': 'object',
                'properties': {'databases': {'type': 'array', 'items': {'type': 'string'}}},
                'required': ['databases'],
                'additionalProperties': False,
            },
        ),
        Tool(
            name='drop_databases',
            description='Drop one or more Snowflake databases.',
            inputSchema={
                'type': 'object',
                'properties': {'databases': {'type': 'array', 'items': {'type': 'string'}}},
                'required': ['databases'],
                'additionalProperties': False,
            },
        ),
        Tool(
            name='create_table',
            description='Execute DDL such as CREATE TABLE.',
            inputSchema={
                'type': 'object',
                'properties': {'query': {'type': 'string'}},
                'required': ['query'],
                'additionalProperties': False,
            },
        ),
        Tool(
            name='read_query',
            description='Execute a SELECT query and return rows.',
            inputSchema={
                'type': 'object',
                'properties': {'query': {'type': 'string'}},
                'required': ['query'],
                'additionalProperties': False,
            },
        ),
        Tool(
            name='write_query',
            description='Execute INSERT, UPDATE, DELETE, COMMENT, or other write SQL.',
            inputSchema={
                'type': 'object',
                'properties': {'query': {'type': 'string'}},
                'required': ['query'],
                'additionalProperties': False,
            },
        ),
    ]


async def _run_server(mode: str) -> None:
    server = Server('toolathlon-snowflake')

    handlers = {
        'create_databases': lambda args: _create_databases(args, mode=mode),
        'drop_databases': lambda args: _drop_databases(args, mode=mode),
        'create_table': lambda args: _create_table(args, mode=mode),
        'read_query': lambda args: _read_query(args, mode=mode),
        'write_query': lambda args: _write_query(args, mode=mode),
    }

    @server.list_tools()
    async def _list_tools() -> list[Tool]:
        return _tool_definitions()

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        handler = handlers.get(name)
        if handler is None:
            raise _error(INVALID_PARAMS, f'Unknown tool: {name}')
        return handler(arguments or {})

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('mock', 'api'), default='mock')
    args = parser.parse_args()

    import asyncio

    asyncio.run(_run_server(args.mode))


if __name__ == '__main__':
    main()
