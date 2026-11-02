#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import McpError
from mcp.types import INTERNAL_ERROR, INVALID_PARAMS, ErrorData, TextContent, Tool


TASK_ID_RE = re.compile(r'tid_(\d+)\.blend$', re.IGNORECASE)


def _error(code: int, message: str) -> McpError:
    return McpError(ErrorData(code=code, message=message))


def _json_content(payload: Any) -> list[TextContent]:
    return [TextContent(type='text', text=json.dumps(payload, ensure_ascii=False, indent=2))]


def _extract_task_id(output_path: str, instructions: str) -> str:
    match = TASK_ID_RE.search(output_path)
    if match:
        return match.group(1)
    match = re.search(r'tid_(\d+)\.blend', instructions, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    raise _error(INVALID_PARAMS, 'output_path must include a filename like tid_1.blend')


def _save_blend_file(arguments: dict[str, Any]) -> dict[str, Any]:
    output_path = str((arguments or {}).get('output_path') or '').strip()
    instructions = str((arguments or {}).get('instructions') or '').strip()
    if not output_path:
        raise _error(INVALID_PARAMS, 'output_path is required')
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    task_id = _extract_task_id(output_path, instructions)
    payload = {
        'format': 'nexus-blender-mock',
        'task_id': task_id,
        'output_path': str(output),
        'saved_at': datetime.now(timezone.utc).isoformat(),
        'instructions': instructions,
        'cwd': os.getcwd(),
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return {
        'status': 'ok',
        'task_id': task_id,
        'output_path': str(output),
        'bytes_written': output.stat().st_size,
    }


async def main() -> None:
    server = Server('universe-blender-server')

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name='save_blend_file',
                description=(
                    'Construct the requested benchmark Blender scene and save a mock .blend file. '
                    'Pass the exact output_path from the task prompt. In mock mode this single tool '
                    'handles the full scene generation workflow.'
                ),
                inputSchema={
                    'type': 'object',
                    'properties': {
                        'output_path': {'type': 'string', 'description': 'Absolute path to the target .blend file.'},
                        'instructions': {
                            'type': 'string',
                            'description': 'Optional full task instructions for traceability.',
                        },
                    },
                    'required': ['output_path'],
                    'additionalProperties': False,
                },
            )
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        if name != 'save_blend_file':
            raise _error(INVALID_PARAMS, f'Unknown tool: {name}')
        try:
            return _json_content(_save_blend_file(arguments or {}))
        except McpError:
            raise
        except Exception as exc:
            raise _error(INTERNAL_ERROR, str(exc)) from exc

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
