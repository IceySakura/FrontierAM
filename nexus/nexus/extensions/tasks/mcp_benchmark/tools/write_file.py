"""
local-write_file tool: write content to a file in the agent workspace.
"""

import logging
import os

from pydantic import Field

from nexus.core import ToolContext, ToolResponse, ToolType, require_context, tool_registry
from nexus.core.types import RobustBaseModel


logger = logging.getLogger(__name__)

WRITE_FILE_WORKSPACE_KEY = '_write_file_workspace'


class WriteFileInput(RobustBaseModel):
    path: str = Field(description='File path relative to the workspace directory (e.g. "result.json")')
    content: str = Field(description='Content to write to the file')


@tool_registry.register(
    'local-write_file',
    meta={
        'name': 'local-write_file',
        'description': (
            'Write content to a file in your workspace directory. '
            'The path should be relative to the workspace directory.'
        ),
        'tool_type': ToolType.LOCAL,
    },
)
async def write_file_tool(params: WriteFileInput, context: ToolContext) -> ToolResponse:
    agent_workspace = context.tool_state.get(WRITE_FILE_WORKSPACE_KEY)
    if not agent_workspace:
        return ToolResponse(
            output=[{'type': 'text', 'text': 'Error: agent workspace not configured.'}],
            meta={'success': False},
        )

    ctx = require_context()
    runtime = await ctx.get_runtime()

    file_path = os.path.join(agent_workspace, params.path)
    dir_path = os.path.dirname(file_path)

    from nexus.runtime.executors.file import FileItem

    await runtime.run_command(command=f'mkdir -p {dir_path}', timeout=10)
    result = await runtime.upload_file([FileItem(path=file_path, content=params.content, encoding='utf8')])
    upload_result = result.results[0]
    if not upload_result.success:
        return ToolResponse(
            output=[{'type': 'text', 'text': f'Error writing file: {upload_result.message}'}],
            meta={'success': False},
        )

    return ToolResponse(
        output=[{'type': 'text', 'text': f'Successfully wrote to {params.path}'}],
        meta={'success': True},
    )
