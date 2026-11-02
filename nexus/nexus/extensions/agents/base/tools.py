from typing import Literal

from pydantic import BaseModel, Field

from nexus.core import ToolContext, ToolResponse, ToolType, tool_registry
from nexus.core.types import RobustBaseModel
from nexus.extensions.agents.base.logger import logger
from nexus.extensions.agents.base.prompt import FINISH_DESCRIPTION, WRITE_TODOS_DESCRIPTION


class TodoItem(BaseModel):
    content: str
    status: Literal['pending', 'in_progress', 'completed']


class WriteTodosInput(BaseModel):
    todos: list[TodoItem] = Field(description='List of tasks with content and status.')


@tool_registry.register(
    'base/write_todos',
    meta={
        'name': 'write_todos',
        'description': WRITE_TODOS_DESCRIPTION,
        'tool_type': ToolType.LOCAL,
    },
)
async def write_todos_tool(params: WriteTodosInput, context: ToolContext) -> ToolResponse:
    """
    Updates the todo list and derives the next active task.
    """
    todos = params.todos
    logger.debug(f'📝 Overwriting {len(todos)} todos')

    # 1. Determine Active Task Strategy
    if found := next((t for t in todos if t.status == 'in_progress'), None):
        active_todo = found.content
    elif found := next((t for t in todos if t.status == 'pending'), None):
        active_todo = f'(Next Pending) {found.content}'
    else:
        active_todo = 'All tasks completed.' if todos else "No active task. Use 'write_todos' to plan."

    # 2. Prepare Data (Convert objects to dicts once for storage)
    todos_data = [t.model_dump() for t in todos]

    return ToolResponse(
        output=[{'type': 'text', 'text': '\n'.join(f'- [{t.status}] {t.content}' for t in todos)}],
        meta={'todos': todos_data, 'success': True},
        context_updates={
            'todos': todos_data,
            'active_todo': active_todo,
        },
    )


class FinishInput(RobustBaseModel):
    message: str = Field(description='Final message to send to the user')


@tool_registry.register(
    'base/finish',
    meta={
        'name': 'finish',
        'description': FINISH_DESCRIPTION,
        'tool_type': ToolType.LOCAL,
    },
)
async def finish_tool(params: FinishInput, context: ToolContext) -> ToolResponse:
    """Signal task completion with a final message to the user."""
    return ToolResponse(output=[{'type': 'text', 'text': params.message}], meta={'finished': True})
