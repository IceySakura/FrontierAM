"""``task_schedule/delete_task@v1.0`` tool definition.

Routing & identity flow identical to ``task_schedule/create_task``
(see that module's docstring): the server-side tool reads
``ToolContext.User.UserId`` from the ContextIn channel, no Params
field for ``user_id``.

Schema mirrors ``deleteTaskSchema`` in
``tool_proxy/toolcalls/task_schedule/delete_task.go`` (iwiki 4020427234).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from nexus.extensions.projects.yb_search.tools.base import YbTool
from nexus.extensions.projects.yb_search.tools.registry import (
    yb_tool_registry,
)


class _TaskScheduleDeleteTaskParams(BaseModel):
    """``task_schedule/delete_task`` 参数，字段顺序对齐 iwiki 4020427234。"""

    task_id: str = Field(..., min_length=1)


@yb_tool_registry.register
class TaskScheduleDeleteTaskTool(YbTool):
    name = 'task_schedule/delete_task'
    version = 'v1.0'
    description = (
        '删除任务：task_id 仅作工具参数，禁止在对用户回复中写出；'
        '可来自同轮或近期 create/list/update 的工具返回（如用户删刚建的任务），'
        '或 list_tasks 后与 title/instruction 唯一对齐。'
        '无可靠 id 或有多条可能时先 list_tasks 再确认。'
    )
    Params = _TaskScheduleDeleteTaskParams
