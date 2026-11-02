"""``task_schedule/list_tasks@v1.0`` tool definition.

Routing & identity flow identical to ``task_schedule/create_task``
(see that module's docstring): the server-side tool reads
``ToolContext.User.UserId`` from the ContextIn channel, no Params
field for ``user_id``.

Schema mirrors ``listTasksSchema`` in
``tool_proxy/toolcalls/task_schedule/list_tasks.go`` (iwiki 4020427234).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from nexus.extensions.projects.yb_search.tools.base import YbTool
from nexus.extensions.projects.yb_search.tools.registry import (
    yb_tool_registry,
)


class _TaskScheduleListTasksParams(BaseModel):
    """``task_schedule/list_tasks`` 参数，字段顺序对齐 iwiki 4020427234。"""

    enabled_only: bool = False
    limit: int = Field(default=30, ge=1, le=100)


@yb_tool_registry.register
class TaskScheduleListTasksTool(YbTool):
    name = 'task_schedule/list_tasks'
    version = 'v1.0'
    description = '列出当前会话中的任务。'
    Params = _TaskScheduleListTasksParams
