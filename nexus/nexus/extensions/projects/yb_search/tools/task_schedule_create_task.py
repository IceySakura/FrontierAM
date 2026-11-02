"""``task_schedule/create_task@v1.0`` tool definition.

Routing & identity:
    Server-side ``task_schedule/*`` tools extract the caller via
    ``ToolContext.User.UserId`` (see ``toolcalls/task_schedule/format.go``
    in tool_proxy). The dispatcher already injects
    ``tool_proxy_context_in.user.user_id`` from YAML into
    ``context_in.user.user_id``, so this tool does **not**
    expose ``user_id`` as a Params field — that would shadow the
    ContextIn channel and is rejected by the server schema
    (``additionalProperties: false``).

Schema mirrors ``createTaskSchema`` in
``tool_proxy/toolcalls/task_schedule/create_task.go`` (iwiki 4020427234).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from nexus.extensions.projects.yb_search.tools.base import YbTool
from nexus.extensions.projects.yb_search.tools.registry import (
    yb_tool_registry,
)


class _TaskScheduleCreateTaskParams(BaseModel):
    """``task_schedule/create_task`` 参数，字段顺序对齐 iwiki 4020427234。"""

    instruction: str
    timezone: str = Field(default='Asia/Shanghai', min_length=1)
    task_kind: Literal['once', 'rrule']
    title: str = Field(..., min_length=1)
    run_at: str
    rrule: str
    dtstart: str
    enabled: bool


@yb_tool_registry.register
class TaskScheduleCreateTaskTool(YbTool):
    name = 'task_schedule/create_task'
    version = 'v1.0'
    description = (
        '创建定时任务。task_kind=once 时使用 run_at；task_kind=rrule 时使用 rrule 与 dtstart。'
        '未使用的时间字段填空字符串。title 非空。rrule 中不要带 RRULE: 前缀，'
        '不要包含 BYHOUR/BYMINUTE（时分秒由 dtstart 承载）。时间字段用紧凑格式 YYYYMMDDTHHMMSS。'
        'timezone 必填，默认 Asia/Shanghai。'
    )
    Params = _TaskScheduleCreateTaskParams
