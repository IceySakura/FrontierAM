"""``task_schedule/update_task@v1.0`` tool definition.

Routing & identity flow identical to ``task_schedule/create_task``
(see that module's docstring): the server-side tool reads
``ToolContext.User.UserId`` from the ContextIn channel, no Params
field for ``user_id``.

Schema mirrors ``updateTaskSchema`` in
``tool_proxy/toolcalls/task_schedule/update_task.go`` (iwiki 4020427234).

Optional field semantics:
    Server-side ``update_task`` does partial update — only fields the
    LLM actually passes are applied; unset fields keep their stored
    value. The server distinguishes "not passed" from "passed false"
    via ``_, hasEnabled := pre.args["enabled"]`` (字段存在性), so
    optional fields here default to ``None``. The dispatcher must dump
    Params with ``exclude_unset=True`` to avoid leaking these defaults
    into ``arguments`` (otherwise every update would toggle
    enable/disable). See ``tool_proxy_dispatcher._build_req``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from nexus.extensions.projects.yb_search.tools.base import YbTool
from nexus.extensions.projects.yb_search.tools.registry import (
    yb_tool_registry,
)


class _TaskScheduleUpdateTaskParams(BaseModel):
    """``task_schedule/update_task`` 参数，字段顺序对齐 iwiki 4020427234。"""

    task_id: str = Field(..., min_length=1)
    instruction: str | None = None
    timezone: str = Field(default='Asia/Shanghai', min_length=1)
    task_kind: Literal['once', 'rrule'] | None = None
    title: str | None = Field(default=None, min_length=1)
    run_at: str | None = None
    rrule: str | None = None
    dtstart: str | None = None
    enabled: bool | None = None


@yb_tool_registry.register
class TaskScheduleUpdateTaskTool(YbTool):
    name = 'task_schedule/update_task'
    version = 'v1.0'
    description = (
        '更新已有任务：只传需要修改的字段，未传的字段自动保留原值。task_id 必填，'
        '其余字段按需传入。task_id 仅作工具参数，禁止在对用户回复中写出。'
    )
    Params = _TaskScheduleUpdateTaskParams
