"""Per-tool definitions for yb_one_agent_toolproxy + the yb_tool_registry.

Tool modules are imported at package load time so their
``@yb_tool_registry.register`` decorators run and populate the registry.
"""

# Importing tool modules runs their @yb_tool_registry.register decorators.
from nexus.extensions.projects.yb_search.tools import (  # noqa: F401,E402
    picture_search,
    task_schedule_create_task,
    task_schedule_delete_task,
    task_schedule_list_tasks,
    task_schedule_update_task,
    video_search,
    web_fetch,
    web_search,
)
from nexus.extensions.projects.yb_search.tools.base import YbTool
from nexus.extensions.projects.yb_search.tools.registry import (
    YbToolRegistry,
    yb_tool_registry,
)


__all__ = ['YbTool', 'YbToolRegistry', 'yb_tool_registry']
