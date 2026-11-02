"""How a Codex turn's items are named in a report.

``backend.steps_of`` reads this module to turn ``item.completed`` events into the
``detail`` shape ``actor.rendering`` reports. The item types and their fields are the
ones ``agents/codex/json_events`` documents against the Codex source.
"""

from collections.abc import Mapping
from typing import Any, cast

from nexus.extensions.tasks.hitl.actor.rendering import detail_from


# Non-speech ``item_type`` values. ``agent_message`` and ``reasoning`` are speech.
TOOL_ITEM_TYPES = frozenset({
    'command_execution',
    'file_change',
    'mcp_tool_call',
    'web_search',
    'collab_tool_call',
    'todo_list',
})

# The field naming what each item was pointed at. An item_type absent here is
# reported by name alone; the documented schema names no fields for ``mcp_tool_call``
# or ``todo_list``.
TOOL_DETAIL_FIELDS: Mapping[str, tuple[str, ...]] = {
    'command_execution': ('command',),
    'web_search': ('query',),
    'collab_tool_call': ('prompt',),
}


def item_detail(item_type: str, item: Mapping[str, object]) -> str:
    """The argument worth naming for this item, or '' when the schema names none."""
    if item_type == 'file_change':
        return ', '.join(_changed_paths(item.get('changes')))
    return detail_from(item, TOOL_DETAIL_FIELDS.get(item_type, ()))


def _changed_paths(changes: object) -> list[str]:
    """The paths a ``file_change`` item names."""
    if not isinstance(changes, (list, tuple)):
        return []
    found: list[str] = []
    for change in changes:
        if isinstance(change, Mapping) and (path := str(cast(Mapping[str, Any], change).get('path') or '')):
            found.append(path)
    return found


__all__ = ['TOOL_DETAIL_FIELDS', 'TOOL_ITEM_TYPES', 'item_detail']
