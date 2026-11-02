"""How a Claude Code turn's tool calls are named in a report.

Both Claude Code backends record ``{'type': 'tool_use', 'name', 'input'}`` entries --
the print backend from the stream-json ``assistant`` events, the PTY one from the
proxy's ``recent_events`` -- and both read this module to turn them into the
``detail`` shape ``actor.rendering`` reports.
"""

from collections.abc import Mapping, Sequence
from typing import Any, cast

from nexus.extensions.tasks.hitl.actor.rendering import detail_from


# The argument worth naming for each tool: what it was pointed at, not how it was
# configured. A tool absent here is reported by name alone, which is what an
# unrecognised tool from an MCP server gets.
TOOL_DETAIL_FIELDS: Mapping[str, tuple[str, ...]] = {
    'Bash': ('command',),
    'Read': ('file_path',),
    'Write': ('file_path',),
    'Edit': ('file_path',),
    'NotebookEdit': ('notebook_path',),
    'Glob': ('pattern',),
    'Grep': ('pattern',),
    'Agent': ('description',),
    'Skill': ('skill',),
    'WebFetch': ('url',),
    'WebSearch': ('query',),
    'TodoWrite': (),
}


def steps_with_detail(entries: object) -> list[dict[str, Any]]:
    """``entries`` with each ``tool_use``'s ``input`` reduced to a ``detail`` string."""
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        return []
    out: list[dict[str, Any]] = []
    for raw_entry in entries:
        if not isinstance(raw_entry, Mapping):
            continue
        entry = cast(Mapping[str, Any], raw_entry)
        if entry.get('type') != 'tool_use':
            out.append(dict(entry))
            continue
        name = str(entry.get('name') or '')
        out.append({
            'type': 'tool_use',
            'name': name,
            'detail': detail_from(entry.get('input'), TOOL_DETAIL_FIELDS.get(name, ())),
        })
    return out


__all__ = ['TOOL_DETAIL_FIELDS', 'steps_with_detail']
