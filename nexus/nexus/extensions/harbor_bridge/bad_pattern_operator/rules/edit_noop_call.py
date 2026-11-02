"""Rule: edit_noop_call."""

from .base import Rule
from ..models import Finding


def scan(ctx):
    out = []
    for e in ctx.calls:
        if (
            e.name.casefold() not in {'edit', 'str_replace', 'filesystem_edit_file', 'apply_patch'}
            or e.arguments is None
        ):
            continue
        old = e.arguments.get('old_string', e.arguments.get('old_str'))
        new = e.arguments.get('new_string', e.arguments.get('new_str'))
        if isinstance(old, str) and isinstance(new, str) and old == new:
            out.append(
                Finding(
                    e.message_index,
                    '编辑调用的 old_string 与 new_string 完全相同',
                    ['tool_calls', e.tool_call_index, 'function', 'arguments'],
                    e.arguments,
                    e.tool_call_index,
                    e.call_id,
                )
            )
    return out


RULE = Rule('edit_noop_call', 'Edit 空改动调用', scan)
