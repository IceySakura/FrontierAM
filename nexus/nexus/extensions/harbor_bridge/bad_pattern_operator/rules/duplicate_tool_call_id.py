"""Rule: duplicate_tool_call_id."""

from .base import Rule


def scan(ctx):
    return list(ctx.duplicate_ids)


RULE = Rule('duplicate_tool_call_id', '重复 tool_call id', scan)
