"""Rule: orphan_tool_response."""

from .base import Rule


def scan(ctx):
    return list(ctx.orphan_responses)


RULE = Rule('orphan_tool_response', '孤立工具回执', scan)
