"""Rule: malformed_tool_call."""

from .base import Rule


def scan(ctx):
    return list(ctx.malformed)


RULE = Rule('malformed_tool_call', '工具调用结构损坏', scan)
