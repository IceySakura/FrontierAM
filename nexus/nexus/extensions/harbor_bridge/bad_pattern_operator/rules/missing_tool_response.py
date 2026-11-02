"""Rule: missing_tool_response."""

from .base import Rule


def scan(ctx):
    return list(ctx.missing_responses)


RULE = Rule('missing_tool_response', '工具调用缺少回执', scan)
