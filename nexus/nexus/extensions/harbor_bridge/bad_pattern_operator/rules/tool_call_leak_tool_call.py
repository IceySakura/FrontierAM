"""Rule: tool_call_leak_tool_call."""

from .base import Rule
from ..models import Finding
from ..patterns import TOOL_TOKEN
from ..utils import call_arguments, walk_strings


def scan(ctx):
    out = []
    for e in ctx.calls:
        for path, value in walk_strings(
            call_arguments(e.call), ['tool_calls', e.tool_call_index, 'function', 'arguments']
        ):
            match = TOOL_TOKEN.search(value)
            if match:
                out.append(
                    Finding(
                        e.message_index,
                        'tool_call 协议标签嵌套泄漏到工具参数',
                        path,
                        match.group(0),
                        e.tool_call_index,
                        e.call_id,
                    )
                )
    return out


RULE = Rule('tool_call_leak_tool_call', 'tool_call 协议嵌套泄漏到参数', scan)
