"""Rule: think_tag_tool_call."""

from .base import Rule
from ..models import Finding
from ..patterns import HASHED_THINK_TAG
from ..utils import call_arguments, walk_strings


def scan(ctx):
    out = []
    for e in ctx.calls:
        for path, value in walk_strings(
            call_arguments(e.call), ['tool_calls', e.tool_call_index, 'function', 'arguments']
        ):
            match = HASHED_THINK_TAG.search(value)
            if match:
                out.append(
                    Finding(
                        e.message_index, 'think 标签泄漏到工具参数', path, match.group(0), e.tool_call_index, e.call_id
                    )
                )
    return out


RULE = Rule('think_tag_tool_call', 'think 标签泄漏到工具参数', scan)
