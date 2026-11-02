"""Rule: tool_call_leak_reasoning."""

from .base import Rule
from ..models import Finding
from ..patterns import TOOL_TOKEN


def scan(ctx):
    out = []
    for i, m in ctx.assistant_messages():
        if isinstance(m.get('tool_calls'), list) and m['tool_calls']:
            continue
        for match in TOOL_TOKEN.finditer(ctx.reasoning(i)):
            if match.group(0).lower().replace(' ', '') in {'<invoke>', '</invoke>'}:
                continue
            out.append(Finding(i, 'tool_call 协议标签泄漏到 reasoning', ['reasoning_content'], match.group(0)))
            break
    return out


RULE = Rule('tool_call_leak_reasoning', 'tool_call 协议泄漏到 reasoning', scan)
