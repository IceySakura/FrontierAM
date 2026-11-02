"""Rule: tool_call_leak_content."""

import re

from .base import Rule
from ..models import Finding
from ..patterns import TOOL_CONTENT_TOKEN


def scan(ctx):
    out = []
    for i, _ in ctx.assistant_messages():
        content = ctx.content(i)
        for match in TOOL_CONTENT_TOKEN.finditer(content):
            token = match.group(0)
            if token.lower().replace(' ', '') in {'<invoke>', '</invoke>'} and re.search(
                r'`[^`\n]*' + re.escape(token) + r'[^`\n]*`', content
            ):
                continue
            out.append(Finding(i, 'tool_call 协议标签泄漏到正文', ['content'], token))
            break
    return out


RULE = Rule('tool_call_leak_content', 'tool_call 协议泄漏到正文', scan)
