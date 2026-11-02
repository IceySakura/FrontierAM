"""Rule: think_tag_content."""

from .base import Rule
from ..models import Finding
from ..patterns import THINK_TAG


def scan(ctx):
    out = []
    for i, _ in ctx.assistant_messages():
        match = THINK_TAG.search(ctx.content(i))
        if match:
            out.append(Finding(i, 'think 标签泄漏到正文', ['content'], match.group(0)))
    return out


RULE = Rule('think_tag_content', 'think 标签泄漏到正文', scan)
