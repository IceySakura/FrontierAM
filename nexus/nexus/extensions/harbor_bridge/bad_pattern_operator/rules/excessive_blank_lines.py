"""Rule: excessive_blank_lines."""

import re

from .base import Rule
from ..models import Finding


def scan(ctx):
    out = []
    for i, _ in ctx.assistant_messages():
        content = ctx.content(i)
        if re.search(r'^\n{8,}|\n{12,}|\n{8,}$', content):
            out.append(Finding(i, '正文出现连续大量空行', ['content'], content))
    return out


RULE = Rule('excessive_blank_lines', '大段空行异常', scan)
