"""Rule: mojibake."""

import re

from .base import Rule
from ..models import Finding


def scan(ctx):
    out = []
    for i, _ in ctx.assistant_messages():
        content = ctx.content(i)
        cleaned = re.sub(r'```.*?```', '', content, flags=re.S)
        if len(cleaned) >= 60 and cleaned.count('�') >= 3:
            out.append(Finding(i, '正文中至少出现 3 个 U+FFFD 替换字符', ['content'], content))
    return out


RULE = Rule('mojibake', '编码乱码', scan)
