"""Rule: punct_flood."""

import re
import unicodedata

from .base import Rule
from ..models import Finding


def scan(ctx):
    out = []
    for i, _ in ctx.assistant_messages():
        content = ctx.content(i)
        cleaned = re.sub(r'```.*?```', '', content, flags=re.S)
        if len(cleaned) < 200:
            continue
        punct = sum(unicodedata.category(ch).startswith(('P', 'S')) for ch in cleaned)
        natural = len(re.findall(r'[A-Za-z\u3400-\u9fff]', cleaned))
        if punct / len(cleaned) >= 0.5 and punct / max(natural, 1) >= 5:
            out.append(Finding(i, '正文被标点或符号大面积淹没', ['content'], content))
    return out


RULE = Rule('punct_flood', '标点或符号刷屏', scan)
