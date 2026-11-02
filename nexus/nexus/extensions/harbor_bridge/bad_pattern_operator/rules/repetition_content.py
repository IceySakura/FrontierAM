"""Rule: repetition_content."""

import re

from collections import Counter

from .base import Rule
from ..models import Finding


def scan(ctx):
    out = []
    for i, _ in ctx.assistant_messages():
        content = ctx.content(i)
        blocks = [re.sub(r'\s+', ' ', x).strip() for x in re.split(r'\n\s*\n', content)]
        counts = Counter(x for x in blocks if len(x) >= 40)
        if any(v >= 3 for v in counts.values()):
            out.append(Finding(i, '同一有效段落在正文中重复至少 3 次', ['content'], content))
    return out


RULE = Rule('repetition_content', '正文复读', scan)
